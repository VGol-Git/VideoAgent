"""Download verified inference assets without Windows symlink privileges.

Reuses complete snapshot files and Hugging Face blobs. Metadata pins a commit;
files are checked against official Git-blob SHA-1 or LFS SHA-256 hashes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "huggingface"
SUFFIXES = {".json", ".safetensors", ".txt", ".jinja", ".bin"}
DOWNLOAD_ATTEMPTS = 3
CHUNK_BYTES = 1024 * 1024
PROGRESS_BYTES = 32 * 1024 * 1024
PROGRESS_SECONDS = 15


def digest(path: Path, size: int, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    if algorithm == "sha1":
        hasher.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verified(path: Path, size: int, algorithm: str, expected_hash: str) -> bool:
    return path.is_file() and path.stat().st_size == size and digest(path, size, algorithm) == expected_hash


def _promote(temporary: Path, target: Path, size: int, algorithm: str, expected_hash: str):
    if not _verified(temporary, size, algorithm, expected_hash):
        raise ValueError(f"Integrity verification failed for {target.name}; partial file retained")
    # A concurrently completed valid target is retained rather than overwritten.
    if target.exists():
        if not _verified(target, size, algorithm, expected_hash):
            raise ValueError(f"Existing {target.name} is invalid; refusing to overwrite")
        return
    temporary.replace(target)


def _download_verified(url: str, target: Path, size: int, algorithm: str, expected_hash: str):
    """Resume one pinned asset, preserving partial bytes across three attempts.

    Range response bounds are checked before appending. A server ignoring Range
    must return a valid full response, which replaces rather than extends the
    partial file. A complete file is promoted only after size and official hash
    verification. Protocol/hash failures are explicit and keep their evidence.
    """
    temporary = target.with_name(target.name + ".download-part")
    if temporary.exists() and temporary.stat().st_size > size:
        raise ValueError(f"Partial {target.name} exceeds expected size; refusing to overwrite")
    if temporary.exists() and temporary.stat().st_size == size:
        _promote(temporary, target, size, algorithm, expected_hash)
        return
    last_error = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        offset = temporary.stat().st_size if temporary.exists() else 0
        print(f"DOWNLOAD_START {target.name} attempt={attempt}/{DOWNLOAD_ATTEMPTS} retained={offset}/{size}", flush=True)
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = response.status
                if response.headers.get("Content-Encoding", "identity").lower() not in ("", "identity"):
                    raise ValueError("Compressed download response cannot preserve byte offsets")
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match:
                        raise ValueError("Partial download requires a valid Content-Range")
                    begin, finish, total = map(int, match.groups())
                    if begin != offset or total != size or not begin <= finish < total:
                        raise ValueError("Content-Range does not match requested offset and expected file size")
                    response_bytes, mode = finish - begin + 1, "ab" if offset else "wb"
                elif status == 200:
                    # An HTTP server may ignore Range; appending a full response
                    # would corrupt the existing prefix.
                    offset, response_bytes, mode = 0, size, "wb"
                else:
                    raise ValueError(f"Unexpected asset download status: {status}")
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None and (not declared_length.isdigit() or int(declared_length) != response_bytes):
                    raise ValueError("Content-Length disagrees with expected response byte range")
                received = 0
                last_progress_bytes, last_progress_time = 0, time.monotonic()
                with temporary.open(mode) as stream:
                    while True:
                        incomplete = None
                        try:
                            chunk = response.read(min(CHUNK_BYTES, response_bytes - received + 1))
                        except http.client.IncompleteRead as exc:
                            chunk, incomplete = exc.partial, exc
                        if received + len(chunk) > response_bytes:
                            raise ValueError("Download body exceeds the verified response byte range")
                        if chunk:
                            stream.write(chunk)
                            received += len(chunk)
                            now = time.monotonic()
                            if received - last_progress_bytes >= PROGRESS_BYTES or now - last_progress_time >= PROGRESS_SECONDS:
                                print(f"DOWNLOAD_PROGRESS {target.name} downloaded={offset + received}/{size}", flush=True)
                                last_progress_bytes, last_progress_time = received, now
                        if incomplete is not None:
                            raise incomplete
                        if not chunk:
                            break
                if received != response_bytes or temporary.stat().st_size != size:
                    last_error = OSError(f"Incomplete download: {temporary.stat().st_size}/{size} bytes")
                else:
                    _promote(temporary, target, size, algorithm, expected_hash)
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead,
                http.client.RemoteDisconnected) as exc:
            last_error = exc
        # Some streams signal truncation after delivering the final bytes.
        # A hash check, rather than a further Range request beyond EOF, decides.
        if temporary.exists() and temporary.stat().st_size == size:
            _promote(temporary, target, size, algorithm, expected_hash)
            return
        retained = temporary.stat().st_size if temporary.exists() else 0
        print(f"DOWNLOAD_RETRY {target.name} attempt={attempt}/{DOWNLOAD_ATTEMPTS} retained={retained}/{size}", flush=True)
    raise OSError(f"Download incomplete after {DOWNLOAD_ATTEMPTS} attempts for {target.name}; partial file retained") from last_error


def download(repo: str, revision: str = "main", output: Path | None = None) -> Path:
    metadata_url = f"https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}?blobs=true"
    with urllib.request.urlopen(metadata_url, timeout=15) as response:
        metadata = json.load(response)
    commit = metadata["sha"]
    repository_cache = CACHE / ("models--" + repo.replace("/", "--"))
    destination = (output or repository_cache / "snapshots" / commit).resolve()
    if not destination.is_relative_to((ROOT / ".cache").resolve()):
        raise ValueError("Model output must stay inside the project's .cache directory")
    destination.mkdir(parents=True, exist_ok=True)
    files = [item for item in metadata["siblings"] if Path(item["rfilename"]).suffix in SUFFIXES]
    manifest = {"repo": repo, "revision": commit, "files": []}

    def prepare(item: dict) -> dict:
        filename = item["rfilename"]
        target = (destination / filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Invalid model filename: {filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        lfs = item.get("lfs")
        expected_size = item["size"]
        algorithm, expected_hash = ("sha256", lfs["sha256"]) if lfs else ("sha1", item["blobId"])
        source = "existing"
        if not target.exists():
            blob = repository_cache / "blobs" / expected_hash
            if _verified(blob, expected_size, algorithm, expected_hash):
                temporary = target.with_name(target.name + ".download-part")
                shutil.copy2(blob, temporary)
                _promote(temporary, target, expected_size, algorithm, expected_hash)
                source = "cached_blob"
            else:
                source = "download"
                url = f"https://huggingface.co/{repo}/resolve/{commit}/{urllib.parse.quote(filename, safe='/')}"
                _download_verified(url, target, expected_size, algorithm, expected_hash)
        if target.stat().st_size != expected_size:
            raise ValueError(f"Existing {filename} has unexpected size; refusing to redownload or overwrite")
        if digest(target, expected_size, algorithm) != expected_hash:
            raise ValueError(f"Existing {filename} has incorrect {algorithm}; refusing to redownload or overwrite")
        print(f"VERIFIED {filename} {expected_size} bytes ({source})", flush=True)
        return {"filename": filename, "size": expected_size, "algorithm": algorithm, "digest": expected_hash}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest["files"] = list(pool.map(prepare, files))
    (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if output is None:
        reference = repository_cache / "refs" / revision
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(commit, encoding="utf-8")
    print(f"MODEL_READY {destination}", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    download(args.repo, args.revision, args.output)
