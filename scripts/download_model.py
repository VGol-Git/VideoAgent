"""Download verified inference assets without Windows symlink privileges.

Reuses complete snapshot files and Hugging Face blobs. Metadata pins a commit;
files are checked against official Git-blob SHA-1 or LFS SHA-256 hashes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "huggingface"
SUFFIXES = {".json", ".safetensors", ".txt", ".jinja", ".bin"}


def digest(path: Path, size: int, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    if algorithm == "sha1":
        hasher.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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
            if blob.is_file() and blob.stat().st_size == expected_size:
                shutil.copy2(blob, target)
                source = "cached_blob"
            else:
                source = "download"
                temporary = target.with_name(target.name + ".download-part")
                url = f"https://huggingface.co/{repo}/resolve/{commit}/{urllib.parse.quote(filename, safe='/')}"
                with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                if temporary.stat().st_size != expected_size or digest(temporary, expected_size, algorithm) != expected_hash:
                    raise ValueError(f"Integrity verification failed for {filename}; partial file retained")
                temporary.replace(target)
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
