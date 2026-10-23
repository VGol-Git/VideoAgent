"""Resumable CPU CLIP/OCR sidecar for observed video frames.

Visual diversity is an independent implementation of the representative-frame
idea used by VideoAgent/VideoTree. This module uses normalized CLIP vectors and
deterministic farthest-point selection, not the original Ward-clustering code.
Dense similarity proposes frames to inspect; it is not evidence of an event or
a probability that the query is true. Five-second sampling can miss events.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter

from .index import finite_number, validate_interval
from .media import extract_frames, file_fingerprint, probe_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class VisualIndexError(RuntimeError):
    pass


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise VisualIndexError("Visual indexing requires NumPy.") from exc
    return np


def _normalize(values):
    np = _np()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise VisualIndexError("Embeddings must be a finite nonempty-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise VisualIndexError("Zero embedding cannot be normalized")
    return values / norms


def _json_write(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def _cached_model(source):
    """Resolve an actual snapshot so Transformers cannot probe the network."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    model_folder = "models--" + str(source).replace("/", "--")
    for cache in (PROJECT_ROOT / ".cache" / "huggingface",
                  PROJECT_ROOT / ".cache" / "huggingface" / "hub"):
        root = cache / model_folder
        reference = root / "refs" / "main"
        if reference.is_file():
            snapshot = root / "snapshots" / reference.read_text(encoding="utf-8").strip()
            if snapshot.is_dir():
                return snapshot.resolve()
        snapshots = list((root / "snapshots").glob("*"))
        if len(snapshots) == 1 and snapshots[0].is_dir():
            return snapshots[0].resolve()
    raise VisualIndexError(f"CLIP model is not cached locally: {source}. Supply --clip-model with its snapshot directory.")


class CPUCLIPEncoder:
    """Lazy local CLIP encoder; no model downloads and no GPU allocation."""
    def __init__(self, model=DEFAULT_CLIP_MODEL, *, device="cpu"):
        if device != "cpu":
            raise ValueError("This sidecar uses CPU; device must be 'cpu'")
        self.model_source = str(model)
        self.device = device
        self._model = self._processor = self._torch = None
        self._path = None
        self._model_info = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise VisualIndexError("CPU CLIP requires torch, transformers, and Pillow.") from exc
        path = _cached_model(self.model_source)
        try:
            processor = CLIPProcessor.from_pretrained(str(path), local_files_only=True)
            model = CLIPModel.from_pretrained(str(path), local_files_only=True,
                                             torch_dtype=torch.float32).to("cpu").eval()
        except Exception as exc:
            raise VisualIndexError(f"Unable to load the local CLIP checkpoint: {exc}") from exc
        self._torch, self._processor, self._model, self._path = torch, processor, model, path

    def encode_images(self, images):
        self._load()
        prepared = []
        for image in images:
            image = image.convert("RGB").copy()
            image.thumbnail((336, 336))
            prepared.append(image)
        if not prepared:
            raise ValueError("At least one image is required")
        inputs = self._processor(images=prepared, return_tensors="pt")
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        return _normalize(features.detach().cpu().numpy())

    def encode_text(self, text):
        self._load()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True,
                                 truncation=True, max_length=77)
        with self._torch.inference_mode():
            features = self._model.get_text_features(**inputs)
        return _normalize(features.detach().cpu().numpy())[0]

    def get_model_info(self):
        self._load()
        if self._model_info is None:
            revision = self._path.name if self._path.parent.name == "snapshots" else "local-files"
            files = [*self._path.glob("*.safetensors"), *self._path.glob("pytorch_model*.bin"),
                     *self._path.glob("*.json"), *self._path.glob("*.txt")]
            self._model_info = {"model_id": DEFAULT_CLIP_MODEL, "source": str(self._path),
                "revision": revision, "device": "cpu", "dtype": "float32",
                "files_sha256": {p.name: file_fingerprint(p) for p in sorted(set(files))},
                "image_max_side": 336, "text_max_tokens": 77}
        return dict(self._model_info)


class VisualIndex:
    def __init__(self, metadata, embeddings, *, encoder=None):
        if metadata.get("schema_version") != 1 or metadata.get("status") != "complete":
            raise VisualIndexError("Visual index must be a complete schema_version=1 sidecar")
        self.metadata = metadata
        self.duration = finite_number(metadata.get("duration"), "duration")
        self.cadence = finite_number(metadata.get("cadence_seconds"), "cadence_seconds")
        if self.duration <= 0 or self.cadence <= 0:
            raise VisualIndexError("Visual index duration and cadence must be positive")
        self.frames = list(metadata.get("frames", []))
        self.embeddings = _normalize(embeddings)
        if len(self.frames) != self.embeddings.shape[0] or not self.frames:
            raise VisualIndexError("Frame and embedding counts must agree and be positive")
        previous, identities = -1., set()
        for frame in self.frames:
            timestamp = finite_number(frame.get("timestamp"), "frame timestamp")
            if not previous < timestamp < self.duration or timestamp < 0:
                raise VisualIndexError("Frame timestamps must increase within the video")
            previous = timestamp
            identity = frame.get("id")
            if not isinstance(identity, str) or not identity or identity in identities:
                raise VisualIndexError("Frame identities must be nonempty and unique")
            identities.add(identity)
            ocr = frame.get("ocr")
            if ocr is not None and abs(finite_number(ocr.get("timestamp"), "OCR timestamp") - timestamp) > 1e-6:
                raise VisualIndexError("OCR timestamp does not match its frame")
        self._encoder = encoder
        self._encoder_verified = False
        self.query_calls = 0

    @classmethod
    def load(cls, path, index=None, *, encoder=None):
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            path = path / "manifest.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if index is not None:
            if metadata.get("index_fingerprint") != index.fingerprint():
                raise VisualIndexError("Visual sidecar belongs to a different transcript/caption index")
            if index.video_sha256 and metadata.get("video_sha256") != index.video_sha256:
                raise VisualIndexError("Visual sidecar belongs to a different video")
            if abs(metadata.get("duration", -1) - index.duration) > 1e-6:
                raise VisualIndexError("Visual sidecar duration does not match the video index")
        artifact = metadata.get("embeddings_file", "embeddings.npz")
        if not isinstance(artifact, str) or Path(artifact).name != artifact:
            raise VisualIndexError("Embeddings must be a sibling file of the manifest")
        array_path = path.parent / artifact
        if metadata.get("embeddings_sha256") != file_fingerprint(array_path):
            raise VisualIndexError("Embedding artifact checksum mismatch")
        np = _np()
        with np.load(array_path, allow_pickle=False) as data:
            embeddings = data["embeddings"].copy()
        return cls(metadata, embeddings, encoder=encoder)

    @property
    def ocr_records(self):
        return [frame["ocr"] for frame in self.frames if frame.get("ocr") is not None]

    def _bounds(self, start, end):
        if (start is None) != (end is None):
            raise ValueError("Visual search requires both start and end")
        return (0., self.duration) if start is None else validate_interval(start, end, self.duration)

    def search(self, query, top_k=10, start=None, end=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must contain 1–500 characters")
        if type(top_k) is not int or not 1 <= top_k <= 1000:
            raise ValueError("top_k must be an integer from 1 to 1000")
        lo, hi = self._bounds(start, end)
        rows = [i for i, frame in enumerate(self.frames) if lo <= frame["timestamp"] < hi]
        if not rows:
            return []
        if self._encoder is None:
            self._encoder = CPUCLIPEncoder(self.metadata["clip_model"]["source"])
        if not self._encoder_verified and self.metadata["clip_model"].get("files_sha256"):
            observed = self._encoder.get_model_info()
            if observed.get("files_sha256") != self.metadata["clip_model"]["files_sha256"]:
                raise VisualIndexError("Query encoder files differ from the model used to build image embeddings")
            self._encoder_verified = True
        np = _np()
        self.query_calls += 1
        text = _normalize(np.asarray(self._encoder.encode_text(query)).reshape(1, -1))[0]
        if text.shape[0] != self.embeddings.shape[1]:
            raise VisualIndexError("Text encoder dimensions differ from stored image vectors")
        scores = self.embeddings[rows] @ text
        ordered = sorted(zip(rows, scores.tolist()), key=lambda pair: (-pair[1], self.frames[pair[0]]["timestamp"]))
        hits = []
        for rank, (row, score) in enumerate(ordered[:top_k], 1):
            frame = self.frames[row]
            at = frame["timestamp"]
            hits.append({"id": frame["id"], "source": "dense_visual", "score": float(score),
                "score_type": "cosine_similarity", "rank": rank, "query": query,
                "start": max(lo, at - self.cadence / 2), "end": min(hi, at + self.cadence / 2),
                "sampled_timestamps": [at], "frame_ids": [frame["id"]],
                "text": f"CLIP retrieval candidate at {at:.3f} seconds; inspect the frame to verify its content.",
                "window_kind": "context_around_sample",
                "provenance": {"index_fingerprint": self.metadata["index_fingerprint"],
                    "video_sha256": self.metadata["video_sha256"],
                    "model_id": self.metadata["clip_model"].get("model_id"),
                    "model_revision": self.metadata["clip_model"].get("revision"),
                    "image_sha256": frame.get("image_sha256"),
                    "warning": "Similarity is a retrieval score, not event presence or confidence."}})
        return hits

    def representatives(self, start, end, count):
        """Return actual timestamps with visual diversity and temporal ties.

        Endpoints seed multi-frame selection. Subsequent frames maximize their
        minimum cosine distance to selected vectors; ties favor uncovered time.
        This is a representative sample, not continuous temporal coverage.
        """
        lo, hi = self._bounds(start, end)
        if type(count) is not int or not 1 <= count <= 64:
            raise ValueError("count must be an integer from 1 to 64")
        rows = [i for i, f in enumerate(self.frames) if lo <= f["timestamp"] < hi]
        if len(rows) <= count:
            return [self.frames[i]["timestamp"] for i in rows]
        np = _np()
        if count == 1:
            mean = self.embeddings[rows].mean(axis=0)
            center = max(rows, key=lambda i: (float(self.embeddings[i] @ mean), -i))
            return [self.frames[center]["timestamp"]]
        selected = [rows[0], rows[-1]]
        while len(selected) < count:
            def value(i):
                distance = float(1 - np.max(self.embeddings[selected] @ self.embeddings[i]))
                temporal = min(abs(self.frames[i]["timestamp"] - self.frames[j]["timestamp"]) for j in selected)
                return (round(distance, 6), temporal, -i)
            selected.append(max((i for i in rows if i not in selected), key=value))
        return sorted(self.frames[i]["timestamp"] for i in selected)


def _journal_records(path):
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    lines, records = raw.splitlines(), []
    for n, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if n != len(lines) - 1:
                raise VisualIndexError("Checkpoint contains a malformed committed record") from exc
            # A power loss can leave the final line unwritten. Only that trailing
            # incomplete record is discarded; previously completed frames remain.
            with path.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return records


def prepare_visual_index(index, output, *, clip_model=DEFAULT_CLIP_MODEL,
                         cadence=5., device="cpu", ocr=False, encoder=None,
                         ocr_adapter=None, on_progress=None):
    """Build or resume frame sidecar. Completed frame work is never repeated.

    Output contains an append-only per-frame checkpoint and a final compressed
    vector matrix. Failures remain explicit and retain completed checkpoints.
    Test injection of encoder/OCR is supported; normal use loads local weights.
    """
    cadence = finite_number(cadence, "cadence")
    if cadence <= 0 or cadence > 3600:
        raise ValueError("cadence must be in (0, 3600] seconds")
    if device != "cpu":
        raise ValueError("Research sidecar preparation supports device='cpu'")
    if not index.video_path or not Path(index.video_path).is_file():
        raise VisualIndexError("Visual sidecar requires the index's source video file")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = probe_video(index.video_path)
    if abs(info["duration"] - index.duration) > max(.1, 2 / info["fps"]):
        raise VisualIndexError("Decoded video duration differs from the input index")
    video_hash = file_fingerprint(index.video_path)
    if index.video_sha256 and video_hash != index.video_sha256:
        raise VisualIndexError("Source video checksum differs from the index")
    config = {"index_fingerprint": index.fingerprint(), "video_sha256": video_hash,
              "clip_model": str(clip_model), "cadence_seconds": cadence, "ocr": bool(ocr),
              "device": device, "ocr_max_side": 1536, "clip_max_side": 336}
    build_path, journal_path = output / "build.json", output / "checkpoint.jsonl"
    manifest_path = output / "manifest.json"
    if build_path.exists():
        build = json.loads(build_path.read_text(encoding="utf-8"))
        if build.get("config") != config:
            raise VisualIndexError("Existing sidecar configuration differs; choose another output directory")
        if manifest_path.exists():
            return VisualIndex.load(manifest_path, index=index, encoder=encoder)
    else:
        if journal_path.exists() or manifest_path.exists():
            raise VisualIndexError("Unrecognized existing sidecar artifacts; choose another output directory")
        build = {"schema_version": 1, "config": config, "created_at": datetime.now(timezone.utc).isoformat(),
                 "failed_attempts": 0, "failed_processing_seconds": 0., "sessions": 0,
                 "failed_decoded_frames": 0, "failed_clip_image_calls": 0, "failed_ocr_calls": 0}
    build["sessions"] += 1
    _json_write(build_path, build)
    records = _journal_records(journal_path)
    targets = [i * cadence for i in range(math.ceil(index.duration / cadence))]
    if len(records) > len(targets):
        raise VisualIndexError("Checkpoint has more frames than the configured cadence")
    for n, record in enumerate(records):
        if record.get("requested_timestamp") != targets[n] or record.get("sequence") != n:
            raise VisualIndexError("Checkpoint sampling order or cadence is inconsistent")
    encoder = encoder or CPUCLIPEncoder(clip_model, device=device)
    model_info = encoder.get_model_info()
    if build.get("clip_model_info") is not None and build["clip_model_info"] != model_info:
        raise VisualIndexError("CLIP checkpoint changed since preparation began; choose another output directory")
    build["clip_model_info"] = model_info
    _json_write(build_path, build)
    if ocr and ocr_adapter is None:
        from .ocr import RapidOCRAdapter
        ocr_adapter = RapidOCRAdapter(max_side=1536)
    for n in range(len(records), len(targets)):
        started = perf_counter()
        decoded_this = clip_calls_this = ocr_calls_this = 0
        committed = False
        try:
            at, image = extract_frames(index.video_path, 0., index.duration, count=1,
                                       max_side=1536, target_timestamp=targets[n])[0]
            decoded_this = 1
            if records and at <= records[-1]["timestamp"]:
                raise VisualIndexError("Cadence is too small to produce distinct chronological video frames")
            frame_id = f"visual_{n:06d}"
            image_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            clip_image = image.copy()
            clip_image.thumbnail((336, 336))
            clip_calls_this = 1
            vector = _normalize(encoder.encode_images([clip_image]))
            if vector.shape[0] != 1:
                raise VisualIndexError("CLIP encoder returned an unexpected number of vectors")
            ocr_calls_this = int(bool(ocr))
            observation = ocr_adapter.recognize(image, at, frame_id) if ocr else None
            record = {"sequence": n, "id": frame_id, "timestamp": at,
                      "requested_timestamp": targets[n], "image_sha256": image_hash,
                      "decoded_size": list(image.size), "ocr": observation,
                      "embedding": vector[0].tolist(), "processing_seconds": perf_counter() - started}
            with journal_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            records.append(record)
            committed = True
            if on_progress:
                on_progress({"completed_frames": len(records), "total_frames": len(targets),
                             "timestamp": at, "frame_seconds": record["processing_seconds"]})
        except Exception as exc:
            build["failed_attempts"] += 1
            build["failed_processing_seconds"] += perf_counter() - started
            if not committed:
                build["failed_decoded_frames"] = build.get("failed_decoded_frames", 0) + max(decoded_this, getattr(exc, "frames_decoded", 0))
                build["failed_clip_image_calls"] = build.get("failed_clip_image_calls", 0) + clip_calls_this
                build["failed_ocr_calls"] = build.get("failed_ocr_calls", 0) + ocr_calls_this
            build["last_error"] = str(exc)
            _json_write(build_path, build)
            raise
    np = _np()
    matrix = _normalize([record["embedding"] for record in records])
    array_path = output / "embeddings.npz"
    with tempfile.NamedTemporaryFile(dir=output, prefix="embeddings.", suffix=".tmp", delete=False) as stream:
        np.savez_compressed(stream, embeddings=matrix)
        temporary = stream.name
    os.replace(temporary, array_path)
    metadata = {"schema_version": 1, "status": "complete", "created_at": build["created_at"],
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "index_fingerprint": index.fingerprint(), "video_sha256": video_hash,
                "duration": index.duration, "cadence_seconds": cadence, "video_metadata": info,
                "clip_model": model_info, "ocr_enabled": bool(ocr), "preparation_config": config,
                "embeddings_file": array_path.name, "embeddings_sha256": file_fingerprint(array_path),
                "embedding_dimensions": matrix.shape[1],
                "frames": [{k: v for k, v in r.items() if k not in ("embedding", "processing_seconds")}
                           for r in records],
                "usage": {"committed_decoded_frames": len(records),
                          "decoded_frames": len(records) + build.get("failed_decoded_frames", 0),
                          "clip_image_calls": len(records) + build.get("failed_clip_image_calls", 0),
                          "ocr_calls": (len(records) if ocr else 0) + build.get("failed_ocr_calls", 0),
                          "processing_seconds": sum(r["processing_seconds"] for r in records),
                          "failed_attempts": build["failed_attempts"],
                          "failed_processing_seconds": build["failed_processing_seconds"],
                          "preparation_sessions": build["sessions"]},
                "coverage_note": "Sparse sampled frames only. Events between samples and exact first appearance are not guaranteed."}
    _json_write(manifest_path, metadata)
    return VisualIndex(metadata, matrix, encoder=encoder)
