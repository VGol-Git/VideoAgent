"""Optional CPU screen-text observations with frame-level provenance.

RapidOCR result conversion is adapted from Grounded Video Agent's
RapidOCRBackend (Apache-2.0); see THIRD_PARTY_NOTICES.txt. OCR recognition
confidence is an engine score, not confidence that an answer is true.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path


class OCRDependencyError(RuntimeError):
    pass


class OCRExecutionError(RuntimeError):
    pass


def _file_identity(path) -> dict:
    """Hash an identified model file without loading it into an inference engine."""
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"Model file changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": after.st_size}


def _config_value(value):
    """Serialize effective parameters without object addresses or opaque sessions."""
    if isinstance(value, Enum):
        return _config_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config_value(item) for item in value]
    return {"unserialized_type": f"{type(value).__module__}.{type(value).__qualname__}"}


class RapidOCRAdapter:
    """Lazy RapidOCR adapter; use preinstalled weights via engine params.

    ``engine`` supports dependency-free test injection. When omitted, calling
    recognize loads the optional rapidocr package and its configured weights;
    no model is loaded during construction. CUDA and DirectML are disabled.
    Input images are never enlarged; the default longest side is 1120 pixels.
    """
    def __init__(self, *, params=None, engine=None, max_side=1120, min_confidence=0.5):
        if type(max_side) is not int or not 720 <= max_side <= 4096:
            raise ValueError("OCR max_side must be an integer from 720 to 4096")
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            raise ValueError("OCR min_confidence must be between 0 and 1")
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("OCR min_confidence must be between 0 and 1")
        self.params = dict(params or {})
        self.params.update({"EngineConfig.onnxruntime.use_cuda": False,
                            "EngineConfig.onnxruntime.use_dml": False})
        self._engine = engine
        self._injected_engine = engine is not None
        self._model_info_cache = None
        self._model_info_engine_id = None
        self.max_side = max_side
        self.min_confidence = float(min_confidence)

    def _load(self):
        if self._engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise OCRDependencyError("Screen OCR requires the optional rapidocr and onnxruntime packages.") from exc
            try:
                self._engine = RapidOCR(params=self.params)
                self._model_info_cache = None
            except Exception as exc:
                raise OCRExecutionError(f"Unable to initialize CPU RapidOCR and its weights: {exc}") from exc
        return self._engine

    def model_info(self) -> dict:
        """Return a cached identity snapshot; never initialize or execute OCR.

        Hash only paths exposed by the actual initialized ONNX sessions. The
        files are an on-disk snapshot, not proof that loaded memory is unchanged.
        Injected engines never inherit identities from installed RapidOCR assets.
        """
        if self._model_info_cache is not None and self._model_info_engine_id == id(self._engine):
            return deepcopy(self._model_info_cache)
        config = {"params": _config_value(self.params), "max_side": self.max_side,
                  "min_confidence": self.min_confidence, "effective": None}
        info = {"engine": "injected" if self._injected_engine else "RapidOCR",
                "package_version": None, "runtime": None if self._injected_engine else "CPU ONNX Runtime",
                "identity_status": "injected_unknown" if self._injected_engine else "not_initialized",
                "model_files": [], "unresolved_components": [], "disabled_components": [],
                "model_hash_policy": "SHA-256 of actual enabled session model files on disk, cached per adapter engine; no inference. Injected or in-memory engines have unknown model identity."}
        if not self._injected_engine:
            try:
                info["package_version"] = version("rapidocr")
            except PackageNotFoundError:
                pass
            if self._engine is not None:
                config["effective"] = _config_value(getattr(self._engine, "cfg", None))
                for component, attribute, enabled_attribute in (
                    ("detector", "text_det", "use_det"),
                    ("classifier", "text_cls", "use_cls"),
                    ("recognizer", "text_rec", "use_rec"),
                ):
                    if getattr(self._engine, enabled_attribute, True) is False:
                        info["disabled_components"].append(component)
                        continue
                    wrapper = getattr(getattr(self._engine, attribute, None), "session", None)
                    session = getattr(wrapper, "session", wrapper)
                    # RapidOCR 3.x wraps ONNX Runtime's InferenceSession, whose
                    # _model_path records the path actually passed at creation.
                    model_path = getattr(session, "_model_path", None)
                    if not isinstance(model_path, (str, Path)) or not str(model_path):
                        info["unresolved_components"].append({"component": component, "reason": "session_model_path_unavailable"})
                        continue
                    try:
                        info["model_files"].append({"component": component, **_file_identity(model_path)})
                    except OSError as exc:
                        info["unresolved_components"].append({"component": component, "reason": str(exc)})
                info["identity_status"] = "partial" if info["unresolved_components"] else "resolved"
        info["config"] = config
        info["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                                         separators=(",", ":")).encode("utf-8")).hexdigest()
        self._model_info_cache = info
        self._model_info_engine_id = id(self._engine)
        return deepcopy(info)

    def recognize(self, image, timestamp: float, frame_id: str | None = None) -> dict:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError("OCR timestamp must be a finite nonnegative number")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("OCR timestamp must be a finite nonnegative number")
        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise OCRDependencyError("Screen OCR requires Pillow and NumPy.") from exc
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                prepared = opened.convert("RGB")
        elif isinstance(image, Image.Image):
            prepared = image.convert("RGB")
        else:
            prepared = Image.fromarray(image).convert("RGB")
        original_size = prepared.size
        prepared.thumbnail((self.max_side, self.max_side))
        width, height = prepared.size
        image_hash = hashlib.sha256(prepared.tobytes()).hexdigest()
        frame_id = str(frame_id or f"frame_{timestamp:.6f}_{image_hash[:12]}")
        try:
            # RapidOCR consumes OpenCV-style BGR NumPy arrays.
            output = self._load()(np.asarray(prepared)[:, :, ::-1].copy())
        except (OCRDependencyError, OCRExecutionError):
            raise
        except Exception as exc:
            raise OCRExecutionError(f"CPU OCR failed for {frame_id}: {exc}") from exc
        if not all(hasattr(output, field) for field in ("boxes", "txts", "scores")):
            raise OCRExecutionError("Unsupported RapidOCR output; expected boxes, txts, and scores")
        boxes = output.boxes if output.boxes is not None else ()
        texts = output.txts if output.txts is not None else ()
        scores = output.scores if output.scores is not None else ()
        if not len(boxes) == len(texts) == len(scores):
            raise OCRExecutionError("RapidOCR returned inconsistent boxes/text/scores lengths")
        lines = []
        for box, text, raw_score in zip(boxes, texts, scores):
            try:
                confidence = float(raw_score)
                polygon = [[float(p[0]), float(p[1])] for p in box]
            except (TypeError, ValueError, IndexError) as exc:
                raise OCRExecutionError("Malformed OCR polygon or confidence") from exc
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise OCRExecutionError("OCR returned an invalid confidence score")
            if len(polygon) < 3 or any(not math.isfinite(v) for p in polygon for v in p):
                raise OCRExecutionError("OCR returned an invalid polygon")
            text = str(text).strip()
            if not text or confidence < self.min_confidence:
                continue
            x1, x2 = max(0., min(p[0] for p in polygon)), min(float(width), max(p[0] for p in polygon))
            y1, y2 = max(0., min(p[1] for p in polygon)), min(float(height), max(p[1] for p in polygon))
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append({"text": text, "confidence": confidence, "polygon": polygon,
                          "bbox": [x1, y1, x2, y2],
                          "bbox_normalized": [x1 / width, y1 / height, x2 / width, y2 / height]})
        model_info = self.model_info()
        return {"id": frame_id, "frame_id": frame_id, "timestamp": float(timestamp),
                "text": "\n".join(line["text"] for line in lines), "lines": lines,
                "width": width, "height": height, "source": "ocr",
                "provenance": {"engine": model_info["engine"], "package_version": model_info["package_version"],
                    "runtime": model_info["runtime"], "model_info": model_info, "image_sha256": image_hash,
                    "original_size": list(original_size), "processed_size": [width, height],
                    "min_confidence": self.min_confidence,
                    "evidence_kind": "sampled_frame_screen_text",
                    "warning": "OCR may misread text; this observation does not establish text duration."}}
