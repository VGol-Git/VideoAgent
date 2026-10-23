"""User-invoked, checkpointed research experiment with numbered run folders.

No model is initialized by importing this module or requesting --help. Each new
invocation allocates a separate run; only --resume PATH reuses an existing one.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

from clipscout.agent import AgentConfig
from clipscout.backends import QWEN_REVISION, QwenBackend
from clipscout.cli import save_json
from clipscout.index import VideoIndex
from clipscout.media import file_fingerprint
from clipscout.research_evaluation import evaluate_research, validate_questions
from clipscout.research_agent import ResearchAgent, RESEARCH_VERSION

PROJECT = Path(__file__).resolve().parents[1]
MODEL_CACHE = PROJECT / ".cache" / "huggingface"
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
RESEARCH_MODES = ("adaptive", "retrieval-only", "static", "uniform")
PACKAGES = ("torch", "torchvision", "transformers", "numpy", "Pillow", "opencv-python-headless",
            "faster-whisper", "ctranslate2", "av", "rapidocr", "onnxruntime", "tokenizers", "safetensors")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def hardware_snapshot():
    """Read-only GPU telemetry; failure to collect it is explicit, not fatal."""
    report = {"captured_at_utc": utc_now(), "cpu": platform.processor(), "cpu_count": os.cpu_count()}
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
                                    "--format=csv,noheader"], capture_output=True, text=True, timeout=8)
        report["gpu_csv_columns"] = ["name", "driver_version", "memory_total", "memory_used", "utilization_gpu"]
        report["gpu_csv"] = completed.stdout.strip()
        if completed.returncode:
            report["gpu_error"] = completed.stderr.strip() or f"exit {completed.returncode}"
    except (OSError, subprocess.TimeoutExpired) as error:
        report["gpu_error"] = f"{type(error).__name__}: {error}"
    return report


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def launch_context():
    """Record only the launcher's explicit demo-stop information, never env data."""
    try:
        value = json.loads(os.environ.get("CLIPSCOUT_LAUNCH_CONTEXT", "{}"))
    except (ValueError, TypeError):
        return {}
    if not isinstance(value, dict):
        return {}
    result = {}
    if type(value.get("stop_demo_requested")) is bool:
        result["stop_demo_requested"] = value["stop_demo_requested"]
    if isinstance(value.get("stopped_demo_pids"), list):
        result["stopped_demo_pids"] = [pid for pid in value["stopped_demo_pids"] if type(pid) is int and pid > 0]
    return result


def scalar_metrics(value, prefix=""):
    """Flatten mapping metrics; lists retain their structured JSON representation."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from scalar_metrics(item, f"{prefix}.{key}" if prefix else str(key))
    elif value is None or isinstance(value, (str, int, float, bool)):
        encoded = "unknown" if value is None else ("true" if value is True else "false" if value is False else str(value))
        yield prefix, encoded


def write_metrics_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def allocate_run(output_root):
    """Exclusive mkdir makes numbering safe against simultaneous starts."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 1_000_000):
        target = root / f"run_{number:03d}"
        try:
            target.mkdir()
            return target
        except FileExistsError:
            continue
    raise RuntimeError("Run-number range exhausted")


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


@contextmanager
def run_log(run_dir):
    with (run_dir / "console.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            yield


def source_files():
    files = list((PROJECT / "clipscout").rglob("*.py")) + [Path(__file__).resolve(),
             PROJECT / "scripts" / "download_model.py", PROJECT / "scripts" / "prepare_research_assets.py"]
    for name in ("run_experiment.ps1", "pyproject.toml", "requirements-research.txt", "THIRD_PARTY_NOTICES.txt"):
        if (PROJECT / name).is_file():
            files.append(PROJECT / name)
    return {p.relative_to(PROJECT).as_posix(): p for p in sorted(set(files))}


def source_hashes():
    return {name: file_fingerprint(path) for name, path in source_files().items()}


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", default="results/research_v2")
    result.add_argument("--resume", metavar="RUN_DIRECTORY", help="Resume exactly this run, preserving all completed predictions")
    result.add_argument("--questions", default="data/assignment_v7/questions.json")
    result.add_argument("--protocol", default="data/assignment_v7/protocol.json")
    result.add_argument("--index", default="data/indexes/user_video.json")
    result.add_argument("--visual-index", default="data/indexes/user_video_visual")
    result.add_argument("--prepare-assets", action="store_true", help="Build/resume the shared CPU CLIP/OCR sidecar if incomplete")
    result.add_argument("--download-models", action="store_true", help="Download verified pinned Qwen/CLIP snapshots only when local assets are missing or incomplete")
    result.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    result.add_argument("--visual-cadence", type=float, default=5.0)
    result.add_argument("--prepare-only", action="store_true", help="Freeze source/data and validate only; do not build assets or initialize models")
    result.add_argument("--metrics-only", action="store_true", help="With --resume, recompute metrics/review templates from saved predictions without loading models or assets")
    result.add_argument("--modes", nargs="+", choices=RESEARCH_MODES, default=list(RESEARCH_MODES),
                        help="UI research modes, all using the same v7 agent and model")
    result.add_argument("--split", choices=("dev", "test"), default="test")
    result.add_argument("--question-ids", nargs="+", help="Optional subset allowed only for the development split")
    result.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--max-new-tokens", type=int, default=768)
    result.add_argument("--max-context-tokens", type=int, default=8192)
    result.add_argument("--image-size", type=int, default=560)
    result.add_argument("--frames-per-clip", type=int, default=4)
    result.add_argument("--max-frames", type=int, default=16)
    result.add_argument("--no-warmup", action="store_true")
    result.add_argument("--reviews-dir", help="Optional completed reviewer JSON files named MODE.reviews.json")
    return result


def select_questions(all_questions, split, selected_ids=None):
    questions = [q for q in validate_questions(all_questions) if q["split"] == split]
    if selected_ids:
        if split != "dev":
            raise ValueError("--question-ids is permitted only for dev, never selective test reporting")
        if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) - {q["id"] for q in questions}:
            raise ValueError("Duplicate or unknown development question IDs")
        questions = [q for q in questions if q["id"] in selected_ids]
    if not questions:
        raise ValueError("Selected split has no questions")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", q["id"]) for q in questions):
        raise ValueError("Question IDs must be safe file names")
    return questions


def make_schedule(questions, modes):
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("Modes must be unique and nonempty")
    if set(modes) - set(RESEARCH_MODES):
        raise ValueError("Only the four v7 UI research modes are supported")
    result = []
    for number, question in enumerate(questions):
        offset = number % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            result.append({"mode": mode, "question_id": question["id"]})
    return result


def stable_configuration(args, index, questions, protocol):
    paths = {key: Path(getattr(args, key)).expanduser().resolve() for key in ("questions", "protocol", "index")}
    if file_fingerprint(paths["questions"]) != protocol["questions_sha256"]:
        raise ValueError("Questions differ from the frozen protocol checksum")
    annotations = (PROJECT / protocol["visual_annotations_file"]).resolve()
    if file_fingerprint(annotations) != protocol["visual_annotations_sha256"]:
        raise ValueError("Visual annotations differ from the frozen protocol checksum")
    reference_assets = {}
    reference_rows = list(protocol.get("annotation_assets", []))
    for kind in ("visual", "timestamp"):
        if protocol.get(f"{kind}_annotations_file"):
            reference_rows.append({"path": protocol[f"{kind}_annotations_file"],
                                   "sha256": protocol[f"{kind}_annotations_sha256"]})
    for row in reference_rows:
        asset = (PROJECT / row["path"]).resolve()
        if not asset.is_relative_to(PROJECT) or file_fingerprint(asset) != row["sha256"]:
            raise ValueError(f"Reference asset is unsafe or differs from the frozen checksum: {row['path']}")
        reference_assets[asset.relative_to(PROJECT).as_posix()] = {"path": str(asset), "sha256": row["sha256"]}
    video_hash = file_fingerprint(index.video_path) if index.video_path else None
    if video_hash != index.video_sha256 or video_hash != protocol["video_sha256"]:
        raise ValueError("Video checksum differs from the index or study reference")
    config = AgentConfig(frames_per_clip=args.frames_per_clip, max_frames=args.max_frames, image_size=args.image_size)
    # Validate backend arguments without constructing/loading model dependencies.
    if not (1 <= args.max_new_tokens <= 2048 and 256 <= args.max_context_tokens <= 32768
            and args.max_new_tokens < args.max_context_tokens):
        raise ValueError("Invalid generation/context token limits")
    if not 1024 <= args.image_size ** 2 <= 1048576:
        raise ValueError("Image pixel cap is outside the Qwen backend limits")
    if not 0 < args.visual_cadence <= 3600:
        raise ValueError("Invalid visual-index cadence")
    return {
        "schema_version": "assignment-v7-run-v1", "split": args.split, "modes": args.modes,
        "question_ids": [q["id"] for q in questions], "schedule": make_schedule(questions, args.modes),
        "files": {key: {"path": str(path), "sha256": file_fingerprint(path)} for key, path in paths.items()},
        "visual_annotations": {"path": str(annotations), "sha256": file_fingerprint(annotations)},
        "reference_assets": reference_assets,
        "index_fingerprint": index.fingerprint(), "video": {"path": index.video_path, "sha256": video_hash, "duration": index.duration},
        "model": {"name": args.model, "revision": QWEN_REVISION if args.model == "Qwen/Qwen3-VL-2B-Instruct" else None,
                  "device": args.device, "local_files_only": args.local_files_only, "max_new_tokens": args.max_new_tokens,
                  "max_context_tokens": args.max_context_tokens, "image_pixel_cap": args.image_size ** 2},
        "agent_config": vars(config), "inspection_cache_enabled": False,
        "visual_preparation": {"path": str(Path(args.visual_index).expanduser().resolve()), "clip_model": args.clip_model,
                               "clip_revision": CLIP_REVISION if args.clip_model == CLIP_MODEL else None,
                               "cadence_seconds": args.visual_cadence, "device": "cpu", "ocr": True},
        "warmup_enabled": not args.no_warmup, "source_sha256": source_hashes(),
        "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": package_versions()},
        "agent_version": RESEARCH_VERSION,
    }


def freeze_run(run_dir, configuration, protocol, all_questions, questions, index):
    destination = run_dir / "manifest.json"
    fingerprint = digest(configuration)
    if destination.exists():
        previous = read_json(destination)
        if previous.get("configuration_sha256") != fingerprint or previous.get("configuration") != configuration:
            raise ValueError("Cannot resume: code, runtime, data, questions, modes or settings changed. Start a new numbered run.")
        verify_snapshot(run_dir, previous)
        return previous
    snapshot = run_dir / "source"
    for relative, source in source_files().items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_fingerprint(target) != configuration["source_sha256"][relative]:
            raise ValueError("Source changed while creating snapshot; start a fresh run")
    save_json(run_dir / "protocol.json", protocol)
    save_json(run_dir / "questions.all.json", all_questions)
    save_json(run_dir / "questions.selected.json", questions)
    shutil.copyfile(configuration["visual_annotations"]["path"], run_dir / "visual_annotations.json")
    # Save an executable index snapshot with the correct relative media path.
    index.save(run_dir / "assets" / "index.json")
    reference_copies = []
    for relative, item in configuration.get("reference_assets", {}).items():
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["path"], target)
        if file_fingerprint(target) != item["sha256"]:
            raise ValueError("Annotation asset changed while freezing the run")
        reference_copies.append(target)
    manifest = {"created_at_utc": utc_now(), "configuration_sha256": fingerprint, "configuration": configuration,
                "notes": ["References and rubric are used by the evaluator/reviewer only; agents receive question text and ID.",
                          "Model weights and source video remain shared; exact references and checksums are recorded."]}
    manifest["snapshot_sha256"] = {p.relative_to(run_dir).as_posix(): file_fingerprint(p)
                                   for p in [*snapshot.rglob("*"), run_dir / "protocol.json", run_dir / "questions.all.json",
                                             run_dir / "questions.selected.json", run_dir / "visual_annotations.json",
                                             run_dir / "assets" / "index.json", *reference_copies] if p.is_file()}
    save_json(destination, manifest)
    return manifest


def verify_snapshot(run_dir, manifest):
    for relative, expected in manifest.get("snapshot_sha256", {}).items():
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir.resolve()) or file_fingerprint(path) != expected:
            raise ValueError(f"Frozen source/data snapshot changed: {relative}")


def check_unchanged(configuration):
    if source_hashes() != configuration["source_sha256"]:
        raise ValueError("Source changed during the run. Completed checkpoints are preserved; start a new run for modified code.")
    for item in [*configuration["files"].values(), configuration["visual_annotations"],
                 *configuration.get("reference_assets", {}).values()]:
        if file_fingerprint(item["path"]) != item["sha256"]:
            raise ValueError("Reference data changed during the run")


def model_readiness(snapshot, repo, revision):
    """Offline completeness check using a prior verified download manifest.

    This checks manifest identity, safe file names, existence and sizes. Content
    hashes were verified by the downloader; this fast check is not a rehash of
    several gigabytes on every invocation. The distinction is recorded per run.
    """
    snapshot = Path(snapshot).resolve()
    manifest_path = snapshot / "download_manifest.json"
    try:
        manifest = read_json(manifest_path)
        if manifest.get("repo") != repo or manifest.get("revision") != revision:
            raise ValueError("Download manifest identity differs from pinned model")
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError("No verified files in download manifest")
        names = []
        for row in records:
            name, size = row.get("filename"), row.get("size")
            if not isinstance(name, str) or type(size) is not int or size < 0:
                raise ValueError("Invalid file record")
            path = (snapshot / name).resolve()
            if not path.is_relative_to(snapshot) or not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"Missing, unsafe or truncated model file: {name}")
            if row.get("algorithm") not in {"sha1", "sha256"} or not isinstance(row.get("digest"), str):
                raise ValueError(f"Missing prior integrity record: {name}")
            names.append(name)
        if "config.json" not in names or not any(name.endswith(".safetensors") or Path(name).name.startswith("pytorch_model") and name.endswith(".bin") for name in names):
            raise ValueError("Model config or inference weights are missing")
        return {"ready": True, "snapshot": str(snapshot), "repo": repo, "revision": revision,
                "download_manifest_sha256": file_fingerprint(manifest_path), "files": records,
                "readiness_check": "Pinned manifest identity, safe paths, file existence and exact sizes; content hashes verified at download time."}
    except (OSError, ValueError, AttributeError, TypeError) as error:
        return {"ready": False, "snapshot": str(snapshot), "repo": repo, "revision": revision,
                "reason": f"{type(error).__name__}: {error}"}


def ensure_models(args, run_dir):
    """Called only in the live-inference branch, after source/data are frozen."""
    required = [("qwen", args.model, QWEN_REVISION if args.model == "Qwen/Qwen3-VL-2B-Instruct" else None)]
    required.append(("clip", args.clip_model, CLIP_REVISION if args.clip_model == CLIP_MODEL else None))
    report = {"started_at_utc": utc_now(), "status": "checking", "models": {}, "download_enabled": bool(args.download_models)}
    report_path = run_dir / "model_preparation.json"
    started = time.perf_counter()
    try:
        for name, repo, revision in required:
            if revision is None:
                local = Path(repo).expanduser().resolve()
                if not local.is_dir() or not (local / "config.json").is_file():
                    raise ValueError(f"Automatic download supports the pinned default {name} model only; provide an existing local snapshot for {repo}")
                weights = sorted([*local.glob("*.safetensors"), *local.glob("pytorch_model*.bin")])
                if not weights:
                    raise ValueError(f"No inference weights found in local model: {local}")
                report["models"][name] = {"ready": True, "snapshot": str(local), "revision": "user-supplied-local",
                    "files_sha256": {p.name: file_fingerprint(p) for p in [local / "config.json", *weights]},
                    "downloaded_this_invocation": False}
                save_json(report_path, report)
                continue
            snapshot = MODEL_CACHE / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
            readiness = model_readiness(snapshot, repo, revision)
            report["models"][name] = readiness
            save_json(report_path, report)
            downloaded = False
            if not readiness["ready"]:
                if not args.download_models:
                    raise FileNotFoundError(f"{name} model is incomplete: {readiness['reason']}. Use --download-models to fetch verified assets.")
                print(f"Preparing pinned {repo}@{revision}; download progress is recorded in this run.", flush=True)
                from scripts.download_model import download
                actual = download(repo, revision=revision)
                if Path(actual).resolve() != snapshot.resolve():
                    raise ValueError("Downloader returned an unexpected snapshot path")
                downloaded = True
                readiness = model_readiness(snapshot, repo, revision)
                if not readiness["ready"]:
                    raise ValueError(f"Model remains incomplete after verified download: {readiness['reason']}")
            readiness["downloaded_this_invocation"] = downloaded
            report["models"][name] = readiness
            target = run_dir / "assets" / "models" / f"{name}.download_manifest.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / "download_manifest.json", target)
            save_json(report_path, report)
        report["status"] = "ready"
        return report["models"]
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(finished_at_utc=utc_now(), seconds_this_invocation=time.perf_counter()-started)
        save_json(report_path, report)


def prepare_assets(args, index, run_dir, configuration, models=None):
    from clipscout.ocr import RapidOCRAdapter
    from clipscout.visual_index import CPUCLIPEncoder, VisualIndex, prepare_visual_index
    cache = Path(args.visual_index).expanduser().resolve()
    manifest_path = cache if cache.suffix.lower() == ".json" else cache / "manifest.json"
    started = time.perf_counter()
    reused = manifest_path.is_file() and read_json(manifest_path).get("status") == "complete"
    if not reused:
        if not args.prepare_assets:
            raise FileNotFoundError("Visual sidecar missing/incomplete; use --prepare-assets to build it")
        print("Preparing CPU CLIP/OCR visual index; completed frame checkpoints are reusable.", flush=True)
        def progress(value):
            save_json(run_dir / "asset_progress.json", {"updated_at_utc": utc_now(), "cache_path": str(manifest_path.parent), **value})
            print(json.dumps({"asset_progress": value}, ensure_ascii=False), flush=True)
        try:
            prepare_visual_index(index, manifest_path.parent, clip_model=args.clip_model, cadence=args.visual_cadence,
                                 device="cpu", ocr=True, on_progress=progress,
                                 encoder=CPUCLIPEncoder(models["clip"]["snapshot"]) if models and "clip" in models else None)
        except BaseException as error:
            save_json(run_dir / "asset_last_invocation.json", {"status": "failed", "cache_path": str(manifest_path.parent),
                      "preparation_seconds_this_invocation": time.perf_counter()-started,
                      "error": f"{type(error).__name__}: {error}", "partial_frame_checkpoints_preserved": True})
            raise
    sidecar = VisualIndex.load(manifest_path, index)
    if not sidecar.metadata.get("ocr_enabled"):
        raise ValueError("Research modes require the visual index with OCR enabled")
    if abs(sidecar.cadence - args.visual_cadence) > 1e-9:
        raise ValueError("Existing visual sidecar uses another cadence; choose a distinct --visual-index path")
    if sidecar.metadata.get("preparation_config", {}).get("clip_model") != args.clip_model:
        raise ValueError("Existing visual sidecar uses another CLIP model; choose a distinct --visual-index path")
    if args.clip_model == CLIP_MODEL and sidecar.metadata.get("clip_model", {}).get("revision") != CLIP_REVISION:
        raise ValueError("Existing visual sidecar uses a different CLIP revision; choose a distinct --visual-index path")
    asset_files = {"manifest.json": manifest_path, "embeddings.npz": manifest_path.parent / sidecar.metadata["embeddings_file"]}
    hashes = {name: file_fingerprint(path) for name, path in asset_files.items()}
    previous_path = run_dir / "asset_manifest.json"
    if previous_path.exists() and read_json(previous_path).get("files_sha256") != hashes:
        raise ValueError("Visual sidecar changed since this run; cannot mix preprocessing versions")
    for name, path in asset_files.items():
        target = run_dir / "assets" / "visual_index" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        elif file_fingerprint(target) != hashes[name]:
            raise ValueError("Saved visual asset snapshot checksum mismatch")
    report = {"required": True, "source_path": str(manifest_path), "files_sha256": hashes,
              "reused_existing_complete_sidecar": reused, "preparation_seconds_this_invocation": time.perf_counter() - started,
              "lifetime_preparation_usage": sidecar.metadata.get("usage"), "metadata_snapshot": "assets/visual_index/manifest.json"}
    if not previous_path.exists():
        save_json(previous_path, report)
    save_json(run_dir / "asset_last_invocation.json", report)
    check_unchanged(configuration)
    return sidecar, RapidOCRAdapter(max_side=1536)


def checkpoint_path(run_dir, mode, identifier):
    return run_dir / "checkpoints" / mode / f"{identifier}.json"


def load_checkpoints(run_dir, configuration):
    fingerprint = digest(configuration)
    records = {mode: {} for mode in configuration["modes"]}
    for job in configuration["schedule"]:
        path = checkpoint_path(run_dir, job["mode"], job["question_id"])
        if not path.exists():
            continue
        checkpoint = read_json(path)
        if checkpoint.get("configuration_sha256") != fingerprint or checkpoint.get("mode") != job["mode"] or checkpoint.get("question_id") != job["question_id"]:
            raise ValueError(f"Checkpoint provenance mismatch: {path}")
        prediction = checkpoint["prediction"]
        if prediction.get("question_id") != job["question_id"] or digest(prediction) != checkpoint.get("prediction_sha256"):
            raise ValueError(f"Checkpoint content mismatch: {path}")
        records[job["mode"]][job["question_id"]] = prediction
    return records


def review_template(questions, predictions):
    by_id = {p["question_id"]: p for p in predictions}
    return [{"question_id": q["id"], "reviewer": None, "method": None,
             "atom_judgments": [{"atom_id": atom["id"], "correct": None, "supported": None,
                                 "notes": "", "reference_atom": atom["text"], "reference_support_ids": atom["support_ids"]}
                                for atom in q["claim_atoms"]],
             "extra_claims_supported": None, "abstention_correct": None, "event_identity_correct": None,
             "context": {"question": q["question"], "modality": q["modality"], "task": q["task"],
                         "supports": q["supports"], "event_reference": q.get("event"),
                         "timestamp_request": q.get("timestamp_request"),
                         "expected_abstention_reason": q.get("expected_abstention_reason"),
                         "prediction": by_id.get(q["id"]),
                         "caution": "Do not use the answering model's own audit as an independent correctness judgment."}}
            for q in questions]


def write_reports(args, run_dir, questions, records):
    summaries = {}
    per_question_csv = []
    for mode, by_id in records.items():
        predictions = [by_id[q["id"]] for q in questions if q["id"] in by_id]
        save_json(run_dir / f"{mode}.predictions.json", predictions)
        reviews = []
        saved_reviews = run_dir / "review_inputs" / f"{mode}.reviews.json"
        path = Path(args.reviews_dir) / f"{mode}.reviews.json" if args.reviews_dir else saved_reviews
        if not path.exists():
            path = saved_reviews
        if path.exists():
            # A copied template may retain completely untouched entries.
            # Ignore only those; partial judgments still require provenance.
            raw_reviews = read_json(path)
            if not isinstance(raw_reviews, list):
                raise ValueError("Review files must contain a list of question reviews")
            reviews = [r for r in raw_reviews if r.get("reviewer") or r.get("method") or
                       any(r.get(field) is not None for field in ("extra_claims_supported", "abstention_correct", "event_identity_correct")) or
                       any(a.get("correct") is not None or a.get("supported") is not None for a in r.get("atom_judgments", []))]
        metrics = evaluate_research(questions, predictions, reviews=reviews,
                                    max_tool_calls=4, max_frames=args.max_frames)
        metrics["review_provenance"] = {"source_file": str(path) if path.exists() else None,
                                        "input_sha256": file_fingerprint(path) if path.exists() else None,
                                        "review_records_sha256": digest(reviews), "review_count": len(reviews)}
        save_json(run_dir / f"{mode}.metrics.json", metrics)
        summaries[mode] = metrics["summary"]
        for row in metrics["per_question"]:
            per_question_csv.extend((mode, row["question_id"], key, value)
                                    for key, value in scalar_metrics(row) if key != "question_id")
        # Separate templates are regenerated; completed reviews live elsewhere.
        save_json(run_dir / "review_templates" / f"{mode}.reviews.template.json", review_template(questions, predictions))
        if path.exists():
            save_json(saved_reviews, reviews)
    by_task = {}
    for mode in records:
        metrics = read_json(run_dir / f"{mode}.metrics.json")
        for task, task_summary in metrics.get("by_task", {}).items():
            by_task.setdefault(task, {})[mode] = task_summary
    save_json(run_dir / "study_summary.json", {"agent_version": RESEARCH_VERSION,
        "by_task": by_task, "by_mode": summaries,
        "notes": ["All research modes use v7 with the same model and question set; this is not a version comparison.",
                  "Timestamp search, reasoning and event appearance timing are reported separately.",
                  "Null semantic scores require explicit reviewer judgments. Templates contain unknowns, not automatic failures.",
                  "Research modes differ in evidence acquisition. Inspect their observations, modalities and costs before interpreting scores."]})
    write_metrics_csv(run_dir / "mode_metrics.csv", ("mode", "metric", "value"),
                      ((mode, key, value) for mode, summary in summaries.items() for key, value in scalar_metrics(summary)))
    write_metrics_csv(run_dir / "task_metrics.csv", ("task", "mode", "metric", "value"),
                      ((task, mode, key, value) for task, modes in by_task.items()
                       for mode, summary in modes.items() for key, value in scalar_metrics(summary)))
    write_metrics_csv(run_dir / "per_question.metrics.csv", ("mode", "question_id", "metric", "value"), per_question_csv)


def execute(args, run_dir, argv):
    state = {"status": "running", "started_at_utc": utc_now(), "run_directory": str(run_dir),
             "phase": "validation", "completed_predictions": 0, "argv": argv, "dry_run": bool(args.prepare_only)}
    if not args.resume:
        save_json(run_dir / "run_status.json", state)
    history_path = run_dir / "execution_history.json"
    history = read_json(history_path) if history_path.exists() else []
    invocation = {"started_at_utc": state["started_at_utc"], "argv": argv, "resume": bool(args.resume), "prepare_only": bool(args.prepare_only)}
    invocation["launch_context"] = launch_context()
    history.append(invocation)
    save_json(history_path, history)
    try:
        if args.metrics_only:
            manifest = read_json(run_dir / "manifest.json")
            configuration = manifest["configuration"]
            verify_snapshot(run_dir, manifest)
            check_unchanged(configuration)
            questions = validate_questions(read_json(run_dir / "questions.selected.json"))
            records = load_checkpoints(run_dir, configuration)
            args.max_frames = configuration["agent_config"]["max_frames"]
            state["completed_predictions"] = sum(len(rows) for rows in records.values())
            write_reports(args, run_dir, questions, records)
            state.update(status="metrics_completed", phase="metrics_only", finished_at_utc=utc_now())
            save_json(run_dir / "metrics_status.json", state)
            print(f"Metrics updated from saved predictions: {run_dir}. No model or asset loading.", flush=True)
            return 0
        index = VideoIndex.load(args.index)
        all_questions, protocol = read_json(args.questions), read_json(args.protocol)
        questions = select_questions(all_questions, args.split, args.question_ids)
        configuration = stable_configuration(args, index, questions, protocol)
        freeze_run(run_dir, configuration, protocol, all_questions, questions, index)
        records = load_checkpoints(run_dir, configuration)
        completed = sum(len(rows) for rows in records.values())
        state.update(completed_predictions=completed, total_predictions=len(configuration["schedule"]), phase="prepared")
        print(f"Run folder: {run_dir}\nSelected {len(questions)} questions; {len(configuration['schedule'])} mode/question pairs; {completed} saved.", flush=True)
        if args.prepare_only:
            state.update(status="prepared", finished_at_utc=utc_now(), note="No asset preparation, model loading, or inference was performed.")
            save_json(run_dir / ("validation_status.json" if args.resume else "run_status.json"), state)
            return 0
        save_json(run_dir / "run_status.json", state)
        pending = [job for job in configuration["schedule"] if job["question_id"] not in records[job["mode"]]]
        if pending:
            save_json(run_dir / "hardware.before.json", hardware_snapshot())
            state["phase"] = "model_preparation"
            save_json(run_dir / "run_status.json", state)
            models = ensure_models(args, run_dir)
            state["phase"] = "asset_preparation"
            save_json(run_dir / "run_status.json", state)
            sidecar, ocr = prepare_assets(args, index, run_dir, configuration, models=models)
            backend = QwenBackend(args.model, device=args.device, local_files_only=args.local_files_only,
                                  max_new_tokens=args.max_new_tokens, max_context_tokens=args.max_context_tokens,
                                  max_image_pixels=args.image_size ** 2)
            config = AgentConfig(**configuration["agent_config"])
            agents = {mode: ResearchAgent(index, backend, config, cache_dir=None,
                                         visual_index=sidecar, ocr=ocr, refine_events=True)
                      for mode in args.modes}
            if not args.no_warmup:
                state["phase"] = "warmup"
                save_json(run_dir / "run_status.json", state)
                from clipscout.media import extract_frames
                started = time.perf_counter()
                images = extract_frames(index.video_path, 0, min(4, index.duration), count=args.frames_per_clip, max_side=args.image_size)
                backend.generate("Reply with READY only.", "Initialize inference.", images=images)
                invocation["warmup"] = {"seconds": time.perf_counter()-started, "frames": len(images), "usage": dict(backend.usage)}
                save_json(history_path, history)
            lookup = {q["id"]: q for q in questions}
            for job in pending:
                check_unchanged(configuration)
                mode, identifier = job["mode"], job["question_id"]
                state.update(phase="inference", current=job)
                save_json(run_dir / "run_status.json", state)
                print(f"[{state['completed_predictions']+1}/{state['total_predictions']}] {mode} / {identifier}", flush=True)
                started = time.perf_counter()
                before = dict(backend.usage)
                try:
                    pred = agents[mode].run(lookup[identifier]["question"], question_id=identifier, mode=mode)
                    if pred.get("provenance", {}).get("prompt_version") != RESEARCH_VERSION or pred.get("mode") != mode:
                        raise ValueError("Agent returned a different version or research mode than requested")
                except Exception as error:
                    traceback.print_exc()
                    pred = {"question_id": identifier, "status": "error", "answer": "", "citations": [],
                            "error": f"{type(error).__name__}: {error}", "outer_exception": True,
                            "tool_calls": None, "frames_used": None, "trace": [],
                            "usage": {key: backend.usage.get(key, 0)-before.get(key, 0) for key in backend.usage},
                            "latency_seconds": time.perf_counter()-started,
                            "accounting_note": "Outer failure: tool/frame counts unavailable, not measured zero."}
                pred["study_mode"] = mode
                pred["study_agent_version"] = RESEARCH_VERSION
                save_json(checkpoint_path(run_dir, mode, identifier),
                          {"saved_at_utc": utc_now(), "configuration_sha256": digest(configuration),
                           "mode": mode, "question_id": identifier, "prediction_sha256": digest(pred), "prediction": pred})
                records[mode][identifier] = pred
                state["completed_predictions"] += 1
                save_json(run_dir / "run_status.json", state)
                write_reports(args, run_dir, questions, records)
                print(json.dumps({"mode": mode, "question_id": identifier, "status": pred["status"],
                                  "latency_seconds": pred.get("latency_seconds")}), flush=True)
            invocation["total_backend_usage"] = dict(backend.usage)
        state["phase"] = "reporting"
        save_json(run_dir / "run_status.json", state)
        write_reports(args, run_dir, questions, records)
        failed = sum(p["status"] == "error" for rows in records.values() for p in rows.values())
        if pending:
            save_json(run_dir / "hardware.after.json", hardware_snapshot())
        state.update(status="completed_with_errors" if failed else "completed", phase="complete",
                     finished_at_utc=utc_now(), error_prediction_count=failed)
        state.pop("current", None)
        save_json(run_dir / "run_status.json", state)
        print(f"Completed with {failed} operational question errors. Outputs and review templates: {run_dir}", flush=True)
        return 2 if failed else 0
    except BaseException as error:
        state.update(status="failed", finished_at_utc=utc_now(), error=f"{type(error).__name__}: {error}")
        text = traceback.format_exc()
        preserve_original = bool(args.resume and (args.metrics_only or state["phase"] == "validation"))
        failure_name = "resume_failure" if preserve_original else "failure"
        (run_dir / f"{failure_name}_traceback.txt").write_text(text, encoding="utf-8")
        save_json(run_dir / ("resume_failure.json" if preserve_original else "run_status.json"), state)
        print(text, file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        invocation.update(finished_at_utc=utc_now(), status=state["status"], completed_predictions=state["completed_predictions"])
        save_json(history_path, history)


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(raw)
    if args.metrics_only and (not args.resume or args.prepare_only):
        raise ValueError("--metrics-only requires --resume PATH and cannot be combined with --prepare-only")
    if args.resume:
        run_dir = Path(args.resume).expanduser().resolve()
        if not (run_dir / "manifest.json").is_file():
            raise ValueError("--resume requires an existing prepared run with manifest.json")
        if read_json(run_dir / "manifest.json").get("configuration", {}).get("schema_version") != "assignment-v7-run-v1":
            raise ValueError("This folder belongs to an earlier study protocol. Start a NEW numbered run without --resume; the shared visual cache is reused independently.")
    else:
        run_dir = allocate_run(args.output_root)
    lock = run_dir / ".run.lock"
    # Never silently take over another live or interrupted writer.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_at_utc": utc_now()}))
    try:
        with run_log(run_dir):
            print(f"ClipScout research run: {run_dir}", flush=True)
            return execute(args, run_dir, raw)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
