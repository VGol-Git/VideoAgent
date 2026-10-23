"""Verify experiment completeness, immutable inputs and saved evidence contracts.

This checks reproducible artifacts; it is deliberately not a semantic grader.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def study_sources(directory, manifest):
    """Check the study snapshot independently of later service development."""
    working_package = Path(__file__).resolve().parents[1] / "clipscout"
    snapshot = Path(directory) / "source" / "clipscout"
    package = snapshot if snapshot.exists() else working_package
    expected_hashes = manifest["provenance"]["source_sha256"]
    for filename, expected in expected_hashes.items():
        source = package / filename
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Study source differs from manifest: {source.resolve()}")
    unexpected = {p.name for p in package.glob("*.py")} - set(expected_hashes)
    if unexpected:
        raise ValueError(f"Unrecorded study source files in {package.resolve()}: {sorted(unexpected)}")
    working_changes = [filename for filename, expected in expected_hashes.items()
                       if not (working_package / filename).is_file()
                       or hashlib.sha256((working_package / filename).read_bytes()).hexdigest() != expected]
    working_changes.extend({p.name for p in working_package.glob("*.py")} - set(expected_hashes))
    return package.resolve(), {
        "study_source_matches_manifest": True,
        "source_code_directory": str(package.resolve()),
        "source_code_origin": "frozen_snapshot" if snapshot.exists() else "working_tree",
        "working_source_directory": str(working_package.resolve()),
        "source_code_unchanged": not working_changes,
        "working_source_matches_manifest": not working_changes,
        "working_source_changed_files": sorted(working_changes),
    }


def verification_helpers(package):
    """Use the hash-validated study implementation for metric recomputation."""
    namespace = "_clipscout_verified_" + hashlib.sha256(str(package).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(
        namespace, package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = module
    spec.loader.exec_module(module)
    evaluation = importlib.import_module(f"{namespace}.evaluation")
    index = importlib.import_module(f"{namespace}.index")
    media = importlib.import_module(f"{namespace}.media")
    return evaluation.evaluate, index.VideoIndex, media.file_fingerprint


def verify(directory, questions_path, index_path):
    directory, questions_path = Path(directory), Path(questions_path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    package, source_report = study_sources(directory, manifest)
    evaluate, VideoIndex, file_fingerprint = verification_helpers(package)
    if hashlib.sha256(questions_path.read_bytes()).hexdigest() != manifest["question_file_sha256"]:
        raise ValueError("Question file changed")
    index = VideoIndex.load(index_path)
    if index.fingerprint() != manifest["index_fingerprint"]:
        raise ValueError("Index changed")
    if file_fingerprint(index.video_path) != index.video_sha256:
        raise ValueError("Source video changed")
    questions = [q for q in json.loads(questions_path.read_text(encoding="utf-8")) if q["id"] in manifest["question_ids"]]
    by_id = {q["id"]: q for q in questions}
    if set(by_id) != set(manifest["question_ids"]):
        raise ValueError("Manifest question IDs do not match dataset")
    counts, statuses = {}, {}
    resume_path = directory / "resume_history.json"
    resume = json.loads(resume_path.read_text(encoding="utf-8")) if resume_path.exists() else None
    if resume:
        initial = json.loads((directory / resume["original_manifest"]).read_text(encoding="utf-8"))
        for key in ("backend", "index_fingerprint", "question_file_sha256", "question_ids", "modes", "vision",
                    "frames_per_clip", "max_frames", "image_size", "max_new_tokens", "max_tool_calls",
                    "cache_enabled", "prompt_version", "provenance"):
            if initial[key] != manifest[key]:
                raise ValueError(f"Configuration changed across process resume: {key}")
    for mode in manifest["modes"]:
        prediction_path = directory / f"{mode}.predictions.json"
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        legacy_predictions = {p["question_id"]: p for p in json.loads(prediction_path.read_text(
            encoding=resume.get("preserved_prediction_input_encoding", "utf-8")))} if resume else {}
        if len(predictions) != len(questions) or {p["question_id"] for p in predictions} != set(by_id):
            raise ValueError(f"Incomplete or duplicate predictions: {mode}")
        saved = json.loads((directory / f"{mode}.metrics.json").read_text(encoding="utf-8"))
        metric_config = {"max_tool_calls": manifest["max_tool_calls"], "max_frames": manifest["max_frames"],
                         "bootstrap_samples": 1000, "seed": 42}
        if saved["configuration"] != metric_config:
            raise ValueError(f"Unexpected metric configuration: {mode}")
        recomputed = evaluate(questions, predictions, **metric_config)
        if recomputed != saved:
            raise ValueError(f"Saved metrics differ from recomputation: {mode}")
        if recomputed["summary"]["budget_violation_rate"] != 0:
            raise ValueError(f"Budget violation in {mode}")
        for prediction in predictions:
            if resume:
                expected_hash = resume["preserved_prediction_sha256"].get(mode, {}).get(prediction["question_id"])
                actual_hash = hashlib.sha256(json.dumps(legacy_predictions[prediction["question_id"]], sort_keys=True, ensure_ascii=False,
                                                        separators=(",", ":")).encode()).hexdigest()
                if expected_hash and actual_hash != expected_hash:
                    raise ValueError("A completed pre-resume prediction was changed")
                utf8_hash = resume.get("preserved_prediction_utf8_sha256", {}).get(mode, {}).get(prediction["question_id"])
                if utf8_hash and hashlib.sha256(json.dumps(prediction, sort_keys=True, ensure_ascii=False,
                                                           separators=(",", ":")).encode("utf-8")).hexdigest() != utf8_hash:
                    raise ValueError("UTF-8 canonical prediction hash differs")
            if prediction["question"] != by_id[prediction["question_id"]]["question"]:
                raise ValueError("A question was altered at inference")
            if prediction["backend"] != manifest["backend"] or prediction["mode"] != mode:
                raise ValueError("Backend or mode differs from manifest")
            if prediction["provenance"]["index_fingerprint"] != manifest["index_fingerprint"]:
                raise ValueError("Prediction used a different index")
            if prediction["provenance"]["prompt_version"] != manifest["prompt_version"]:
                raise ValueError("Prediction used a different prompt")
            if prediction["tool_calls"] != len(prediction["trace"]):
                raise ValueError("Tool trace counter mismatch")
            evidence = prediction["evidence"]
            for citation in prediction["citations"]:
                observation = evidence[citation["evidence_id"]]
                if not 0 <= observation["start"] <= citation["start"] < citation["end"] <= observation["end"] <= index.duration:
                    raise ValueError("Citation escaped its evidence interval")
            for observation in evidence.values():
                if any(not observation["start"] <= t < observation["end"] for t in observation.get("sampled_timestamps", [])):
                    raise ValueError("Frame timestamp escaped its inspected interval")
        counts[mode] = len(predictions)
        statuses[mode] = dict(Counter(p["status"] for p in predictions))
    report = {"verified": True, "prediction_counts": counts, "total_predictions": sum(counts.values()),
              "statuses": statuses, **source_report, "question_file_unchanged": True,
              "index_and_video_unchanged": True, "metrics_recomputed": True, "budget_violations": 0,
              "citation_contracts_valid": True,
              "resumed_completed_predictions_unchanged": True if resume else None,
              "process_count": 2 if resume else 1,
              "note": "Structural/provenance validation only; does not establish factual correctness or semantic evidence support."}
    (directory / "artifact_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    verify(args.directory, args.questions, args.index)
