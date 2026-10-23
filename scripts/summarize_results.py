"""Produce a compact, reproducible comparison from saved experiment outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

from clipscout.evaluation import bootstrap_ci, evaluate


def summarize(directory, questions_path=None):
    directory = Path(directory)
    rows = []
    per_question = {}
    predictions_by_mode = {}
    for path in sorted(directory.glob("*.metrics.json")):
        mode = path.name.removesuffix(".metrics.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["summary"]
        per_question[mode] = {r["question_id"]: r for r in report["per_question"]}
        predictions = json.loads((directory / f"{mode}.predictions.json").read_text(encoding="utf-8"))
        predictions_by_mode[mode] = predictions
        times = sorted(p["latency_seconds"] for p in predictions)
        row = {"mode": mode, **metrics, "median_latency_seconds": statistics.median(times) if times else None,
               "p95_latency_seconds": times[min(len(times)-1, int(.95*len(times)))] if times else None,
               "total_model_calls": sum(p.get("usage", {}).get("model_calls", 0) for p in predictions),
               "total_input_tokens": sum(p.get("usage", {}).get("input_tokens", 0) for p in predictions),
               "total_output_tokens": sum(p.get("usage", {}).get("output_tokens", 0) for p in predictions),
               "controller_decision_count": sum(len(p.get("decisions", [])) for p in predictions),
               "invalid_decision_count": sum(bool(d.get("error")) for p in predictions for d in p.get("decisions", [])),
               "questions_with_invalid_decisions": sum(any(d.get("error") for d in p.get("decisions", [])) for p in predictions)}
        rows.append(row)
    if not rows:
        raise ValueError("No completed per-mode metrics found")
    (directory / "comparison.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    paired = {}
    if "adaptive" in per_question:
        adaptive = per_question["adaptive"]
        for mode, baseline in per_question.items():
            if mode == "adaptive":
                continue
            if set(adaptive) != set(baseline):
                raise ValueError("Paired comparison requires exactly the same question IDs")
            paired[mode] = {}
            for metric in ("task_success_proxy", "token_f1", "temporal_iou", "temporal_recall_at_0_3", "frames_used", "latency_seconds"):
                differences = [adaptive[q][metric] - baseline[q][metric] for q in sorted(adaptive)
                               if adaptive[q].get(metric) is not None and baseline[q].get(metric) is not None]
                paired[mode][metric] = bootstrap_ci(differences, samples=1000, seed=42)
    (directory / "paired_differences.json").write_text(json.dumps({
        "direction": "adaptive minus named baseline", "comparisons": paired,
        "note": "Exploratory paired percentile bootstrap over questions from one video. Questions are dependent; intervals are not evidence of cross-video generalization. No correction for multiple comparisons; latency has one run per question."
    }, indent=2), encoding="utf-8")
    if questions_path:
        questions_path = Path(questions_path)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if hashlib.sha256(questions_path.read_bytes()).hexdigest() != manifest["question_file_sha256"]:
            raise ValueError("Question file differs from frozen experiment")
        questions = [q for q in json.loads(questions_path.read_text(encoding="utf-8")) if q["id"] in manifest["question_ids"]]
        sources = sorted({q.get("annotation_notes", {}).get("source", "unspecified") for q in questions})
        modality = {}
        for source in sources:
            subset = [q for q in questions if q.get("annotation_notes", {}).get("source", "unspecified") == source]
            ids = {q["id"] for q in subset}
            modality[source] = {mode: evaluate(subset, [p for p in predictions if p["question_id"] in ids])["summary"]
                                for mode, predictions in predictions_by_mode.items()}
        (directory / "modality_metrics.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")
    print("Mode | Answered | Errors | F1 | tIoU | Recall@.3 | Tools | Frames | Median s")
    for r in rows:
        def number(key):
            return "n/a" if r[key] is None else f"{r[key]:.3f}"
        print(f"{r['mode']} | {r['answered_count']}/{r['example_count']} | {r['error_count']} | " +
              " | ".join(number(k) for k in ("token_f1", "temporal_iou", "temporal_recall_at_0_3", "mean_tool_calls", "mean_frames_used", "median_latency_seconds")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--questions", help="Frozen question file for separate visual/ASR-reference summaries")
    arguments = parser.parse_args()
    summarize(arguments.directory, arguments.questions)
