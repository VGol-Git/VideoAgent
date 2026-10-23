"""Read a completed study and write an external diagnostic, without rerunning it.

This tool never loads a model, executes an agent, rewrites predictions, or
recalculates archived metrics. Classification describes recorded terminal
failures, not a counterfactual score after repairing them.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clipscout.research_evaluation import ERROR_CATEGORIES, MODEL_PROTOCOL_CATEGORIES, classify_prediction_error


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status_counts(records):
    return {status: sum(p.get("status") == status for p in records)
            for status in ("answered", "insufficient_evidence", "error", "missing")}


def _group(records):
    errors = [p for p in records if p.get("status") == "error"]
    counts = Counter(classify_prediction_error(p) for p in errors)
    return {"prediction_count": len(records), "status_counts": _status_counts(records),
            "terminal_error_categories": {c: counts[c] for c in ERROR_CATEGORIES},
            "error_types": dict(Counter(str(p.get("error_type", "unknown")) for p in errors)),
            "error_messages": dict(Counter(str(p.get("error", "")) for p in errors)),
            "model_protocol_error_count": sum(counts[c] for c in MODEL_PROTOCOL_CATEGORIES),
            "runtime_error_count": counts["backend"] + counts["infrastructure"],
            "unknown_error_count": counts["unknown"]}


def diagnose_run(run_directory: str | Path) -> dict:
    """Return a diagnosis based only on existing frozen question/output files."""
    run = Path(run_directory).resolve()
    questions_path = run / "questions.selected.json"
    questions = _read(questions_path)
    if not isinstance(questions, list):
        raise ValueError("questions.selected.json must contain a list")
    by_id = {q["id"]: q for q in questions}
    if len(by_id) != len(questions):
        raise ValueError("duplicate frozen question IDs")
    prediction_files = sorted(run.glob("*.predictions.json"))
    if not prediction_files:
        raise ValueError("No saved *.predictions.json files found")
    inputs = [questions_path]
    modes, all_records, errors, event_cases, modality_cases, placeholders = {}, [], [], [], [], []
    reviewer_coverage = {}
    for path in prediction_files:
        mode = path.name.removesuffix(".predictions.json")
        saved = _read(path)
        if not isinstance(saved, list):
            raise ValueError(f"{path.name} must contain a list")
        predictions = {p["question_id"]: p for p in saved}
        if len(predictions) != len(saved) or not set(predictions) <= set(by_id):
            raise ValueError(f"{path.name}: duplicate or unknown question IDs")
        inputs.append(path)
        records = [predictions.get(qid, {"question_id": qid, "status": "missing"}) for qid in by_id]
        all_records.extend(records)
        modes[mode] = {**_group(records), "by_task": {
            task: _group([p for p in records if by_id[p["question_id"]]["task"] == task])
            for task in ("timestamp_search", "reasoning", "onset", "unanswerable")}}
        metrics_path = run / f"{mode}.metrics.json"
        if metrics_path.exists():
            metrics = _read(metrics_path)
            inputs.append(metrics_path)
            reviewer_coverage[mode] = metrics.get("summary", {}).get("review_coverage")
        else:
            reviewer_coverage[mode] = None
        for p in records:
            qid = p["question_id"]
            q, plan = by_id[qid], p.get("plan") or {}
            if p.get("status") == "error":
                errors.append({"mode": mode, "question_id": qid, "task": q["task"],
                               "category": classify_prediction_error(p), "error_type": p.get("error_type"),
                               "error": p.get("error")})
            if q["task"] == "onset":
                event_cases.append({"mode": mode, "question_id": qid, "status": p["status"],
                                    "planned_task": plan.get("task"), "planned_modality": plan.get("modality"),
                                    "reference_modality": q["modality"], "event_task_mismatch": plan.get("task") != "event",
                                    "explicit_event_present": isinstance(p.get("event"), dict),
                                    "answer_present": bool(p.get("answer"))})
            modality_cases.append({"mode": mode, "question_id": qid, "reference_modality": q["modality"],
                                   "planned_modality": plan.get("modality"), "status": p["status"]})
            requirements = plan.get("requirements", {})
            values = list(requirements.values()) if isinstance(requirements, dict) else requirements
            if "distinctive search terms" in plan.get("queries", []) or "one fact that must be established" in (values or []):
                placeholders.append({"mode": mode, "question_id": qid, "status": p["status"],
                                     "queries": plan.get("queries"), "requirements": requirements})
    status_path = run / "run_status.json"
    run_status = _read(status_path) if status_path.exists() else None
    if status_path.exists():
        inputs.append(status_path)
    return {"diagnostic_version": "clipscout-saved-run-audit-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run": str(run), "source_run_status": run_status,
            "method": "Read-only inspection of frozen questions, saved predictions and saved metric review counts. No agent/model execution and no archived metric recalculation.",
            "input_files": [{"path": str(p.relative_to(run)), "sha256": _sha(p)} for p in inputs],
            "summary": _group(all_records), "by_mode": modes,
            "by_task": {task: _group([p for p in all_records if by_id[p["question_id"]]["task"] == task])
                        for task in ("timestamp_search", "reasoning", "onset", "unanswerable")},
            "terminal_errors": errors, "onset": {
                "case_count": len(event_cases), "planned_event_count": sum(not c["event_task_mismatch"] for c in event_cases),
                "explicit_event_count": sum(c["explicit_event_present"] for c in event_cases),
                "missing_event_output_count": sum(not c["explicit_event_present"] for c in event_cases), "cases": event_cases},
            "gold_modality_vs_planned": {
                "counts": dict(Counter(f"{c['reference_modality']} -> {c['planned_modality'] or 'missing'}" for c in modality_cases)),
                "cases": modality_cases,
                "interpretation": "Routing diagnostic, not automatic correctness. A transcript plan cannot establish a visual-only occurrence; missing plans include planner failures."},
            "placeholder_plans": {"case_count": len(placeholders), "cases": placeholders},
            "saved_semantic_review_coverage": reviewer_coverage,
            "limitations": [
                "status=answered establishes delivery, not semantic correctness. Abstention is a distinct outcome, not a runtime exception.",
                "No semantic scores or successes are assigned by this diagnosis, including apparently correct drafts rejected by validation.",
                "operational_error_rate in archived metrics counts all status=error outputs; model/protocol failures are not infrastructure outages.",
                "Legacy accuracy_known may be zero solely from automatic failure scores with zero reviewed answers. Unreviewed answer quality remains unknown.",
                "Missing explicit event outputs are end-to-end failures and cannot measure the accuracy of an event-refinement stage that was not reached.",
                "Point/range temporal compliance is distinct from correct scene descriptions. Model self-audits are not independent reference checks.",
                "Do not remove failed questions from denominators, convert rejected drafts into successes, or mix repaired reruns into this archived run."]}


def render_text(report):
    lines = [f"Saved-run diagnosis: {report['source_run']}", report["method"], "",
             "Mode | answered | insufficient_evidence | error | missing"]
    for mode, data in report["by_mode"].items():
        counts = data["status_counts"]
        lines.append(f"{mode} | {counts['answered']} | {counts['insufficient_evidence']} | {counts['error']} | {counts['missing']}")
    lines.extend(["", "Task | answered | insufficient_evidence | error | missing"])
    for task, data in report["by_task"].items():
        counts = data["status_counts"]
        lines.append(f"{task} | {counts['answered']} | {counts['insufficient_evidence']} | {counts['error']} | {counts['missing']}")
    lines.extend(["", "Terminal failure categories:"])
    lines.extend(f"  {category}: {count}" for category, count in report["summary"]["terminal_error_categories"].items())
    onset = report["onset"]
    lines.extend(["", f"Onset: {onset['case_count']} cases, {onset['planned_event_count']} event plans, {onset['explicit_event_count']} explicit event outputs.",
                  f"Plans containing copied schema placeholders: {report['placeholder_plans']['case_count']}.", "", "Saved semantic review coverage:"])
    for mode, coverage in report["saved_semantic_review_coverage"].items():
        lines.append(f"  {mode}: {json.dumps(coverage, ensure_ascii=False)}")
    lines.extend(["", "Interpretation limits:", *[f"- {note}" for note in report["limitations"]]])
    return "\n".join(lines) + "\n"


def write_diagnosis(report, output_directory):
    run, output = Path(report["source_run"]).resolve(), Path(output_directory).resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError("Diagnosis output must be outside the source run directory")
    destinations = [output / "diagnosis.json", output / "diagnosis.txt"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Diagnosis already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    destinations[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    destinations[1].write_text(render_text(report), encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.run_dir.resolve().parent.parent / "diagnostics" / args.run_dir.name
    report = diagnose_run(args.run_dir)
    destination = write_diagnosis(report, output)
    print(render_text(report))
    print(f"Diagnosis saved outside archived run: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
