"""Dependency-free evaluation of answers, temporal evidence, and retrieval.

Lexical answer scores and citation geometry are proxies, not semantic verification.
Every answerable question remains in answer/grounding denominators even if its
prediction is missing, abstained, or errored. Unanswerable questions are evaluated
separately and in the all-example task-success proxy.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
import math
import random
import re
from typing import Any


_CATEGORIES = {"localization", "reasoning", "temporal", "unanswerable", "visual"}
_STATUSES = {"answered", "insufficient_evidence", "error"}
_THRESHOLDS = (0.3, 0.5)
_NOTES = [
    "Token F1, exact match, and required-term agreement are lexical proxies; they do not establish factual correctness.",
    "Temporal scores assess citation geometry only. Evidence IDs are not checked against a source ledger, and semantic claim support is not verified.",
    "Missing, abstained, and error predictions score zero on answerable-question metrics; errors do not count as correct abstentions.",
    "Citation intervals are deduplicated by geometry. One citation can match at most one reference interval, including for multi-hop recall.",
    "Bootstrap intervals describe resampling uncertainty for the supplied examples, not model-training uncertainty or benchmark representativeness.",
]


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return number


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _interval(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain [start, end]")
    start = _finite_number(value[0], f"{label}.start", nonnegative=True)
    end = _finite_number(value[1], f"{label}.end", nonnegative=True)
    if end <= start:
        raise ValueError(f"{label} must have positive duration")
    return start, end


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def token_f1(prediction: str, reference: str) -> float:
    """Case/punctuation-normalized bag-of-token F1, with Unicode word tokens."""
    predicted, expected = _tokens(prediction), _tokens(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    common = sum((Counter(predicted) & Counter(expected)).values())
    return 2.0 * common / (len(predicted) + len(expected))


def temporal_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Intersection over union for two validated, positive-duration intervals."""
    a, b = _interval(first, "first interval"), _interval(second, "second interval")
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if intersection == 0:
        return 0.0
    # For overlapping intervals the union is their enclosing interval. This
    # avoids overflow from adding two large, otherwise finite durations.
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union


def _assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Maximum-weight one-to-one assignment, padded with zero-weight dummies.

    Rectangular Hungarian algorithm; polynomial time even with many references.
    The result excludes dummy columns and zero-weight assignments.
    """
    row_count = len(weights)
    column_count = len(weights[0]) if weights else 0
    if not row_count or not column_count:
        return []
    columns = column_count + row_count
    costs = [[-value for value in row] + [0.0] * row_count for row in weights]
    u, v = [0.0] * (row_count + 1), [0.0] * (columns + 1)
    matched_row, previous = [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        column = 0
        distances = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta, next_column = math.inf, 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = costs[active_row - 1][candidate - 1] - u[active_row] - v[candidate]
                if reduced < distances[candidate]:
                    distances[candidate], previous[candidate] = reduced, column
                if distances[candidate] < delta:
                    delta, next_column = distances[candidate], candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    distances[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while column:
            before = previous[column]
            matched_row[column] = matched_row[before]
            column = before
    return [
        (matched_row[column] - 1, column - 1)
        for column in range(1, column_count + 1)
        if matched_row[column] and weights[matched_row[column] - 1][column - 1] > 0
    ]


def _grounding(reference: list[tuple[float, float]], predicted: list[tuple[float, float]]) -> dict[str, Any]:
    predicted = list(dict.fromkeys(predicted))
    scores = [[temporal_iou(expected, actual) for actual in predicted] for expected in reference]
    assignments = _assignment(scores)
    total_iou = sum(scores[row][column] for row, column in assignments)
    result: dict[str, Any] = {
        "reference_interval_count": len(reference),
        "unique_citation_interval_count": len(predicted),
        "temporal_iou": total_iou / len(reference) if reference else None,
        "citation_iou_precision": total_iou / len(predicted) if reference and predicted else (0.0 if reference else None),
        "matched_intervals": [
            {"reference_index": row, "citation_index": column, "iou": scores[row][column]}
            for row, column in sorted(assignments)
        ],
    }
    for threshold in _THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        threshold_scores = [[float(score >= threshold) for score in row] for row in scores]
        matches = _assignment(threshold_scores)
        result[f"temporal_hit_at_{suffix}"] = float(bool(matches)) if reference else None
        result[f"temporal_recall_at_{suffix}"] = len(matches) / len(reference) if reference else None
        result[f"all_evidence_hit_at_{suffix}"] = float(len(matches) == len(reference)) if reference else None
    # Onset error is conditional on an actual geometric match, never fabricated
    # as zero for a missing answer. The explicit count prevents denominator hiding.
    onset_errors = [abs(reference[row][0] - predicted[column][0]) for row, column in assignments]
    result["onset_error_seconds"] = sum(onset_errors) / len(onset_errors) if onset_errors else None
    result["onset_matched_interval_count"] = len(onset_errors)
    return result


def _questions(questions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for question in questions:
        if not isinstance(question, Mapping):
            raise ValueError("questions must contain objects")
        identifier = _identifier(question.get("id"), "question.id")
        if identifier in seen:
            raise ValueError(f"duplicate question id: {identifier}")
        seen.add(identifier)
        _identifier(question.get("question"), f"{identifier}.question")
        if question.get("category") not in _CATEGORIES:
            raise ValueError(f"{identifier}: invalid category")
        if question.get("split") not in {"dev", "test"}:
            raise ValueError(f"{identifier}: split must be dev or test")
        if not isinstance(question.get("answerable"), bool):
            raise ValueError(f"{identifier}.answerable must be boolean")
        answers = question.get("reference_answers")
        if not isinstance(answers, list) or any(not isinstance(answer, str) for answer in answers):
            raise ValueError(f"{identifier}.reference_answers must be a list of strings")
        if question["answerable"] and not any(_tokens(answer) for answer in answers):
            raise ValueError(f"{identifier}: answerable questions require a nonempty reference answer")
        intervals = question.get("evidence_intervals")
        if not isinstance(intervals, list):
            raise ValueError(f"{identifier}.evidence_intervals must be a list")
        validated_intervals = [_interval(item, f"{identifier}.evidence_intervals") for item in intervals]
        if len(set(validated_intervals)) != len(validated_intervals):
            raise ValueError(f"{identifier}: duplicate reference intervals")
        terms = question.get("required_terms", [])
        if not isinstance(terms, list) or any(
            not isinstance(group, list) or not group or any(not isinstance(term, str) or not _tokens(term) for term in group)
            for group in terms
        ):
            raise ValueError(f"{identifier}.required_terms must be nonempty groups of nonempty strings")
        validated.append({**question, "evidence_intervals": validated_intervals, "required_terms": terms})
    return validated


def _predictions(predictions: Sequence[Mapping[str, Any]], ids: set[str]) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ValueError("predictions must contain objects")
        identifier = _identifier(prediction.get("question_id"), "prediction.question_id")
        if identifier not in ids:
            raise ValueError(f"unknown prediction question_id: {identifier}")
        if identifier in validated:
            raise ValueError(f"duplicate prediction question_id: {identifier}")
        if prediction.get("status") not in _STATUSES:
            raise ValueError(f"{identifier}: invalid prediction status")
        if not isinstance(prediction.get("answer"), str):
            raise ValueError(f"{identifier}.answer must be a string")
        citations = prediction.get("citations", [])
        if not isinstance(citations, list):
            raise ValueError(f"{identifier}.citations must be a list")
        spans = []
        for citation in citations:
            if not isinstance(citation, Mapping):
                raise ValueError(f"{identifier}: citations must be objects")
            _identifier(citation.get("evidence_id"), f"{identifier}.evidence_id")
            span = _interval([citation.get("start"), citation.get("end")], f"{identifier}.citation")
            spans.append(span)
        tool_calls = _count(prediction.get("tool_calls", 0), f"{identifier}.tool_calls")
        frames = _count(prediction.get("frames_used", 0), f"{identifier}.frames_used")
        latency = _finite_number(prediction.get("latency_seconds", 0), f"{identifier}.latency_seconds", nonnegative=True)
        trace = prediction.get("trace", [])
        if not isinstance(trace, list) or any(not isinstance(event, Mapping) for event in trace):
            raise ValueError(f"{identifier}.trace must be a list of objects")
        validated[identifier] = {
            **prediction, "intervals": list(dict.fromkeys(spans)), "tool_calls": tool_calls,
            "frames_used": frames, "latency_seconds": latency, "trace": trace,
        }
    return validated


def _required_term_agreement(answer: str, groups: list[list[str]]) -> bool:
    haystack = " " + " ".join(_tokens(answer)) + " "
    return all(any(" " + " ".join(_tokens(term)) + " " in haystack for term in group) for group in groups)


def bootstrap_ci(values: Iterable[float], *, samples: int = 1000, confidence: float = 0.95, seed: int = 42) -> dict[str, Any]:
    """Percentile bootstrap CI for an arithmetic mean, deterministically seeded."""
    samples = _count(samples, "samples")
    if samples < 1:
        raise ValueError("samples must be positive")
    confidence = _finite_number(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    data = [_finite_number(value, "bootstrap value") for value in values]
    if not data:
        return {"mean": None, "low": None, "high": None, "n": 0, "samples": samples, "confidence": confidence}
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(data) for _ in data) / len(data) for _ in range(samples))
    def quantile(probability: float) -> float:
        position = probability * (len(means) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        return means[lower] + (means[upper] - means[lower]) * (position - lower)
    tail = (1.0 - confidence) / 2
    return {"mean": sum(data) / len(data), "low": quantile(tail), "high": quantile(1 - tail), "n": len(data), "samples": samples, "confidence": confidence}


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    grounded = [row for row in answerable if row["reference_interval_count"]]
    answered = [row for row in rows if row["status"] == "answered"]
    answered_answerable = [row for row in answerable if row["status"] == "answered"]
    multi_hop = [row for row in grounded if row["reference_interval_count"] > 1]
    present = [row for row in rows if row["status"] != "missing"]
    result: dict[str, Any] = {
        "example_count": len(rows), "answerable_count": len(answerable), "unanswerable_count": len(unanswerable),
        "grounding_example_count": len(grounded), "multi_hop_example_count": len(multi_hop),
        "prediction_count": len(present), "missing_prediction_count": len(rows) - len(present),
        "error_count": sum(row["status"] == "error" for row in rows), "answered_count": len(answered),
        "token_f1": _mean(answerable, "token_f1"), "exact_match": _mean(answerable, "exact_match"),
        "lexical_correctness_proxy": _mean(answerable, "lexical_correctness_proxy"),
        "task_success_proxy": _mean(rows, "task_success_proxy"),
        "answerable_coverage": len(answered_answerable) / len(answerable) if answerable else None,
        "unanswerable_abstention_accuracy": _mean(unanswerable, "correct_abstention"),
        "selective_risk_proxy": 1.0 - _mean(answered, "task_success_proxy") if answered else None,
        "selective_risk_denominator": len(answered),
        "temporal_iou": _mean(grounded, "temporal_iou"),
        "citation_iou_precision": _mean(grounded, "citation_iou_precision"),
        "onset_error_seconds": _mean(grounded, "onset_error_seconds"),
        "onset_error_example_count": sum(row["onset_error_seconds"] is not None for row in grounded),
        "onset_matched_interval_count": sum(row["onset_matched_interval_count"] for row in grounded),
        "tool_budget_violation_count": sum(row["tool_budget_violation"] for row in rows),
        "frame_budget_violation_count": sum(row["frame_budget_violation"] for row in rows),
        "budget_violation_rate": _mean(rows, "budget_violation"),
        "tool_call_underreport_count": sum(row["tool_call_underreported"] for row in rows),
        "mean_tool_calls": _mean(present, "tool_calls"), "total_tool_calls": sum(row["tool_calls"] for row in rows),
        "mean_frames_used": _mean(present, "frames_used"), "total_frames_used": sum(row["frames_used"] for row in rows),
        "mean_latency_seconds": _mean(present, "latency_seconds"),
        "denominators": {
            "answer_metrics": len(answerable), "grounding_metrics": len(grounded),
            "task_success_and_budget_rate": len(rows), "unanswerable_abstention": len(unanswerable),
            "usage_means": len(present), "multi_hop_metrics": len(multi_hop),
        },
    }
    for threshold in _THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        for metric in ("temporal_hit", "temporal_recall", "all_evidence_hit"):
            key = f"{metric}_at_{suffix}"
            result[key] = _mean(grounded, key)
        result[f"multi_hop_recall_at_{suffix}"] = _mean(multi_hop, f"temporal_recall_at_{suffix}")
        result[f"multi_hop_success_at_{suffix}"] = _mean(multi_hop, f"all_evidence_hit_at_{suffix}")
    return result


def evaluate(
    questions: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], *,
    max_tool_calls: int = 4, max_frames: int = 16, bootstrap_samples: int = 1000, seed: int = 42,
) -> dict[str, Any]:
    """Evaluate predictions; reject duplicate/unknown IDs and malformed numeric data.

    ``lexical_correctness_proxy`` uses all required-term groups when supplied,
    otherwise normalized exact match to any reference. ``task_success_proxy``
    additionally requires correct abstention on unanswerable examples. Neither
    is an expert or model-based factuality score. ``trace`` is a list of tool
    calls: failed calls count, and observed trace length overrides underreporting.
    """
    max_tool_calls, max_frames = _count(max_tool_calls, "max_tool_calls"), _count(max_frames, "max_frames")
    validated = _questions(questions)
    predicted = _predictions(predictions, {question["id"] for question in validated})
    rows = []
    for question in validated:
        prediction = predicted.get(question["id"])
        status = prediction["status"] if prediction else "missing"
        answered = status == "answered"
        answer = prediction["answer"] if prediction else ""
        correct_abstention = not question["answerable"] and status == "insufficient_evidence"
        references = question["reference_answers"]
        exact_match = float(any(_tokens(answer) == _tokens(reference) for reference in references)) if answered else 0.0
        lexical = float(_required_term_agreement(answer, question["required_terms"])) if question["required_terms"] and answered else exact_match
        task_success = lexical if question["answerable"] else float(correct_abstention)
        reported_tools = prediction["tool_calls"] if prediction else 0
        trace_count = len(prediction["trace"]) if prediction else 0
        tool_calls = max(reported_tools, trace_count)
        frames = prediction["frames_used"] if prediction else 0
        row = {
            "question_id": question["id"], "category": question["category"], "split": question["split"],
            "answerable": question["answerable"], "status": status,
            "token_f1": max((token_f1(answer, reference) for reference in references), default=0.0) if answered and question["answerable"] else (0.0 if question["answerable"] else None),
            "exact_match": exact_match if question["answerable"] else None,
            "lexical_correctness_proxy": lexical if question["answerable"] else None,
            "task_success_proxy": task_success, "correct_abstention": float(correct_abstention) if not question["answerable"] else None,
            "tool_calls": tool_calls, "reported_tool_calls": reported_tools, "tool_call_underreported": trace_count > reported_tools,
            "frames_used": frames, "latency_seconds": prediction["latency_seconds"] if prediction else 0.0,
            "tool_budget_violation": tool_calls > max_tool_calls, "frame_budget_violation": frames > max_frames,
            "budget_violation": float(tool_calls > max_tool_calls or frames > max_frames),
            **_grounding(question["evidence_intervals"], prediction["intervals"] if prediction and answered else []),
        }
        rows.append(row)
    summary = _summary(rows)
    summary["confidence_intervals"] = {
        key: bootstrap_ci([row[key] for row in rows if row.get(key) is not None and (key != "temporal_iou" or row["answerable"])], samples=bootstrap_samples, seed=seed)
        for key in ("token_f1", "exact_match", "task_success_proxy", "temporal_iou")
    }
    return {
        "schema_version": "1.0", "summary": summary,
        "per_category": {category: _summary([row for row in rows if row["category"] == category]) for category in sorted({row["category"] for row in rows})},
        "per_question": rows,
        "configuration": {"max_tool_calls": max_tool_calls, "max_frames": max_frames, "bootstrap_samples": bootstrap_samples, "seed": seed},
        "notes": list(_NOTES),
    }


def retrieval_eval(
    questions: Sequence[Mapping[str, Any]], search_fn: Callable[..., Any], k: int = 5,
) -> dict[str, Any]:
    """Evaluate independent retrieval from original questions, never agent traces.

    Calls ``search_fn(question_text, k=k)`` exactly once per question. The return
    value is a sequence of mappings with ``start``/``end`` or an object containing
    a ``hits`` list. Exceptions are reported and score zero, never dropped.
    Ground-truth answers and timestamps are never passed to ``search_fn``.
    """
    k = _count(k, "k")
    if k < 1:
        raise ValueError("k must be positive")
    validated = _questions(questions)
    rows = []
    for question in validated:
        error, spans = None, []
        try:
            hits = search_fn(question["question"], k=k)
            if isinstance(hits, Mapping):
                hits = hits.get("hits")
            if not isinstance(hits, (list, tuple)):
                raise ValueError("search_fn must return a sequence of hits or {'hits': [...]} ")
            for hit in hits[:k]:
                if isinstance(hit, Mapping):
                    value = [hit.get("start"), hit.get("end")]
                else:
                    value = [getattr(hit, "start", None), getattr(hit, "end", None)]
                spans.append(_interval(value, "retrieval hit"))
        except Exception as exc:
            error, spans = f"{type(exc).__name__}: {exc}", []
        rows.append({
            "question_id": question["id"], "category": question["category"], "answerable": question["answerable"],
            "query": question["question"], "retrieval_error": error, **_grounding(question["evidence_intervals"], spans),
        })
    eligible = [row for row in rows if row["answerable"] and row["reference_interval_count"]]
    multi_hop = [row for row in eligible if row["reference_interval_count"] > 1]
    summary: dict[str, Any] = {
        "example_count": len(rows), "grounding_example_count": len(eligible), "multi_hop_example_count": len(multi_hop),
        "retrieval_error_count": sum(row["retrieval_error"] is not None for row in rows), "temporal_iou": _mean(eligible, "temporal_iou"),
    }
    for threshold in _THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        summary[f"hit_at_{k}_iou_{suffix}"] = _mean(eligible, f"temporal_hit_at_{suffix}")
        summary[f"evidence_recall_at_{k}_iou_{suffix}"] = _mean(eligible, f"temporal_recall_at_{suffix}")
        summary[f"all_evidence_hit_at_{k}_iou_{suffix}"] = _mean(eligible, f"all_evidence_hit_at_{suffix}")
        summary[f"multi_hop_success_at_{k}_iou_{suffix}"] = _mean(multi_hop, f"all_evidence_hit_at_{suffix}")
    return {"schema_version": "1.0", "k": k, "summary": summary, "per_question": rows, "notes": ["Independent original-question retrieval; reference evidence is used only for scoring.", *_NOTES[1:]]}
