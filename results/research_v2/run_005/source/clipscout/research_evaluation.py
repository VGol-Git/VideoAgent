"""Evaluate explicit event predictions and reviewer-scored reasoning.

This protocol deliberately does not turn word overlap, valid citations, or an
LLM's confidence into semantic correctness. References may be uncertain visual
transition brackets or ASR cue spans; neither implies frame/word-exact truth.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any

from .evaluation import _assignment, temporal_iou


MODALITIES = {"visual", "audio", "transcript", "text", "audiovisual"}
TASKS = {"timestamp_search", "reasoning", "onset", "unanswerable"}
TOLERANCES = (1, 2, 5)
ERROR_CATEGORIES = ("planner", "format", "evidence", "time_policy", "budget", "model_protocol", "backend", "infrastructure", "unknown")
MODEL_PROTOCOL_CATEGORIES = {"planner", "format", "evidence", "time_policy", "budget", "model_protocol"}

DIAGNOSTIC_DEFINITIONS = {
    "timestamp_search": "Explicit point requests require actual sampled visual timestamps within the predeclared tolerance, linked to final citations. Range requests require cited, modality-compatible source observations in every required reference support, and all cited/source intervals and observed frames inside the requested range. Transcript cases use cue intervals, not visual frames. Requested timestamps alone are never observations. Unknown telemetry remains null; known missing/error/abstention is failure. These are temporal compliance diagnostics, not semantic answer correctness or continuous range coverage.",
    "semantic_review_coverage": "A named review is present only with reviewer and method. Complete atom reviews require boolean correct/supported for every required atom plus extra_claims_supported; event review completeness requires event_identity_correct; negative controls require abstention_correct. Missing/error/abstained task outputs can have known zero task scores without being reviewed. Legacy accuracy_known therefore includes automatic failures and is not reviewed answer quality. semantic_reviewed_only includes only complete named semantic reviews, stays null with zero reviews, and reports its denominator alongside coverage.",
    "retrieval": "Candidates follow chronological search calls and returned rank; duplicate physical regions are removed. Recall@k is the maximum one-to-one fraction of required supports matched in the first k unique candidates. A hit requires compatible known modality and either an actual sampled timestamp in the reference interval or interval IoU>=0.3. These are localization diagnostics, not semantic correctness.",
    "reciprocal_rank": "Inverse rank of the first candidate matching any required support; zero for a recorded search with no relevant hit. Questions without required supports or without a recorded search have null retrieval metrics.",
    "ndcg_at_5": "Binary unique-support gain, discounted by log2(rank+1). At most one previously unmatched support earns gain for each candidate (highest geometric score breaks ambiguity); ideal DCG contains min(5, number of required supports) gains. Repeated evidence for the same support cannot increase gain.",
    "citation_geometry": "Unique final citation intervals with known/inferred-from-ledger modality are matched to required supports at interval IoU>=0.3. Precision counts matching citations divided by unique valid citations plus invalid citations. Support recall uses one-to-one assignment. Unknown modality earns no modality-qualified credit; temporal-only precision is reported separately. Neither is entailment or event-onset accuracy.",
    "cost": "Only observed finite nonnegative telemetry contributes. Missing fields remain null. Latency quantiles use linear interpolation at (n-1)*p; every distribution reports measured and missing counts. Counts include failed tool attempts where recorded. inspection_frames is selected model observations; decoded_frames is its retained legacy alias, not all codec work. raw_decoder_frames_decoded separately records codec decoding including seek preroll when available.",
    "duplicate_frames": "(selected inspection frame count - unique sampled timestamp count) / selected inspection frame count, only when both counts are observed and consistent. Zero selected frames and zero unique samples yield zero. Unobserved timestamps do not imply all frames were unique. This metric excludes codec seek preroll.",
    "rates": "Response failure and abstention rates use non-missing predictions as their denominator. operational_error_rate is a retained legacy alias of response_failure_rate, counting every status=error, including model/protocol rejection; it must not be described as infrastructure failure. error_breakdown separates terminal planner/format/evidence/time-policy/budget/model-protocol failures from backend and infrastructure exceptions; unknown provenance remains unknown. Tool failure rates use attempts with known ok/error outcomes. Format repair requires an explicitly classified format/validation rejection followed by another decision; unclassified errors leave that rate unknown. Decision-repair counts are reported separately.",
    "budget": "Hard limits are at most four attempted tools (or a stricter configured limit) and the configured selected-inspection-frame limit, default 16 model observations. Codec seek preroll can decode additional frames and is reported separately, not charged to this inspection budget. Reported counts are checked against trace counts; the larger tool count is used. Unknown telemetry cannot establish budget compliance.",
    "joint_reviewed": "Reasoning joint success requires a complete named semantic review, supported correct required/extra claims, and all required geometric supports. Event joint success requires an explicit boolean event-identity review and an explicit within-scope, modality-compatible timing prediction. Unreviewed quality remains null; citation times are never substituted for event predictions.",
}


def classify_prediction_error(prediction: Mapping[str, Any]) -> str | None:
    """Classify the recorded terminal failure; never reinterpret it as success.

    This is a transparent diagnostic taxonomy, not an inferred root-cause label.
    Backend and infrastructure categories require an explicit type/message;
    an invalid model response is not evidence of a hardware or service outage.
    """
    if prediction.get("status") != "error":
        return None
    kind = str(prediction.get("error_type", "")).lower()
    message = str(prediction.get("error", "")).lower()
    if kind == "planning_error":
        return "planner"
    if kind == "decision_budget":
        return "budget"
    if kind in {"tool_protocol_error", "tool_argument_error", "duplicate_tool_call"}:
        return "model_protocol"
    if kind == "audit_error":
        # An audit can fail before generation (backend) or after it (format /
        # validation). Preserve that recorded distinction instead of treating
        # every verifier exception as either a model mistake or an outage.
        decisions = prediction.get("decisions", [])
        for decision in reversed(decisions if isinstance(decisions, list) else []):
            if (isinstance(decision, Mapping) and decision.get("role") == "verifier"
                    and decision.get("error_type") in {"backend_error", "format_error", "validation_error"}):
                return classify_prediction_error({"status": "error", "error_type": decision["error_type"],
                                                  "error": decision.get("error", "")})
    if kind in {"backend_error", "generation_error", "model_load_error"} or any(
        term in message for term in ("cuda", "out of memory", "device-side assert", "cublas", "model weights")
    ):
        return "backend"
    if kind in {"infrastructure_error", "io_error", "network_error", "permission_error", "filenotfounderror", "permissionerror", "connectionerror", "timeouterror"} or any(
        term in message for term in ("permission denied", "no such file", "connection refused", "connection timed out", "disk full")
    ):
        return "infrastructure"
    if prediction.get("failure_origin") == "tool_execution":
        # A generic tool failure lacks enough provenance to distinguish media,
        # retrieval or inference runtime issues. Do not blame model formatting
        # merely because an older finalizer used invalid_model_response.
        return "unknown"
    if kind in {"format_error", "json_error", "parse_error", "jsondecodeerror"} or any(
        term in message for term in ("expecting ',' delimiter", "expecting property name", "expecting value", "unterminated string", "extra data:", "invalid json", "malformed json")
    ):
        return "format"
    if any(term in message for term in ("no tools remain", "budget exhausted", "frame budget", "tool budget", "tool limit")):
        return "budget"
    if any(term in message for term in ("temporal overview", "observation window", "outside requested", "outside the requested", "temporal constraint", "outside scope", "timestamp scope")):
        return "time_policy"
    if any(term in message for term in ("evidence ids", "evidence_id", "evidence id", "unknown evidence", "observed evidence", "unobserved evidence")):
        return "evidence"
    if kind in {"invalid_model_response", "validation_error", "schema_error"}:
        return "model_protocol"
    return "unknown"


def _error_breakdown(rows):
    observed = sum(row["status"] != "missing" for row in rows)
    counts = {category: sum(row.get("error_category") == category for row in rows) for category in ERROR_CATEGORIES}
    failed = sum(counts.values())
    protocol = sum(counts[c] for c in MODEL_PROTOCOL_CATEGORIES)
    runtime = counts["backend"] + counts["infrastructure"]
    return {"observed_prediction_count": observed, "error_prediction_count": failed,
            "category_counts": counts, "model_protocol_error_count": protocol,
            "runtime_error_count": runtime, "unknown_error_count": counts["unknown"],
            "response_failure_rate": failed / observed if observed else None,
            "model_protocol_failure_rate": protocol / observed if observed else None,
            "runtime_failure_rate": runtime / observed if observed else None}


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def _span(value: Any, label: str, *, point_allowed: bool = False) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain [start, end]")
    start, end = (_number(item, label) for item in value)
    if end < start or (end == start and not point_allowed):
        raise ValueError(f"{label} has invalid order or duration")
    return start, end


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _modality_matches(expected: str, actual: str) -> bool:
    # These relationships name available source modalities, not an assertion
    # that an ASR cue boundary is calibrated against an audio waveform.
    return actual == expected or actual == "audiovisual" or (
        expected in {"text", "transcript"} and actual in {"text", "transcript", "audio"}
    )


def validate_questions(questions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate annotation contracts without promoting draft references to gold."""
    result, seen = [], set()
    for q in questions:
        if not isinstance(q, Mapping):
            raise ValueError("questions must contain objects")
        identifier = _text(q.get("id"), "question.id")
        if identifier in seen:
            raise ValueError(f"duplicate question id: {identifier}")
        seen.add(identifier)
        _text(q.get("question"), f"{identifier}.question")
        if q.get("task") not in TASKS or q.get("split") not in {"dev", "test"}:
            raise ValueError(f"{identifier}: invalid task or split")
        if q.get("modality") not in MODALITIES or not isinstance(q.get("answerable"), bool):
            raise ValueError(f"{identifier}: invalid modality or answerable")
        if (q["task"] == "unanswerable") != (not q["answerable"]):
            raise ValueError(f"{identifier}: task and answerable disagree")
        supports = q.get("supports", [])
        atoms = q.get("claim_atoms", [])
        if not isinstance(supports, list) or not isinstance(atoms, list):
            raise ValueError(f"{identifier}: supports and claim_atoms must be lists")
        support_ids = set()
        for support in supports:
            sid = _text(support.get("id"), f"{identifier}.support.id")
            if sid in support_ids:
                raise ValueError(f"{identifier}: duplicate support id")
            support_ids.add(sid)
            _span([support.get("start"), support.get("end")], f"{identifier}.{sid}")
            if support.get("modality") not in MODALITIES or not isinstance(support.get("required", True), bool):
                raise ValueError(f"{identifier}: invalid support modality or required flag")
        atom_ids = set()
        for atom in atoms:
            aid = _text(atom.get("id"), f"{identifier}.atom.id")
            if aid in atom_ids:
                raise ValueError(f"{identifier}: duplicate atom id")
            atom_ids.add(aid)
            _text(atom.get("text"), f"{identifier}.{aid}.text")
            references = atom.get("support_ids", [])
            if not isinstance(references, list) or not references or any(sid not in support_ids for sid in references):
                raise ValueError(f"{identifier}.{aid}: unknown or missing supports")
        if q["task"] in {"reasoning", "timestamp_search"} and not atoms:
            raise ValueError(f"{identifier}: reasoning/timestamp_search requires a prewritten atomic rubric")
        if q["task"] == "timestamp_search":
            request = q.get("timestamp_request")
            if not isinstance(request, Mapping) or request.get("kind") not in {"points", "range"}:
                raise ValueError(f"{identifier}: timestamp_search requires points/range timestamp_request")
            if request["kind"] == "points":
                points = request.get("points")
                if not isinstance(points, list) or not points:
                    raise ValueError(f"{identifier}: timestamp_request.points must be nonempty")
                values = [_number(t, "timestamp point") for t in points]
                if len(set(values)) != len(values):
                    raise ValueError(f"{identifier}: duplicate timestamp points")
                tolerance = _number(request.get("point_tolerance_seconds"), "point tolerance")
                if not 0 < tolerance <= 1:
                    raise ValueError(f"{identifier}: point tolerance must be in (0, 1]")
            else:
                _span(request.get("scope"), f"{identifier}.timestamp_request.scope")
                if request.get("points"):
                    raise ValueError(f"{identifier}: range request cannot also specify points")
        if q["task"] == "onset":
            event = q.get("event")
            if not isinstance(event, Mapping):
                raise ValueError(f"{identifier}: onset requires event reference")
            for field in ("label", "selector", "annotation_status"):
                _text(event.get(field), f"{identifier}.event.{field}")
            if event.get("reference_kind") not in {"transcript_cue", "visual_transition_bracket", "audio_alignment_bracket", "text_cue"}:
                raise ValueError(f"{identifier}: invalid event reference kind")
            scope = _span(event.get("scope"), f"{identifier}.event.scope")
            ref = _span(event.get("reference_bracket"), f"{identifier}.event.reference_bracket", point_allowed=True)
            if ref[0] < scope[0] or ref[1] > scope[1]:
                raise ValueError(f"{identifier}: event reference outside scope")
        result.append({**q, "supports": supports, "claim_atoms": atoms})
    return result


def _by_id(records: Sequence[Mapping[str, Any]], ids: set[str], label: str) -> dict[str, Mapping[str, Any]]:
    result = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} records must be objects")
        identifier = record.get("question_id")
        if identifier not in ids or identifier in result:
            raise ValueError(f"{label}: unknown or duplicate question_id {identifier!r}")
        result[identifier] = record
    return result


def _support_coverage(q: Mapping[str, Any], p: Mapping[str, Any]) -> dict[str, Any]:
    supports = [s for s in q["supports"] if s.get("required", True)]
    citations, invalid, unknown_modality = [], 0, 0
    if p.get("status") == "answered":
        raw = p.get("citations", [])
        if not isinstance(raw, list):
            raw, invalid = [], 1
        for c in raw:
            try:
                span = _span([c.get("start"), c.get("end")], "citation")
                modality = c.get("modality")
                if modality is not None and modality not in MODALITIES:
                    raise ValueError("invalid citation modality")
                if modality is None:
                    unknown_modality += 1
                # Duplicate IDs or interval spellings cannot inflate coverage.
                if not any(previous_span == span for previous_span, _ in citations):
                    citations.append((span, modality))
            except (ValueError, AttributeError):
                invalid += 1
    weights = [[temporal_iou([s["start"], s["end"]], span)
                if modality is None or _modality_matches(s["modality"], modality) else 0.0
                for span, modality in citations] for s in supports]
    matched = _assignment([[float(v >= 0.3) for v in row] for row in weights])
    matched_ids = [supports[row]["id"] for row, _ in matched]
    return {
        "required_support_count": len(supports),
        "matched_support_ids": matched_ids,
        "support_recall_at_0_3": len(matched) / len(supports) if supports else None,
        "all_supports_hit_at_0_3": float(len(matched) == len(supports)) if supports else None,
        "citation_count": len(citations), "invalid_citation_count": invalid,
        "citation_unknown_modality_count": unknown_modality,
    }


def _distance(point: float, span: tuple[float, float]) -> float:
    return max(span[0] - point, point - span[1], 0.0)


def _event_score(q: Mapping[str, Any], p: Mapping[str, Any], review: Mapping[str, Any] | None) -> dict[str, Any]:
    reference = q["event"]
    ref = _span(reference["reference_bracket"], "reference", point_allowed=True)
    scope = _span(reference["scope"], "scope")
    result = {
        "reference_kind": reference["reference_kind"],
        "annotation_status": reference["annotation_status"],
        "reference_bracket": list(ref), "reference_width_seconds": ref[1] - ref[0],
        "explicit_prediction": False, "prediction_valid": False,
        "modality_matches": False, "within_scope": False,
        "event_identity_reviewed": review.get("event_identity_correct") if review else None,
        "absolute_error_to_reference_bracket_seconds": None,
        "worst_endpoint_error_seconds": None,
        "predicted_bracket_width_seconds": None, "bracket_covers_reference": None,
        **{f"success_at_{tol}s": 0.0 for tol in TOLERANCES},
    }
    if p.get("status") != "answered":
        result["failure_reason"] = "missing_abstained_or_error"
        return result
    event = p.get("event")
    if not isinstance(event, Mapping):
        result["failure_reason"] = "no_explicit_event_prediction"
        return result
    result["explicit_prediction"] = True
    try:
        if event.get("modality") not in MODALITIES:
            raise ValueError("event modality is missing or invalid")
        time = event.get("time_seconds")
        bracket = event.get("bracket")
        if time is None and bracket is None:
            raise ValueError("event requires time_seconds or bracket")
        if time is not None:
            time = _number(time, "event.time_seconds")
        if bracket is not None:
            predicted = _span(bracket, "event.bracket", point_allowed=True)
            if time is not None and not predicted[0] <= time <= predicted[1]:
                raise ValueError("event.time_seconds is outside its bracket")
            result["predicted_bracket_width_seconds"] = predicted[1] - predicted[0]
            result["bracket_covers_reference"] = predicted[0] <= ref[0] and predicted[1] >= ref[1]
        else:
            predicted = (time, time)
        estimate = time if time is not None else predicted[0] + (predicted[1] - predicted[0]) / 2
        result.update({
            "prediction_valid": True,
            "modality_matches": _modality_matches(q["modality"], event["modality"]),
            "within_scope": scope[0] <= predicted[0] <= predicted[1] <= scope[1],
            "predicted_bracket": list(predicted), "estimate_seconds": estimate,
            "absolute_error_to_reference_bracket_seconds": _distance(estimate, ref),
            "worst_endpoint_error_seconds": max(_distance(x, ref) for x in predicted),
        })
        eligible = result["modality_matches"] and result["within_scope"] and result["event_identity_reviewed"] is not False
        for tolerance in TOLERANCES:
            # Both endpoints must lie within the tolerated reference bracket.
            # A giant interval with a lucky midpoint never receives full credit.
            result[f"success_at_{tolerance}s"] = float(eligible and result["worst_endpoint_error_seconds"] <= tolerance)
        if not eligible:
            result["failure_reason"] = "modality_scope_or_reviewed_identity_mismatch"
    except ValueError as error:
        result["failure_reason"] = str(error)
    return result


def _semantic_score(q: Mapping[str, Any], p: Mapping[str, Any], review: Mapping[str, Any] | None) -> dict[str, Any]:
    atoms = q["claim_atoms"]
    result = {"reviewer": review.get("reviewer") if review else None,
              "method": review.get("method") if review else None,
              "atom_count": len(atoms), "reviewed_atom_count": 0,
              "atom_judgments": [], "semantic_success": None,
              "supported_semantic_success": None, "abstention_correct": None}
    if q["task"] == "unanswerable":
        if p.get("status") in {"missing", "error", "answered"}:
            result["abstention_correct"] = 0.0
        elif review and review.get("abstention_correct") is not None:
            result["abstention_correct"] = float(review["abstention_correct"])
        return result
    if q["task"] not in {"reasoning", "timestamp_search"}:
        return result
    if p.get("status") != "answered":
        result.update(semantic_success=0.0, supported_semantic_success=0.0)
        return result
    judgments = {j["atom_id"]: j for j in review.get("atom_judgments", [])} if review else {}
    for atom in atoms:
        judgment = judgments.get(atom["id"], {})
        correct, supported = judgment.get("correct"), judgment.get("supported")
        result["atom_judgments"].append({"atom_id": atom["id"], "correct": correct, "supported": supported})
        result["reviewed_atom_count"] += int(correct is not None and supported is not None)
    correct = [j["correct"] for j in result["atom_judgments"]]
    supported = [j["supported"] for j in result["atom_judgments"]]
    extras = review.get("extra_claims_supported") if review else None
    if False in correct or extras is False:
        result["semantic_success"] = 0.0
    elif all(v is True for v in correct) and extras is True:
        result["semantic_success"] = 1.0
    if result["semantic_success"] == 0.0 or False in supported:
        result["supported_semantic_success"] = 0.0
    elif result["semantic_success"] == 1.0 and all(v is True for v in supported):
        result["supported_semantic_success"] = 1.0
    return result


def _validate_review(q: Mapping[str, Any], review: Mapping[str, Any]) -> None:
    for field in ("reviewer", "method"):
        _text(review.get(field), f"review.{field}")
    for field in ("abstention_correct", "event_identity_correct", "extra_claims_supported"):
        if review.get(field) is not None and not isinstance(review[field], bool):
            raise ValueError(f"review.{field} must be boolean or null")
    judgments = review.get("atom_judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("review.atom_judgments must be a list")
    ids, seen = {a["id"] for a in q["claim_atoms"]}, set()
    for item in judgments:
        aid = item.get("atom_id")
        if aid not in ids or aid in seen:
            raise ValueError("review has unknown or duplicate atom_id")
        seen.add(aid)
        for field in ("correct", "supported"):
            if item.get(field) is not None and not isinstance(item[field], bool):
                raise ValueError(f"review.{field} must be boolean or null")


def _known_summary(values: list[float | None]) -> dict[str, Any]:
    known = [v for v in values if v is not None]
    n, total = len(values), sum(known)
    return {"question_count": n, "known_count": len(known), "unknown_count": n - len(known),
            "success_count": total,
            "accuracy_all": total / n if n and len(known) == n else None,
            "accuracy_known": total / len(known) if known else None,
            "accuracy_lower_bound": total / n if n else None,
            "accuracy_upper_bound": (total + n - len(known)) / n if n else None}


def _observed_number(value, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    if integer and int(value) != value:
        return None
    return int(value) if integer else float(value)


def _distribution(values):
    known = sorted(v for v in values if v is not None)
    def quantile(fraction):
        if not known:
            return None
        position = (len(known) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return known[low] + (known[high] - known[low]) * (position - low)
    return {"measured_count": len(known), "missing_count": len(values) - len(known),
            "sum": sum(known) if known else None,
            "mean": statistics.mean(known) if known else None,
            "median": statistics.median(known) if known else None,
            "p90": quantile(.90), "p95": quantile(.95)}


def _source_modalities(record):
    aliases = {"caption": "visual", "ocr": "visual", "dense_visual": "visual"}
    names = []
    for field in ("modality", "source"):
        if isinstance(record.get(field), str):
            names.extend(record[field].split("+"))
    if isinstance(record.get("modalities"), list):
        names.extend(x for x in record["modalities"] if isinstance(x, str))
    return tuple(sorted({aliases.get(x, x) for x in names if aliases.get(x, x) in MODALITIES}))


def _candidate_regions(record, *, citation=False, ledger=None):
    """Represent source-specific geometry, not the union of fused modalities."""
    if not isinstance(record, Mapping):
        raise ValueError("evidence item must be an object")
    if not citation and isinstance(record.get("evidence"), list) and record["evidence"]:
        return [r for item in record["evidence"] for r in _candidate_regions(item)]
    start, end = _span([record.get("start"), record.get("end")], "evidence interval")
    modalities = _source_modalities(record)
    if citation and not modalities and ledger is not None:
        evidence = ledger.get(record.get("evidence_id"))
        if isinstance(evidence, Mapping):
            modalities = _source_modalities(evidence)
    samples = ()
    if not citation and record.get("sampled_timestamps") is not None:
        values = record["sampled_timestamps"]
        if not isinstance(values, (list, tuple)):
            raise ValueError("sampled_timestamps must be a list")
        samples = tuple(sorted({_number(value, "sample timestamp") for value in values}))
        if any(not start <= value < end for value in samples):
            raise ValueError("sample outside evidence interval")
    return [(start, end, modalities, samples)]


def _geometry(support, regions, *, ignore_modality=False):
    best = 0.
    for start, end, modalities, samples in regions:
        if not ignore_modality and not any(_modality_matches(support["modality"], m) for m in modalities):
            continue
        if samples:
            score = float(any(support["start"] <= t < support["end"] for t in samples))
        else:
            score = temporal_iou([support["start"], support["end"]], [start, end])
        best = max(best, score)
    return best


def _rank_metrics(supports, candidates):
    if not supports:
        return {**{f"recall_at_{k}": None for k in (1, 3, 5)}, "reciprocal_rank": None,
                "ndcg_at_5": None, "unique_support_coverage": None, "matched_support_ids": []}
    weights = [[_geometry(support, regions) for regions in candidates] for support in supports]
    result = {}
    for k in (1, 3, 5):
        pairs = _assignment([[float(value >= .3) for value in row[:k]] for row in weights])
        result[f"recall_at_{k}"] = len(pairs) / len(supports)
    relevant = [any(row[column] >= .3 for row in weights) for column in range(len(candidates))]
    result["reciprocal_rank"] = next((1 / (i + 1) for i, match in enumerate(relevant) if match), 0.)
    consumed, dcg = set(), 0.
    for column in range(min(5, len(candidates))):
        possible = [i for i in range(len(supports)) if i not in consumed and weights[i][column] >= .3]
        if possible:
            selected = max(possible, key=lambda i: (weights[i][column], -i))
            consumed.add(selected)
            dcg += 1 / math.log2(column + 2)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(supports))))
    result["ndcg_at_5"] = dcg / ideal
    pairs = _assignment([[float(value >= .3) for value in row] for row in weights])
    result["matched_support_ids"] = [supports[i]["id"] for i, _ in pairs]
    result["unique_support_coverage"] = len(pairs) / len(supports)
    return result


def _retrieval_diagnostics(q, p):
    supports = [s for s in q["supports"] if s.get("required", True)]
    trace = p.get("trace")
    searches = [row for row in trace if isinstance(row, Mapping) and row.get("tool") == "search"] if isinstance(trace, list) else []
    all_candidates, seen, per_search = [], set(), []
    invalid, unknown, duplicate = 0, 0, 0
    for number, call in enumerate(searches, 1):
        result = call.get("result", {})
        hits = result.get("hits", []) if isinstance(result, Mapping) else []
        if not isinstance(hits, list):
            invalid += 1
            hits = []
        candidates, local_seen = [], set()
        for hit in hits:
            try:
                regions = _candidate_regions(hit)
            except (ValueError, TypeError, AttributeError):
                invalid += 1
                continue
            identity = tuple(sorted(regions))
            if identity in local_seen:
                duplicate += 1
                continue
            local_seen.add(identity)
            candidates.append(regions)
            unknown += int(not any(region[2] for region in regions))
            if identity not in seen:
                all_candidates.append(regions)
                seen.add(identity)
            else:
                duplicate += 1
        args = call.get("args", {})
        per_search.append({"search_number": number, "query": args.get("query") if isinstance(args, Mapping) else None,
                           "unique_candidate_count": len(candidates), **_rank_metrics(supports, candidates)})
    metrics = _rank_metrics(supports, all_candidates) if searches else _rank_metrics([], [])
    return {"required_support_count": len(supports), "trace_available": isinstance(trace, list),
            "search_call_count": len(searches) if isinstance(trace, list) else None,
            "unique_candidate_count": len(all_candidates), "invalid_candidate_count": invalid,
            "unknown_modality_candidate_count": unknown, "duplicate_candidate_count": duplicate,
            "per_search": per_search, **metrics}


def _citation_diagnostics(q, p):
    supports = [s for s in q["supports"] if s.get("required", True)]
    raw = p.get("citations", []) if p.get("status") == "answered" else []
    invalid, duplicates, seen, candidates = 0, 0, set(), []
    if not isinstance(raw, list):
        invalid, raw = 1, []
    ledger = p.get("evidence") if isinstance(p.get("evidence"), Mapping) else None
    for item in raw:
        try:
            regions = _candidate_regions(item, citation=True, ledger=ledger)
        except (ValueError, TypeError, AttributeError):
            invalid += 1
            continue
        identity = tuple(sorted(regions))
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        candidates.append(regions)
    weights = [[_geometry(s, c) for c in candidates] for s in supports]
    temporal = [[_geometry(s, c, ignore_modality=True) for c in candidates] for s in supports]
    matching = sum(any(row[i] >= .3 for row in weights) for i in range(len(candidates)))
    matching_temporal = sum(any(row[i] >= .3 for row in temporal) for i in range(len(candidates)))
    pairs = _assignment([[float(value >= .3) for value in row] for row in weights])
    denominator = len(candidates) + invalid
    return {"required_support_count": len(supports), "unique_citation_count": len(candidates),
            "invalid_citation_count": invalid, "duplicate_citation_count": duplicates,
            "unknown_modality_citation_count": sum(not any(r[2] for r in c) for c in candidates),
            "precision_denominator": denominator, "matching_citation_count": matching,
            "matching_temporal_only_count": matching_temporal,
            "precision_at_0_3": matching / denominator if supports and denominator else None,
            "temporal_only_precision_at_0_3": matching_temporal / denominator if supports and denominator else None,
            "support_recall_at_0_3": len(pairs) / len(supports) if supports else None,
            "all_supports_hit_at_0_3": float(len(pairs) == len(supports)) if supports else None,
            "matched_support_ids": [supports[i]["id"] for i, _ in pairs]}


def _cost_diagnostics(p, max_tool_calls, max_frames):
    trace = p.get("trace")
    trace = trace if isinstance(trace, list) else None
    usage = p.get("usage") if isinstance(p.get("usage"), Mapping) else {}
    reported_tools = _observed_number(p.get("tool_calls"), integer=True)
    traced_tools = len(trace) if trace is not None else None
    tools = max(v for v in (reported_tools, traced_tools) if v is not None) if any(v is not None for v in (reported_tools, traced_tools)) else None
    frames = _observed_number(p.get("frames_used", p.get("frames_decoded")), integer=True)
    samples = p.get("unique_sampled_timestamps")
    unique = None
    if isinstance(samples, list) and all(_observed_number(t) is not None for t in samples):
        unique = len(set(samples))
    errors = []
    if reported_tools is not None and traced_tools is not None and reported_tools != traced_tools:
        errors.append("tool_calls differs from recorded trace length")
    if frames is not None and unique is not None and unique > frames:
        errors.append("unique frame timestamps exceed reported decoded frames")
    duplicates = (frames - unique) if frames is not None and unique is not None and unique <= frames else None
    fraction = duplicates / frames if duplicates is not None and frames else (0. if duplicates == 0 else None)
    tool_rows = []
    for call in trace or []:
        if not isinstance(call, Mapping):
            errors.append("malformed tool trace entry")
            continue
        failed = (not call["ok"]) if isinstance(call.get("ok"), bool) else (True if call.get("error") else None)
        tool_rows.append({"tool": str(call.get("tool", "unknown")), "failed": failed,
                          "latency_seconds": _observed_number(call.get("latency_seconds"))})
    decisions = p.get("decisions")
    repair_attempts = format_repairs = format_rejections = unknown_rejections = None
    if isinstance(decisions, list):
        repair_attempts = format_repairs = format_rejections = unknown_rejections = 0
        for i, decision in enumerate(decisions):
            if not isinstance(decision, Mapping) or not decision.get("error"):
                continue
            retry = i < len(decisions) - 1
            repair_attempts += int(retry)
            category = decision.get("error_type")
            if category in {"format_error", "validation_error", "schema_error", "json_error", "parse_error"}:
                format_rejections += 1
                format_repairs += int(retry)
            elif category not in {"backend_error", "generation_error", "operational_error"}:
                unknown_rejections += 1
        if unknown_rejections:
            format_repairs = None
    tool_violation = tools > max_tool_calls if tools is not None else None
    frame_violation = frames > max_frames if frames is not None else None
    violation = True if True in (tool_violation, frame_violation) else (None if None in (tool_violation, frame_violation) else False)
    return {"latency_seconds": _observed_number(p.get("latency_seconds")),
            **{name: _observed_number(usage.get(name), integer=True) for name in ("model_calls", "input_tokens", "output_tokens")},
            "tool_calls": tools, "reported_tool_calls": reported_tools, "traced_tool_calls": traced_tools,
            "decoded_frames": frames, "inspection_frames": frames,
            "raw_decoder_frames_decoded": _observed_number(p.get("raw_decoder_frames_decoded"), integer=True),
            "unique_frames": unique, "duplicate_frames": duplicates,
            "duplicate_frame_fraction": fraction, "tools": tool_rows,
            "decision_repair_attempt_count": repair_attempts,
            "format_repair_attempt_count": format_repairs, "format_rejection_count": format_rejections,
            "unclassified_decision_rejection_count": unknown_rejections,
            "tool_limit": max_tool_calls, "frame_limit": max_frames,
            "tool_budget_violation": tool_violation, "frame_budget_violation": frame_violation,
            "hard_budget_violation": violation, "telemetry_inconsistencies": errors}


def _timestamp_score(q, p):
    """Measure observed temporal grounding without inferring answer semantics."""
    request = q["timestamp_request"]
    points = request.get("points", [])
    scope = request.get("scope")
    tolerance = request.get("point_tolerance_seconds", 0.)
    visual = q["modality"] in {"visual", "audiovisual"}
    ledger = p.get("evidence")
    ledger = ledger if isinstance(ledger, Mapping) else None
    citations = p.get("citations", [])
    citations = citations if isinstance(citations, list) else [None]
    source_regions, linked_regions, invalid_sources, invalid_citations = [], [], 0, 0
    citation_ok, citation_widths = [], []

    def in_scope(start, end):
        return (scope[0] <= start <= end <= scope[1]) if scope else any(start <= t < end for t in points)

    def regions_with_cues(item):
        regions = _candidate_regions(item)
        if isinstance(item.get("transcript_segments"), list):
            for cue in item["transcript_segments"]:
                a, b = _span([cue.get("start"), cue.get("end")], "transcript cue")
                # A source window only exposes its intersecting cue portion.
                for start, end, modalities, _ in regions[:1]:
                    if any(_modality_matches("transcript", m) for m in modalities):
                        lo, hi = max(a, start), min(b, end)
                        if lo < hi:
                            regions.append((lo, hi, ("transcript",), ()))
        return regions

    for item in (ledger or {}).values():
        try:
            source_regions.extend(regions_with_cues(item))
        except (ValueError, TypeError, AttributeError):
            invalid_sources += 1
    for citation in citations:
        try:
            start, end = _span([citation.get("start"), citation.get("end")], "citation")
            citation_ok.append(in_scope(start, end))
            citation_widths.append(end - start)
            evidence = ledger.get(citation.get("evidence_id")) if ledger is not None else None
            if isinstance(evidence, Mapping):
                for a, b, modalities, samples in regions_with_cues(evidence):
                    lo, hi = max(a, start), min(b, end)
                    if lo < hi:
                        linked_regions.append((lo, hi, modalities, tuple(t for t in samples if lo <= t < hi)))
        except (ValueError, TypeError, AttributeError):
            invalid_citations += 1

    raw_samples = p.get("unique_sampled_timestamps")
    observed_samples = None
    if isinstance(raw_samples, list) and all(_observed_number(t) is not None for t in raw_samples):
        observed_samples = sorted(set(raw_samples))
    # Samples explicitly recorded in retrieved evidence are valid source
    # observations too, but are not counted as newly decoded frame cost.
    source_samples = sorted({t for _, _, _, samples in source_regions for t in samples})
    if observed_samples is not None:
        observed_samples = sorted(set(observed_samples).union(source_samples))
    elif source_samples:
        observed_samples = source_samples
    elif p.get("frames_used", p.get("frames_decoded")) == 0:
        observed_samples = []
    compatible = [r for r in linked_regions if any(_modality_matches(q["modality"], m) for m in r[2])]
    linked_samples = sorted({t for _, _, _, samples in compatible for t in samples})
    citation_count = len(citation_ok) + invalid_citations
    citation_compliance = (sum(citation_ok) / citation_count) if citation_count else 0.
    source_compliance = (float(not invalid_sources and all(in_scope(a, b) for a, b, _, _ in source_regions))
                         if ledger is not None else None)
    frame_compliance = None
    if visual and observed_samples is not None:
        frame_compliance = float(all((scope[0] <= t <= scope[1]) if scope else any(abs(t - point) <= tolerance for point in points)
                                     for t in observed_samples))
    required_supports = [s for s in q["supports"] if s.get("required", True)]
    support_hits = []
    for support in required_supports:
        regions = [r for r in compatible if any(_modality_matches(support["modality"], m) for m in r[2])]
        if support["modality"] == "visual":
            hit = any(support["start"] <= t < support["end"] for r in regions for t in r[3])
        else:
            hit = any(temporal_iou([support["start"], support["end"]], [r[0], r[1]]) >= .3 for r in regions)
        support_hits.append(bool(hit))
    observation_known = bool(linked_samples) or observed_samples is not None if visual else ledger is not None
    if points:
        point_hits = [any(abs(t - point) <= tolerance for t in linked_samples) if visual else
                      any(a <= point < b for a, b, _, _ in compatible) for point in points]
        target_coverage = sum(point_hits) / len(points) if observation_known else None
    else:
        point_hits = []
        target_coverage = sum(support_hits) / len(support_hits) if observation_known and support_hits else None
    temporal = None
    components = [citation_compliance, source_compliance, target_coverage]
    if visual:
        components.append(frame_compliance)
    if p.get("status") != "answered" or any(v == 0 for v in components):
        temporal = 0.
    elif all(v is not None for v in components):
        temporal = float(all(v == 1 for v in components))
    distances = [min(abs(t - point) for t in observed_samples) for point in points] if observed_samples else []
    return {"kind": request["kind"], "requested_points": points, "requested_scope": scope,
            "point_tolerance_seconds": tolerance if points else None,
            "actual_frame_timestamps_observed": observed_samples,
            "observed_frame_count": len(observed_samples) if observed_samples is not None else None,
            "cited_compatible_source_count": len(compatible), "cited_actual_frame_count": len(linked_samples),
            "citation_count": citation_count, "invalid_citation_count": invalid_citations,
            "invalid_source_count": invalid_sources, "citation_scope_precision": citation_compliance,
            "source_scope_compliance": source_compliance, "frame_scope_compliance": frame_compliance,
            "point_hits": point_hits if observation_known else None,
            "target_coverage": target_coverage, "temporal_success": temporal,
            "required_observation_count": len(required_supports),
            "observed_support_ids": [s["id"] for s, hit in zip(required_supports, support_hits) if hit],
            "mean_point_distance_seconds_conditional": statistics.mean(distances) if distances else None,
            "max_citation_width_seconds": max(citation_widths) if citation_widths else None}


def _review_coverage(q, semantic, review):
    complete = False
    if review:
        if q["task"] in {"reasoning", "timestamp_search"}:
            complete = semantic["reviewed_atom_count"] == semantic["atom_count"] and isinstance(review.get("extra_claims_supported"), bool)
        elif q["task"] == "onset":
            complete = isinstance(review.get("event_identity_correct"), bool)
        else:
            complete = isinstance(review.get("abstention_correct"), bool)
    return {"named_review_present": review is not None, "complete_semantic_review": complete}


def _joint_diagnostics(row, review):
    result = {"reasoning_and_support_success": None,
              "timestamp_content_and_temporal_success": None,
              **{f"event_identity_and_timing_success_at_{t}s": None for t in TOLERANCES}}
    if row["task"] == "reasoning" and review:
        complete = row["semantic"]["reviewed_atom_count"] == row["semantic"]["atom_count"] and isinstance(review.get("extra_claims_supported"), bool)
        semantic = row["semantic"]["supported_semantic_success"]
        geometry = row["diagnostics"]["citation_geometry"]["all_supports_hit_at_0_3"]
        if complete and semantic is not None and geometry is not None:
            result["reasoning_and_support_success"] = float(bool(semantic) and bool(geometry))
    if row["task"] == "onset" and review and isinstance(review.get("event_identity_correct"), bool):
        for tolerance in TOLERANCES:
            result[f"event_identity_and_timing_success_at_{tolerance}s"] = float(review["event_identity_correct"] and row["event"][f"success_at_{tolerance}s"])
    if row["task"] == "timestamp_search" and row["review_coverage"]["complete_semantic_review"]:
        semantic = row["semantic"]["supported_semantic_success"]
        temporal = row["timestamp_search"]["temporal_success"]
        if semantic is not None and temporal is not None:
            result["timestamp_content_and_temporal_success"] = float(bool(semantic) and bool(temporal))
    return result


def _aggregate_diagnostics(rows):
    n = len(rows)
    observed = [r for r in rows if r["status"] != "missing"]
    costs = [r["diagnostics"]["cost"] for r in rows]
    retrieval = [r["diagnostics"]["retrieval"] for r in rows]
    citations = [r["diagnostics"]["citation_geometry"] for r in rows]
    tool_names = sorted({t["tool"] for c in costs for t in c["tools"]})
    by_tool = {}
    for name in tool_names:
        attempts = [t for c in costs for t in c["tools"] if t["tool"] == name]
        known = [t for t in attempts if t["failed"] is not None]
        failed = sum(t["failed"] for t in known)
        by_tool[name] = {"attempt_count": len(attempts), "known_outcome_count": len(known),
                         "failure_count": failed, "failure_rate": failed / len(known) if known else None,
                         "latency_seconds": _distribution([t["latency_seconds"] for t in attempts])}
    matched = sum(c["matching_citation_count"] for c in citations if c["required_support_count"])
    citation_denominator = sum(c["precision_denominator"] for c in citations if c["required_support_count"])
    duplicates = [(c["duplicate_frames"], c["decoded_frames"]) for c in costs if c["duplicate_frames"] is not None]
    duplicate_denominator = sum(total for _, total in duplicates)
    format_known = [c for c in costs if c["format_repair_attempt_count"] is not None]
    budget_known = [c for c in costs if c["hard_budget_violation"] is not None]
    return {
        "retrieval": {"annotated_question_count": sum(bool(r["required_support_count"]) for r in retrieval),
            "recorded_search_question_count": sum(bool(r["search_call_count"]) for r in retrieval),
            **{metric: _distribution([r[metric] for r in retrieval]) for metric in
               ("recall_at_1", "recall_at_3", "recall_at_5", "reciprocal_rank", "ndcg_at_5", "unique_support_coverage")}},
        "citation_geometry": {"annotated_question_count": sum(bool(c["required_support_count"]) for c in citations),
            "matching_citation_count": matched, "precision_denominator": citation_denominator,
            "micro_precision_at_0_3": matched / citation_denominator if citation_denominator else None,
            **{metric: _distribution([c[metric] for c in citations]) for metric in
               ("precision_at_0_3", "temporal_only_precision_at_0_3", "support_recall_at_0_3", "all_supports_hit_at_0_3")}},
        "cost": {**{metric: _distribution([c[metric] for c in costs]) for metric in
                    ("latency_seconds", "model_calls", "input_tokens", "output_tokens", "decoded_frames", "inspection_frames", "raw_decoder_frames_decoded",
                     "unique_frames", "tool_calls", "duplicate_frame_fraction", "decision_repair_attempt_count")},
            "aggregate_duplicate_frame_fraction": sum(d for d, _ in duplicates) / duplicate_denominator if duplicate_denominator else (0. if duplicates else None),
            "duplicate_frame_denominator": duplicate_denominator,
            "duplicate_frame_measured_question_count": len(duplicates), "per_tool": by_tool,
            "telemetry_inconsistent_question_count": sum(bool(c["telemetry_inconsistencies"]) for c in costs)},
        "rates": {"question_count": n, "observed_prediction_count": len(observed),
            "missing_prediction_count": n - len(observed),
            "response_failure_rate": sum(r["status"] == "error" for r in observed) / len(observed) if observed else None,
            "operational_error_rate": sum(r["status"] == "error" for r in observed) / len(observed) if observed else None,
            "abstention_rate": sum(r["status"] == "insufficient_evidence" for r in observed) / len(observed) if observed else None,
            "format_repair_measured_count": len(format_known), "format_repair_unknown_count": n - len(format_known),
            "format_repair_rate": sum(c["format_repair_attempt_count"] > 0 for c in format_known) / len(format_known) if format_known else None,
            "hard_budget_measured_count": len(budget_known), "hard_budget_unknown_count": n - len(budget_known),
            "hard_budget_violation_count": sum(c["hard_budget_violation"] is True for c in costs),
            "hard_budget_violation_rate": sum(c["hard_budget_violation"] for c in budget_known) / len(budget_known) if budget_known else None},
        "joint_reviewed": {
            "reasoning_and_support": _known_summary([r["diagnostics"]["joint_reviewed"]["reasoning_and_support_success"] for r in rows if r["task"] == "reasoning"]),
            "timestamp_content_and_temporal": _known_summary([r["diagnostics"]["joint_reviewed"]["timestamp_content_and_temporal_success"] for r in rows if r["task"] == "timestamp_search"]),
            **{f"event_identity_and_timing_at_{t}s": _known_summary([r["diagnostics"]["joint_reviewed"][f"event_identity_and_timing_success_at_{t}s"] for r in rows if r["task"] == "onset"]) for t in TOLERANCES}},
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasoning = [r for r in rows if r["task"] == "reasoning"]
    timestamp = [r for r in rows if r["task"] == "timestamp_search"]
    negative = [r for r in rows if r["task"] == "unanswerable"]
    events = [r["event"] for r in rows if r["task"] == "onset"]
    errors = [e["absolute_error_to_reference_bracket_seconds"] for e in events
              if e["prediction_valid"] and e["modality_matches"]]
    widths = [e["predicted_bracket_width_seconds"] for e in events if e["predicted_bracket_width_seconds"] is not None]
    cover = [float(e["bracket_covers_reference"]) for e in events if e["bracket_covers_reference"] is not None]
    supports = [r["support_coverage"] for r in rows if r["answerable"] and r["support_coverage"]["required_support_count"]]
    return {
        "question_count": len(rows),
        "error_breakdown": _error_breakdown(rows),
        "status_counts": {s: sum(r["status"] == s for r in rows) for s in ("answered", "insufficient_evidence", "error", "missing")},
        "reasoning": _known_summary([r["semantic"]["semantic_success"] for r in reasoning]),
        "supported_reasoning": _known_summary([r["semantic"]["supported_semantic_success"] for r in reasoning]),
        "semantic_reviewed_only": {
            "reasoning": _known_summary([r["semantic"]["semantic_success"] for r in reasoning if r["review_coverage"]["complete_semantic_review"]]),
            "supported_reasoning": _known_summary([r["semantic"]["supported_semantic_success"] for r in reasoning if r["review_coverage"]["complete_semantic_review"]]),
            "timestamp_search": _known_summary([r["semantic"]["semantic_success"] for r in timestamp if r["review_coverage"]["complete_semantic_review"]]),
            "supported_timestamp_search": _known_summary([r["semantic"]["supported_semantic_success"] for r in timestamp if r["review_coverage"]["complete_semantic_review"]]),
            "negative_abstention": _known_summary([r["semantic"]["abstention_correct"] for r in negative if r["review_coverage"]["complete_semantic_review"]])},
        "timestamp_search": {
            "question_count": len(timestamp),
            "semantic": _known_summary([r["semantic"]["semantic_success"] for r in timestamp]),
            "supported_semantic": _known_summary([r["semantic"]["supported_semantic_success"] for r in timestamp]),
            "temporal": _known_summary([r["timestamp_search"]["temporal_success"] for r in timestamp]),
            **{field: _distribution([r["timestamp_search"][field] for r in timestamp]) for field in
               ("target_coverage", "citation_scope_precision", "source_scope_compliance", "frame_scope_compliance",
                "mean_point_distance_seconds_conditional", "max_citation_width_seconds")}},
        "review_coverage": {
            "question_count": len(rows),
            "named_review_count": sum(r["review_coverage"]["named_review_present"] for r in rows),
            "complete_semantic_review_count": sum(r["review_coverage"]["complete_semantic_review"] for r in rows),
            "complete_semantic_review_fraction": sum(r["review_coverage"]["complete_semantic_review"] for r in rows) / len(rows) if rows else None,
            "required_atom_count": sum(r["semantic"]["atom_count"] for r in rows),
            "reviewed_atom_count": sum(r["semantic"]["reviewed_atom_count"] for r in rows)},
        "negative_abstention": _known_summary([r["semantic"]["abstention_correct"] for r in negative]),
        "onset": {
            "question_count": len(events),
            "explicit_prediction_count": sum(e["explicit_prediction"] for e in events),
            "valid_prediction_count": sum(e["prediction_valid"] for e in events),
            "modality_mismatch_count": sum(e["prediction_valid"] and not e["modality_matches"] for e in events),
            "error_measured_count": len(errors),
            "mean_absolute_error_seconds_conditional": statistics.mean(errors) if errors else None,
            "median_absolute_error_seconds_conditional": statistics.median(errors) if errors else None,
            "bracket_count": len(widths), "mean_bracket_width_seconds": statistics.mean(widths) if widths else None,
            "reference_bracket_coverage_conditional": statistics.mean(cover) if cover else None,
            **{f"success_at_{t}s": sum(e[f"success_at_{t}s"] for e in events) / len(events) if events else None for t in TOLERANCES},
        },
        "support_geometry": {"question_count": len(supports),
            "macro_support_recall_at_0_3": statistics.mean(s["support_recall_at_0_3"] for s in supports) if supports else None,
            "all_supports_hit_at_0_3": statistics.mean(s["all_supports_hit_at_0_3"] for s in supports) if supports else None},
        "diagnostics": _aggregate_diagnostics(rows),
    }


def evaluate_research(questions: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
                      reviews: Sequence[Mapping[str, Any]] | None = None, *,
                      max_tool_calls: int = 4, max_frames: int = 16) -> dict[str, Any]:
    """Score a supplied split; missing predictions stay in every task denominator.

    Event predictions use ``event={time_seconds and/or bracket, modality}``.
    Review records require question_id, reviewer and method. Atomic reasoning
    judgments contain atom_id/correct/supported, plus extra_claims_supported for
    a complete answer audit. Unknown judgments remain unknown with score bounds.
    """
    if type(max_tool_calls) is not int or not 1 <= max_tool_calls <= 4:
        raise ValueError("max_tool_calls must be an integer from 1 to 4")
    if type(max_frames) is not int or not 0 <= max_frames <= 1024:
        raise ValueError("max_frames must be an integer from 0 to 1024")
    validated = validate_questions(questions)
    ids = {q["id"] for q in validated}
    predicted, reviewed = _by_id(predictions, ids, "prediction"), _by_id(reviews or [], ids, "review")
    rows = []
    for q in validated:
        p = predicted.get(q["id"], {"status": "missing"})
        if p.get("status") not in {"answered", "insufficient_evidence", "error", "missing"}:
            raise ValueError(f"{q['id']}: invalid prediction status")
        review = reviewed.get(q["id"])
        if review is not None:
            _validate_review(q, review)
        row = {"question_id": q["id"], "task": q["task"], "split": q["split"],
               "modality": q["modality"], "answerable": q["answerable"], "status": p["status"],
               "error_category": classify_prediction_error(p),
               "support_coverage": _support_coverage(q, p), "semantic": _semantic_score(q, p, review)}
        row["review_coverage"] = _review_coverage(q, row["semantic"], review)
        if q["task"] == "onset":
            row["event"] = _event_score(q, p, review)
        if q["task"] == "timestamp_search":
            row["timestamp_search"] = _timestamp_score(q, p)
        row["diagnostics"] = {"retrieval": _retrieval_diagnostics(q, p),
                              "citation_geometry": _citation_diagnostics(q, p),
                              "cost": _cost_diagnostics(p, max_tool_calls, max_frames)}
        row["diagnostics"]["joint_reviewed"] = _joint_diagnostics(row, review)
        rows.append(row)
    return {"protocol_version": "assignment-v7.0", "summary": _aggregate(rows),
            "by_task": {task: _aggregate([r for r in rows if r["task"] == task]) for task in sorted(TASKS)},
            "by_modality": {m: _aggregate([r for r in rows if r["modality"] == m]) for m in sorted({r["modality"] for r in rows})},
            "by_split": {s: _aggregate([r for r in rows if r["split"] == s]) for s in sorted({r["split"] for r in rows})},
            "per_question": rows,
            "diagnostic_definitions": DIAGNOSTIC_DEFINITIONS,
            "notes": [
                "Reasoning correctness comes only from named reviewer annotations; unknown judgments remain unknown. Missing/error/abstained answers are task failures.",
                "Timestamp-search content uses the same named atomic review; temporal compliance uses recorded actual frames and cited source intervals separately. Sampled range evidence is not continuous coverage.",
                "Support scores measure one-to-one citation geometry at IoU>=0.3, not semantic entailment; missing citation modalities are explicitly counted.",
                "Onset scores require an explicit event prediction; citation starts and numbers in prose are never substituted.",
                "Onset success uses distance of both predicted bracket endpoints to the uncertain reference bracket; wide brackets cannot win by containing everything.",
                "All onset questions remain in success denominators. Absolute-error means are conditional, include wrong-occurrence errors, and expose measured counts.",
                "ASR cue references measure transcript-cue localization, not verified word-level audio timing. Visual bracket precision is limited by its annotation width.",
                "Event timing metrics do not establish semantic event identity without an explicit identity review. Reference-bracket coverage must be read together with bracket width and tolerance success.",
                "This evaluator does not confer independence, blindness, human validation, or held-out status on the supplied data or reviews.",
            ]}
