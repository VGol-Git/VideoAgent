"""Evidence contracts for event localization, without model or decoder calls.

A sampled positive frame establishes presence at that frame. An absent frame
followed by a positive frame brackets a *local* transition, conditional on the
frame labels being correct. Neither proves the event's first appearance in the
whole video. Transcript matching has its own claim type and cue uncertainty.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from .index import finite_number, validate_interval


STATUSES = frozenset(("present", "absent", "uncertain"))
MAX_ALIASES = 8
MAX_OBSERVATIONS = 512


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def validate_event_query(event_query: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Validate one event phrase or up to eight explicit, alternative aliases.

    Alias expansion is caller supplied and recorded, never inferred from gold
    labels. The same contract is used for visual and transcript events.
    """
    values = [event_query] if isinstance(event_query, str) else event_query
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= MAX_ALIASES:
        raise ValueError("event_query must be a phrase or a list of 1–8 aliases")
    aliases = []
    normalized = set()
    for value in values:
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 500 or not _words(value):
            raise ValueError("Each event phrase must contain 1–500 characters and a word")
        phrase = value.strip()
        key = _words(phrase)
        if key in normalized:
            raise ValueError("Event aliases must be distinct")
        normalized.add(key)
        aliases.append(phrase)
    return tuple(aliases)


def _bounds(start, end) -> tuple[float, float]:
    start, end = finite_number(start, "start"), finite_number(end, "end")
    if not 0 <= start < end:
        raise ValueError("Event scope requires 0 <= start < end")
    return start, end


def _observation(value, start: float, end: float) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Frame observations must be objects")
    if set(value) - {"timestamp", "status", "support", "source_id", "modality"}:
        raise ValueError("Unknown frame observation field")
    timestamp = finite_number(value.get("timestamp"), "observation timestamp")
    if not start <= timestamp < end:
        raise ValueError("Observation timestamp lies outside the event scope")
    if not isinstance(value.get("status"), str) or value["status"] not in STATUSES:
        raise ValueError("Observation status must be present, absent, or uncertain")
    support = value.get("support", "")
    if not isinstance(support, str) or len(support) > 2000:
        raise ValueError("Observation support must be text of at most 2000 characters")
    if value.get("modality", "visual") != "visual":
        raise ValueError("Only visual observations can establish a visual transition")
    normalized = {"timestamp": timestamp, "status": value["status"], "support": support.strip(),
                  "modality": "visual"}
    if "source_id" in value:
        source = value["source_id"]
        if not isinstance(source, str) or not source.strip() or len(source) > 200:
            raise ValueError("source_id must be nonempty text of at most 200 characters")
        normalized["source_id"] = source
    return normalized


def validate_frame_observations(observations: list[dict], sampled_timestamps: Sequence[float],
                                start: float, end: float) -> list[dict]:
    """Require exactly one valid label for every actually supplied frame.

    Model timestamps may differ by at most one microsecond from a supplied
    timestamp; accepted values are replaced with the original decoder values.
    Missing, duplicate, fabricated, or out-of-scope timestamps are rejected.
    This verifies provenance and schema, not the model's visual judgement.
    """
    start, end = _bounds(start, end)
    if not isinstance(observations, list) or not 1 <= len(observations) <= MAX_OBSERVATIONS:
        raise ValueError("Provide 1–512 frame observations")
    if not isinstance(sampled_timestamps, (list, tuple)) or len(sampled_timestamps) != len(observations):
        raise ValueError("Every sampled frame needs exactly one observation")
    actual = [finite_number(t, "sampled timestamp") for t in sampled_timestamps]
    if len(set(actual)) != len(actual) or any(not start <= t < end for t in actual):
        raise ValueError("Sampled timestamps must be distinct and inside the event scope")
    remaining = set(actual)
    normalized = []
    for value in observations:
        observation = _observation(value, start, end)
        matches = [t for t in remaining if abs(t - observation["timestamp"]) <= 1e-6]
        if len(matches) != 1:
            raise ValueError("Observation must reference one distinct supplied frame timestamp")
        observation["timestamp"] = matches[0]
        remaining.remove(matches[0])
        normalized.append(observation)
    return sorted(normalized, key=lambda value: value["timestamp"])


def summarize_event(event_query, observations: list[dict], *, start: float, end: float,
                    history: Sequence[dict] = ()) -> dict:
    """Merge verified observations and describe the earliest sampled presence.

    Call ``validate_frame_observations`` on each new model response first.
    ``start``/``end`` are the overall localization scope, which can contain
    multiple narrower inspections. ``history`` contains prior observations in
    that same scope. Conflicting labels for one frame fail closed.

    A local onset bracket requires an absent observation immediately preceding
    the earliest positive one; an intervening uncertain label blocks that
    claim. Repeated appearances remain local evidence, never global-first proof.
    """
    aliases = validate_event_query(event_query)
    start, end = _bounds(start, end)
    if not isinstance(observations, (list, tuple)) or not isinstance(history, (list, tuple)):
        raise ValueError("observations and history must be lists or tuples")
    if len(observations) + len(history) > MAX_OBSERVATIONS:
        raise ValueError("Event history exceeds 512 observations")
    merged = {}
    for raw in [*history, *observations]:
        value = _observation(raw, start, end)
        previous = merged.get(value["timestamp"])
        if previous is not None:
            if previous["status"] != value["status"]:
                raise ValueError("Conflicting labels for the same frame; resolve before localization")
            continue
        merged[value["timestamp"]] = value
    ordered = sorted(merged.values(), key=lambda value: value["timestamp"])
    positive_index = next((i for i, value in enumerate(ordered) if value["status"] == "present"), None)
    first = ordered[positive_index] if positive_index is not None else None
    before = ordered[positive_index - 1] if positive_index is not None and positive_index > 0 else None
    bracket = ([before["timestamp"], first["timestamp"]]
               if before and before["status"] == "absent" else None)
    uncertain = any(value["status"] == "uncertain" for value in ordered)
    claim = "local_transition_bracket" if bracket else "first_observed" if first else "not_observed"
    preceding_absent = [value["timestamp"] for value in ordered
                        if first and value["timestamp"] < first["timestamp"] and value["status"] == "absent"]
    candidate = ([max(preceding_absent, default=start), first["timestamp"]] if first else None)
    if candidate and candidate[0] >= candidate[1]:
        candidate = None
    return {
        "event_query": list(aliases), "scope": {"start": start, "end": end}, "modality": "visual",
        "claim": claim, "first_observed_timestamp": first["timestamp"] if first else None,
        "onset_bracket": bracket, "bracket_width_seconds": bracket[1] - bracket[0] if bracket else None,
        "refinement_candidate": candidate, "global_first_verified": False,
        "sampled_timestamps": [value["timestamp"] for value in ordered], "observations": ordered,
        "uncertain_observations": sum(value["status"] == "uncertain" for value in ordered),
        "presence_status": "observed" if first else "uncertain" if uncertain else "not_observed",
        "coverage_note": (
            "Labels describe sampled frames only and may be mistaken. A local transition bracket is conditional "
            "on those labels, and does not prove the first occurrence in the video. Unseen gaps may contain "
            "earlier or repeated events. No positive sample is not proof of absence."
        ),
    }


def refinement_interval(localization: dict, *, previous_interval=None, min_width: float = 0.05):
    """Return a narrower candidate inspection interval, or None if exhausted.

    The right endpoint is an already-observed positive frame. The decoder can
    sample strictly inside the returned half-open interval while the historic
    positive frame remains available to the next summary. ``previous_interval``
    enforces monotonic refinement; an expanding proposal raises ValueError.
    """
    min_width = finite_number(min_width, "min_width")
    if min_width <= 0:
        raise ValueError("min_width must be positive")
    if not isinstance(localization, dict) or localization.get("modality") != "visual":
        raise ValueError("Refinement requires visual localization")
    scope = localization.get("scope", {})
    start, end = _bounds(scope.get("start"), scope.get("end"))
    if previous_interval is not None:
        if not isinstance(previous_interval, (list, tuple)) or len(previous_interval) != 2:
            raise ValueError("previous_interval must contain start and end")
        previous_start, previous_end = _bounds(*previous_interval)
        if not start <= previous_start < previous_end <= end:
            raise ValueError("Previous interval lies outside the localization scope")
    else:
        previous_start, previous_end = start, end
    candidate = localization.get("refinement_candidate")
    if candidate is None:
        return None
    if not isinstance(candidate, (list, tuple)) or len(candidate) != 2:
        raise ValueError("Malformed refinement candidate")
    left, right = _bounds(*candidate)
    if not previous_start <= left < right <= previous_end:
        raise ValueError("Refinement must not expand the previous inspection interval")
    if right - left < min_width or (left == previous_start and right == previous_end):
        return None
    return left, right


def transcript_event(index, event_query, start: float = 0, end: float | None = None,
                     match: str = "phrase", max_history: int = 20) -> dict:
    """Exhaustively match supplied aliases against whole, in-scope text cues.

    ``phrase`` matches a contiguous normalized word sequence; ``terms`` requires
    every query word in a cue. Aliases are alternatives in either mode. Matching
    is lexical and therefore does not prove that no paraphrased mention exists.
    Captions are never used as transcript evidence. Crossing boundary cues are
    excluded without clipping or relabeling their text. Return the earliest
    matching cue, immediate in-scope neighbors, and up to ``max_history`` matches.

    Cue boundaries are transcript estimates, not word-aligned audio onset. This
    result cannot establish when the event appears visually or actually occurs.
    """
    aliases = validate_event_query(event_query)
    end = index.duration if end is None else end
    start, end = validate_interval(start, end, index.duration)
    if match not in ("phrase", "terms"):
        raise ValueError("match must be phrase or terms")
    if type(max_history) is not int or not 1 <= max_history <= 100:
        raise ValueError("max_history must be an integer from 1 to 100")
    cues = sorted(index.contained(start, end, "transcript"), key=lambda cue: (cue.start, cue.end, cue.id))
    queries = [_words(alias) for alias in aliases]
    matches = []
    first_position = None

    def record(cue):
        return {"source_id": cue.id, "start": cue.start, "end": cue.end, "text": cue.text,
                "modality": "transcript"}

    for position, cue in enumerate(cues):
        words = _words(cue.text)
        hits = []
        for alias, query in zip(aliases, queries):
            found = (any(words[i:i + len(query)] == query for i in range(len(words) - len(query) + 1))
                     if match == "phrase" else set(query).issubset(words))
            if found:
                hits.append(alias)
        if hits:
            if first_position is None:
                first_position = position
            matches.append({**record(cue), "matched_aliases": hits})
    first = matches[0] if matches else None
    neighbors = []
    if first_position is not None:
        neighbors = [record(cues[i]) for i in (first_position - 1, first_position + 1) if 0 <= i < len(cues)]
    excluded = sum(cue.source == "transcript" and cue.start < end and cue.end > start
                   and not start <= cue.start < cue.end <= end for cue in index.segments)
    return {
        "event_query": list(aliases), "scope": {"start": start, "end": end}, "modality": "transcript",
        "claim": "first_transcript_mention" if first else "not_observed", "match": match,
        "first_matching_cue": first, "first_matching_cue_start": first["start"] if first else None,
        "first_observed_timestamp": None, "onset_bracket": None, "global_first_verified": False,
        "earliest_match_in_index": bool(first), "audio_onset_verified": False,
        "cue_onset_uncertainty": ({"start": first["start"], "end": first["end"],
                                   "width_seconds": first["end"] - first["start"],
                                   "basis": "transcript cue extent; word onset and ASR alignment error unverified"}
                                  if first else None),
        "source_ids": [value["source_id"] for value in matches[:max_history]],
        "matches": matches[:max_history], "matching_cue_count": len(matches),
        "history_truncated": len(matches) > max_history, "neighbors": neighbors,
        "examined_cue_count": len(cues), "excluded_boundary_cues": excluded,
        "coverage_note": (
            "Earliest lexical match among fully contained transcript cues. Phrases split across cues, "
            "paraphrases, excluded boundary cues, and transcription errors can hide earlier mentions. "
            "Cue timestamps do not establish exact audio onset or visual appearance."
        ),
    }
