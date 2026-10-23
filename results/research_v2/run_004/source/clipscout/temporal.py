"""Conservative parsing of explicit video times, not ordinary durations.

Supports numeric English/Russian units and MM:SS / HH:MM:SS timecodes.
Unit expressions need a point cue (at / на / timestamp); unanchored durations
and ranges are handled separately by ``parse_temporal_scope``.
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass

_NUMBER = r"[-+]?\d+(?:[.,]\d+)?"
_HOUR = r"hours?|hrs?|h|час(?:а|ов|е|ах)?|ч"
_MINUTE = r"minutes?|mins?|m|минут(?:а|ы|у|е|ах)?|мин|м"
_SECOND = r"seconds?|secs?|s|секунд(?:а|ы|у|е|ах)?|сек|с"
_ORDINAL = r"(?:-?(?:й|я|ю|ой|ей|ую|го))?"
_PART = rf"({_NUMBER}){_ORDINAL}\s*({_HOUR}|{_MINUTE}|{_SECOND})(?![^\W\d_])"
_JOIN = r"\s*(?:(?:and|и)\s+)?"
_H, _M, _S = [rf"{_NUMBER}{_ORDINAL}\s*(?:{unit})(?![^\W\d_])" for unit in (_HOUR, _MINUTE, _SECOND)]
_UNIT_GROUP = re.compile(rf"(?<![\w.])(?:{_H}(?:{_JOIN}{_M})?(?:{_JOIN}{_S})?|{_M}(?:{_JOIN}{_S})?|{_S})", re.I)
_PARTS = re.compile(_PART, re.I)
_CLOCK = re.compile(r"(?<![\w.:])[-+]?\d+:\d{2}(?::\d{2})?(?:\.\d+)?(?![\w:])")
_POINT_CUE = re.compile(
    r"(?:\bat|\baround|\btimestamp|\btimecode|\btime|\bна|\bв|\bоколо|\bмомент(?:е)?|\bтаймкод(?:е)?)"
    r"\s*(?:the\s+)?(?:отметке\s*)?[:=]?\s*$", re.I)
_RANGE_CUE = re.compile(r"(?:\bfrom|\bbetween|\bс|\bмежду)\s*$", re.I)
_CLOCK_CUE = re.compile(r"(?:\bcompare|\bdescribe|\binspect|\bshow|\bсравни(?:те)?|\bопиши(?:те)?|\bпокажи(?:те)?)\s*$", re.I)


def explicit_moments(question: str) -> tuple[float, ...]:
    """Return ordered unique point targets; do not silently select just one.

    Negative/out-of-duration targets are returned for the caller to reject.
    Invalid clock fields raise ValueError rather than falling back to search.
    """
    found = []
    candidates = sorted([(m.start(), pattern, m) for pattern in (_CLOCK, _UNIT_GROUP)
                         for m in pattern.finditer(question)], key=lambda item: item[0])
    range_members = set()
    for i in range(len(candidates) - 1):
        left, right = candidates[i][2], candidates[i + 1][2]
        separator = question[left.end():right.start()]
        if (re.fullmatch(r"\s*(?:[-–—]|to|through|до)\s*", separator, re.I) or
                (_RANGE_CUE.search(question[:left.start()]) and re.fullmatch(r"\s*(?:and|и)\s*", separator, re.I))):
            range_members.update((i, i + 1))
    previous_accepted_end = None
    for i, (_, pattern, match) in enumerate(candidates):
        if i in range_members:
            continue
        prefix = question[:match.start()]
        # A range has different semantics from a point inspection.
        if _RANGE_CUE.search(prefix) or re.match(r"\s*(?:[-–—]|to\b|through\b|до\b)", question[match.end():], re.I):
            continue
        continuation = (previous_accepted_end is not None and
                        re.fullmatch(r"\s*(?:and|vs\.?|versus|и|,)\s*", question[previous_accepted_end:match.start()], re.I))
        if pattern is _UNIT_GROUP:
            if not _POINT_CUE.search(prefix) and not continuation:
                continue
            total = 0.0
            for amount, unit in _PARTS.findall(match.group()):
                unit = unit.lower()
                scale = 3600 if unit.startswith(("h", "ч")) else 60 if unit.startswith(("m", "м")) else 1
                total += float(amount.replace(",", ".")) * scale
        else:
            if not (not prefix.strip() or _POINT_CUE.search(prefix) or _CLOCK_CUE.search(prefix) or continuation):
                continue
            fields = match.group().lstrip("+-").split(":")
            if any(float(field) >= 60 for field in fields[1:]):
                raise ValueError("Timecode minute/second fields must be below 60.")
            total = 0.0
            for field in fields:
                total = total * 60 + float(field)
            if match.group().startswith("-"):
                total = -total
        found.append((match.start(), total))
        previous_accepted_end = match.end()
    return tuple(dict.fromkeys(value for _, value in sorted(found)))


@dataclass(frozen=True)
class TemporalScope:
    """A requested half-open video interval, measured in seconds."""

    kind: str
    start: float
    end: float
    interpretation: str


# Keep interval grammar separate from the already-supported point grammar.
# Ordinal minute numbers denote timestamp boundaries, not one-based minute bins:
# "20th to 21st minute" means 20:00 to 21:00, explicitly reported to the caller.
_SCOPE_ORDINAL = r"(?:st|nd|rd|th|-?(?:й|я|ю|ой|ей|ую|го))?"
_SCOPE_NUMBER = rf"{_NUMBER}{_SCOPE_ORDINAL}"
_SCOPE_PART = rf"({_NUMBER}){_SCOPE_ORDINAL}\s*({_HOUR}|{_MINUTE}|{_SECOND})(?![^\W\d_])"
_SCOPE_PARTS = re.compile(_SCOPE_PART, re.I)
_SH, _SM, _SS = [rf"{_SCOPE_NUMBER}\s*(?:{unit})(?![^\W\d_])"
                 for unit in (_HOUR, _MINUTE, _SECOND)]
_SCOPE_UNITS = rf"(?:{_SH}(?:{_JOIN}{_SM})?(?:{_JOIN}{_SS})?|{_SM}(?:{_JOIN}{_SS})?|{_SS})"
_SCOPE_CLOCK = r"[-+]?\d+:\d+(?::\d+)?(?:[.,]\d+)?"
_SCOPE_VALUE = rf"(?:the\s+)?(?:{_SCOPE_CLOCK}|{_SCOPE_UNITS}|{_SCOPE_NUMBER})(?![\w:]|\.\d)"
_SCOPE_RANGE = re.compile(
    rf"(?<![\w.:])(?P<left>{_SCOPE_VALUE})\s*"
    rf"(?P<separator>[-–—]|\bto\b|\buntil\b|\bthrough\b|\bдо\b|\bпо\b|\band\b|\bи\b)"
    rf"\s*(?P<right>{_SCOPE_VALUE})", re.I)
_SCOPE_PREFIX = re.compile(r"(?:\bfrom|\bbetween|\bс|\bот|\bмежду)\s*$", re.I)
_SCOPE_OPEN = re.compile(
    rf"(?P<cue>\bafter|\bbefore|\buntil|\bпосле|\bдо)\s+(?P<value>{_SCOPE_VALUE})", re.I)
_SCOPE_START = re.compile(
    rf"(?:\bfrom|\bbetween|\bс|\bот|\bмежду)\s+(?P<value>{_SCOPE_VALUE})", re.I)
_SCOPE_CLOCK_INTENT = re.compile(
    r"(?:\bfrom|\bbetween|\bafter|\bbefore|\buntil|\bс|\bот|\bмежду|\bпосле|\bдо)"
    r"\s+(?:the\s+)?(?P<value>[-+]?\d+(?::[\d.,]*)+)", re.I)
_SCOPE_SHARED_UNIT_INTENT = re.compile(
    rf"\s*(?:[-–—]|\bto\b|\buntil\b|\bthrough\b|\band\b|\bдо\b|\bпо\b|\bи\b)"
    rf"\s*(?:the\s+)?[^\s?!.]+\s*(?:{_HOUR}|{_MINUTE}|{_SECOND})(?![^\W\d_])", re.I)


def _scope_value(text: str, shared_scale: float | None = None) -> tuple[float | None, float | None]:
    """Parse one endpoint and expose a unit only when safe to share it."""
    text = re.sub(r"^the\s+", "", text.strip(), flags=re.I)
    if ":" in text:
        fields = text.lstrip("+-").replace(",", ".").split(":")
        if len(fields) not in (2, 3) or any(float(field) >= 60 for field in fields[1:]):
            raise ValueError("Timecode minute/second fields must be below 60.")
        total = 0.0
        for field in fields:
            total = total * 60 + float(field)
        return (-total if text.startswith("-") else total), None
    parts = _SCOPE_PARTS.findall(text)
    if parts:
        if any(float(amount.replace(",", ".")) < 0 for amount, _ in parts):
            raise ValueError("Video time components must not be negative.")
        scales = [3600.0 if unit.lower().startswith(("h", "ч")) else
                  60.0 if unit.lower().startswith(("m", "м")) else 1.0 for _, unit in parts]
        return sum(float(amount.replace(",", ".")) * scale
                   for (amount, _), scale in zip(parts, scales)), scales[0] if len(parts) == 1 else None
    if shared_scale is None:
        return None, None
    number = re.match(_NUMBER, text)
    return float(number.group().replace(",", ".")) * shared_scale, shared_scale


def parse_temporal_scope(question: str, duration: float) -> TemporalScope | None:
    """Recognize numeric EN/RU ranges and before/after video boundaries.

    A unit may be shared by two range endpoints; a clock is always MM:SS or
    HH:MM:SS. Bare quantities and ordinary durations are not video constraints.
    Unbounded "after" extends to the end, "before" starts at zero. Recognized
    invalid or multiple intervals fail closed instead of falling back to search.
    Point requests remain the responsibility of ``explicit_moments``.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be finite and positive.")
    found: list[tuple[int, int, TemporalScope]] = []
    # Recognizable malformed clocks must not disappear into ordinary search
    # merely because their field count prevents the main grammar matching.
    for match in _SCOPE_CLOCK_INTENT.finditer(question):
        clock = match["value"].rstrip(".,")
        if not re.fullmatch(_SCOPE_CLOCK, clock):
            raise ValueError("Use MM:SS or HH:MM:SS for a video timecode.")
        _scope_value(clock)

    def scope(kind: str, start: float, end: float, detail: str) -> TemporalScope:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError(f"Requested interval must satisfy 0 <= start < end <= {duration:g} seconds.")
        return TemporalScope(kind, float(start), float(end),
                             f"Requested video interval: {start:g}–{end:g} seconds. {detail}")

    for match in _SCOPE_RANGE.finditer(question):
        if match["separator"].lower() in ("and", "и") and not _SCOPE_PREFIX.search(question[:match.start()]):
            continue
        left, left_scale = _scope_value(match["left"])
        right, right_scale = _scope_value(match["right"])
        if left is None and right is None:
            continue  # e.g. "between 2 and 3 agents", not timestamps
        if left is None:
            left, _ = _scope_value(match["left"], right_scale)
        if right is None:
            right, _ = _scope_value(match["right"], left_scale)
        if left is None or right is None:
            raise ValueError("Both interval endpoints need a time unit or timecode.")
        if re.search(r"(?:\bnot|\bне)\s+(?:(?:from|between|at|с|от|между|на)\s+)?$",
                     question[:match.start()], re.I):
            raise ValueError("Negated time constraints are ambiguous; use an explicit start and end.")
        found.append((match.start(), match.end(), scope(
            "range", left, right,
            "Minute/hour numbers, including ordinal wording, are timestamp boundaries.")))

    for match in _SCOPE_OPEN.finditer(question):
        if any(begin <= match.start() < end for begin, end, _ in found):
            continue  # "until" / "до" already consumed as a range separator
        value, _ = _scope_value(match["value"])
        if value is None:
            continue
        if re.search(r"(?:\bnot|\bне)\s*$", question[:match.start()], re.I):
            raise ValueError("Negated time constraints are ambiguous; use an explicit start and end.")
        after = match["cue"].lower() in ("after", "после")
        found.append((match.start(), match.end(), scope(
            "after" if after else "before", value if after else 0, duration if after else value,
            "After means through the end of the video." if after else "Before means from the start of the video.")))

    # Catch incomplete numeric time ranges such as "from 10:10 to ..." rather
    # than turning their recognizable time constraint into a free text search.
    for match in _SCOPE_START.finditer(question):
        value, _ = _scope_value(match["value"])
        recognized = value is not None or _SCOPE_SHARED_UNIT_INTENT.match(question[match.end():])
        if recognized and not any(begin <= match.start("value") < end for begin, end, _ in found):
            raise ValueError("A time range needs two valid endpoints with units or timecodes.")
    if len(found) > 1:
        raise ValueError("Use one video interval per question; multiple time constraints are not supported.")
    if found:
        # A point elsewhere in a mixed request must not be silently discarded.
        # Mask the actual interval text first: "at 20:00 until 21:00" contains
        # a range endpoint that the separate point parser otherwise recognizes.
        remaining = list(question)
        for begin, end, _ in found:
            remaining[begin:end] = " " * (end - begin)
        if explicit_moments("".join(remaining)):
            raise ValueError("Ask about a video interval or exact moments separately; mixed time requests are not supported.")
    return found[0][2] if found else None
