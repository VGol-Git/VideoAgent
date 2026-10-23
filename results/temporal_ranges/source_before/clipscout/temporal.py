"""Conservative parsing of explicit video moments, not ordinary durations.

Supports numeric English/Russian units and MM:SS / HH:MM:SS timecodes.
Unit expressions need a point cue (at / на / timestamp); unanchored durations
and ranges remain ordinary agent questions instead of being misrouted.
"""
from __future__ import annotations

import re

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
