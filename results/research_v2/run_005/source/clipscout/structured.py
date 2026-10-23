"""Conservative JSON framing for model output, independent of role schemas.

Only a complete object's trailing quote, one missing root brace after a closed
nested value, or an enclosing Markdown JSON fence can be normalized. Nothing
fills in text, numbers, keys, values, or missing nested delimiters. Callers must
retain the original output, record returned repair metadata, and still validate
the complete role/action schema and evidence references.
"""
from __future__ import annotations

import json


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def _load(text):
    value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


def _structure(text):
    """Return root end and unfinished stack, respecting JSON string escapes."""
    if not text.startswith("{"):
        return None, None, False
    stack = []
    quoted = escaped = False
    for position, character in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            expected = "{" if character == "}" else "["
            if not stack or stack.pop() != expected:
                return None, None, False
            if not stack:
                return position + 1, [], False
    return None, stack, quoted


def parse_structured_json(raw: str) -> tuple[dict, dict | None]:
    """Parse one object and return ``(value, repair_metadata_or_None)``.

    Strict JSON succeeds unchanged. Accepted framing changes are limited to:

    * ``remove_trailing_quote``: a valid complete root followed by just one
      double quote (apart from whitespace), as can occur in a stop token;
    * ``close_root_object``: append exactly one ``}`` when only the root is
      unclosed and the last token already closed a nested object or array;
    * ``remove_code_fence``: a whole, otherwise valid object in one enclosing
      `````json`` or plain Markdown fence, preserving previous compatibility.

    Metadata contains ``kind`` and the full ``framed_json``. Original ``raw``
    must be kept separately by the caller. Repairs do not combine. Ambiguous
    outputs (including internal stray quotes, prose, two objects, partial
    strings/numbers, or duplicate keys) raise rather than guessing intent.
    """
    if not isinstance(raw, str):
        raise TypeError("Structured model output must be a string")
    text = raw.strip()
    try:
        return _load(text), None
    except json.JSONDecodeError as original_error:
        if text.startswith(("```json\n", "```\n", "```json\r\n", "```\r\n")) and text.endswith("```"):
            framed = text.split("\n", 1)[1][:-3].strip()
            value = _load(framed)
            return value, {"kind": "remove_code_fence", "framed_json": framed}
        root_end, stack, quoted = _structure(text)
        if root_end is not None and text[root_end:].strip() == '"':
            framed = text[:root_end]
            value = _load(framed)
            return value, {"kind": "remove_trailing_quote", "framed_json": framed,
                           "removed_suffix": text[root_end:]}
        if root_end is None and stack == ["{"] and not quoted and text.endswith(("}", "]")):
            framed = text + "}"
            value = _load(framed)
            return value, {"kind": "close_root_object", "framed_json": framed,
                           "appended_suffix": "}"}
        raise original_error
