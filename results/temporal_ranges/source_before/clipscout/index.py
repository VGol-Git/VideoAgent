"""Timestamp-preserving transcript storage and dependency-free retrieval."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import html
import json
import math
import re

STOP = set("a an the is are was were be been being to of in on at for from with and or it this that these those what when where which who why how does do did can could would should tell me video lecture about as by we you i they he she s t".split())


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE) if w not in STOP]


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_interval(start, end, duration, max_span=None):
    start, end = finite_number(start, "start"), finite_number(end, "end")
    if not 0 <= start < end <= duration:
        raise ValueError(f"Require 0 <= start < end <= {duration:.3f}")
    if max_span is not None and end - start > max_span:
        raise ValueError(f"Clip must be at most {max_span} seconds; narrow the interval")
    return start, end


@dataclass(frozen=True)
class Segment:
    id: str
    start: float
    end: float
    text: str
    source: str = "transcript"


def timestamp(value: str) -> float:
    fields = value.replace(",", ".").split(":")
    if len(fields) not in (2, 3):
        raise ValueError(f"Invalid timestamp {value!r}")
    total = 0.0
    for field in fields:
        total = total * 60 + float(field)
    if not math.isfinite(total) or total < 0:
        raise ValueError("Invalid subtitle timestamp")
    return total


def read_subtitles(path: str | Path) -> list[Segment]:
    """Read SRT/WebVTT; preserve cue boundaries, discard cue metadata."""
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().splitlines()
        if not lines or lines[0].startswith(("NOTE", "STYLE", "REGION")):
            continue
        for n, line in enumerate(lines):
            match = re.match(r"\s*([\d:.,]+)\s+-->\s+([\d:.,]+)", line)
            if not match:
                continue
            start, end = map(timestamp, match.groups())
            if end <= start:
                raise ValueError("Subtitle end must be after start")
            words = html.unescape(re.sub(r"<[^>]+>", "", " ".join(lines[n + 1:]))).strip()
            if words:
                cues.append(Segment(f"T{len(cues)+1:05}", start, end, words))
            break
    if not cues:
        raise ValueError("No subtitle cues found (expected SRT or WebVTT)")
    return sorted(cues, key=lambda s: (s.start, s.end))


class VideoIndex:
    def __init__(self, duration: float, segments: list[Segment], video_path: str | None = None,
                 video_sha256: str = "", title: str = "Video", chunk_seconds: float = 30):
        self.duration = finite_number(duration, "duration")
        chunk_seconds = finite_number(chunk_seconds, "chunk_seconds")
        if self.duration <= 0 or chunk_seconds <= 0:
            raise ValueError("Duration and chunk_seconds must be positive")
        ids = set()
        for segment in segments:
            validate_interval(segment.start, segment.end, self.duration)
            if segment.id in ids or not segment.id:
                raise ValueError("Segment IDs must be unique and non-empty")
            if segment.source not in ("transcript", "caption") or not isinstance(segment.text, str):
                raise ValueError("Invalid segment source or text")
            ids.add(segment.id)
        self.segments = sorted(segments, key=lambda s: (s.start, s.id))
        self.video_path = video_path
        self.video_sha256 = video_sha256
        self.title = title
        self.chunk_seconds = chunk_seconds
        self.chunks = self._chunks()
        self._terms = [Counter(tokens(c["text"])) for c in self.chunks]
        self._df = Counter(t for terms in self._terms for t in terms)
        self._avg = sum(sum(t.values()) for t in self._terms) / max(1, len(self._terms))

    def _chunks(self):
        groups = {}
        for s in self.segments:
            key = (s.source, int(s.start // self.chunk_seconds))
            groups.setdefault(key, []).append(s)
        return [{"id": f"{source}:{bucket}", "start": min(s.start for s in group),
                 "end": max(s.end for s in group), "text": " ".join(s.text for s in group),
                 "source": source, "segment_ids": [s.id for s in group]}
                for (source, bucket), group in sorted(groups.items())]

    def overlapping(self, start: float, end: float, source: str | None = None):
        validate_interval(start, end, self.duration)
        return [s for s in self.segments if s.start < end and s.end > start and (source is None or s.source == source)]

    def search(self, query: str, top_k: int = 5, source: str = "all") -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must contain 1–500 characters")
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer from 1 to 10")
        if source not in ("all", "transcript", "caption"):
            raise ValueError("source must be all, transcript or caption")
        terms = set(tokens(query))
        scored = []
        n = len(self.chunks)
        for i, (chunk, frequencies) in enumerate(zip(self.chunks, self._terms)):
            if source != "all" and chunk["source"] != source:
                continue
            length = sum(frequencies.values())
            score = 0.0
            for term in terms:
                tf = frequencies[term]
                if tf:
                    idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                    score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / max(1, self._avg)))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda x: (-x[0], self.chunks[x[1]]["start"]))
        # BM25 scores are relevance scores, never probabilities/confidences.
        return [{**self.chunks[i], "score": round(score, 5), "text": self.chunks[i]["text"][:1800]}
                for score, i in scored[:top_k]]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        video = self.video_path
        if video:
            import os
            video = os.path.relpath(Path(video).resolve(), path.parent.resolve())
        data = {"schema_version": 1, "duration": self.duration, "title": self.title,
                "video_path": video, "video_sha256": self.video_sha256,
                "chunk_seconds": self.chunk_seconds, "segments": [asdict(s) for s in self.segments]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.pop("schema_version", None) != 1:
            raise ValueError("Unsupported index schema")
        if data.get("video_path"):
            data["video_path"] = str((path.parent / data["video_path"]).resolve())
        data["segments"] = [Segment(**s) for s in data["segments"]]
        return cls(**data)

    def fingerprint(self):
        payload = {"video": self.video_sha256, "duration": self.duration, "chunk_seconds": self.chunk_seconds,
                   "segments": [asdict(s) for s in self.segments]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
