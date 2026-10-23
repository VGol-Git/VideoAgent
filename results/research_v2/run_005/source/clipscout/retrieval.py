"""Modality-aware lexical retrieval, rank fusion, and temporal diversification.

Weighted reciprocal-rank fusion is adapted from Grounded Video Agent's
CandidateFusionCapability (Apache-2.0). See THIRD_PARTY_NOTICES.txt. This
implementation keeps whole-cue bounds, records every contribution, and adds
explicit query decomposition and greedy temporal diversification. Scores are
ranking values, never probabilities or measures of factual support.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
import math

from .index import finite_number, tokens, validate_interval


SOURCES = ("transcript", "caption", "ocr", "dense_visual")
MAX_SEARCH_QUERIES = 8


def _queries(query, queries):
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("query must contain 1–500 characters")
    if queries is None:
        queries = []
    if not isinstance(queries, (tuple, list)) or len(queries) >= MAX_SEARCH_QUERIES:
        raise ValueError(f"query plus queries must contain at most {MAX_SEARCH_QUERIES} explicit queries")
    result = []
    for item in [query, *queries]:
        if not isinstance(item, str) or not item.strip() or len(item) > 500:
            raise ValueError("Each subquery must contain 1–500 characters")
        item = item.strip()
        if item.casefold() not in {q.casefold() for q in result}:
            result.append(item)
    return result


def _bm25(records, query):
    counts = [Counter(tokens(record["text"])) for record in records]
    df = Counter(term for item in counts for term in item)
    n = len(records)
    average = sum(sum(item.values()) for item in counts) / max(1, n)
    scored = []
    for record, item in zip(records, counts):
        length = sum(item.values())
        score = 0.0
        for term in set(tokens(query)):
            tf = item[term]
            if tf:
                idf = math.log(1 + (n - df[term] + .5) / (df[term] + .5))
                score += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * length / max(1, average)))
        if score > 0:
            scored.append((record, score))
    return sorted(scored, key=lambda item: (-item[1], item[0]["start"], item[0]["id"]))


def _ocr_candidates(records, duration, start, end):
    candidates, seen = [], set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("OCR records must be mappings")
        at = finite_number(record.get("timestamp"), "OCR timestamp")
        if not 0 <= at < duration:
            raise ValueError("OCR timestamp must lie inside the video")
        if not start <= at < end:
            continue
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("OCR record text must be a string")
        if not text.strip():
            continue
        identity = str(record.get("id") or record.get("frame_id") or
                       hashlib.sha256(f"{at}:{text}".encode()).hexdigest()[:20])
        identity = "ocr:" + identity
        if identity in seen:
            raise ValueError("OCR frame IDs must be unique")
        seen.add(identity)
        # These bounds propose surrounding context to inspect. The actual OCR
        # evidence is one sampled frame, not text continuously present for 10s.
        candidates.append({"id": identity, "start": max(start, at - 5),
                           "end": min(end, at + 5), "source": "ocr",
                           "text": text, "segment_ids": [identity],
                           "sampled_timestamps": [at],
                           "window_kind": "context_around_sample",
                           "ocr_lines": list(record.get("lines", [])),
                           "ocr_provenance": record.get("provenance", {})})
    return candidates


def _iou(left, right):
    overlap = max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
    union = max(left["end"], right["end"]) - min(left["start"], right["start"])
    return overlap / union if union else 0.0


def _visual_candidates(records, duration, start, end, default_query):
    # A frame can legitimately match several planner subqueries. Its physical
    # identity is shared, but each query supplies a separate ranked list. Exact
    # repeats within that list must not increase reciprocal-rank weight.
    candidates, physical_frames = {}, {}
    for record in records:
        if not isinstance(record, Mapping) or record.get("source", "dense_visual") != "dense_visual":
            raise ValueError("Visual hits must be dense_visual mappings")
        lower, upper = validate_interval(record.get("start"), record.get("end"), duration)
        timestamps = record.get("sampled_timestamps")
        if not isinstance(timestamps, (list, tuple)) or not timestamps:
            raise ValueError("Visual hits must contain actual sampled_timestamps")
        timestamps = [finite_number(t, "Visual sample timestamp") for t in timestamps]
        if any(not lower <= t < upper for t in timestamps):
            raise ValueError("Visual sample timestamp is outside its evidence window")
        # Context windows may straddle the request; actual sampled evidence may
        # not. Keep all-or-nothing samples instead of relabelling their time.
        if any(not start <= t < end for t in timestamps):
            continue
        identity = record.get("id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Visual hit IDs must be nonempty strings")
        physical_identity = (lower, upper, tuple(timestamps))
        if identity in physical_frames and physical_frames[identity] != physical_identity:
            raise ValueError("Repeated visual frame ID has conflicting timestamps or window bounds")
        physical_frames[identity] = physical_identity
        query = record.get("query", default_query)
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("Visual hit query must contain 1–500 characters")
        query = query.strip()
        text = record.get("text", "")
        if not isinstance(text, str):
            raise ValueError("Visual hit text must be a string")
        score = finite_number(record.get("score"), "Visual retrieval score")
        candidate = {**record, "id": "dense_visual:" + identity, "query": query,
            "start": max(start, lower), "end": min(end, upper), "text": text,
            "source": "dense_visual", "segment_ids": [identity],
            "sampled_timestamps": timestamps, "window_kind": "context_around_sample",
            "candidate_original_range": [lower, upper], "dense_similarity": score}
        key = (identity, query.casefold())
        previous = candidates.get(key)
        if previous is None or score > previous["dense_similarity"]:
            candidates[key] = candidate
    return sorted(candidates.values(), key=lambda r: (-r["dense_similarity"], r["start"], r["id"]))


def _cluster(fused):
    """Fuse overlapping channels while retaining separate evidence items.

    Same-channel hits never merge: nearby but distinct events must survive.
    A new member must overlap every existing member, preventing chain merges.
    """
    clusters = []
    for item in sorted(fused.values(), key=lambda x: (-x["score"], x["start"], x["id"])):
        compatible = [cluster for cluster in clusters
                      if item["source"] not in {c["source"] for c in cluster}
                      and all(_iou(item, c) >= .5 for c in cluster)]
        if compatible:
            max(compatible, key=lambda c: sum(_iou(item, v) for v in c)).append(item)
        else:
            clusters.append([item])
    output = []
    for cluster in clusters:
        evidence = [{key: value for key, value in item.items()
                     if key not in {"score", "ranking_provenance", "matched_queries"}}
                    for item in cluster]
        sources = sorted({item["source"] for item in cluster})
        identity = cluster[0]["id"] if len(cluster) == 1 else "fusion:" + hashlib.sha256(
            "|".join(sorted(item["id"] for item in cluster)).encode()).hexdigest()[:16]
        output.append({"id": identity,
                       "start": min(item["start"] for item in cluster),
                       "end": max(item["end"] for item in cluster),
                       "text": cluster[0]["text"] if len(cluster) == 1 else "\n".join(
                           f"[{item['source']}] {item['text']}" for item in cluster),
                       "source": sources[0] if len(sources) == 1 else "hybrid",
                       "modalities": sources,
                       "segment_ids": [s for item in cluster for s in item["segment_ids"]],
                       "sampled_timestamps": sorted({t for item in cluster
                                                     for t in item.get("sampled_timestamps", [])}),
                       "score": sum(item["score"] for item in cluster),
                       "score_type": "weighted_rrf",
                       "ranking_provenance": [p for item in cluster for p in item["ranking_provenance"]],
                       "matched_queries": list(dict.fromkeys(q for item in cluster for q in item["matched_queries"])),
                       "evidence": evidence})
    return output


def hybrid_retrieve(index, query: str, top_k: int = 5, source: str = "all",
                    start: float | None = None, end: float | None = None, *,
                    queries: list[str] | None = None, ocr_records: Iterable[Mapping] = (),
                    visual_hits: Iterable[Mapping] = (),
                    modality_weights: Mapping[str, float] | None = None,
                    rrf_k: float = 60, diversity_weight: float = .35) -> list[dict]:
    """Retrieve independent modalities and explicit subqueries, then fuse.

    ``queries`` contains planner-provided subquestions, not generated synonyms.
    ``ocr_records`` accepts ``RapidOCRAdapter.recognize`` results. ``visual_hits``
    are precomputed CLIP retrieval results for the original query; optionally
    include their ``query`` when supplying hits for a particular subquery.
    ``source='visual'`` selects captions, OCR, and dense visual hits only.
    OCR sampling
    does not establish duration; its candidate window is labelled in evidence.
    Optional temporal bounds apply before scoring and preserve whole cues.
    """
    query_list = _queries(query, queries)
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer from 1 to 10")
    if source not in ("all", "visual", *SOURCES):
        raise ValueError("source must be all, visual, transcript, caption, ocr, or dense_visual")
    if (start is None) != (end is None):
        raise ValueError("Temporal search requires both start and end")
    lo, hi = (0.0, index.duration) if start is None else validate_interval(start, end, index.duration)
    rrf_k = finite_number(rrf_k, "rrf_k")
    diversity_weight = finite_number(diversity_weight, "diversity_weight")
    if rrf_k <= 0 or not 0 <= diversity_weight <= 1:
        raise ValueError("Require rrf_k > 0 and 0 <= diversity_weight <= 1")
    weights = {channel: 1.0 for channel in SOURCES}
    for channel, weight in (modality_weights or {}).items():
        if channel not in SOURCES:
            raise ValueError("Unknown modality weight")
        weight = finite_number(weight, f"{channel} weight")
        if weight < 0:
            raise ValueError("Modality weights cannot be negative")
        weights[channel] = weight

    # Rebuild each channel separately so its BM25 statistics cannot be diluted
    # by unrelated modalities or by text outside the requested time range.
    chunks = index._chunks(index.contained(lo, hi))
    channels = {channel: [dict(c) for c in chunks if c["source"] == channel]
                for channel in ("transcript", "caption")}
    channels["ocr"] = _ocr_candidates(ocr_records, index.duration, lo, hi)
    channels["dense_visual"] = _visual_candidates(visual_hits, index.duration, lo, hi, query_list[0])
    fused = {}
    for channel, records in channels.items():
        eligible = source in ("all", channel) or (source == "visual" and channel != "transcript")
        if not eligible or weights[channel] == 0:
            continue
        for subquery in query_list:
            ranked = ([(r, r["dense_similarity"]) for r in records
                       if r["query"].casefold() == subquery.casefold()]
                      if channel == "dense_visual" else _bm25(records, subquery))
            for rank, (record, retrieval_score) in enumerate(ranked[:50], 1):
                contribution = weights[channel] / (rrf_k + rank) / len(query_list)
                item = fused.setdefault(record["id"], {**{k: v for k, v in record.items() if k != "score"}, "score": 0.0,
                                        "ranking_provenance": [], "matched_queries": []})
                item["score"] += contribution
                item["matched_queries"].append(subquery)
                item["ranking_provenance"].append({"channel": channel, "query": subquery,
                    "rank": rank, "retrieval_score": retrieval_score,
                    "channel_score_type": "cosine_similarity" if channel == "dense_visual" else "bm25",
                    "weight": weights[channel],
                    "rrf_contribution": contribution, "rrf_k": rrf_k})

    remaining, selected = _cluster(fused), []
    max_score = max((item["score"] for item in remaining), default=1.0)
    while remaining and len(selected) < top_k:
        for item in remaining:
            overlap = max((_iou(item, prev) for prev in selected), default=0.0)
            # Reward a subquestion not yet represented; diversify only ranking,
            # never delete close observations required by a temporal relation.
            covered = {q for prev in selected for q in prev["matched_queries"]}
            novelty = len(set(item["matched_queries"]) - covered) / len(query_list)
            item["selection_score"] = ((1 - diversity_weight) * item["score"] / max_score
                                       + diversity_weight * (novelty - overlap))
            item["temporal_overlap_penalty"] = overlap
        winner = min(remaining, key=lambda item: (-item["selection_score"], -item["score"],
                                                  item["start"], item["id"]))
        selected.append(winner)
        remaining.remove(winner)
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
        item["retrieval_method"] = "per_modality_bm25+optional_dense_visual+weighted_rrf+temporal_diversity"
    return selected
