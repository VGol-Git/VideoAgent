# Temporal constraints in the service — 2026-10-22

The current service uses `clipscout-v6-temporal-scopes`. Explicit numeric ranges and before/after questions now constrain evidence acquisition and citation validation. This extends the earlier [exact-moment fix](EXACT_MOMENTS.md): that document's statement that ranges and before/after questions follow the general search path describes v5, not v6.

## Supported interpretations

| Request | Interpretation in Adaptive agent |
|---|---|
| `at 10 minutes 10 seconds`, `at 10:10` | Direct inspection of the nearest verified frame to 610 seconds. |
| `at 800 seconds` | Direct inspection of the nearest verified frame to 800 seconds. |
| `from the 20th to the 21st minute` | 20:00–21:00, or 1200–1260 seconds. Ordinal wording denotes timestamp boundaries, not one-based minute bins. |
| `from the 15th to the 20th minute` | 15:00–20:00, or 900–1200 seconds. |
| `after 15 minutes` | From 15:00 through the end of the video; not just the next scene. |
| `before 15 minutes` | From the beginning of the video to 15:00. |

The parser supports numeric English/Russian units and timecodes. A unit can be shared by range endpoints, as in `15–20 minutes`. Intervals use an inclusive start and exclusive end for frame sampling. The interpreted bounds are exposed in `temporal_scope` and displayed with the answer. Recognized invalid intervals, malformed timecodes, and multiple interval constraints are rejected rather than silently becoming unrestricted text searches. Ordinary durations are not automatically treated as video positions. This is a conservative parser, not general understanding of every temporal expression or event-relative request.

Point lookup remains a separate path. The requested time and actual decoded frame timestamp are retained separately, and an answered point query must cite evidence containing both. Multiple explicit points require evidence for each within the call/frame budget.

## Evidence acquisition and validation

In Adaptive agent, an explicit interval bypasses unrestricted semantic search. The executor distributes inspection windows across that interval and obtains fresh sampled frames from the original video. With the demo's defaults, it uses up to **4 windows, 16 frames total, and 120 seconds per window**. The budgets are shared across windows; a lower frame budget reduces the number of windows and allocates frames before inspection.

For 20:00–21:00, the default windows are 20:00–20:15, 20:15–20:30, 20:30–20:45, and 20:45–21:00. Each receives four sampled frames. A five-minute request can use four adjacent 75-second windows. A longer request, such as 15:00 through the end of this 42:55 video, receives distributed windows with uninspected gaps between them.

The safeguards operate independently of the model's wording:

- Inspections outside the requested bounds are rejected before video decoding.
- Bounded text retrieval rebuilds candidates from whole cues contained in the requested interval. Crossing cues are excluded rather than relabeled with narrower timestamps while retaining outside text.
- Transcript text attached to a scoped inspection is likewise restricted to whole cues contained in its window.
- For an adaptive interval overview, the finalizer receives the inspected evidence ledger. Search candidates cannot replace those observations in this path. Retrieval only deliberately uses bounded text evidence.
- Citations must reference actual evidence IDs and remain within both their evidence range and the requested interval. Cited sampled frames are also checked against the requested bounds.
- An answered adaptive interval overview must retain citations to every inspected window. Missing window citations trigger bounded response repair or an error.

The API returns `coverage.inspection_intervals`, `coverage.omitted_intervals`, `coverage.sampled_frame_count`, and an executor-authored explanation in `coverage.note`. The answer is labeled as selected observations with partial coverage. **`complete_visual_coverage` remains false even when the inspection windows are adjacent**: observing four images from a window does not mean watching every frame inside it. An empty `omitted_intervals` list means the nominal window ranges cover the request, not that every intervening event was observed.

## Research modes

The three baselines retain their acquisition policies while respecting explicit interval bounds:

| Mode | Behavior for an explicit interval |
|---|---|
| Adaptive agent | Directly inspects distributed windows across the requested interval. Requires citations to all inspected windows for an answered overview. |
| Retrieval only | Searches contained text/caption cues within the interval; no new visual inspection. |
| Search + fixed inspection | Performs one bounded search, then inspects up to three retrieved intervals, within the shared budgets. |
| Uniform inspection | Intersects its original fixed video-wide positions with the requested interval. It may have no eligible position in a narrow request, such as 20:00–21:00 in this video. |

Baselines are not converted into the adaptive interval planner or direct point lookup. No eligible evidence can mean an insufficient-evidence response. Their metadata uses `bounded_baseline`, and it reports their actual sampled windows and omissions. Use Adaptive agent when asking for a distributed overview of a specified interval.

## Verification and remaining limits

The [unit/integration test log](../results/temporal_ranges/unit_tests.txt) records **127 passing tests**. These use scripted model responses where inference is needed; they verify routing, parser behavior, frame budgets, filtering, wrong-time rejection, coverage metadata, and response contracts. They do not measure the VLM's ability to read slides correctly.

Only the user's three interval questions are used for the new live checks:

| Question | Required interval | Saved output location |
|---|---|---|
| What is displayed on the screen from the 20th to the 21st minute | 1200–1260 s | [20_to_21](../results/temporal_ranges/live/20_to_21) |
| What happens in the video after 15 minutes? | 900 s–video end | [after_15](../results/temporal_ranges/live/after_15) |
| What happens in the video from the 15th to the 20th minute? | 900–1200 s | [15_to_20](../results/temporal_ranges/live/15_to_20) |

Each output directory contains the unedited `response.json`, the actual sampled frames re-extracted for diagnostic review, and a `sampled_frames.png` contact sheet. [The check script](../scripts/check_temporal_ranges.py) saves each response before checking it and refuses to overwrite completed responses. All three requests completed with status `answered` and passed the [structural checks](../results/temporal_ranges/live/checks.json). Each used four inspections and sixteen frames, with all sampled timestamps and citations inside the requested bounds. Latencies were 87.36 s, 184.99 s, and 178.97 s respectively; the first request included model loading. The report records source hashes, response hashes, run IDs, and structural checks, not a factual-accuracy score.

**Correct temporal routing does not guarantee a factually correct description.** Diagnostic review of the [first saved response](../results/temporal_ranges/live/20_to_21/response.json) found residual model errors: it calls a five-stage diagram a four-step process, describes an additional “Revise” step appearing without supporting that change, and attributes a detail to exactly 20:45 without verifying that exact instant. Correctly bounded citations do not validate those statements. The response is retained unchanged so these limitations remain inspectable.

The after-15-minute answer covers different sampled portions through the ending, but misidentifies a Hermes documentation page as GitHub and uses window boundaries as if they were verified event onset times. The 15:00–20:00 answer broadly identifies the bot/team demonstration and workflow slides, but calls Hermes “Slack,” conflates two distinct diagrams, and makes an unsupported absence claim in its last sentence. These are model description/finalization failures; the executor-authored partial-coverage label does not make them true. [Diagnostic semantic review](../results/temporal_ranges/live/semantic_review.json) records the comparisons with all three contact sheets. This was an assistant review of sampled frames and supplied ASR, not an independent human evaluation or full-video review. The three responses should not be reported as three factually perfect answers.

Sparse frames also cannot establish an exact transition time, all events in an interval, continuous activity, or absence throughout an unobserved gap. Small on-screen text and names may be misread. Structural validation verifies the location and provenance of evidence; it does not automatically verify semantic entailment, event counts, or each sentence of the answer. The final description therefore still needs comparison with the saved frames or source video.

## Research provenance

This is a service regression fix, not a rerun of the historical 52-prediction study. The study used `clipscout-v4-finalizer`; its predictions, metrics, manifests, and original source snapshot remain unchanged. Its benchmark scores do not evaluate v6 behavior.

[provenance.json](../results/temporal_ranges/provenance.json) records the current package hashes and checks that the historical artifacts remain unchanged, the frozen v4 source matches the original study manifest, and the pre-change v5 package matches the earlier moment-check hashes. The pre-change package is retained in [source_before/clipscout](../results/temporal_ranges/source_before/clipscout). The historical study source is in [results/corrected/source/clipscout](../results/corrected/source/clipscout), and the earlier two point checks remain in [results/moment_fix/verified](../results/moment_fix/verified).

The tested v6 package, browser HTML and targeted check script are frozen in [source_v6](../results/temporal_ranges/source_v6), with SHA-256 hashes in its manifest. After all three checks, current source and saved response hashes were verified against `live/checks.json`.
