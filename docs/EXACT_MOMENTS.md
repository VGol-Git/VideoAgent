# Exact-moment service fix — 2026-10-22

The service now resolves explicit timestamps before semantic search in **Adaptive agent** mode. Examples: `at 10 minutes 10 seconds`, `at 800 seconds`, `at 10:10`, `at 00:10:10`, and equivalent Russian expressions for 10 minutes 10 seconds and the 800th second. Numeric units need a point cue; bare timecodes and explicit comparison requests are also supported. Duration, before/after, and range questions retain the general agent path. This is a conservative numeric parser, not unrestricted temporal-language understanding.

The old failure searched the words in the question and inspected 1080–1140 seconds, then described those images as 10:10. Bounds checking alone accepted the citation. The new route directly calls `inspect_clip` with the requested timestamp and a narrow containing interval, extracts one nearest video frame, and finalizes from its observation. The decoder checks its actual presentation timestamp within one frame interval; a missing or inaccurate timestamp fails explicitly. Requested and actual times remain separate in the JSON. An answered result must cite evidence containing both the requested time and the verified decoded frame. Out-of-video moments and requests exceeding the budget are rejected before inference. Multiple explicit moments require evidence for each.

The target frame is decoded at up to twice the normal thumbnail side, capped at 1536 pixels, and passed through the backend's **unchanged** per-image pixel cap (313600 pixels for the running 560 configuration). With this landscape video, the backend receives approximately 746×420 rather than 560×315 pixels. This improves small-text reading while using one image. Ordinary sparse inspections keep their existing dimensions and policies. The trace records both limits. The UI labels target citations with the requested moment and seeks to the actual frame; precise frame seconds are shown separately.

The three research baselines retain their acquisition policies. They do not silently become direct timestamp lookup. Their answers are also checked against requested moments and cannot cite unrelated evidence as a verified target frame. Use **Adaptive agent** for this feature.

## Targeted verification

Only the two user-specified questions were used for live regression checks. The complete 52-prediction study was **not rerun**.

| Question | Requested time | Actual frame | Checked content |
|---|---:|---:|---|
| Describe what was on the slide at 10 minutes 10 seconds | 610 s | 610.0094 s | “GOAL LOOPS PROVIDE AUTONOMOUS QUALITY CONTROL”; trigger/goal, spawn sub-agents, evaluate criteria, retry and exit paths |
| Describe what was on the slide at 800 seconds | 800 s | 799.9992 s | An editor dialog for `wanderloots-tutorials · SOUL.md`, Hermes Agent instructions, and a Save SOUL.md button |

Final unedited responses, reference frames and structural check report are in [results/moment_fix/verified](../results/moment_fix/verified). Each response uses one tool call and one frame. The assistant compared the answers with the original frames; this was a diagnostic inspection, not blind human grading. The second view is an editor, although the model still uses the question's generic word “slide”. The first answer describes the retry box as a fourth stage; the slide itself numbers three main steps. Timing is verified automatically; fine-grained semantic accuracy is not guaranteed.

Development attempts are retained: `results/moment_fix/610.json` is the first correct-frame inspection followed by an erroneous abstention over a 9.4 ms offset; `results/moment_fix/final` contains the next pair with correct timing but low-resolution reading mistakes at 800 seconds; `results/moment_fix/verified` contains the final pair after improving single-frame readability. Total: five live responses across these two questions, not a benchmark rerun. Connection failures before inference produced no model responses.

All **103 unit/integration tests** passed, including generated-video frame identity and timestamp checks, both question formats, wrong-time citation rejection, range/duration parsing, budget limits, long questions, and HTTP contracts. The test suite uses scripted model responses. Log: [unit_tests.txt](../results/moment_fix/unit_tests.txt).

## Research provenance

The completed report describes `clipscout-v4-finalizer`. Its original package is now stored byte-for-byte in [results/corrected/source/clipscout](../results/corrected/source/clipscout), verified against every source hash in the original manifest before editing. The current service is `clipscout-v5-explicit-moments`; its behavior is not represented by the old benchmark scores.

Existing predictions, metrics, manifests and verification reports were not overwritten. [provenance.json](../results/moment_fix/provenance.json) records snapshot validation, matching prior artifact hashes and current service source hashes. The study verifier can use the frozen package and explicitly reports differences from the working service.
