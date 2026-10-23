# ClipScout

A local video agent for a course project: the model chooses `search(query)` and `inspect_clip(start, end, question)`, receives observations, and answers with temporal citations. Each question allows **at most four attempted tool calls**, including invalid arguments and failures. The pretrained Qwen3-VL-2B is used without fine-tuning.

**Service update, 2026-10-22:** version v6 handles exact moments, ranges, and `before` / `after` constraints. In Adaptive agent, a point is routed to the nearest verified frame, while an interval is routed to several windows within its bounds. Search, inspection, and citations respect the requested time bounds; the interface explicitly shows partial coverage. All 127 software tests passed. [Point queries: v5 verification](docs/EXACT_MOMENTS.md); [ranges and three user questions: v6 verification](docs/TEMPORAL_SCOPES.md). The study tables below refer to the frozen v4 version; the full experiment was not repeated for v5/v6.

Implemented features include video processing, ASR, a temporal index, four evidence acquisition policies, an English browser demo, action logging, and evaluation. **52 real trajectories (4 modes × 13 questions)** and their diagnostic audit are complete; **87 software tests** passed. [REPORT.md](REPORT.md) contains tables, figures, and limitations. Adaptive achieved F1 0.267 and tIoU 0.261 with four frames per question, but these proxies and a single-video pilot do not establish general superiority.

## Data and study scope

The source is the user-provided file `../Subagents vs Agent Teams 🧠 Hermes Bots Goal Loops Kanban Graphs.mp4`: **2575.006 seconds, 42:55**, 1920×1080, approximately 29.97 FPS. Provenance and SHA-256 are recorded in [data/source.json](data/source.json). This is one natural video; the results do not automatically generalize to other videos.

A pilot set of **20 questions: 7 dev and 13 test** was prepared, with answers and intervals annotated by the assistant using ASR and selected frames. This is **not an independently human-validated benchmark**: 12 speech references have ASR-supported silver status, and 8 visual references rely on inspected frames. ASR supplies both some reference answers and retrieval text, so both may share an error. Negative questions do not establish that an answer is absent from the entire video. The dataset is saved in [data/questions.json](data/questions.json), and its file hash is recorded in the run manifest. The visual draft `data/visual_questions_draft.json` is neither a separate benchmark nor model output.

The first run with prompt v3 stopped after 13 adaptive, 13 retrieval-only, and one static answer: finalizer instructions conflicted with the tool-use system prompt, and 12 of 13 retrieval-only answers ended in format errors. The partial pilot, its source snapshot, and audits are saved in [results/test](results/test). The corrected v4 finalizer and bounded format retries were checked across all four modes in [results/corrected](results/corrected). This is a **posthoc protocol-corrected repeat, not an independent heldout test**: questions and references were unchanged, but the correction followed inspection of the first run's failures. Completeness is confirmed by [artifact_verification.json](results/corrected/artifact_verification.json).

## Quick start on this computer

Run all commands from the `ClipScout` folder. `run.ps1` selects the local `.venv`, then the course environment, and loads additional packages from `.runtime-packages`. The demo requires prepared `data/indexes/user_video.json` and `data/questions.json` files.

```powershell
.\run.ps1 --help
.\run.ps1 serve --index data/indexes/user_video.json --questions data/questions.json --backend local --device cuda --local-files-only --max-new-tokens 384 --image-size 560 --frames-per-clip 4 --max-frames 16 --port 7871
```

Open the [local demo](http://127.0.0.1:7871). It provides the source video, seeking through temporal citations, a tool log, and four modes. The first question includes loading the model into the GPU; `Ctrl+C` stops the server. `--local-files-only` uses existing local weights without downloading them during inference.

The interface is in English. Its translation was checked in the browser without additional model queries: [screenshot](results/demo_english_ui.png), [UI check](results/demo_ui_check.json), [HTTP check](results/demo_http_check.json). Port 7871 is set explicitly to avoid a conflict with another local application on 7860.

```powershell
.\run.ps1 ask --index data/indexes/user_video.json --question "How are subagents different from agent teams?" --mode adaptive --backend local --device cuda --local-files-only --max-new-tokens 384 --image-size 560 --frames-per-clip 4 --max-frames 16 --output runs/example.json
.\run.ps1 evaluate --index data/indexes/user_video.json --questions data/questions.json --split test --modes adaptive retrieval-only static uniform --output results/reproduction --backend local --device cuda --local-files-only --max-new-tokens 384 --image-size 560 --frames-per-clip 4 --max-frames 16
```

The explicitly specified **384 output tokens and 560 px** are the main study configuration. The image size was reduced during development because of the 8 GB GPU memory limit: a static trial at 840 px used 7891 of 8192 MiB and took 92.4 s. This was dev tuning, not a test result. CLI defaults may differ. Use a new output folder for a repeat: the CLI refuses to overwrite an existing experiment; `--resume` continues the same experiment with an unchanged manifest, source code, and runtime. The inspection cache is disabled by default. Warmup is one multimodal generation with text and four real frames; its time and usage are recorded separately from question timing.

## Models and environment

- VLM: `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`, local CUDA inference, BF16/SDPA, `do_sample=False`.
- ASR: `Systran/faster-whisper-base.en`, revision `3d3d5dee26484f91867d81cb899cfcf72b96be6c`, `.cache/whisper-base.en`, CPU/int8, language `en`.
- Python 3.12; tested dependency versions are listed in [requirements-runtime.txt](requirements-runtime.txt). Core logic and index, agent, and metric tests use the standard library; media processing and inference require optional packages.

For a separate environment, first install compatible CUDA builds of torch 2.6.0 / torchvision 0.21.0 using the [PyTorch instructions](https://pytorch.org/get-started/previous-versions/#v260), then install the project package and dependencies. CUDA 12.4 was used on this computer.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-runtime.txt -e ".[local,asr,analysis]"
.\.venv\Scripts\python scripts/download_model.py --revision 89644892e4d85e24eaac8bacfd4f463576704203
.\.venv\Scripts\python scripts/download_model.py --repo Systran/faster-whisper-base.en --revision 3d3d5dee26484f91867d81cb899cfcf72b96be6c --output .cache/whisper-base.en
```

Model downloads require network access only during preparation. The script saves regular files without requiring Windows symlink privileges, reuses cached files, checks official Git SHA-1/LFS SHA-256 hashes, and writes `download_manifest.json`. Complete dependency installation and weight downloads before the offline demo.

## Video preparation

For the supplied file, read the path from the manifest without manually typing its name:

```powershell
$source = Get-Content -LiteralPath data/source.json -Raw | ConvertFrom-Json
$video = $source.video_path
.\run.ps1 transcribe --video "$video" --model .cache/whisper-base.en --language en --device cpu --output data/transcripts/user_video.vtt
.\run.ps1 caption --video "$video" --output data/captions/user_video.json --interval-seconds 60 --frames-per-segment 2 --backend local --device cuda --local-files-only --max-new-tokens 192 --image-size 560
.\run.ps1 index --video "$video" --subtitles data/transcripts/user_video.vtt --captions data/captions/user_video.json --output data/indexes/user_video.json
```

Existing SRT/VTT files can replace ASR. For another video, change the path and output artifact names. ASR is an automatic transcript that may contain errors; frame captions are VLM observations rather than original speech. The index is built without reference answers or intervals. ASR, captioning, and indexing costs are accounted for separately from query latency.

The prepared caption file contains 43 entries and 86 frames; preprocessing used 192 new tokens and 560 px, separately from the main 384/560 query configuration. The first 42 windows used endpoint positions; the last was recovered using interior positions after a decoding failure. Total time and tokens for the original process are unknown (`null`). Use the saved captions to reproduce the study exactly; the current captioning command samples interior positions in every window and will create a different index. See the [caption manifest](data/captions/user_video.json.manifest.json).

## How the budget is spent

| Example permitted trajectory | Tool attempts | Maximum new frames |
|---|---:|---:|
| `search → inspect → inspect → inspect` | 4 | 12 |
| `inspect → inspect → inspect → inspect` | 4 | 16 |
| `search → invalid call → inspect → inspect` | 4 | 8 |

This illustrates constraints, **not measured results**. Four calls × at most four frames gives an upper bound of 16; search extracts no frames but consumes the shared call budget. An equal ceiling therefore does not imply equal actual policy costs. One inspection covers at most 120 seconds. Sparse sampling does not mean continuous viewing of that interval.

BM25 searches transcripts and visual captions grouped into 30-second chunks. The controller receives the question, remaining budget, and observations. Action generation stops at the boundary of one JSON object; this is not grammar-constrained decoding, and the runtime checks syntax and arguments separately. `inspect_clip` extracts frames with actual timestamps. Final validation permits only existing observation IDs and intervals within their associated observations. It **does not establish the semantic truth of an answer**, complete video coverage, or observation of every second in the cited interval.

## Four modes and metrics

`adaptive` chooses subsequent actions based on observations; `static` searches and performs fixed inspections of the results; `uniform` inspects four predefined windows; `retrieval-only` answers from retrieved text without fresh visual inspection. The last mode can read previously prepared visual captions, so it is a baseline without new frames, not necessarily a subtitle-only baseline. One model is used; actual calls, frames, and latency are recorded separately.

Metrics include lexical token F1/EM and required-term agreement, temporal IoU, hit/recall at IoU ≥0.3/0.5, multi-hop recall with one-to-one interval matching, conditional onset error, answerable coverage, and abstention. Errors and missing predictions remain in the denominators of their respective tasks. Lexical scores and citation geometry **do not replace human factuality assessment**; a full manual evaluation of claim support is not yet claimed.

Matched-frame comparison, repeated latency runs under an unchanged protocol, evidence gate ablation, and evaluation on external videos are **future experiments, not completed results of this pilot**. The posthoc protocol correction does not count as an independent repeated latency measurement. The assistant's visual audit is stored separately and is neither human nor blind evaluation. Measurement status and exact settings for each run are in [REPORT.md](REPORT.md) and the corresponding `manifest.json`.

Performance conditions changed during the corrected repeat: a competing `llama-server`, noticed during `static/visual_02`, was stopped with user permission on 2026-10-21 at 20:12:07 UTC. The study continued without source changes. The process's presence and load from the start of the run were not instrumented; adaptive and retrieval-only finished before it was stopped. Recorded latency reflects uncontrolled conditions and is not a rigorous comparison of policy speed.

The computation process was later restarted with `--resume` to release GPU memory. 40 completed answers, including `uniform/visual_01`, were retained; the unfinished `uniform/visual_02` was rerun, and its partial cost is unknown. Code, model, questions, and parameters were preserved. The first manifest is in `results/corrected/manifest.initial_process.json`, with history in `resume_history.json`; the new process performed its own warmup. Total cost therefore cannot be calculated from final answers and a single warmup alone.

## Checks and documentation

```powershell
python -m unittest discover -s tests -v
```

**87 tests** passed before the corrected run. They check software contracts, not the accuracy of model answers. All four modes are complete; the following commands recheck artifacts, recompute the summary, and generate figures without new model queries:

```powershell
python -m scripts.verify_study results/corrected --questions data/questions.json --index data/indexes/user_video.json
python -m scripts.summarize_results results/corrected --questions data/questions.json
python -m scripts.plot_results results/corrected
```

For your own run, replace `results/corrected` with `results/reproduction`. Use the project environment's Python; on this computer, `..\..\.venv\Scripts\python.exe` can be used with local packages enabled by `$env:PYTHONPATH = "$PWD;$PWD\.runtime-packages"`. Plotting requires the `analysis` extra (`matplotlib==3.10.8`). The verifier checks completeness, hashes, budgets, and citation structure; it is not a semantic grader. The partial pilot in `results/test` intentionally fails the four-mode completeness check.

- [Study protocol and limitations](docs/RESEARCH_PLAN.md)
- [Demo walkthrough](docs/DEMO_WALKTHROUGH.md)
- [Annotation provenance and future independent validation](docs/ANNOTATION_GUIDE.md)
- [Review of four projects](docs/RELATED_WORK.md)
- [Alignment with lectures and the assignment](docs/COURSE_ALIGNMENT.md)

Source code is distributed under Apache-2.0; the license of the user-created repository is preserved. **The source video was provided by the user; redistribution permission has not been obtained.** The code license and MIT licenses of individual reference projects do not license this video. The source video, weights, and local caches are excluded from Git. Diagnostic figures contain selected frames; rights to the source material remain with their owners. An external endpoint is enabled only by explicitly setting `--backend compatible --endpoint ... --model ...`; the study commands above run locally.
