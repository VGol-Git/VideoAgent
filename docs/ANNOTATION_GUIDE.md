# Annotation and independent answer validation

[questions.json](../data/questions.json) contains a frozen list of **20 questions**: 17 answerable and 3 unanswerable; dev — 7, test — 13. Categories: visual — 2, localization — 5, reasoning — 5, temporal — 5, unanswerable — 3. Wording, answers, and all other fields were copied unchanged from two drafts. The [manifest](../data/annotation_manifest.json) records SHA-256 hashes of the final file, source drafts, full ASR transcript, and frame sampling logs.

These annotations cover one 2575.006-second video and are **not an independently validated benchmark**. Eight visual questions were created by the assistant from decoded frames; twelve speech questions have **ASR-supported silver** status. Audio was not independently listened to when creating the speech references. This status must remain visible in the report.

## Reproducing the original annotations

1. Use the same source MP4 and verify its SHA-256 against the manifest. Record the decoder version and run settings. Video duration comes from decoder metadata; the end of the last ASR cue is not the end of the video.
2. For visual annotations, open [visual_questions_draft.json](../data/visual_questions_draft.json). The initial overview used 60-second steps; selected scenes were refined at 10- and 5-second steps. Each question's `annotation_notes` contains actual frame timestamps, files, and adjacent observations that bracket scene boundaries. Full lists are in `.cache/annotation/*_manifest.json`; these are local reproducible artifacts that may be absent from a clean checkout.
3. To extract a frame again, use `clipscout.media.extract_frames(video, t, t + 0.1, count=1, max_side=1280)`, where `t` is `requested` from the corresponding log. Save the returned timestamp with the image: it may differ slightly from the requested time. Assess the natural duration of the visible scene rather than turning one inspected frame into a ground-truth interval. For a new independent check, watch the video continuously around both boundaries; the original sparse sampling does not replace this.
4. For speech annotations, use [speech_questions_draft.json](../data/speech_questions_draft.json) and the full `data/transcripts/user_video.json`. The current set has 878 ASR cues; IDs and bounds were checked against the final file. The transcript was generated with Whisper `base.en`, CPU/int8, language `en`; settings are saved in `data/transcripts/user_video.manifest.json`. A new ASR version may produce different bounds and IDs and must not silently replace the frozen set.
5. For each question, check the specified `evidence_cue_ids` and the coherent speech passage. Do not infer a fact from ambiguous recognition. References about tool behavior are phrased as the video author's explanation, not an independent verification of an external product. Intervals cover complete explanations; audio-alignment accuracy was not measured manually.
6. Negative questions have an explicit scope: a particular diagram or comparison. Empty `reference_answers` and `evidence_intervals` mean that the requested value has not been established. `annotation_notes` retains the review context. The absence of a number within this scope does not establish its absence from the entire video.
7. Before the experiment, check ID uniqueness, the schema, and `0 <= start < end < duration`; freeze the dataset and checksum. In this set, the related Email/Article scenes are in dev; speech questions do not overlap the 1152.5–1237.5-second interval. The first delegation example and its token comparison are also entirely in dev. Recurring concepts still connect parts of the same video, so test does not measure transfer to new videos.

The larger frame sample used for **annotation** is outside the model's budget: when answering, the agent is limited to four tool calls and sixteen frames. Annotation information must not enter the model prompt. Browser examples show the question, not the reference answer.

## Independent review rubric

The reviewer first watches the source clip with audio and checks the reference itself. They then assess the anonymized answer, citations, and actual frames/cues obtained by the system. The mode name and automatic scores are hidden until grading is complete. Two independent reviewers are preferable; disagreements are retained and resolved in a separate discussion. Assistant review is useful for diagnostics but is not described as independent human evaluation.

| Field | 0 | 1 | 2 |
|---|---|---|---|
| `answer_correctness` | Incorrect answer, or a substantive claim contradicts the source | The main idea is partly correct, with a significant omission or inaccuracy | All required facts are correct; scope and qualifications are preserved |
| `evidence_support` | A citation does not support the key claim, or evidence is fabricated | Only some substantive claims are supported | Every substantive claim is supported by evidence actually available to the system |
| `temporal_order` | The order is wrong, or a temporal relationship is fabricated | The order is incomplete or ambiguous | All requested events are ordered correctly |

For non-temporal questions, `temporal_order = null`. For an answerable question, abstention receives `answer_correctness = 0` but is recorded separately as abstention rather than hallucination. For an unanswerable question, `abstention_correct = 1` only for a substantive refusal to give an unsupported answer; an execution error is not a successful abstention. A number inferred from missing data is an unsupported claim even if a valid citation appears beside it.

Strict semantic success on an answerable question requires `answer_correctness = 2`, `evidence_support = 2`, and, where applicable, `temporal_order = 2`. A useful audit row contains `question_id`, `mode`, `run_id`, the three scores, `abstention_correct`, `unsupported_claims`, `reviewer`, and `notes`. For annotation errors, add `reference_issue`; after correction, release a new dataset version and reevaluate all compared modes rather than tailoring the reference to one answer.

## What automatic metrics cannot establish

An existing citation ID and matching timestamps check provenance and interval geometry. They do not prove that the citation supports the answer. Keyword overlap does not distinguish before from after, negation from affirmation, or a cost estimate from actual measured usage. Four frames do not establish the absence of an event between them.

Speech questions and search may use **the same erroneous ASR transcript**. High agreement with a silver reference then shows agreement with a shared source of errors. Independent validation must include listening to the audio, and results should be broken down by visual/speech source, category, and split. Scores from one video and small groups do not support a broad claim of method superiority.
