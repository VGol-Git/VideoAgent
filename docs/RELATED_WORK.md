## Ideas used

| Work | Description in the supplied README | Decision for ClipScout | Substantial difference |
|---|---|---|---|
|  An `overview`, `skim`, `focus` sequence; searching for question-critical evidence; saving predictions and trajectories | Low-cost search followed by targeted `inspect_clip`; subsequent queries can be refined using observations | Two tools and at most four calls per question; the executor accounts for the budget |
|  A ReAct controller, a tool schema separate from the model, hierarchical memory, model-aware caching | A typed action contract, interchangeable model backends, compact observation history, and an action log | A small local runtime without a mandatory expert server or multi-GPU infrastructure |
|  A Master Agent coordinates a Grounding Agent and a Vision Agent; localization is separate from visual extraction; step limits; RL controller training | Candidate search is separate from direct visual observation; the question context sets the inspection goal | No training or RL is required: the policy of a pretrained model is tested under a small fixed budget |
|  Deterministic media processing, a searchable timeline, argument validation, budgets, evidence citations, and logging | Verifiable observation provenance, actual timestamps, argument validation, and formal answer-citation checks | An educational experiment with a substantially smaller budget; inspecting individual frames is not described as watching the entire interval |

## Important distinctions

A search result is a candidate for verification. High similarity does not prove that an event occurred, and a VLM description does not guarantee a correct interpretation of a frame. ClipScout's evidence gate checks that an observation ID exists and that a citation interval is contained in its associated observation range. It does not verify that every cited second matches an actually sampled frame and cannot deterministically establish the semantic truth of an arbitrary answer.
