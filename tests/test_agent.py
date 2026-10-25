"""Behavioral regression tests without model weights or network access."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clipscout.agent import AgentConfig, FINAL_SYSTEM, SYSTEM, VideoAgent, parse_action
from clipscout.index import Segment, VideoIndex


def tool(name, **args):
    return {"tool": name, "args": args}


def answer(evidence_id="E001", **citation):
    return {"action": "answer", "status": "answered", "answer": "The inspected lecture discusses entropy.",
            "citations": [{"evidence_id": evidence_id, **citation}]}


class ScriptedBackend:
    """Separates controller decisions from visual observations."""

    name = "test-scripted-vlm"
    max_new_tokens = 128

    def __init__(self, responses, fail_vision=False):
        self.responses = iter(responses)
        self.fail_vision = fail_vision
        self.visual_calls = 0
        self.states = []
        self.systems = []
        self.usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}

    def generate(self, system, user, images=None):
        self.usage["model_calls"] += 1
        if images is not None:
            self.visual_calls += 1
            if self.fail_vision:
                raise RuntimeError("simulated VLM failure after frames were decoded")
            return "The slide shows the word entropy."
        self.systems.append(system)
        self.states.append(json.loads(user))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def small_index(video_path=None, text="Entropy measures uncertainty."):
    return VideoIndex(120, [Segment("T1", 0, 10, text),
                            Segment("T2", 35, 45, "Entropy and probability distributions."),
                            Segment("T3", 65, 75, "Conditional entropy is explained."),
                            Segment("T4", 95, 105, "Independent events have additive entropy.")],
                      video_path=video_path, video_sha256="test-video-content")


class AgentBehaviorTests(unittest.TestCase):
    def test_action_aliases_preserve_arguments_and_legacy_spelling(self):
        for name, args in (("search", {"query": "entropy", "top_k": 2, "source": "transcript"}),
                           ("inspect_clip", {"start": 1.5, "end": 9.5, "question": "Read the title"})):
            with self.subTest(name=name):
                expected = {"tool": name, "args": args}
                self.assertEqual(parse_action(json.dumps({"action": name, "args": args})), expected)
                self.assertEqual(parse_action(json.dumps(expected)), expected)

    def test_action_aliases_execute_search_inspection_and_cited_answer(self):
        backend = ScriptedBackend([
            {"action": "search", "args": {"query": "entropy", "top_k": 1, "source": "transcript"}},
            {"action": "inspect_clip", "args": {"start": 1, "end": 9, "question": "Explain entropy"}},
            answer(start=1, end=9),
        ])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("Find entropy")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual([event["tool"] for event in result["trace"]], ["search", "inspect_clip"])
        self.assertEqual(len(result["trace"][0]["result"]["hits"]), 1)
        self.assertEqual(result["trace"][1]["args"], {"start": 1, "end": 9, "question": "Explain entropy"})
        self.assertEqual(result["citations"], [{"evidence_id": "E001", "start": 1.0, "end": 9.0}])

    def test_aliases_share_validation_duplicate_detection_and_four_attempt_budget(self):
        backend = ScriptedBackend([
            {"action": "search", "args": {"query": "entropy", "top_k": True}},
            tool("search", query="entropy", top_k=True),
            {"action": "inspect_clip", "args": {"start": True, "end": 10}},
            {"action": "search", "args": {"query": "entropy", "source": "unknown"}},
            {"action": "inspect_clip", "args": {"start": 0, "end": 10}},
            {"action": "answer", "status": "insufficient_evidence", "answer": "No verified evidence.", "citations": []},
        ])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("Find entropy")
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(len(result["trace"]), 4)
        self.assertTrue(all(not event["ok"] for event in result["trace"]))
        self.assertIn("Duplicate", result["trace"][1]["error"])
        self.assertEqual(result["evidence"], {})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual([state["calls_remaining"] for state in backend.states], [4, 3, 2, 1, 0, 0])
        self.assertEqual(result["decisions"][-2]["error_type"], "invalid_model_response")

    def test_unknown_action_is_not_coerced_into_a_tool(self):
        unknown = {"action": "delete_video", "args": {"path": "video.mp4"}}
        self.assertEqual(parse_action(json.dumps(unknown)), unknown)
        backend = ScriptedBackend([unknown, unknown, unknown])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("Find entropy")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "invalid_model_response")
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(result["trace"], [])
        self.assertEqual(backend.usage["model_calls"], 3)

    def test_ambiguous_action_and_tool_discriminants_are_rejected(self):
        for action, name in (("search", "inspect_clip"), ("search", "search"), ("answer", "search")):
            with self.subTest(action=action, tool=name), self.assertRaises(ValueError):
                parse_action(json.dumps({"action": action, "tool": name, "args": {"query": "entropy"}}))

    def test_four_attempts_include_invalid_and_duplicate_requests(self):
        backend = ScriptedBackend([
            tool("search", query="entropy", top_k=True),
            tool("unsupported"), tool("unsupported"),
            tool("inspect_clip", start=115, end=125),
            tool("search", query="entropy"),
            {"action": "answer", "status": "insufficient_evidence", "answer": "No verified evidence.", "citations": []},
        ])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("Find entropy")
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(len(result["trace"]), 4)
        self.assertTrue(all(not event["ok"] for event in result["trace"]))
        self.assertIn("Duplicate", result["trace"][2]["error"])
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(backend.states[-1]["calls_remaining"], 0)

    def test_empty_search_can_recover_then_inspect_and_answer(self):
        backend = ScriptedBackend([
            tool("search", query="zxqv-no-matches"),
            tool("search", query="entropy"),
            tool("inspect_clip", start=0, end=10), answer(start=0, end=10),
        ])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("Find entropy")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["tool_calls"], 3)
        self.assertTrue(result["trace"][0]["result"]["empty"])
        self.assertFalse(result["trace"][1]["result"]["empty"])
        self.assertEqual(list(result["evidence"]), ["E001"])
        self.assertEqual(result["evidence"]["E001"]["modality"], "transcript")
        self.assertEqual(result["frames_used"], 0)

    def test_malformed_json_stops_without_tools_or_unbounded_retries(self):
        backend = ScriptedBackend(["not JSON", "[]", '{"tool":'])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("Find entropy")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["error_type"], "invalid_model_response")
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(backend.usage["model_calls"], 3)
        self.assertEqual(len(result["decisions"]), 3)
        self.assertTrue(all("error" in decision for decision in result["decisions"]))
        self.assertIn("Invalid model response", result["stop_reason"])

    def test_uninspected_search_results_cannot_be_cited(self):
        backend = ScriptedBackend([tool("search", query="entropy"), answer(), answer(), answer()])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("Find entropy")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "invalid_model_response")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["evidence"], {})
        self.assertEqual(result["citations"], [])

    def test_unknown_outside_and_nonfinite_citations_are_rejected(self):
        agent = VideoAgent(small_index(), ScriptedBackend([]), AgentConfig(use_vision=False))
        ledger = {"E001": {"start": 30, "end": 40}}
        bad = [answer("unknown"), answer(start=29, end=40), answer(start=30, end=41),
               answer(start=40, end=30), answer(start=True, end=40),
               answer(start=float("nan"), end=40), answer(start=30, end=float("inf"))]
        for action in bad:
            with self.subTest(action=action), self.assertRaises(ValueError):
                agent._validate_answer(action, ledger)
        accepted = agent._validate_answer(answer(start=32, end=35), ledger)
        self.assertEqual(accepted["citations"][0]["start"], 32)

    def test_json_nonfinite_constants_are_not_actions(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_action('{"tool":"inspect_clip","args":{"start":' + value + ',"end":10}}')

    def test_visual_backend_failure_is_terminal_and_charges_decoded_frames(self):
        responses = [tool("inspect_clip", start=i * 10, end=i * 10 + 5) for i in range(4)]
        responses.append({"action": "answer", "status": "insufficient_evidence", "answer": "No verified observation.", "citations": []})
        backend = ScriptedBackend(responses, fail_vision=True)
        config = AgentConfig(frames_per_clip=2, max_frames=4)
        with patch("clipscout.media.extract_frames", return_value=[(1.0, None), (2.0, None)]) as extraction:
            result = VideoAgent(small_index("mock-video.mp4"), backend, config).run("Find entropy")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "backend_error")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["frames_used"], 2)
        self.assertEqual(extraction.call_count, 1)
        self.assertEqual(backend.visual_calls, 1)
        self.assertEqual(result["trace"][0]["error_type"], "backend_error")
        self.assertEqual(result["evidence"], {})

    def test_visual_cache_reuses_only_matching_model_and_index(self):
        cache_root = Path(__file__).resolve().parents[1] / ".cache"
        cache_root.mkdir(exist_ok=True)
        responses = [tool("inspect_clip", start=0, end=10), answer()] * 2
        backend = ScriptedBackend(responses)
        with tempfile.TemporaryDirectory(dir=cache_root) as directory:
            with patch("clipscout.media.extract_frames", return_value=[(1.0, None), (2.0, None)]):
                agent = VideoAgent(small_index("mock.mp4"), backend, AgentConfig(frames_per_clip=2), directory)
                first = agent.run("Find entropy")
                second = agent.run("Find entropy")
                self.assertFalse(first["evidence"]["E001"]["cache_hit"])
                self.assertTrue(second["evidence"]["E001"]["cache_hit"])
                self.assertEqual(backend.visual_calls, 1)
                self.assertEqual(second["usage"]["model_calls"], 2)
                changed = ScriptedBackend([tool("inspect_clip", start=0, end=10), answer()])
                changed.name = "different-model"
                third = VideoAgent(small_index("mock.mp4"), changed, AgentConfig(frames_per_clip=2), directory).run("Find entropy")
                self.assertFalse(third["evidence"]["E001"]["cache_hit"])
                self.assertEqual(changed.visual_calls, 1)
                changed_index = ScriptedBackend([tool("inspect_clip", start=0, end=10), answer()])
                fourth = VideoAgent(small_index("mock.mp4", text="Corrected entropy subtitle."),
                                    changed_index, AgentConfig(frames_per_clip=2), directory).run("Find entropy")
                self.assertFalse(fourth["evidence"]["E001"]["cache_hit"])
                self.assertEqual(changed_index.visual_calls, 1)

    def test_partial_decoding_failure_is_charged_without_entering_ledger(self):
        from clipscout.media import MediaError

        error = MediaError("failed while decoding second sample")
        error.frames_decoded = 1
        error.sampled_timestamps = [1.0]
        backend = ScriptedBackend([tool("inspect_clip", start=0, end=10),
                                   tool("inspect_clip", start=20, end=30),
                                   {"action": "answer", "status": "insufficient_evidence",
                                    "answer": "No verified observation.", "citations": []}])
        with patch("clipscout.media.extract_frames", side_effect=error) as extraction:
            result = VideoAgent(small_index("mock.mp4"), backend,
                                AgentConfig(frames_per_clip=2, max_frames=1)).run("Find entropy")
        self.assertEqual(result["frames_used"], 1)
        self.assertEqual(result["unique_sampled_timestamps"], [1.0])
        self.assertEqual(extraction.call_count, 1)
        self.assertEqual(backend.visual_calls, 0)
        self.assertEqual(result["evidence"], {})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "media_error")

    def test_uniform_baseline_charges_frames_when_visual_model_fails(self):
        backend = ScriptedBackend([{"action": "answer", "status": "insufficient_evidence",
                                    "answer": "No verified observation.", "citations": []}], fail_vision=True)
        with patch("clipscout.media.extract_frames", return_value=[(1.0, None), (2.0, None)]) as extraction:
            result = VideoAgent(small_index("mock.mp4"), backend,
                                AgentConfig(frames_per_clip=2, max_frames=4)).run("entropy", mode="uniform")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "backend_error")
        self.assertEqual(result["frames_used"], 2)
        self.assertEqual(extraction.call_count, 1)
        self.assertEqual(backend.visual_calls, 1)
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(backend.usage["model_calls"], 1)

    def test_decoder_failures_can_recover_but_cannot_reset_frame_budget(self):
        from clipscout.media import MediaError

        def fail_after_decoding(path, start, end, count, max_side):
            error = MediaError("simulated partial decoder failure")
            error.frames_decoded = count
            error.sampled_timestamps = [start + 0.5 + i for i in range(count)]
            raise error

        actions = [tool("inspect_clip", start=10 * i, end=10 * i + 5) for i in range(4)]
        actions.append({"action": "answer", "status": "insufficient_evidence",
                        "answer": "No usable observations available.", "citations": []})
        backend = ScriptedBackend(actions)
        with patch("clipscout.media.extract_frames", side_effect=fail_after_decoding) as extraction:
            result = VideoAgent(small_index("mock.mp4"), backend,
                                AgentConfig(frames_per_clip=2, max_frames=4)).run("entropy")
        self.assertEqual(result["frames_used"], 4)
        self.assertEqual(extraction.call_count, 2)
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(backend.visual_calls, 0)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "media_error")

    def test_controller_backend_failure_is_error_without_format_retries(self):
        backend = ScriptedBackend([tool("search", query="entropy"), RuntimeError("endpoint unavailable")])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("entropy")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "backend_error")
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["citations"], [])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(backend.usage["model_calls"], 2)
        self.assertIn("endpoint unavailable", result["stop_reason"])
        self.assertEqual(result["decisions"][-1]["error_type"], "backend_error")

    def test_format_retry_limit_is_hard_and_cumulative(self):
        for retries in (0, 3):
            with self.subTest(retries=retries):
                backend = ScriptedBackend(["invalid JSON"] * (retries + 1))
                result = VideoAgent(small_index(), backend,
                                    AgentConfig(use_vision=False, max_format_retries=retries)).run("entropy")
                self.assertEqual(result["status"], "error")
                self.assertEqual(backend.usage["model_calls"], retries + 1)
                self.assertEqual(result["tool_calls"], 0)
        backend = ScriptedBackend(["invalid JSON", tool("search", query="entropy"), "invalid JSON",
                                   tool("inspect_clip", start=0, end=10), "invalid JSON"])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("entropy")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(backend.usage["model_calls"], 5)

    def test_genuine_abstention_and_valid_tool_budget_exhaustion_remain_insufficient(self):
        abstention = {"action": "answer", "status": "insufficient_evidence",
                      "answer": "No matching event was established by these observations.", "citations": []}
        for mode in ("adaptive", "retrieval-only", "static", "uniform"):
            with self.subTest(mode=mode):
                result = VideoAgent(small_index(), ScriptedBackend([abstention]),
                                    AgentConfig(use_vision=False)).run("unfindable", mode=mode)
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertNotIn("error_type", result)
        backend = ScriptedBackend([tool("search", query=f"unfindable{i}") for i in range(4)] + [abstention])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("unfindable")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["tool_calls"], 4)
        self.assertTrue(all(event["ok"] for event in result["trace"]))

    def test_baseline_backend_and_response_failures_are_errors(self):
        for mode in ("retrieval-only", "static", "uniform"):
            for response, error_type in ((RuntimeError("endpoint unavailable"), "backend_error"),
                                         ("not JSON", "invalid_model_response"),
                                         (answer("unknown-id"), "invalid_model_response"),
                                         ({"status": "insufficient_evidence", "answer": "missing action",
                                           "citations": []}, "invalid_model_response"),
                                         ({"action": "answer", "status": "error", "answer": "model-chosen error",
                                           "citations": []}, "invalid_model_response")):
                with self.subTest(mode=mode, response=response):
                    attempts = 1 if isinstance(response, Exception) else 2 if mode == "uniform" else 3
                    backend = ScriptedBackend([response] * attempts)
                    result = VideoAgent(small_index(), backend,
                                        AgentConfig(use_vision=False)).run("entropy", mode=mode)
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error_type"], error_type)
                    self.assertEqual(result["answer"], "")
                    self.assertEqual(result["citations"], [])
                    self.assertEqual(backend.usage["model_calls"], attempts)
                    self.assertEqual(len(result["decisions"]), attempts)
                    self.assertEqual(result["decisions"][0]["error_type"], error_type)

    def test_finalization_prompt_contains_only_final_answer_shapes(self):
        shapes = [json.loads(line) for line in FINAL_SYSTEM.splitlines() if line.startswith("{")]
        self.assertEqual(len(shapes), 2)
        self.assertEqual({shape["status"] for shape in shapes}, {"answered", "insufficient_evidence"})
        for shape in shapes:
            self.assertEqual(shape["action"], "answer")
            self.assertNotIn("tool", shape)
            self.assertNotIn("args", shape)

    def test_baseline_finalization_repairs_without_tools_and_charges_all_generation(self):
        class MeteredBackend(ScriptedBackend):
            def generate(self, system, user, images=None):
                self.usage["input_tokens"] += 31
                self.usage["output_tokens"] += 7
                return super().generate(system, user, images)

        for mode, expected_calls in (("retrieval-only", 1), ("static", 4), ("uniform", 4)):
            with self.subTest(mode=mode):
                bad_tool = {"action": "inspect_clip", "args": {"start": 0, "end": 10}}
                bad_citation = answer("unknown-id")
                backend = MeteredBackend([bad_tool, bad_citation, answer()])
                backend.usage.update(model_calls=10, input_tokens=100, output_tokens=200)
                result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("entropy", mode=mode)
                self.assertEqual(result["status"], "answered")
                self.assertEqual(result["tool_calls"], expected_calls)
                self.assertEqual(len(result["decisions"]), 3)
                self.assertEqual(result["usage"], {"model_calls": 3, "input_tokens": 93, "output_tokens": 21})
                self.assertEqual(backend.systems, [FINAL_SYSTEM] * 3)
                self.assertTrue(all(state["calls_remaining"] == 0 for state in backend.states))
                self.assertEqual(json.loads(backend.states[1]["previous_invalid_response"]), bad_tool)
                self.assertEqual(json.loads(backend.states[2]["previous_invalid_response"]), bad_citation)
                self.assertTrue(backend.states[1]["last_error"])
                self.assertEqual(backend.states[0]["inspected_evidence"], backend.states[2]["inspected_evidence"])

    def test_baseline_finalization_obeys_configured_retry_cap(self):
        for retries in (0, 3):
            with self.subTest(retries=retries):
                backend = ScriptedBackend([tool("search", query="entropy")] * (retries + 1))
                result = VideoAgent(small_index(), backend,
                                    AgentConfig(use_vision=False, max_format_retries=retries)).run("entropy", mode="retrieval-only")
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_type"], "invalid_model_response")
                self.assertEqual(result["tool_calls"], 1)
                self.assertEqual(result["usage"]["model_calls"], retries + 1)
                self.assertEqual(len(result["decisions"]), retries + 1)

    def test_adaptive_switches_to_finalization_at_four_calls_and_repairs_without_a_fifth(self):
        forbidden_fifth = tool("inspect_clip", start=95, end=105)
        backend = ScriptedBackend([tool("search", query="entropy"),
                                   tool("inspect_clip", start=0, end=10),
                                   tool("inspect_clip", start=35, end=45),
                                   tool("inspect_clip", start=65, end=75),
                                   forbidden_fifth, answer("E003", start=65, end=75)])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("entropy")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(len(result["evidence"]), 3)
        self.assertEqual(result["usage"]["model_calls"], 6)
        self.assertEqual(backend.systems, [SYSTEM] * 4 + [FINAL_SYSTEM] * 2)
        self.assertEqual([state["calls_remaining"] for state in backend.states], [4, 3, 2, 1, 0, 0])
        self.assertEqual(json.loads(backend.states[-1]["previous_invalid_response"]), forbidden_fifth)
        self.assertEqual(result["decisions"][4]["error_type"], "invalid_model_response")

    def test_rejected_adaptive_answer_repairs_in_finalization_with_no_available_tools(self):
        abstention = {"action": "answer", "status": "insufficient_evidence",
                      "answer": "I have no inspected evidence.", "citations": []}
        backend = ScriptedBackend([tool("search", query="entropy"), answer(),
                                   tool("inspect_clip", start=0, end=10), abstention])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("entropy")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["evidence"], {})
        self.assertEqual(backend.systems, [SYSTEM, SYSTEM, FINAL_SYSTEM, FINAL_SYSTEM])
        self.assertEqual([state["calls_remaining"] for state in backend.states], [4, 3, 0, 0])
        self.assertTrue(all("No tool" in state["instruction"] for state in backend.states[2:]))

    def test_finalization_does_not_reset_adaptive_format_retry_budget(self):
        backend = ScriptedBackend(["not JSON", tool("search", query="entropy"),
                                   tool("inspect_clip", start=0, end=10),
                                   tool("inspect_clip", start=35, end=45),
                                   tool("inspect_clip", start=65, end=75),
                                   answer("unknown-id"), answer("unknown-id")])
        result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("entropy")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "invalid_model_response")
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(result["usage"]["model_calls"], 7)
        self.assertEqual(sum("error" in decision for decision in result["decisions"]), 3)
        self.assertEqual(backend.systems[-2:], [FINAL_SYSTEM, FINAL_SYSTEM])

    def test_execution_and_format_failures_do_not_earn_negative_abstention_credit(self):
        from clipscout.evaluation import evaluate

        question = {"id": "negative", "question": "Find the nonexistent event.", "category": "unanswerable",
                    "split": "test", "answerable": False, "reference_answers": [], "evidence_intervals": []}
        genuine_abstention = {"action": "answer", "status": "insufficient_evidence",
                              "answer": "The event is not established by available evidence.", "citations": []}
        cases = [([RuntimeError("backend unavailable")], 0), (["bad JSON"] * 3, 0),
                 ([answer("invented")] * 3, 0), ([genuine_abstention], 1)]
        for responses, expected_credit in cases:
            with self.subTest(responses=responses):
                result = VideoAgent(small_index(), ScriptedBackend(responses),
                                    AgentConfig(use_vision=False)).run(question["question"], question["id"])
                summary = evaluate([question], [result], bootstrap_samples=10)["summary"]
                self.assertEqual(summary["unanswerable_abstention_accuracy"], expected_credit)
                self.assertEqual(summary["task_success_proxy"], expected_credit)

    def test_corrupt_cached_observation_is_error_with_charged_frames(self):
        cache_root = Path(__file__).resolve().parents[1] / ".cache"
        cache_root.mkdir(exist_ok=True)
        backend = ScriptedBackend([tool("inspect_clip", start=0, end=10), answer(),
                                   tool("inspect_clip", start=0, end=10)])
        with tempfile.TemporaryDirectory(dir=cache_root) as directory:
            with patch("clipscout.media.extract_frames", return_value=[(1.0, None), (2.0, None)]):
                agent = VideoAgent(small_index("mock.mp4"), backend, AgentConfig(frames_per_clip=2), directory)
                self.assertEqual(agent.run("entropy")["status"], "answered")
                cached = next(Path(directory).glob("*.json"))
                cached.write_text('{"observation": null}', encoding="utf-8")
                result = agent.run("entropy")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "inspection_error")
        self.assertIn("cache_read", result["stop_reason"])
        self.assertEqual(result["frames_used"], 2)
        self.assertEqual(result["evidence"], {})
        self.assertEqual(backend.visual_calls, 1)

    def test_baselines_keep_the_same_call_budget_and_label_retrieval(self):
        for mode, expected_calls in (("retrieval-only", 1), ("static", 4), ("uniform", 4)):
            with self.subTest(mode=mode):
                backend = ScriptedBackend([answer()])
                result = VideoAgent(small_index(), backend, AgentConfig(use_vision=False)).run("entropy", mode=mode)
                self.assertEqual(result["tool_calls"], expected_calls)
                self.assertEqual(result["status"], "answered")
                self.assertEqual(result["frames_used"], 0)
                self.assertEqual(backend.usage["model_calls"], 1)
                if mode == "retrieval-only":
                    self.assertTrue(all(e["retrieval_only"] for e in result["evidence"].values()))
                else:
                    self.assertTrue(all(e["modality"] == "transcript" for e in result["evidence"].values()))

    def test_configuration_rejects_booleans_nonfinite_and_invalid_retry_limits(self):
        invalid = [{"max_tool_calls": True}, {"max_tool_calls": 5}, {"frames_per_clip": True},
                   {"max_frames": False}, {"max_clip_seconds": True},
                   {"max_clip_seconds": float("nan")}, {"max_clip_seconds": float("inf")},
                   {"max_format_retries": -1}, {"max_format_retries": True},
                   {"max_format_retries": 1.5}, {"max_format_retries": 4},
                   {"max_format_retries": float("nan")}, {"use_vision": "false"}]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AgentConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
