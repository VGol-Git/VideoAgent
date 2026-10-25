"""Uniform-only serving fixes, tested without model weights, CUDA or network."""

from contextlib import contextmanager
import unittest
from unittest.mock import patch

from clipscout.agent import AgentConfig, UNIFORM_POLICY_VERSION, VideoAgent
from clipscout.index import Segment, VideoIndex
from test_agent import ScriptedBackend, answer


DURATION = 2575.0057667
ABSTENTION = {"action": "answer", "status": "insufficient_evidence",
              "answer": "The sampled evidence does not establish the answer.", "citations": []}


def decode_samples(path, start, end, count, max_side, **kwargs):
    target = kwargs.get("target_timestamp")
    if target is not None:
        return [(target + .009, None)]
    return [(start + (i + .5) * (end - start) / count, None) for i in range(count)]


def index():
    # Text is plentiful in the test index; selection must still use the
    # question and fixed policy, never indexed relevance or an answer key.
    segments = [Segment(f"T{i}", float(i), float(i + 1), "Narration about entropy.")
                for i in range(2575)]
    return VideoIndex(DURATION, segments, video_path="mock.mp4", video_sha256="test-video")


class UniformAgentTests(unittest.TestCase):
    def agent(self, responses, **config):
        return VideoAgent(index(), ScriptedBackend(responses), AgentConfig(**config))

    def run_without_search(self, agent, question):
        with patch("clipscout.media.extract_frames", side_effect=decode_samples) as decode, \
                patch.object(agent.index, "search", side_effect=AssertionError("Uniform must not search")):
            result = agent.run(question, mode="uniform")
        self.assertEqual(result["mode"], "uniform")
        self.assertEqual(result["uniform_policy_version"], UNIFORM_POLICY_VERSION)
        return result, decode

    def test_point_question_decodes_and_cites_verified_target_frame(self):
        agent = self.agent([answer()])
        result, decode = self.run_without_search(agent, "What is on the slide at 10:10?")
        self.assertEqual(result["status"], "answered", result)
        self.assertEqual(result["routing"], "uniform_explicit_timestamp")
        self.assertEqual(result["requested_timestamps"], [610])
        self.assertEqual(result["frames_used"], 1)
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(decode.call_args.kwargs["target_timestamp"], 610)
        evidence = result["evidence"]["E001"]
        self.assertTrue(evidence["target_frame_verified"])
        self.assertEqual(evidence["requested_timestamp"], 610)
        self.assertEqual(evidence["target_frame_timestamp"], 610.009)
        self.assertEqual(agent.backend.states[0]["requested_timestamps"], [610])
        supplied = agent.backend.states[0]["inspected_evidence"]["E001"]
        self.assertTrue(supplied["target_frame_verified"])
        self.assertEqual(supplied["target_frame_timestamp"], 610.009)

    def test_multiple_points_require_each_target_within_the_same_budget(self):
        final = answer()
        final["citations"].append({"evidence_id": "E002"})
        agent = self.agent([final], max_frames=2)
        result, decode = self.run_without_search(agent, "Compare at 10:10 and 13:20.")
        self.assertEqual(result["status"], "answered", result)
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["frames_used"], 2)
        self.assertEqual([call.kwargs["target_timestamp"] for call in decode.call_args_list], [610, 800])
        self.assertEqual({cite["evidence_id"] for cite in result["citations"]}, {"E001", "E002"})

    def test_impossible_or_over_budget_points_do_not_decode(self):
        cases = (("at -1 second", {}), ("at 3000 seconds", {}),
                 ("at 10:99", {}), ("at 10:10 and 13:20", {"max_frames": 1}),
                 ("at 1:00 and 2:00 and 3:00 and 4:00 and 5:00", {}),
                 ("at 10:10", {"max_frames": 0}))
        for question, config in cases:
            with self.subTest(question=question, config=config):
                agent = self.agent([ABSTENTION], **config)
                with patch("clipscout.media.extract_frames") as decode, \
                        patch.object(agent.index, "search", side_effect=AssertionError("No search")):
                    result = agent.run(question, mode="uniform")
                self.assertEqual(result["status"], "insufficient_evidence", result)
                self.assertEqual(result["tool_calls"], 0)
                self.assertEqual(result["frames_used"], 0)
                decode.assert_not_called()
                self.assertEqual(agent.backend.visual_calls, 0)

    def test_scoped_uniform_distributes_windows_inside_requested_bounds(self):
        cases = (("What is visible between 20:00 and 21:00?", 1200, 1260),
                 ("What is visible after 30:00?", 1800, DURATION),
                 ("What is visible between 17:15 and 17:25?", 1035, 1045))
        for question, start, end in cases:
            with self.subTest(question=question):
                agent = self.agent([answer()])
                result, decode = self.run_without_search(agent, question)
                self.assertEqual(result["status"], "answered", result)
                self.assertEqual(result["routing"], "uniform_temporal_scope")
                self.assertGreater(result["tool_calls"], 0)
                self.assertLessEqual(result["tool_calls"], 4)
                self.assertLessEqual(result["frames_used"], 16)
                intervals = [(e["start"], e["end"]) for e in result["evidence"].values()]
                self.assertEqual(intervals[0][0], start)
                self.assertAlmostEqual(intervals[-1][1], end, delta=.001)
                for lo, hi in intervals:
                    self.assertTrue(start <= lo < hi <= end)
                    self.assertLessEqual(hi - lo, 120)
                for timestamp in result["unique_sampled_timestamps"]:
                    self.assertTrue(start <= timestamp < end)
                self.assertEqual(decode.call_count, result["tool_calls"])
                self.assertFalse(result["coverage"]["complete_visual_coverage"])

    def test_scope_onset_preserves_question_and_does_not_require_all_windows(self):
        question = 'After 19:00, when does the "Email to Decision Brief" slide appear?'
        agent = self.agent([answer()])
        result, _ = self.run_without_search(agent, question)
        self.assertEqual(result["status"], "answered", result)
        self.assertEqual(len(result["evidence"]), 4)
        self.assertEqual(len(result["citations"]), 1)
        state = agent.backend.states[0]
        self.assertEqual(state["question"], question)
        self.assertNotIn("Describe the inspected windows in order", state["instruction"])
        self.assertEqual(len(result["decisions"]), 1)

    def test_unscoped_global_sampling_preserves_historical_fractions(self):
        agent = self.agent([answer()])
        result, decode = self.run_without_search(agent, "Explain the worker roles.")
        self.assertEqual(result["status"], "answered", result)
        self.assertEqual(result["routing"], "uniform_global")
        self.assertEqual(result["tool_calls"], 4)
        self.assertEqual(result["frames_used"], 16)
        expected = [(max(0, DURATION * fraction - 15), min(DURATION, DURATION * fraction + 15))
                    for fraction in (.125, .375, .625, .875)]
        actual = [(call.args[1], call.args[2]) for call in decode.call_args_list]
        self.assertEqual(actual, expected)
        self.assertTrue(all("target_timestamp" not in call.kwargs for call in decode.call_args_list))

    def test_low_frame_budgets_never_attempt_impossible_extra_inspections(self):
        for question in ("Explain the worker roles.", "What is visible after 15:00?"):
            for budget in (0, 1, 3):
                with self.subTest(question=question, max_frames=budget):
                    agent = self.agent([answer() if budget else ABSTENTION], max_frames=budget)
                    result, decode = self.run_without_search(agent, question)
                    self.assertEqual(result["status"], "answered" if budget else "insufficient_evidence", result)
                    self.assertLessEqual(result["frames_used"], budget)
                    self.assertLessEqual(result["tool_calls"], min(4, budget))
                    self.assertEqual(result["tool_calls"], decode.call_count)
                    self.assertTrue(all(event["ok"] for event in result["trace"]))
                    self.assertTrue(all(call.kwargs["count"] >= 1 for call in decode.call_args_list))

    def test_two_equivalent_invalid_json_answers_stop_without_third_generation(self):
        invalid = {"action": "answer", "status": "answered", "answer": "Unsupported answer", "citations": []}
        reordered = '{"citations": [], "answer": "Unsupported answer", "status": "answered", "action": "answer"}'
        agent = self.agent([invalid, reordered, answer()], use_vision=False, max_format_retries=3)
        result = agent.run("Explain entropy.", mode="uniform")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "invalid_model_response")
        self.assertEqual(len(result["decisions"]), 2)
        self.assertEqual(agent.backend.usage["model_calls"], 2)
        self.assertEqual(result["citations"], [])

    def test_repeated_malformed_output_stops_and_does_not_earn_abstention_credit(self):
        agent = self.agent(["not JSON", "  not JSON\n", ABSTENTION], use_vision=False)
        result = agent.run("Explain entropy.", mode="uniform")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "invalid_model_response")
        self.assertEqual(len(result["decisions"]), 2)
        self.assertEqual(result["citations"], [])

    def test_distinct_invalid_answers_obey_retry_cap_and_valid_repair_still_works(self):
        for responses, status, attempts in ((["not JSON", "[]", '{"action":'], "error", 3),
                                            (["not JSON", answer()], "answered", 2)):
            with self.subTest(responses=responses):
                agent = self.agent(responses, use_vision=False)
                result = agent.run("Explain entropy.", mode="uniform")
                self.assertEqual(result["status"], status, result)
                self.assertEqual(len(result["decisions"]), attempts)
                self.assertEqual(agent.backend.usage["model_calls"], attempts)

    def test_adaptive_keeps_existing_identical_invalid_retry_policy(self):
        agent = self.agent(["not JSON"] * 3, use_vision=False)
        result = agent.run("Explain entropy.", mode="adaptive")
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(result["decisions"]), 3)
        self.assertEqual(agent.backend.usage["model_calls"], 3)
        self.assertNotIn("uniform_policy_version", result)

    def test_backend_uniform_session_is_scoped_to_uniform_requests(self):
        class SessionBackend(ScriptedBackend):
            def __init__(self):
                super().__init__([answer(), ABSTENTION])
                self.session_active = False
                self.sessions = []
                self.call_sessions = []

            @contextmanager
            def uniform_session(self):
                self.sessions.append("enter")
                self.session_active = True
                try:
                    yield
                finally:
                    self.session_active = False
                    self.sessions.append("exit")

            def generate(self, system, user, images=None):
                self.call_sessions.append(self.session_active)
                return super().generate(system, user, images)

        backend = SessionBackend()
        agent = VideoAgent(index(), backend, AgentConfig(use_vision=False))
        self.assertEqual(agent.run("Explain entropy.", mode="uniform")["status"], "answered")
        self.assertEqual(agent.run("Explain entropy.", mode="adaptive")["status"], "insufficient_evidence")
        self.assertEqual(backend.sessions, ["enter", "exit"])
        self.assertEqual(backend.call_sessions, [True, False])
        self.assertFalse(backend.session_active)


if __name__ == "__main__":
    unittest.main()
