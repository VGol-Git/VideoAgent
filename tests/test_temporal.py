"""Exact-moment regression checks without model inference."""
import unittest
from unittest.mock import patch

from clipscout.agent import AgentConfig, VideoAgent
from clipscout.index import Segment, VideoIndex
from clipscout.temporal import explicit_moments
from test_agent import ScriptedBackend, answer


class MomentParserTests(unittest.TestCase):
    def test_equivalent_numeric_formats_and_languages(self):
        cases = {
            "Describe what was on the slide at 10 minutes 10 seconds": (610,),
            "Describe what was on the slide at 800 seconds": (800,),
            "What is shown at 10:10?": (610,), "at 00:10:10": (610,),
            "at 10m10s": (610,), "at 1 hour 2 minutes 3 seconds": (3723,),
            "Что на 10 минутах 10 секундах?": (610,), "Что на 800-й секунде?": (800,),
            "На отметке 10 минут 10 секунд": (610,), "timestamp: 0 seconds": (0,),
            "at 10:10.5": (610.5,), "at 800.5 seconds": (800.5,),
            "Compare at 610 seconds and 800 seconds": (610, 800),
            "Compare 10:10 and 13:20": (610, 800), "at 10:10 (610 seconds)": (610,),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(explicit_moments(question), expected)

    def test_durations_and_ranges_are_not_point_lookups(self):
        for question in ("Find a 10 minute lecture", "wait 800 seconds", "a 10 second animation",
                         "from 10:10 to 13:20", "between 10:10 and 13:20", "at 10:10–13:20",
                         "между 10:10 и 13:20", "from 610 seconds to 800 seconds",
                         "What changes after 10:10?", "What happened before 13:20?",
                         "Find the slide showing a duration of 10:10"):
            with self.subTest(question=question):
                self.assertEqual(explicit_moments(question), ())

    def test_invalid_clock_and_negative_points_are_not_silently_searched(self):
        with self.assertRaises(ValueError):
            explicit_moments("at 10:99")
        self.assertEqual(explicit_moments("at -10 seconds"), (-10,))


class MomentAgentTests(unittest.TestCase):
    def make_agent(self, responses, **config):
        index = VideoIndex(2000, [Segment("T1", 609, 611, "Narration.")],
                           video_path="mock.mp4", video_sha256="test-video")
        return VideoAgent(index, ScriptedBackend(responses), AgentConfig(**config))

    def test_user_questions_bypass_search_and_inspect_target_before_finalizing(self):
        for target, question in [(610, "Describe what was on the slide at 10 minutes 10 seconds"),
                                 (800, "Describe what was on the slide at 800 seconds")]:
            agent = self.make_agent([answer()])
            with patch("clipscout.media.extract_frames", return_value=[(target + .009, None)]) as decode, \
                    patch.object(agent.index, "search", side_effect=AssertionError("Point questions must not search")):
                result = agent.run(question)
            self.assertEqual(result["status"], "answered")
            self.assertEqual(result["requested_timestamps"], [target])
            self.assertEqual(result["frames_used"], 1)
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(decode.call_args.kwargs["target_timestamp"], target)
            self.assertEqual(result["evidence"]["E001"]["target_frame_timestamp"], target + .009)
            self.assertEqual(agent.backend.states[0]["requested_timestamps"], [target])

    def test_old_wrong_time_citation_and_nominal_interval_without_target_frame_fail(self):
        agent = self.make_agent([])
        cases = [
            {"start": 1080, "end": 1140, "sampled_timestamps": [1087]},
            {"start": 600, "end": 620, "sampled_timestamps": [602, 607, 612, 617]},
            {"start": 609.5, "end": 610.5, "sampled_timestamps": [], "target_frame_verified": True,
             "requested_timestamp": 610, "target_frame_timestamp": 610.009},
        ]
        for evidence in cases:
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                agent._validate_answer(answer(), {"E001": evidence}, (610,))

    def test_citation_must_cover_both_requested_and_actual_time(self):
        agent = self.make_agent([])
        evidence = {"start": 609.5, "end": 610.5, "sampled_timestamps": [610.009],
                    "target_frame_verified": True, "requested_timestamp": 610, "target_frame_timestamp": 610.009}
        with self.assertRaises(ValueError):
            agent._validate_answer(answer(start=610.005, end=610.5), {"E001": evidence}, (610,))
        self.assertEqual(agent._validate_answer(answer(), {"E001": evidence}, (610,))["status"], "answered")

    def test_impossible_time_and_over_budget_moments_never_call_model(self):
        for question in ("at 2000 seconds", "at -1 second", "at 10:99",
                         "at 1:00 and 2:00 and 3:00 and 4:00 and 5:00"):
            agent = self.make_agent([])
            result = agent.run(question)
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertEqual(result["tool_calls"], 0)
            self.assertEqual(agent.backend.usage["model_calls"], 0)

    def test_multiple_moments_require_citations_to_each(self):
        agent = self.make_agent([])
        ledger = {"E001": {"start": 609.5, "end": 610.5, "sampled_timestamps": [610.009],
                           "target_frame_verified": True, "requested_timestamp": 610, "target_frame_timestamp": 610.009}}
        with self.assertRaises(ValueError):
            agent._validate_answer(answer(), ledger, (610, 800))

    def test_valid_long_question_retains_full_context(self):
        question = "Describe at 10:10. " + "Additional context. " * 60
        agent = self.make_agent([answer()])
        with patch("clipscout.media.extract_frames", return_value=[(610.009, None)]):
            result = agent.run(question)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["trace"][0]["args"]["question"], question)

    def test_baseline_stays_retrieval_only_but_cannot_assert_wrong_moment(self):
        agent = self.make_agent([answer()] * 3)
        result = agent.run("Narration at 10:10", mode="retrieval-only")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["frames_used"], 0)
        self.assertEqual([event["tool"] for event in result["trace"]], ["search"])
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
