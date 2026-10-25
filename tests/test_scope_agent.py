"""Temporal routing regressions using scripted answers and mocked video decoding."""

import json
import unittest
from unittest.mock import patch

from clipscout.agent import AgentConfig, VideoAgent
from clipscout.index import Segment, VideoIndex
from clipscout.temporal import parse_temporal_scope
from test_agent import ScriptedBackend, answer


DURATION = 2575.0057667
USER_CASES = (
    ("What is displayed on the screen from the 20th to the 21st minute", 1200, 1260),
    ("What happens in the video after 15 minutes?", 900, DURATION),
    ("What happens in the video from the 15th to the 20th minute?", 900, 1200),
)


def overview(count):
    response = answer()
    response["citations"] = [{"evidence_id": f"E{i + 1:03}"} for i in range(count)]
    return response


def no_evidence():
    return {"action": "answer", "status": "insufficient_evidence",
            "answer": "No inspected evidence is available inside this interval.", "citations": []}


def decode_midpoints(path, start, end, count, max_side, **kwargs):
    """Honor the real decoder's count and half-open interval contract."""
    return [(start + (i + .5) * (end - start) / count, None) for i in range(count)]


def scoped_index(start=900, end=1200):
    span = end - start
    cues = [Segment(f"inside{i}", start + span * fraction,
                    start + span * fraction + 1,
                    "The video screen displayed shows what happens: entropy.", source)
            for i, (fraction, source) in enumerate(((.05, "transcript"),
                                                    (.4, "caption"),
                                                    (.8, "transcript")))]
    if start > 1:
        cues.extend([
            Segment("outside", 0, 1, "OUTSIDE_SECRET video screen displayed happens " * 20),
            Segment("cross_start", start - 1, start + 1,
                    "CROSS_START_SECRET video screen displayed happens"),
        ])
    if end < DURATION - 1:
        cues.extend([
            Segment("after", end + .1, end + .9, "AFTER_SECRET video screen displayed happens " * 20),
            Segment("cross_end", end - 1, end + 1,
                    "CROSS_END_SECRET video screen displayed happens"),
        ])
    return VideoIndex(DURATION, cues, video_path="mock.mp4", video_sha256="test-video")


class ScopeAgentTests(unittest.TestCase):
    def make_agent(self, responses, start=900, end=1200, **config):
        return VideoAgent(scoped_index(start, end), ScriptedBackend(responses), AgentConfig(**config))

    def assert_bounds(self, result, start, end):
        self.assertEqual((result["temporal_scope"]["start"], result["temporal_scope"]["end"]),
                         (start, end))
        for event in result["trace"]:
            if event["tool"] == "inspect_clip":
                self.assertLessEqual(start, event["args"]["start"])
                self.assertLessEqual(event["args"]["end"], end)
                self.assertLessEqual(event["args"]["end"] - event["args"]["start"], 120)
            for hit in event.get("result", {}).get("hits", []):
                self.assertLessEqual(start, hit["start"])
                self.assertLessEqual(hit["end"], end)
        for evidence in result["evidence"].values():
            self.assertLessEqual(start, evidence["start"])
            self.assertLessEqual(evidence["end"], end)
            for timestamp in evidence.get("sampled_timestamps", []):
                self.assertLessEqual(start, timestamp)
                self.assertLess(timestamp, end)
            for cue in evidence.get("transcript_segments", []):
                self.assertLessEqual(start, cue["start"])
                self.assertLessEqual(cue["end"], end)
        for citation in result["citations"]:
            self.assertLessEqual(start, citation["start"])
            self.assertLessEqual(citation["end"], end)
        for timestamp in result["unique_sampled_timestamps"]:
            self.assertLessEqual(start, timestamp)
            self.assertLess(timestamp, end)

    def test_three_user_questions_route_directly_and_distribute_default_budget(self):
        for question, start, end in USER_CASES:
            with self.subTest(question=question):
                agent = self.make_agent([overview(4)], start, end)
                with patch("clipscout.media.extract_frames", side_effect=decode_midpoints) as decode, \
                        patch.object(agent.index, "search", side_effect=AssertionError("No unrestricted search")):
                    result = agent.run(question)
                self.assertEqual(result["status"], "answered", result)
                self.assertEqual(result["routing"], "temporal_scope")
                self.assertEqual(result["tool_calls"], 4)
                self.assertEqual(result["frames_used"], 16)
                self.assertEqual(decode.call_count, 4)
                self.assertTrue(all(event["ok"] for event in result["trace"]))
                intervals = result["coverage"]["inspection_intervals"]
                self.assertEqual(intervals[0][0], start)
                self.assertAlmostEqual(intervals[-1][1], end, delta=.001)
                self.assertEqual(len({tuple(pair) for pair in intervals}), 4)
                self.assertEqual(len(result["citations"]), 4)
                self.assertFalse(result["coverage"]["complete_visual_coverage"])
                self.assertEqual(result["coverage"]["sampled_frame_count"], 16)
                self.assertIn("partial coverage", result["answer"])
                self.assert_bounds(result, start, end)

    def test_after_means_remaining_video_with_explicit_uninspected_gaps(self):
        question, start, end = USER_CASES[1]
        agent = self.make_agent([overview(4)], start, end)
        with patch("clipscout.media.extract_frames", side_effect=decode_midpoints):
            result = agent.run(question)
        coverage = result["coverage"]
        self.assertEqual(result["temporal_scope"]["kind"], "after")
        self.assertIn("end of the video", result["temporal_scope"]["interpretation"])
        self.assertGreater(len(coverage["omitted_intervals"]), 0)
        self.assertIn("not inspected", coverage["note"])
        self.assertIn("not continuous", coverage["note"])
        windows = coverage["inspection_intervals"]
        for gap_start, gap_end in coverage["omitted_intervals"]:
            self.assertTrue(start <= gap_start < gap_end <= end)
            self.assertFalse(any(window_start < gap_end and window_end > gap_start
                                 for window_start, window_end in windows))

    def test_contiguous_short_windows_still_do_not_claim_full_visual_coverage(self):
        question, start, end = USER_CASES[0]
        agent = self.make_agent([overview(4)], start, end)
        with patch("clipscout.media.extract_frames", side_effect=decode_midpoints):
            result = agent.run(question)
        self.assertEqual(result["coverage"]["omitted_intervals"], [])
        self.assertFalse(result["coverage"]["complete_visual_coverage"])
        self.assertIn("not continuous", result["coverage"]["note"])

    def test_before_request_stays_between_video_start_and_boundary(self):
        agent = self.make_agent([overview(4)], 0, 900)
        with patch("clipscout.media.extract_frames", side_effect=decode_midpoints):
            result = agent.run("What happens before 15 minutes?")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["temporal_scope"]["kind"], "before")
        self.assert_bounds(result, 0, 900)

    def test_outside_inspection_is_rejected_before_decoder_or_model(self):
        agent = self.make_agent([])
        scope = parse_temporal_scope(USER_CASES[2][0], DURATION)
        for start, end in ((360, 420), (2520, 2550), (899, 901), (1199, 1201)):
            with self.subTest(start=start), patch("clipscout.media.extract_frames") as decode:
                with self.assertRaisesRegex(ValueError, "outside.*scope"):
                    agent._inspect({"start": start, "end": end}, "Describe screen", 4, scope)
                decode.assert_not_called()
        self.assertEqual(agent.backend.usage["model_calls"], 0)

    def test_old_six_minute_and_forty_two_minute_citations_are_rejected(self):
        agent = self.make_agent([])
        scope = parse_temporal_scope(USER_CASES[2][0], DURATION)
        for start, end in ((360, 420), (2520, 2550)):
            ledger = {"E001": {"start": start, "end": end, "sampled_timestamps": [start + 1]}}
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, "outside.*scope"):
                agent._validate_answer(answer(), ledger, temporal_scope=scope)

    def test_ledger_range_cannot_hide_a_sample_outside_requested_bounds(self):
        agent = self.make_agent([])
        scope = parse_temporal_scope(USER_CASES[2][0], DURATION)
        for timestamp in (899.99, 1200, 2520):
            ledger = {"E001": {"start": 900, "end": 930, "sampled_timestamps": [timestamp]}}
            with self.subTest(timestamp=timestamp), self.assertRaisesRegex(ValueError, "frame outside"):
                agent._validate_answer(answer(), ledger, temporal_scope=scope)

    def test_temporal_overview_retries_if_model_omits_inspected_windows(self):
        agent = self.make_agent([overview(1), overview(4)])
        with patch("clipscout.media.extract_frames", side_effect=decode_midpoints):
            result = agent.run(USER_CASES[2][0])
        self.assertEqual(result["status"], "answered")
        self.assertEqual(len(result["decisions"]), 2)
        self.assertIn("must cite each inspected window", result["decisions"][0]["error"])
        self.assertEqual({cite["evidence_id"] for cite in result["citations"]}, set(result["evidence"]))
        self.assertEqual(result["tool_calls"], 4)

    def test_baselines_never_retrieve_inspect_or_cite_outside_scope(self):
        for question, start, end in USER_CASES:
            for mode in ("retrieval-only", "static", "uniform"):
                with self.subTest(question=question, mode=mode):
                    # Uniform now distributes its deterministic windows inside
                    # the requested interval, including short intervals.
                    count = 4 if mode == "uniform" else 3
                    agent = self.make_agent([overview(count) if count else no_evidence()], start, end)
                    with patch("clipscout.media.extract_frames", side_effect=decode_midpoints):
                        result = agent.run(question, mode=mode)
                    self.assertEqual(result["status"], "answered" if count else "insufficient_evidence", result)
                    self.assertLessEqual(result["tool_calls"], 4)
                    self.assertLessEqual(result["frames_used"], 16)
                    self.assertTrue(all(event["ok"] for event in result["trace"]))
                    self.assert_bounds(result, start, end)
                    state_text = json.dumps(agent.backend.states)
                    self.assertNotIn("_SECRET", state_text)
                    self.assertNotIn("_SECRET", json.dumps(result["trace"]))
                    if mode == "retrieval-only":
                        self.assertEqual(result["frames_used"], 0)
                        self.assertEqual(agent.backend.visual_calls, 0)
                    self.assertFalse(result["coverage"]["complete_visual_coverage"])

    def test_adaptive_low_frame_budgets_are_distributed_without_partial_exhaustion(self):
        for budget in (0, 1, 2, 3, 5):
            with self.subTest(budget=budget):
                calls = min(4, budget)
                agent = self.make_agent([overview(calls)] if calls else [], max_frames=budget)
                with patch("clipscout.media.extract_frames", side_effect=decode_midpoints) as decode:
                    result = agent.run(USER_CASES[2][0])
                self.assertEqual(result["status"], "answered" if calls else "insufficient_evidence", result)
                self.assertEqual(result["frames_used"], budget)
                self.assertEqual(result["tool_calls"], calls)
                self.assertEqual(decode.call_count, calls)
                self.assertTrue(all(event["ok"] for event in result["trace"]))
                self.assertTrue(all(call.kwargs["count"] >= 1 for call in decode.call_args_list))
                self.assert_bounds(result, 900, 1200)

    def test_baseline_low_frame_budgets_do_not_attempt_impossible_extra_inspections(self):
        for mode in ("static", "uniform"):
            with self.subTest(mode=mode):
                agent = self.make_agent([no_evidence()], 900, DURATION, max_frames=2)
                with patch("clipscout.media.extract_frames", side_effect=decode_midpoints):
                    result = agent.run(USER_CASES[1][0], mode=mode)
                self.assertLessEqual(result["frames_used"], 2)
                self.assertTrue(all(event["ok"] for event in result["trace"]), result["trace"])
                self.assert_bounds(result, 900, DURATION)


if __name__ == "__main__":
    unittest.main()
