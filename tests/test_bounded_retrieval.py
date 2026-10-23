"""Temporal retrieval must preserve whole-cue provenance inside the request."""

from copy import deepcopy
import unittest

from clipscout.index import Segment, VideoIndex


class BoundedRetrievalTests(unittest.TestCase):
    def make_index(self):
        return VideoIndex(180, [
            Segment("outside", 0, 5, "goal goal goal loop graph"),
            Segment("cross_start", 8, 12, "boundarysecret goal"),
            Segment("inside", 12, 18, "goal loop"),
            Segment("cross_end", 18, 24, "endsecret graph"),
            Segment("later", 60, 70, "goal graph"),
            Segment("caption", 12, 18, "goal boxes", "caption"),
        ], chunk_seconds=30)

    def test_outside_high_scoring_results_cannot_be_returned(self):
        results = self.make_index().search("goal graph", start=10, end=20)
        self.assertTrue(results)
        self.assertEqual({sid for result in results for sid in result["segment_ids"]},
                         {"inside", "caption"})
        self.assertTrue(all(10 <= result["start"] < result["end"] <= 20 for result in results))

    def test_chunks_are_rebuilt_without_outside_or_crossing_cue_text(self):
        index = self.make_index()
        result = index.search("goal", source="transcript", start=10, end=20)[0]
        self.assertEqual(result["text"], "goal loop")
        self.assertEqual(result["segment_ids"], ["inside"])
        self.assertEqual((result["start"], result["end"]), (12, 18))
        self.assertEqual(index.search("boundarysecret", start=10, end=20), [])
        self.assertEqual(index.search("endsecret", start=10, end=20), [])

    def test_exact_cue_bounds_are_inclusive(self):
        index = self.make_index()
        self.assertEqual([cue.id for cue in index.contained(12, 18, "transcript")], ["inside"])
        self.assertEqual(index.search("boxes", source="caption", start=12, end=18)[0]["segment_ids"],
                         ["caption"])

    def test_empty_window_or_no_matching_terms_returns_no_hits(self):
        index = self.make_index()
        self.assertEqual(index.search("goal", start=100, end=110), [])
        self.assertEqual(index.search("graph", start=12, end=18), [])
        self.assertEqual(index.contained(100, 110), [])

    def test_search_requires_both_finite_valid_bounds(self):
        index = self.make_index()
        invalid = [(None, 20), (10, None), (-1, 20), (20, 20), (21, 20),
                   (0, 181), (True, 20), (0, False), ("10", 20),
                   (float("nan"), 20), (0, float("inf"))]
        for start, end in invalid:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                index.search("goal", start=start, end=end)

    def test_contained_rejects_invalid_source_and_interval(self):
        index = self.make_index()
        with self.assertRaises(ValueError):
            index.contained(0, 20, "invented")
        with self.assertRaises(ValueError):
            index.contained(-1, 20)

    def test_bounded_search_does_not_mutate_original_index(self):
        index = self.make_index()
        original_chunks = deepcopy(index.chunks)
        original_segments = list(index.segments)
        original_fingerprint = index.fingerprint()
        original_hits = index.search("goal graph")
        index.search("goal", start=10, end=20)
        index.search("goal", start=60, end=70)
        self.assertEqual(index.chunks, original_chunks)
        self.assertEqual(index.segments, original_segments)
        self.assertEqual(index.fingerprint(), original_fingerprint)
        self.assertEqual(index.search("goal graph"), original_hits)
        self.assertEqual(index.search("goal graph", start=None, end=None), original_hits)


if __name__ == "__main__":
    unittest.main()
