"""Tests for subtitle provenance, temporal boundaries and retrieval behavior."""

from pathlib import Path
import tempfile
import unittest

from clipscout.index import Segment, VideoIndex, finite_number, read_subtitles, validate_interval


class IndexTests(unittest.TestCase):
    def make_index(self, **kwargs):
        return VideoIndex(120, [Segment("before", 0, 10, "Before interval"),
                               Segment("inside", 10, 20, "Entropy and uncertainty"),
                               Segment("after", 20, 30, "After interval"),
                               Segment("caption", 12, 18, "Entropy formula on a blackboard", "caption")], **kwargs)

    def test_overlap_requires_both_boundaries_and_excludes_touching_cues(self):
        index = self.make_index()
        self.assertEqual([s.id for s in index.overlapping(10, 20, "transcript")], ["inside"])
        self.assertEqual({s.id for s in index.overlapping(10, 20)}, {"inside", "caption"})
        self.assertEqual(index.overlapping(90, 100), [])

    def test_search_filters_modalities_and_returns_no_fabricated_hits(self):
        index = self.make_index(chunk_seconds=10)
        transcript = index.search("entropy uncertainty", source="transcript")
        caption = index.search("blackboard", source="caption")
        self.assertEqual(transcript[0]["segment_ids"], ["inside"])
        self.assertEqual(caption[0]["source"], "caption")
        self.assertGreater(transcript[0]["score"], 0)
        self.assertEqual(index.search("unfindable zxqw"), [])
        self.assertEqual(index.search("the and is"), [])
        self.assertEqual(index.search("entropy", source="transcript"), index.search("ENTROPY", source="transcript"))

    def test_search_preserves_non_latin_words(self):
        index = VideoIndex(20, [Segment("ru", 1, 4, "Энтропия измеряет неопределённость")])
        self.assertEqual(index.search("ЭНТРОПИЯ")[0]["segment_ids"], ["ru"])

    def test_search_rejects_type_and_range_errors(self):
        index = self.make_index()
        for args in ({"query": ""}, {"query": None}, {"query": "a" * 501},
                     {"query": "entropy", "top_k": True}, {"query": "entropy", "top_k": 0},
                     {"query": "entropy", "top_k": 11}, {"query": "entropy", "source": "invented"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                index.search(**args)

    def test_numeric_guards_reject_nan_inf_booleans_and_numeric_strings(self):
        for value in (True, False, float("nan"), float("inf"), -float("inf"), "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                finite_number(value, "value")
        for start, end in ((0, 0), (-1, 2), (10, 2), (0, 121), (True, 2), (0, float("nan"))):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_interval(start, end, 120)
        with self.assertRaises(ValueError):
            validate_interval(0, 61, 120, max_span=60)

    def test_chunk_configuration_must_be_finite_and_not_boolean(self):
        for value in (True, False, 0, -1, float("nan"), float("inf"), "30"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_index(chunk_seconds=value)

    def test_invalid_segments_are_rejected(self):
        bad_groups = [[Segment("same", 1, 3, "a"), Segment("same", 4, 5, "b")],
                      [Segment("", 1, 3, "a")], [Segment("outside", 119, 125, "a")],
                      [Segment("unknown", 1, 3, "a", "prediction")],
                      [Segment("not-text", 1, 3, None)]]
        for segments in bad_groups:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                VideoIndex(120, segments)

    def test_fingerprint_changes_with_content_and_retrieval_configuration(self):
        first = self.make_index(chunk_seconds=10)
        self.assertEqual(first.fingerprint(), self.make_index(chunk_seconds=10).fingerprint())
        self.assertNotEqual(first.fingerprint(), self.make_index(chunk_seconds=60).fingerprint())
        edited = VideoIndex(120, [Segment("new", 1, 4, "Changed transcript")])
        self.assertNotEqual(first.fingerprint(), edited.fingerprint())

    def test_subtitles_and_saved_index_preserve_provenance(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            directory = Path(directory)
            subtitle = directory / "captions.vtt"
            subtitle.write_text("WEBVTT\n\nNOTE ignored metadata\nignore this\n\n"
                                "first\n00:01.000 --> 00:03.250 align:start\n<b>Entropy</b> &amp; uncertainty\n\n"
                                "00:05.500 --> 00:07.000\nSecond line\ncontinued\n", encoding="utf-8")
            cues = read_subtitles(subtitle)
            self.assertEqual(len(cues), 2)
            self.assertEqual((cues[0].start, cues[0].end), (1, 3.25))
            self.assertEqual(cues[0].text, "Entropy & uncertainty")
            self.assertEqual(cues[1].text, "Second line continued")
            video = directory / "media.mp4"
            video.touch()
            index = VideoIndex(10, cues, video_path=str(video), video_sha256="abc", title="test")
            output = directory / "nested" / "index.json"
            index.save(output)
            restored = VideoIndex.load(output)
            self.assertEqual(restored.fingerprint(), index.fingerprint())
            self.assertEqual(Path(restored.video_path), video.resolve())
            self.assertEqual(restored.title, "test")


if __name__ == "__main__":
    unittest.main()
