"""Regression checks for explicit interval routing, without model inference."""
import unittest
from dataclasses import FrozenInstanceError

from clipscout.temporal import parse_temporal_scope


class TemporalScopeTests(unittest.TestCase):
    def test_three_original_user_questions(self):
        cases = {
            "What is displayed on the screen from the 20th to the 21st minute": ("range", 1200, 1260),
            "What happens in the video after 15 minutes?": ("after", 900, 2575),
            "What happens in the video from the 15th to the 20th minute?": ("range", 900, 1200),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                result = parse_temporal_scope(question, 2575)
                self.assertEqual((result.kind, result.start, result.end), expected)
                self.assertIn("seconds", result.interpretation)

    def test_shared_units_clocks_and_russian(self):
        cases = {
            "from 20 to 21 minutes": (1200, 1260),
            "Describe from the 20th to the 21st minute.": (1200, 1260),
            "between 20 and 21 minutes": (1200, 1260),
            "from 20 minutes until 21": (1200, 1260),
            "at 20:00–21:00": (1200, 1260),
            "at 20:00 until 21:00": (1200, 1260),
            "20–21 minutes": (1200, 1260),
            "from 00:20:00 to 00:21:00": (1200, 1260),
            "from 1 minute 20 seconds to 2 minutes 30 seconds": (80, 150),
            "from 1m20s through 2m30s": (80, 150),
            "from 1.5 to 2 minutes": (90, 120),
            "с 20-й по 21-ю минуту": (1200, 1260),
            "с 20 до 21 минуты": (1200, 1260),
            "между 20:00 и 21:00": (1200, 1260),
            "от 20 минут до 21 минуты": (1200, 1260),
            "before 15 minutes": (0, 900),
            "until 15:00": (0, 900),
            "до 15-й минуты": (0, 900),
            "после 15 минут": (900, 2575),
            "after 800 seconds": (800, 2575),
            "after 800 seconds.": (800, 2575),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                result = parse_temporal_scope(question, 2575)
                self.assertEqual((result.start, result.end), expected)

    def test_points_and_ordinary_durations_are_not_intervals(self):
        for question in ("Describe at 10 minutes 10 seconds", "What is shown at 800 seconds?",
                         "at 10:10", "Find a 10 minute lecture", "Wait 800 seconds",
                         "Compare at 10:10 and 13:20", "Compare 20 and 21 slides",
                         "from 20 to 21 agents", "between 20 and 21 agents",
                         "What happens after the user waits 15 minutes?", "a 10-second animation"):
            with self.subTest(question=question):
                self.assertIsNone(parse_temporal_scope(question, 2575))

    def test_invalid_constraints_do_not_fall_back_to_search(self):
        for question in ("from 21 to 20 minutes", "from -1 to 2 minutes", "after -15 minutes",
                         "from 1 minute -10 seconds to 2 minutes",
                         "from 20 to 50 minutes", "before 0 seconds", "after 2575 seconds",
                         "from 10:99 to 13:20", "after 10:99", "from 10:10 to nowhere",
                         "from 15 minutes to", "from 20:00 to 21", "from 1:00 to 2:00 and from 3:00 to 4:00",
                         "after 10 minutes or before 5 minutes", "not after 15 minutes"):
            with self.subTest(question=question), self.assertRaises(ValueError):
                parse_temporal_scope(question, 2575)

    def test_scope_is_immutable_and_ordinal_convention_is_explicit(self):
        result = parse_temporal_scope("from the 20th to the 21st minute", 2575)
        self.assertIn("ordinal", result.interpretation)
        with self.assertRaises(FrozenInstanceError):
            result.start = 0

    def test_malformed_and_mixed_time_requests_are_not_silently_ignored(self):
        for question in ("Compare at 10:10 with what happens from 15 to 20 minutes",
                         "before 15 minutes and at 10:10", "after 00:15:00:00", "after 10:",
                         "from the 20th to the xx minute", "not from 15 to 20 minutes",
                         "не с 15-й по 20-ю минуту"):
            with self.subTest(question=question), self.assertRaises(ValueError):
                parse_temporal_scope(question, 2575)


if __name__ == "__main__":
    unittest.main()
