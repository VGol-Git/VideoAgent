"""Media correctness checks; codec-dependent checks skip if OpenCV is absent."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clipscout.media import MediaError, extract_frames, file_fingerprint, probe_video


class MediaBoundaryTests(unittest.TestCase):
    def test_invalid_requests_fail_before_opening_video(self):
        for start, end in [(0, 0), (2, 1), (-1, 1), (True, 2), (0, float("inf")), (float("nan"), 2)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                extract_frames("missing.mp4", start, end)
        for count in (0, -1, 65, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                extract_frames("missing.mp4", 0, 1, count=count)

    def test_fingerprint_uses_content_not_filename(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as directory:
            path = Path(directory) / "same-name.bin"
            path.write_bytes(b"first")
            first = file_fingerprint(path, chunk_size=2)
            path.write_bytes(b"second")
            second = file_fingerprint(path, chunk_size=2)
            self.assertNotEqual(first, second)
            self.assertEqual(len(first), 64)

    def test_invalid_target_requests_fail_before_opening_video(self):
        for target in (True, False, float("inf"), float("nan"), "not a time", -0.1, 2):
            with self.subTest(target=target), self.assertRaises(ValueError):
                extract_frames("missing.mp4", 0, 2, target_timestamp=target)
        with self.assertRaises(ValueError):
            extract_frames("missing.mp4", 1, 2, target_timestamp=0.5)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is optional")
class RealVideoTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        self.cv2 = cv2
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "numbered.avi"
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
        if not writer.isOpened():
            self.skipTest("MJPG writer is unavailable in this OpenCV build")
        try:
            for index in range(20):
                image = np.full((64, 96, 3), index * 10, dtype=np.uint8)
                writer.write(image)
        finally:
            writer.release()

    def test_sampling_is_distinct_bounded_and_inside_interval(self):
        metadata = probe_video(self.path)
        self.assertAlmostEqual(metadata["duration"], 2.0, places=3)
        self.assertEqual(metadata["width"], 96)
        frames = extract_frames(self.path, 0.2, 0.8, count=64, max_side=48)
        self.assertEqual(len(frames), 6)
        timestamps = [timestamp for timestamp, _ in frames]
        self.assertEqual(len(timestamps), len(set(timestamps)))
        self.assertTrue(all(0.2 - 1e-8 <= t < 0.8 for t in timestamps))
        for timestamp, image in frames:
            self.assertLessEqual(max(image.size), 48)
            # Each source frame's pixel value encodes its true time. Check that
            # returned evidence corresponds to the label, allowing JPEG noise.
            self.assertAlmostEqual(image.getpixel((0, 0))[0], timestamp * 100, delta=4)

    def test_out_of_duration_and_subframe_intervals_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_frames(self.path, 0, 2.1)
        with self.assertRaises(MediaError):
            extract_frames(self.path, 0.01, 0.02)

    def test_target_sampling_decodes_the_requested_frame_not_uniform_midpoints(self):
        # Uniform samples in [0,2) with count=4 are .2, .7, 1.2, 1.7.
        # A target at .44 must instead return the nearest existing frame, .4.
        frames = extract_frames(self.path, 0, 2, count=4, max_side=48, target_timestamp=0.44)
        self.assertEqual(len(frames), 1)
        timestamp, image = frames[0]
        self.assertAlmostEqual(timestamp, 0.4)
        self.assertAlmostEqual(image.getpixel((0, 0))[0], 40, delta=4)
        self.assertLessEqual(max(image.size), 48)

    def test_target_sampling_at_start_and_near_end_uses_existing_frames(self):
        for target, expected in ((0, 0), (1.99, 1.9)):
            with self.subTest(target=target):
                frames = extract_frames(self.path, 0, 2, count=64, target_timestamp=target)
                self.assertEqual(len(frames), 1)
                timestamp, image = frames[0]
                self.assertAlmostEqual(timestamp, expected)
                self.assertAlmostEqual(image.getpixel((0, 0))[0], expected * 100, delta=4)

    def test_target_outside_video_and_nearest_frame_outside_interval_are_rejected(self):
        with self.assertRaises(ValueError):
            extract_frames(self.path, 0, 2.1, target_timestamp=2.05)
        # The nearest frame to .21 is .2; .3 is in the interval but must not be
        # silently substituted as if it were the nearest frame to the target.
        with self.assertRaises(MediaError):
            extract_frames(self.path, 0.205, 0.4, target_timestamp=0.21)

    def test_target_requires_decoder_timestamp_when_frame_index_is_nonzero(self):
        capture = self.cv2.VideoCapture(str(self.path))

        class MissingTimestamp:
            def get(_, property_id):
                return 0 if property_id == self.cv2.CAP_PROP_POS_MSEC else capture.get(property_id)

            def set(_, *args):
                return capture.set(*args)

            def read(_):
                return capture.read()

            def release(_):
                capture.release()

        with patch("clipscout.media._open_video", return_value=(self.cv2, MissingTimestamp())):
            with self.assertRaises(MediaError) as error:
                extract_frames(self.path, 0, 2, target_timestamp=0.5)
        self.assertEqual(error.exception.frames_decoded, 1)
        self.assertEqual(error.exception.sampled_timestamps, [])

    def test_target_rejects_actual_pts_error_greater_than_one_high_fps_frame(self):
        import numpy as np

        path = Path(self.directory.name) / "thirty-fps.avi"
        writer = self.cv2.VideoWriter(str(path), self.cv2.VideoWriter_fourcc(*"MJPG"), 30, (96, 64))
        if not writer.isOpened():
            self.skipTest("MJPG writer is unavailable in this OpenCV build")
        try:
            for index in range(60):
                writer.write(np.full((64, 96, 3), index * 3, dtype=np.uint8))
        finally:
            writer.release()
        capture = self.cv2.VideoCapture(str(path))

        class InaccurateTimestamp:
            def get(_, property_id):
                value = capture.get(property_id)
                # 60ms is less than the old generic 100ms tolerance, but more
                # than one 30fps frame: it cannot support this target moment.
                return value + 60 if property_id == self.cv2.CAP_PROP_POS_MSEC else value

            def set(_, *args):
                return capture.set(*args)

            def read(_):
                return capture.read()

            def release(_):
                capture.release()

        with patch("clipscout.media._open_video", return_value=(self.cv2, InaccurateTimestamp())):
            with self.assertRaises(MediaError) as error:
                extract_frames(path, 0, 2, target_timestamp=0.5)
        self.assertIn("does not match", str(error.exception))
        self.assertEqual(error.exception.frames_decoded, 1)

    def test_sparse_midpoints_do_not_seek_an_unreadable_last_frame(self):
        capture = self.cv2.VideoCapture(str(self.path))
        seek_indices = []

        class FragileLastFrame:
            def get(_, property_id):
                return capture.get(property_id)

            def set(_, property_id, value):
                if property_id == self.cv2.CAP_PROP_POS_FRAMES:
                    seek_indices.append(value)
                return capture.set(property_id, value)

            def read(_):
                if seek_indices[-1] == 19:
                    return False, None
                return capture.read()

            def release(_):
                capture.release()

        with patch("clipscout.media._open_video", return_value=(self.cv2, FragileLastFrame())):
            frames = extract_frames(self.path, 0, 2, count=2)
        self.assertEqual(seek_indices, [5, 15])
        self.assertEqual([timestamp for timestamp, _ in frames], [0.5, 1.5])
        self.assertTrue(all(0 < timestamp < 2 for timestamp, _ in frames))
        for timestamp, image in frames:
            self.assertAlmostEqual(image.getpixel((0, 0))[0], timestamp * 100, delta=4)

    def test_start_boundary_is_strict_when_just_after_a_frame(self):
        frames = extract_frames(self.path, 0.20000000001, 0.4, count=64)
        self.assertEqual(len(frames), 1)
        self.assertTrue(all(0.20000000001 <= timestamp < 0.4 for timestamp, _ in frames))

    def test_inaccurate_seek_is_not_reported_as_evidence(self):
        capture = self.cv2.VideoCapture(str(self.path))
        real_get = capture.get

        class IncorrectPosition:
            def get(_, property_id):
                return 999 if property_id == self.cv2.CAP_PROP_POS_FRAMES else real_get(property_id)

            def set(_, *args):
                return capture.set(*args)

            def read(_):
                return capture.read()

            def release(_):
                capture.release()

        with patch("clipscout.media._open_video", return_value=(self.cv2, IncorrectPosition())):
            with self.assertRaises(MediaError) as error:
                extract_frames(self.path, 0, 1, count=1)
        self.assertEqual(error.exception.frames_decoded, 1)
        self.assertEqual(error.exception.sampled_timestamps, [])

    def test_decoded_frame_is_charged_even_when_image_conversion_fails(self):
        with patch("PIL.Image.fromarray", side_effect=RuntimeError("image conversion failed")):
            with self.assertRaises(RuntimeError) as error:
                extract_frames(self.path, 0, 1, count=3)
        self.assertEqual(error.exception.frames_decoded, 1)
        self.assertEqual(error.exception.sampled_timestamps, [0.1])


if __name__ == "__main__":
    unittest.main()
