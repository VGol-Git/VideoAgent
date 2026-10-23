"""Loopback demo contracts: video ranges, origin checks and serialized inference."""
from http.client import HTTPConnection
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import threading
import unittest

from clipscout.server import make_server


class FakeAgent:
    def __init__(self, video):
        self.index = SimpleNamespace(video_path=str(video), title="Test video", duration=1800)
        self.backend = SimpleNamespace(name="test-fixture")
        self.config = SimpleNamespace(use_vision=True, max_tool_calls=4, max_frames=16,
                                      frames_per_clip=4, max_clip_seconds=120)
        self.calls = []

    def run(self, question, mode):
        self.calls.append((question, mode))
        return {"answer": "<script>alert('not executable')</script>", "status": "answered",
                "question": question, "mode": mode, "trace": [], "evidence": {},
                "citations": [], "tool_calls": 0, "frames_used": 0}


class DemoServerTests(unittest.TestCase):
    def setUp(self):
        scratch_root = Path(__file__).resolve().parents[1] / ".cache" / "server-tests"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.temp = TemporaryDirectory(dir=scratch_root)
        self.root = Path(self.temp.name)
        video = self.root / "test.mp4"
        video.write_bytes(bytes(range(100)))
        questions = self.root / "questions.json"
        questions.write_text(json.dumps({"questions": [{"id": "q1", "question": "Where?",
                                                        "category": "localization", "answer": "GOLD_SECRET",
                                                        "evidence_intervals": [[12, 19]]}]}), encoding="utf-8")
        self.agent = FakeAgent(video)
        self.server = make_server(self.agent, port=0, run_dir=self.root / "runs", questions_path=questions)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        status, result_headers, data = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return status, result_headers, data

    def post(self, data, headers=None):
        return self.request("POST", "/api/ask", json.dumps(data), {"Content-Type": "application/json", **(headers or {})})

    def test_info_and_examples_exclude_paths_and_gold(self):
        status, _, data = self.request("GET", "/api/info")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(data)["video_available"])
        self.assertNotIn(str(self.root).encode(), data)
        status, _, data = self.request("GET", "/api/examples")
        self.assertEqual(status, 200)
        self.assertNotIn(b"GOLD_SECRET", data)
        self.assertNotIn(b"evidence_intervals", data)
        self.assertEqual(json.loads(data)["questions"][0]["question"], "Where?")

    def test_video_byte_ranges_and_head(self):
        status, headers, body = self.request("GET", "/video", headers={"Range": "bytes=10-19"})
        self.assertEqual((status, body), (206, bytes(range(10, 20))))
        self.assertEqual(headers["Content-Range"], "bytes 10-19/100")
        status, _, body = self.request("GET", "/video", headers={"Range": "bytes=-5"})
        self.assertEqual((status, body), (206, bytes(range(95, 100))))
        status, headers, body = self.request("HEAD", "/video")
        self.assertEqual((status, headers["Content-Length"], body), (200, "100", b""))
        for value in ("bytes=100-110", "bytes=3-1", "bytes=-0", "bytes=1-2,3-4"):
            with self.subTest(value=value):
                self.assertEqual(self.request("GET", "/video", headers={"Range": value})[0], 416)

    def test_only_exact_routes_and_no_path_reading(self):
        for path in ("/../../questions.json", "/video?path=questions.json", "/api/info?secret=1", "/static/", "/runs/"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_foreign_origin_and_rebinding_host_rejected(self):
        data = {"question": "Where?", "mode": "adaptive"}
        self.assertEqual(self.post(data, {"Origin": "https://untrusted.example"})[0], 403)
        self.assertEqual(self.post(data, {"Origin": "null"})[0], 403)
        self.assertEqual(self.post(data, {"Host": "untrusted.example"})[0], 403)
        self.assertEqual(self.post(data, {"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.agent.calls, [])

    def test_valid_ask_runs_and_saves_exact_response(self):
        status, _, data = self.post({"question": "  Where?  ", "mode": "static"}, {"Origin": self.server.origin})
        self.assertEqual(status, 200)
        result = json.loads(data)
        self.assertEqual(self.agent.calls, [("Where?", "static")])
        saved = list((self.root / "runs").glob("*.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(json.loads(saved[0].read_text(encoding="utf-8")), result)
        self.assertNotIn("run_path", result)

    def test_validation_and_busy_do_not_run_inference(self):
        invalid = [{"question": ""}, {"question": "x" * 2001}, {"question": "Q", "mode": []},
                   {"question": "Q", "mode": "bad"}, {"question": "Q", "video_path": "other.mp4"}]
        for payload in invalid:
            self.assertEqual(self.post(payload)[0], 400)
        self.assertEqual(self.request("POST", "/api/ask", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/ask", "x" * 20000, {"Content-Type": "application/json"})[0], 413)
        with self.server.inference_lock:
            self.assertEqual(self.post({"question": "Q"})[0], 409)
        self.assertEqual(self.agent.calls, [])

    def test_ui_is_self_contained_and_uses_text_content(self):
        status, headers, data = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"textContent", data)
        self.assertNotIn(b"innerHTML", data)
        self.assertNotIn(b"<script src=", data)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_non_loopback_bind_is_rejected(self):
        with self.assertRaises(ValueError):
            make_server(self.agent, host="0.0.0.0", port=0, run_dir=self.root / "other")


if __name__ == "__main__":
    unittest.main()
