"""Uniform failure isolation without video, torch, model weights or network."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clipscout import cli
from clipscout.agent import UNIFORM_POLICY_VERSION


class FakeBackend:
    name = "fake-local-vlm"
    UNIFORM_OBSERVATION_TOKENS = 256
    UNIFORM_ACTION_TOKENS = 384
    UNIFORM_CONTEXT_TOKENS = 6144
    max_context_tokens = 8192

    def __init__(self, fail_warmup=False):
        self.usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.uniform_entries = 0
        self.uniform_active = False
        self.warmup_active = []
        self.fail_warmup = fail_warmup

    @contextmanager
    def uniform_session(self):
        self.uniform_entries += 1
        self.uniform_active = True
        try:
            yield
        finally:
            self.uniform_active = False

    def generate(self, *args, **kwargs):
        self.warmup_active.append(self.uniform_active)
        if self.fail_warmup:
            raise RuntimeError("CUDA out of memory after clean reload retry")
        return "READY"


class FakeIndex:
    video_path = None

    def fingerprint(self):
        return "frozen-video-and-index"

    def search(self, *args, **kwargs):
        return []


class FakeAgent:
    def __init__(self, results=None, **backend_options):
        self.backend = FakeBackend(**backend_options)
        self.index = FakeIndex()
        self.calls = []
        self.results = list(results or [])

    def run(self, question, question_id, mode):
        self.calls.append((question_id, mode))
        result = self.results.pop(0) if self.results else "answered"
        if isinstance(result, Exception):
            raise result
        prediction = {"question_id": question_id, "status": result, "answer": "Entropy.",
                      "citations": [], "tool_calls": 0, "frames_used": 0, "latency_seconds": 0.1}
        if result == "error":
            prediction["error"] = "RuntimeError: CUDA out of memory after clean reload retry"
        return prediction


class UniformCliRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions_path = self.root / "questions.json"
        self.out = self.root / "run"
        self.questions_path.write_text(json.dumps([
            {"id": f"Q{number}", "split": "test", "question": "Explain entropy.",
             "category": "reasoning", "answerable": True,
             "reference_answers": ["Entropy."], "evidence_intervals": [[0, 10]], "required_terms": []}
            for number in range(1, 4)
        ]), encoding="utf-8")

    def run_cli(self, agent, modes=("uniform",), resume=False):
        args = ["evaluate", "--index", "unused.json", "--questions", str(self.questions_path),
                "--output", str(self.out), "--text-only", "--modes", *modes]
        if resume:
            args.append("--resume")
        with patch.object(cli, "make_agent", return_value=agent), patch.object(cli, "experiment_provenance", return_value={"source": "frozen"}):
            return cli.main(args)

    def read(self, name):
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def test_persistent_uniform_oom_saves_one_prediction_and_stops_all_modes(self):
        for result in ("error", RuntimeError("CUDA out of memory after clean reload retry"),
                       RuntimeError("CUDA error: out of memory during cleanup")):
            with self.subTest(result=str(result)):
                self.out = self.root / ("run_error" if isinstance(result, str) else
                                        "run_cleanup" if "during cleanup" in str(result) else "run_exception")
                agent = FakeAgent([result])
                with self.assertRaises(SystemExit) as raised:
                    self.run_cli(agent, ("uniform", "adaptive"))
                self.assertEqual(raised.exception.code, 3)
                self.assertEqual(agent.calls, [("Q1", "uniform")])
                self.assertEqual(len(self.read("uniform.predictions.json")), 1)
                self.assertTrue((self.out / "uniform.metrics.json").is_file())
                self.assertFalse((self.out / "adaptive.predictions.json").exists())
                manifest = self.read("manifest.json")
                self.assertEqual(manifest["status"], "failed_cuda_oom")
                self.assertFalse(manifest["complete"])
                self.assertEqual(manifest["failed_question_id"], "Q1")
                self.assertEqual(manifest["uniform_policy_version"], UNIFORM_POLICY_VERSION)

    def test_adaptive_exception_handling_still_continues(self):
        agent = FakeAgent([RuntimeError("CUDA out of memory"), "answered", "answered"])
        self.run_cli(agent, ("adaptive",))
        self.assertEqual(agent.calls, [(f"Q{n}", "adaptive") for n in range(1, 4)])
        self.assertEqual([p["status"] for p in self.read("adaptive.predictions.json")], ["error", "answered", "answered"])
        self.assertEqual(agent.backend.warmup_active, [False])
        self.assertNotIn("uniform_policy_version", self.read("manifest.json"))

    def test_uniform_only_warmup_uses_uniform_policy(self):
        agent = FakeAgent()
        self.run_cli(agent)
        self.assertEqual(agent.backend.warmup_active, [True])
        self.assertFalse(agent.backend.uniform_active)
        self.assertEqual(len(self.read("uniform.predictions.json")), 3)
        self.assertEqual(self.read("manifest.json")["uniform_inference_limits"], {
            "observation_max_new_tokens": 256, "action_max_new_tokens": 384,
            "max_context_tokens": 6144, "cuda_oom_retries_per_generation": 1})

    def test_uniform_warmup_oom_saves_failure_without_starting_questions(self):
        agent = FakeAgent(fail_warmup=True)
        with self.assertRaises(SystemExit) as raised:
            self.run_cli(agent)
        self.assertEqual(raised.exception.code, 3)
        self.assertEqual(agent.calls, [])
        self.assertEqual(self.read("uniform.predictions.json"), [])
        self.assertTrue((self.out / "uniform.metrics.json").exists())
        self.assertEqual(self.read("manifest.json")["failure_stage"], "warmup")

    def test_uniform_resume_rejects_old_policy_before_warmup(self):
        self.run_cli(FakeAgent())
        manifest = self.read("manifest.json")
        manifest.pop("uniform_policy_version")
        (self.out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        agent = FakeAgent()
        with self.assertRaises(SystemExit) as raised:
            self.run_cli(agent, resume=True)
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(agent.backend.warmup_active, [])
        self.assertEqual(agent.calls, [])


if __name__ == "__main__":
    unittest.main()
