"""Uniform-only decode budgets and OOM recovery, without ML dependencies."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

from clipscout.backends import BackendError, BackendOOMError, QwenBackend, is_cuda_oom


class FakeIds:
    def __init__(self, size):
        self.shape = (1, size)

    def __getitem__(self, key):
        return FakeIds(self.shape[-1] - key[1].start)


class FakeBatch(dict):
    def __init__(self, size):
        super().__init__(input_ids=FakeIds(size), token_type_ids=object())
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


class FakeProcessor:
    tokenizer = object()

    def __init__(self, size=100):
        self.size = size
        self.batches = []

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, **kwargs):
        batch = FakeBatch(self.size)
        self.batches.append(batch)
        return batch

    def batch_decode(self, output_ids, **kwargs):
        return ['{"action":"answer"}']


class TensorLifetime:
    pass


class UniformBackendTests(unittest.TestCase):
    def backend(self, *, size=100, outcomes=None, **kwargs):
        backend = QwenBackend(device="cuda", max_new_tokens=kwargs.pop("max_new_tokens", 768),
                              local_files_only=True, **kwargs)
        backend._torch = SimpleNamespace(inference_mode=nullcontext,
                                         cuda=SimpleNamespace(empty_cache=Mock()))
        processor = FakeProcessor(size)
        calls, loads = [], []
        outcomes = iter(outcomes or [None])

        def generate(**options):
            calls.append(options)
            outcome = next(outcomes, None)
            if callable(outcome):
                outcome()
            elif outcome is not None:
                raise outcome
            return FakeIds(size + 7)

        def load():
            if backend._model is None:
                loads.append("cuda")
                backend._processor = processor
                backend._model = SimpleNamespace(generate=generate)

        backend._load = load
        return backend, processor, calls, loads

    def test_adaptive_kwargs_usage_and_context_remain_unchanged(self):
        backend, _, calls, loads = self.backend(size=6500)
        with patch("clipscout.backends._prepare_images", return_value=[]):
            result = backend.generate("system", "user")
        self.assertEqual(result, '{"action":"answer"}')
        self.assertEqual(loads, ["cuda"])
        self.assertEqual(calls[0]["max_new_tokens"], 768)
        self.assertEqual({k: calls[0][k] for k in (
            "do_sample", "temperature", "top_p", "top_k", "use_cache")},
            {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 50, "use_cache": True})
        self.assertNotIn("stopping_criteria", calls[0])
        self.assertNotIn("token_type_ids", calls[0])
        self.assertEqual(backend.usage, {"model_calls": 1, "input_tokens": 6500, "output_tokens": 7})
        self.assertEqual(backend.generation_events, [])

    def test_uniform_limits_events_and_nested_scope_restore(self):
        backend, _, calls, _ = self.backend()
        stopping = object()
        with patch("clipscout.backends._prepare_images", return_value=[]), \
                patch("clipscout.backends._json_stopping_criteria", return_value=stopping):
            with self.assertRaisesRegex(ValueError, "leave scope"):
                with backend.uniform_session():
                    backend.generate("system", "user")
                    with backend.uniform_session():
                        backend.generate_action("system", "user")
                    self.assertEqual(backend._uniform_depth, 1)
                    raise ValueError("leave scope")
            self.assertEqual(backend._uniform_depth, 0)
            backend.generate("system", "user")
        self.assertEqual([call["max_new_tokens"] for call in calls], [256, 384, 768])
        self.assertIs(calls[1]["stopping_criteria"], stopping)
        self.assertEqual(backend.max_new_tokens, 768)
        self.assertEqual(backend.max_context_tokens, 8192)
        self.assertEqual([e["stage"] for e in backend.generation_events], ["observation", "action"])
        for event in backend.generation_events:
            self.assertEqual(event["status"], "ok")
            self.assertEqual(event["input_tokens"], 100)
            self.assertEqual(event["output_tokens"], 7)
            self.assertEqual(event["max_context_tokens"], 6144)
            self.assertGreaterEqual(event["elapsed_seconds"], 0)
            self.assertFalse(event["oom"])
            self.assertFalse(event["retry"])

    def test_limits_never_increase_configured_budgets(self):
        backend, _, calls, _ = self.backend(max_new_tokens=128, max_context_tokens=256)
        with patch("clipscout.backends._prepare_images", return_value=[]), backend.uniform_session():
            backend.generate("system", "user")
        self.assertEqual(calls[0]["max_new_tokens"], 128)
        self.assertEqual(backend.generation_events[0]["max_context_tokens"], 256)

    def test_uniform_context_is_rejected_before_gpu_transfer(self):
        backend, processor, calls, _ = self.backend(size=6000)
        with patch("clipscout.backends._prepare_images", return_value=[]), backend.uniform_session():
            with self.assertRaisesRegex(BackendError, "6000 input \\+ 256 output > 6144"):
                backend.generate("system", "user")
        self.assertEqual(calls, [])
        self.assertEqual(processor.batches[0].devices, [])
        self.assertEqual(backend.generation_events[0]["status"], "error")
        backend._torch.cuda.empty_cache.assert_not_called()

    def test_oom_releases_failed_tensors_before_cleanup_and_reloads_once(self):
        tensor_refs = []

        def fail_with_tensor():
            failed_tensor = TensorLifetime()
            tensor_refs.append(weakref.ref(failed_tensor))
            raise RuntimeError("CUDA out of memory. Tried to allocate a tensor.")

        backend, processor, calls, loads = self.backend(outcomes=[fail_with_tensor, None])
        cleanups = []

        def collect():
            self.assertIsNone(tensor_refs[0]())
            self.assertIsNone(backend._model)
            self.assertIsNone(backend._processor)
            cleanups.append("collect")

        def empty_cache():
            self.assertEqual(cleanups, ["collect"])
            cleanups.append("empty_cache")

        backend._torch.cuda.empty_cache.side_effect = empty_cache
        with patch("clipscout.backends._prepare_images", return_value=[]), \
                patch("clipscout.backends.gc.collect", side_effect=collect), backend.uniform_session():
            result = backend.generate("system", "user", iter([(1, object())]))
        self.assertEqual(result, '{"action":"answer"}')
        self.assertEqual(loads, ["cuda", "cuda"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(cleanups, ["collect", "empty_cache"])
        self.assertEqual([batch.devices for batch in processor.batches], [["cuda"], ["cuda"]])
        self.assertEqual([e["status"] for e in backend.generation_events], ["cuda_oom", "ok"])
        self.assertEqual([e["retry"] for e in backend.generation_events], [False, True])
        self.assertEqual(backend.usage, {"model_calls": 2, "input_tokens": 200, "output_tokens": 7})

    def test_repeated_oom_is_typed_clean_error_with_only_two_attempts(self):
        backend, _, calls, loads = self.backend(outcomes=[
            RuntimeError("CUDA out of memory first"), RuntimeError("CUDA out of memory second")])
        with patch("clipscout.backends._prepare_images", return_value=[]), \
                patch("clipscout.backends.gc.collect") as collect, backend.uniform_session():
            with self.assertRaisesRegex(BackendOOMError, "CUDA out of memory after a clean uniform retry") as caught:
                backend.generate("system", "user")
        self.assertIsNone(caught.exception.__context__)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(backend._model)
        self.assertIsNone(backend._processor)
        self.assertEqual(len(calls), 2)
        self.assertEqual(loads, ["cuda", "cuda"])
        self.assertEqual(collect.call_count, 2)
        self.assertEqual(backend._torch.cuda.empty_cache.call_count, 2)
        self.assertTrue(all(e["oom"] for e in backend.generation_events))
        self.assertTrue(is_cuda_oom(caught.exception))

    def test_failed_cuda_cleanup_is_terminal_typed_oom_without_retry(self):
        backend, _, calls, _ = self.backend(outcomes=[RuntimeError("CUDA error: out of memory")])
        backend._torch.cuda.empty_cache.side_effect = RuntimeError("CUDA error: out of memory during cleanup")
        with patch("clipscout.backends._prepare_images", return_value=[]), backend.uniform_session():
            with self.assertRaisesRegex(BackendOOMError, "uniform recovery failed") as caught:
                backend.generate("system", "user")
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(len(calls), 1)
        self.assertIn("during cleanup", backend.generation_events[-1]["recovery_error"])

    def test_partial_load_oom_is_also_released_before_clean_reload(self):
        backend, _, calls, loads = self.backend()
        original_load = backend._load
        partial_refs = []
        attempts = []

        def load():
            attempts.append("load")
            if len(attempts) == 1:
                partial_weights = TensorLifetime()
                partial_refs.append(weakref.ref(partial_weights))
                raise RuntimeError("CUDA out of memory while loading weights")
            original_load()

        backend._load = load

        def collect():
            self.assertIsNone(partial_refs[0]())
            self.assertIsNone(backend._model)

        with patch("clipscout.backends._prepare_images", return_value=[]), \
                patch("clipscout.backends.gc.collect", side_effect=collect), backend.uniform_session():
            backend.generate("system", "user")
        self.assertEqual(attempts, ["load", "load"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(loads, ["cuda"])
        self.assertEqual([e["input_tokens"] for e in backend.generation_events], [0, 100])

    def test_non_oom_error_propagates_without_retry_or_cleanup(self):
        expected = RuntimeError("kernel launch failed")
        backend, _, calls, loads = self.backend(outcomes=[expected])
        with patch("clipscout.backends._prepare_images", return_value=[]), \
                patch("clipscout.backends.gc.collect") as collect, backend.uniform_session():
            with self.assertRaises(RuntimeError) as caught:
                backend.generate("system", "user")
        self.assertIs(caught.exception, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(loads, ["cuda"])
        collect.assert_not_called()
        backend._torch.cuda.empty_cache.assert_not_called()
        self.assertEqual(backend.generation_events[0]["status"], "error")

    def test_adaptive_oom_keeps_original_exception_without_recovery(self):
        expected = RuntimeError("CUDA out of memory")
        backend, _, calls, _ = self.backend(outcomes=[expected])
        with patch("clipscout.backends._prepare_images", return_value=[]), \
                patch("clipscout.backends.gc.collect") as collect:
            with self.assertRaises(RuntimeError) as caught:
                backend.generate("system", "user")
        self.assertIs(caught.exception, expected)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_new_tokens"], 768)
        self.assertEqual(backend.generation_events, [])
        collect.assert_not_called()
        backend._torch.cuda.empty_cache.assert_not_called()

    def test_one_shot_images_reach_both_attempts_unchanged(self):
        backend, _, _, _ = self.backend(outcomes=[RuntimeError("CUDA out of memory"), None])
        images = [(1.0, object()), (2.0, object())]
        seen = []

        def prepare(supplied, *args):
            seen.append(list(supplied))
            return supplied

        with patch("clipscout.backends._prepare_images", side_effect=prepare), \
                patch("clipscout.backends.gc.collect"), backend.uniform_session():
            backend.generate("system", "user", iter(images))
        self.assertEqual(seen, [images, images])

    def test_oom_detection_supports_torch_type_and_saved_error_string(self):
        oom_type = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch"})
        self.assertTrue(is_cuda_oom(oom_type("allocation failed")))
        self.assertTrue(is_cuda_oom("BackendOOMError: CUDA out of memory after a clean uniform retry"))
        self.assertFalse(is_cuda_oom(RuntimeError("CPU out of memory")))
        self.assertFalse(is_cuda_oom(RuntimeError("CUDA kernel launch failed")))


if __name__ == "__main__":
    unittest.main()
