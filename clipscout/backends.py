"""Explicit local or endpoint-backed VLM adapters with bounded inputs.

Heavy dependencies are loaded only when needed. Model weights always use a
cache inside this project. No backend silently falls back to a remote service.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import gc
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


class BackendError(RuntimeError):
    """The configured backend could not complete a generation."""


class BackendOOMError(BackendError):
    """CUDA generation still failed after one clean, same-request retry."""


def is_cuda_oom(error):
    """Recognize CUDA OOMs without importing torch or retaining exceptions."""
    message = str(error).lower()
    if "cuda out of memory" in message or "cuda error: out of memory" in message:
        return True
    kind = type(error)
    return kind.__name__ == "OutOfMemoryError" and kind.__module__.startswith("torch")


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    return value


def _prepare_images(images, max_images, max_pixels):
    from PIL import Image

    images = [] if images is None else list(images)
    if len(images) > max_images:
        raise ValueError(f"At most {max_images} images are allowed per model call.")
    prepared = []
    for timestamp, source in images:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Image timestamps must be finite nonnegative seconds.")
        if not isinstance(source, Image.Image):
            raise TypeError("Images must be PIL.Image.Image instances.")
        image = source.convert("RGB")
        width, height = image.size
        if width < 1 or height < 1 or max(width, height) / min(width, height) > 200:
            raise ValueError("Image dimensions are empty or too elongated.")
        if width * height > max_pixels:
            scale = math.sqrt(max_pixels / (width * height))
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))),
                                 Image.Resampling.LANCZOS)
        prepared.append((timestamp, image))
    return prepared


def _validate_text(system, user, maximum):
    if not isinstance(system, str) or not isinstance(user, str):
        raise TypeError("system and user must be strings.")
    if not user.strip():
        raise ValueError("The user prompt must not be empty.")
    if len(system) + len(user) > maximum:
        raise ValueError(f"Text exceeds the configured {maximum}-character input limit.")


def json_object_complete(text: str) -> bool:
    """Whether a root JSON object's closing brace has been generated.

    This is a structural stopping predicate, not a JSON/schema validator. It
    respects nested objects/arrays, quoted braces and escaped quotes. Only
    whitespace may precede the object; malformed output remains the agent's
    responsibility. No substring is extracted or substituted for model output.
    """
    text = text.lstrip()
    if not text.startswith("{"):
        return False
    stack = []
    quoted = escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            expected = "{" if character == "}" else "["
            if not stack or stack.pop() != expected:
                return False
            if not stack:
                return True
    return False


def _json_stopping_criteria(tokenizer, prompt_length):
    """Build the actual generation-time criterion without eager ML imports."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopAtRootObject(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            # QwenBackend generates a single sequence. Decode only generated
            # tokens: JSON examples in the system prompt must never stop it.
            generated_text = tokenizer.decode(
                input_ids[0, prompt_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            return json_object_complete(generated_text)

    return StoppingCriteriaList([StopAtRootObject()])


class QwenBackend:
    """Local Qwen3-VL inference using Transformers and one explicit device.

    First generation downloads missing model files to the project cache unless
    local_files_only=True. On CUDA, BF16 and SDPA are used; CPU uses float32.
    Oversized contexts fail explicitly, preserving image/token alignment.
    """

    UNIFORM_OBSERVATION_TOKENS = 256
    UNIFORM_ACTION_TOKENS = 384
    UNIFORM_CONTEXT_TOKENS = 6144

    def __init__(self, model="Qwen/Qwen3-VL-2B-Instruct", cache_dir=None,
                 device="cuda", max_new_tokens=512, max_context_tokens=8192,
                 max_images=16, max_image_pixels=560 * 560, local_files_only=False):
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be 'cuda' or 'cpu'.")
        self.model_id = str(model)
        self.revision = QWEN_REVISION if self.model_id == "Qwen/Qwen3-VL-2B-Instruct" else None
        self.name = f"qwen:{self.model_id}@{self.revision or 'user-selected'}:{device}"
        self.device = device
        self.max_new_tokens = _integer(max_new_tokens, "max_new_tokens", 1, 2048)
        self.max_context_tokens = _integer(max_context_tokens, "max_context_tokens", 256, 32768)
        if self.max_new_tokens >= self.max_context_tokens:
            raise ValueError("max_new_tokens must be less than max_context_tokens.")
        self.max_images = _integer(max_images, "max_images", 1, 64)
        self.max_image_pixels = _integer(max_image_pixels, "max_image_pixels", 1024, 1048576)
        self.cache_dir = Path(cache_dir or PROJECT_ROOT / ".cache" / "huggingface").resolve()
        if not self.cache_dir.is_relative_to(PROJECT_ROOT):
            raise ValueError("Model cache_dir must be inside the ClipScout project.")
        self.local_files_only = bool(local_files_only)
        self.usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._model = self._processor = self._torch = None
        self._uniform_depth = 0
        self.generation_events = []

    @contextmanager
    def uniform_session(self):
        """Enable bounded decoding and OOM recovery only for uniform policy.

        The configured limits are never increased and are not mutated. Nested
        sessions are supported; exceptions also restore the caller's scope.
        """
        self._uniform_depth += 1
        try:
            yield self
        finally:
            self._uniform_depth -= 1

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise BackendError("QwenBackend requires torch, matching torchvision, and transformers>=4.57.1.") from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise BackendError("CUDA was requested but is unavailable; choose device='cpu' explicitly.")
        if self.device == "cuda" and not torch.cuda.is_bf16_supported():
            raise BackendError("The selected CUDA device does not support BF16.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        common = {"cache_dir": str(self.cache_dir), "local_files_only": self.local_files_only,
                  "trust_remote_code": False}
        source = self.model_id
        if self.revision:
            snapshot = self.cache_dir / ("models--" + self.model_id.replace("/", "--")) / "snapshots" / self.revision
            if snapshot.is_dir():
                # Transformers 4.57 can probe optional processor files remotely even
                # with local_files_only. A concrete snapshot path avoids that probe.
                source = str(snapshot)
        if self.revision and source == self.model_id:
            common["revision"] = self.revision
        processor = AutoProcessor.from_pretrained(
            source, use_fast=False, min_pixels=32 * 32,
            max_pixels=self.max_image_pixels, **common)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            source, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa", **common)
        model = model.to(self.device).eval()
        self._torch, self._processor, self._model = torch, processor, model

    def generate(self, system: str, user: str, images=None) -> str:
        return self._generate(system, user, images)

    def generate_action(self, system: str, user: str) -> str:
        """Stop controller decoding when its first root object closes.

        The model output is returned unchanged. Schema checking stays in the
        agent; captions and visual observations use unrestricted ``generate``.
        """
        system += "\nReturn exactly one JSON object beginning with '{'. Do not use Markdown fences or emit a second object."
        return self._generate(system, user, stop_json=True)

    def _generate(self, system: str, user: str, images=None, stop_json=False) -> str:
        if not self._uniform_depth:
            return self._generate_once(system, user, images, stop_json)
        # Materialize one-shot image iterables once so recovery repeats the
        # identical request rather than silently losing visual evidence.
        images = None if images is None else list(images)
        for attempt in (1, 2):
            result, oom_message = self._uniform_attempt(system, user, images, stop_json, attempt)
            if oom_message is None:
                return result
            # The attempt helper has returned and cleared its exception and
            # generation frames before models/caches are released here.
            recovery_error = None
            try:
                self._recover_cuda_oom()
            except Exception as exc:
                recovery_error = f"{type(exc).__name__}: {exc}"
                traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = exc.__cause__ = exc.__context__ = None
            if recovery_error is not None:
                self.generation_events[-1]["recovery_error"] = recovery_error
                raise BackendOOMError("CUDA out of memory; uniform recovery failed: " + recovery_error)
        raise BackendOOMError(
            "CUDA out of memory after a clean uniform retry of the same request: " + oom_message)

    def _uniform_attempt(self, system, user, images, stop_json, attempt):
        output_limit = min(self.max_new_tokens, self.UNIFORM_ACTION_TOKENS if stop_json
                           else self.UNIFORM_OBSERVATION_TOKENS)
        context_limit = min(self.max_context_tokens, self.UNIFORM_CONTEXT_TOKENS)
        event = {"stage": "action" if stop_json else "observation", "attempt": attempt,
                 "retry": attempt > 1, "oom": False, "status": "ok", "input_tokens": 0,
                 "output_tokens": 0, "max_new_tokens": output_limit,
                 "max_context_tokens": context_limit}
        began = time.perf_counter()
        try:
            return self._generate_once(system, user, images, stop_json,
                                       max_new_tokens=output_limit, max_context_tokens=context_limit,
                                       attempt_event=event), None
        except Exception as exc:
            if self.device != "cuda" or not is_cuda_oom(exc):
                event.update(status="error", error_type=type(exc).__name__)
                raise
            message = str(exc)
            event.update(status="cuda_oom", oom=True, error_type=type(exc).__name__, error=message)
            # An OOM traceback owns the failed generate()/to()/load() frames,
            # including their GPU tensors. Do not collect/cache-clear while
            # these frames are still live, or retain them in a retry exception.
            traceback.clear_frames(exc.__traceback__)
            exc.__traceback__ = exc.__cause__ = exc.__context__ = None
            return None, message
        finally:
            event["elapsed_seconds"] = round(time.perf_counter() - began, 4)
            self.generation_events.append(event)

    def _recover_cuda_oom(self):
        self._model = self._processor = None
        torch = self._torch or sys.modules.get("torch")
        gc.collect()
        if torch is not None:
            torch.cuda.empty_cache()

    def _generate_once(self, system: str, user: str, images=None, stop_json=False,
                       *, max_new_tokens=None, max_context_tokens=None, attempt_event=None) -> str:
        # Defaults deliberately retain the original adaptive generation path.
        output_limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        context_limit = self.max_context_tokens if max_context_tokens is None else max_context_tokens
        _validate_text(system, user, context_limit * 4)
        images = _prepare_images(images, self.max_images, self.max_image_pixels)
        self._load()
        content = [{"type": "text", "text": user}]
        for timestamp, _ in images:
            content.extend([{"type": "text", "text": f"Frame at {timestamp:.3f} seconds:"},
                            {"type": "image"}])
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": content}]
        prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        arguments = {"text": [prompt], "return_tensors": "pt", "padding": True}
        if images:
            arguments["images"] = [image for _, image in images]
        inputs = self._processor(**arguments)
        inputs.pop("token_type_ids", None)
        input_count = int(inputs["input_ids"].shape[-1])
        if attempt_event is not None:
            attempt_event["input_tokens"] = input_count
        if input_count + output_limit > context_limit:
            raise BackendError(
                f"Context budget exceeded: {input_count} input + {output_limit} output "
                f"> {context_limit}. Reduce evidence or images.")
        inputs = inputs.to(self.device)
        generation_options = {}
        if stop_json:
            generation_options["stopping_criteria"] = _json_stopping_criteria(
                self._processor.tokenizer, input_count)
        self.usage["model_calls"] += 1
        self.usage["input_tokens"] += input_count
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=output_limit,
                                             do_sample=False, temperature=1.0, top_p=1.0, top_k=50,
                                             use_cache=True, **generation_options)
        output_ids = generated[:, input_count:]
        self.usage["output_tokens"] += int(output_ids.shape[-1])
        if attempt_event is not None:
            attempt_event["output_tokens"] = int(output_ids.shape[-1])
        result = self._processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        if not result:
            raise BackendError("The local model returned an empty response.")
        return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OpenAICompatibleBackend:
    """Opt-in compatible chat endpoint. Pass a base /v1 URL or full endpoint.

    Uses token/api_key or API_KEY from the environment. No credential is required
    for an explicitly chosen unauthenticated local server. Character and image
    limits bound requests; token accounting relies on the server's usage fields.
    """

    def __init__(self, endpoint: str, model: str, api_key=None, token=None,
                 max_new_tokens=512, timeout=120, max_input_chars=32000,
                 max_images=16, max_image_pixels=560 * 560):
        endpoint = endpoint.rstrip("/")
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("endpoint must be an explicit HTTP(S) URL without embedded credentials.")
        if parsed.query or parsed.fragment:
            raise ValueError("endpoint must not contain a query or fragment.")
        self.endpoint = endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"
        self.model_id = str(model)
        self.name = f"compatible:{model}@{self.endpoint}"
        self._token = token or api_key or os.environ.get("API_KEY")
        self.max_new_tokens = _integer(max_new_tokens, "max_new_tokens", 1, 2048)
        self.max_input_chars = _integer(max_input_chars, "max_input_chars", 1, 128000)
        self.max_images = _integer(max_images, "max_images", 1, 64)
        self.max_image_pixels = _integer(max_image_pixels, "max_image_pixels", 1024, 1048576)
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 600:
            raise ValueError("timeout must be between 0 and 600 seconds.")
        self.usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}

    def generate(self, system: str, user: str, images=None) -> str:
        _validate_text(system, user, self.max_input_chars)
        images = _prepare_images(images, self.max_images, self.max_image_pixels)
        content = [{"type": "text", "text": user}]
        for timestamp, image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.extend([{"type": "text", "text": f"Frame at {timestamp:.3f} seconds:"},
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}])
        payload = {"model": self.model_id,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": content if images else user}],
                   "max_tokens": self.max_new_tokens, "temperature": 0}
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        request = urllib.request.Request(self.endpoint,
                                         data=json.dumps(payload).encode("utf-8"), headers=headers)
        self.usage["model_calls"] += 1
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=self.timeout) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise BackendError("Endpoint response exceeds the 4 MiB limit.")
            result = json.loads(raw)
            text = result["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise BackendError("Endpoint returned empty or non-text content.")
            usage = result.get("usage") or {}
            for local, remote in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
                value = usage.get(remote, 0)
                if isinstance(value, int) and value >= 0:
                    self.usage[local] += value
            return text.strip()
        except urllib.error.HTTPError as exc:
            raise BackendError(f"Configured endpoint returned HTTP {exc.code}.") from exc
        except urllib.error.URLError as exc:
            raise BackendError("Could not connect to the configured endpoint.") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise BackendError("Configured endpoint returned an invalid chat response.") from exc
