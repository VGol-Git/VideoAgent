"""Dependency-free, loopback-only browser demo for an existing VideoAgent."""
from __future__ import annotations

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import mimetypes
import re
import threading
import uuid

MODES = ("adaptive", "retrieval-only", "static", "uniform")
MAX_BODY_BYTES = 16_384


def _examples(path):
    """Never send gold answers or annotations to the demo client."""
    if path is None:
        return []
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    data = ([json.loads(line) for line in text.splitlines() if line.strip()]
            if source.suffix.lower() == ".jsonl" else json.loads(text))
    if isinstance(data, dict):
        data = data.get("questions", [])
    if not isinstance(data, list):
        raise ValueError("Questions must be a list, JSONL, or an object with a questions list")
    return [{"id": str(item.get("id", "")), "question": item["question"],
             "type": str(item.get("type", item.get("category", "")))}
            for item in data if isinstance(item, dict) and isinstance(item.get("question"), str)][:100]


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, agent, host="127.0.0.1", port=7860, run_dir="runs/demo", questions_path=None):
        if host != "127.0.0.1":
            raise ValueError("The unauthenticated demo binds only to 127.0.0.1")
        self.agent = agent
        self.inference_lock = threading.Lock()
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.examples = _examples(questions_path)
        self.html = (Path(__file__).parent / "static" / "index.html").read_bytes()
        self.video = Path(agent.index.video_path).resolve() if agent.index.video_path else None
        super().__init__((host, port), DemoHandler)
        self.expected_host = f"127.0.0.1:{self.server_port}"
        self.origin = f"http://{self.expected_host}"


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer
    server_version = "ClipScout"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format, *args):
        # Do not log user questions, model outputs, or filesystem paths.
        return

    def _headers(self, status, content_type, length, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _json(self, status, data, extra=None):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body), extra)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _local_request(self):
        # Host validation also blocks browser DNS-rebinding access to localhost.
        if self.headers.get("Host") != self.server.expected_host:
            self._json(403, {"error": "Открывайте демо по его адресу 127.0.0.1."})
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != self.server.origin:
            self._json(403, {"error": "Запрос с другого сайта запрещён."})
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._json(403, {"error": "Запрос с другого сайта запрещён."})
            return False
        return True

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._local_request():
            return
        if self.path in ("/", "/index.html"):
            self._headers(200, "text/html; charset=utf-8", len(self.server.html))
            if self.command != "HEAD":
                self.wfile.write(self.server.html)
        elif self.path == "/api/info":
            from .agent import PROMPT_VERSION
            agent = self.server.agent
            config = agent.config
            self._json(200, {"title": agent.index.title, "duration": agent.index.duration,
                             "backend": agent.backend.name, "prompt_version": PROMPT_VERSION,
                             "mode": "visual" if config.use_vision else "transcript-only",
                             "video_available": bool(self.server.video and self.server.video.is_file()),
                             "limits": {"tool_calls": config.max_tool_calls, "frames": config.max_frames,
                                        "frames_per_clip": config.frames_per_clip,
                                        "clip_seconds": config.max_clip_seconds}, "modes": list(MODES)})
        elif self.path == "/api/examples":
            self._json(200, {"questions": self.server.examples})
        elif self.path == "/video":
            self._video()
        else:
            self._json(404, {"error": "Страница не найдена."})

    def _video(self):
        path = self.server.video
        if not path or not path.is_file():
            self._json(404, {"error": "К индексу не подключён доступный видеофайл."})
            return
        try:
            stream = path.open("rb")
        except OSError:
            self._json(404, {"error": "Видео недоступно."})
            return
        with stream:
            size = stream.seek(0, 2)
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested is not None:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
                try:
                    if not match or not any(match.groups()) or size == 0:
                        raise ValueError
                    left, right = match.groups()
                    if left:
                        start = int(left)
                        end = min(int(right), size - 1) if right else size - 1
                    else:
                        suffix = int(right)
                        if suffix <= 0:
                            raise ValueError
                        start = max(0, size - suffix)
                    if start >= size or start > end:
                        raise ValueError
                except ValueError:
                    self._json(416, {"error": "Недопустимый диапазон видео."}, {"Content-Range": f"bytes */{size}"})
                    return
            extra = {"Accept-Ranges": "bytes"}
            if requested is not None:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            length = max(0, end - start + 1)
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._headers(206 if requested is not None else 200, content_type, length, extra)
            if self.command == "HEAD":
                return
            stream.seek(start)
            try:
                while length:
                    chunk = stream.read(min(length, 256 * 1024))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    length -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                pass  # Seeking or closing the tab normally cancels a stream.

    def do_POST(self):
        if not self._local_request():
            return
        if self.path != "/api/ask":
            self._json(404, {"error": "Страница не найдена."})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Ожидается application/json."})
            return
        if self.headers.get("Transfer-Encoding") is not None:
            self._json(400, {"error": "Требуется Content-Length без Transfer-Encoding."})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(411, {"error": "Требуется корректный Content-Length."})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(413, {"error": "Запрос слишком большой или пустой."})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete body")
            def reject_constant(value):
                raise ValueError(value)
            data = json.loads(body.decode("utf-8"), parse_constant=reject_constant)
            if not isinstance(data, dict) or set(data) - {"question", "mode"}:
                raise ValueError("Unexpected fields")
            question, mode = data.get("question"), data.get("mode", "adaptive")
            if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
                raise ValueError("Invalid question")
            if not isinstance(mode, str) or mode not in MODES:
                raise ValueError("Invalid mode")
        except (UnicodeDecodeError, ValueError, TimeoutError):
            self._json(400, {"error": "Укажите вопрос длиной 1–2000 символов и допустимый режим."})
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self._json(409, {"error": "Модель уже отвечает на вопрос. Дождитесь завершения."}, {"Retry-After": "5"})
            return
        try:
            result = self.server.agent.run(question.strip(), mode=mode)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
            result = {**result, "run_id": run_id}
            payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            (self.server.run_dir / f"{run_id}.json").write_text(payload, encoding="utf-8")
        except Exception as exc:
            self._json(500, {"error": f"Не удалось завершить запрос ({type(exc).__name__}). Проверьте доступность модели и каталог результатов."})
            return
        finally:
            self.server.inference_lock.release()
        self._json(200, result)


def make_server(agent, host="127.0.0.1", port=7860, run_dir="runs/demo", questions_path=None):
    """Construct without blocking; port=0 is useful for integration tests."""
    return DemoServer(agent, host, port, run_dir, questions_path)


def serve(agent, host="127.0.0.1", port=7860, run_dir="runs/demo", questions_path=None):
    server = make_server(agent, host, port, run_dir, questions_path)
    print(f"ClipScout: {server.origin} · {agent.backend.name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
