"""Command-line entry points. Core tests do not require model or video packages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import sys
import platform
import tempfile
import time

from .agent import AgentConfig, VideoAgent, PROMPT_VERSION
from .index import Segment, VideoIndex, read_subtitles


def experiment_provenance():
    """Record exact source and installed runtime versions without reading secrets."""
    package = Path(__file__).resolve().parent
    versions = {}
    for name in ("torch", "torchvision", "transformers", "numpy", "Pillow", "opencv-python-headless", "faster-whisper", "ctranslate2", "av"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = None
    return {"platform": platform.platform(), "packages": versions,
            "source_sha256": {str(p.relative_to(package)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(package.rglob("*.py"))}}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def backend_options(parser):
    parser.add_argument("--backend", choices=["local", "compatible"], default="local")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--endpoint", help="Explicit compatible endpoint (only used with --backend compatible)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=560, help="Maximum inspection frame side in pixels")
    parser.add_argument("--local-files-only", action="store_true", help="Never download weights during inference")


def make_backend(args):
    from .backends import OpenAICompatibleBackend, QwenBackend
    if args.backend == "compatible":
        if not args.endpoint:
            raise ValueError("--endpoint is required for a compatible backend")
        return OpenAICompatibleBackend(args.endpoint, args.model, max_new_tokens=args.max_new_tokens, max_image_pixels=args.image_size**2)
    return QwenBackend(args.model, device=args.device, max_new_tokens=args.max_new_tokens,
                       local_files_only=args.local_files_only, max_image_pixels=args.image_size**2)


def agent_options(parser):
    backend_options(parser)
    parser.add_argument("--index", required=True)
    parser.add_argument("--text-only", action="store_true", help="Inspect subtitle cues without decoding frames")
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--cache", help="Optional inspection cache directory; omit for cold evaluation")


def make_agent(args):
    index = VideoIndex.load(args.index)
    if index.video_path:
        from .media import file_fingerprint
        if not Path(index.video_path).is_file():
            raise ValueError(f"Indexed video missing: {index.video_path}")
        if file_fingerprint(index.video_path) != index.video_sha256:
            raise ValueError("Video content changed since indexing; rebuild the index")
    return VideoAgent(index, make_backend(args), AgentConfig(use_vision=not args.text_only,
                      frames_per_clip=args.frames_per_clip, max_frames=args.max_frames, image_size=args.image_size), args.cache)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="clipscout", description="Search, inspect and evaluate long video with a bounded VLM agent.")
    sub = parser.add_subparsers(dest="command", required=True)
    index_p = sub.add_parser("index", help="Index timestamped subtitles and/or sampled visual captions")
    index_p.add_argument("--video", required=True)
    index_p.add_argument("--subtitles")
    index_p.add_argument("--captions", help="Precomputed caption records (JSON array)")
    index_p.add_argument("--output", required=True)
    index_p.add_argument("--chunk-seconds", type=float, default=30)
    caption_p = sub.add_parser("caption", help="Create a visual-only search index from sparse VLM captions")
    backend_options(caption_p)
    caption_p.add_argument("--video", required=True)
    caption_p.add_argument("--output", required=True)
    caption_p.add_argument("--interval-seconds", type=float, default=60)
    caption_p.add_argument("--frames-per-segment", type=int, default=2)
    asr_p = sub.add_parser("transcribe", help="Optional local faster-whisper transcription")
    asr_p.add_argument("--video", required=True)
    asr_p.add_argument("--output", required=True)
    asr_p.add_argument("--model", default="base.en")
    asr_p.add_argument("--language", default="en")
    asr_p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ask_p = sub.add_parser("ask")
    agent_options(ask_p)
    ask_p.add_argument("--question", required=True)
    ask_p.add_argument("--mode", choices=["adaptive", "retrieval-only", "static", "uniform"], default="adaptive")
    ask_p.add_argument("--output")
    eval_p = sub.add_parser("evaluate")
    agent_options(eval_p)
    eval_p.add_argument("--questions", required=True)
    eval_p.add_argument("--output", required=True, help="Directory for per-mode predictions, metrics and manifest")
    eval_p.add_argument("--modes", nargs="+", choices=["adaptive", "retrieval-only", "static", "uniform"], default=["adaptive", "retrieval-only", "static", "uniform"])
    eval_p.add_argument("--split", choices=["dev", "test", "all"], default="test")
    eval_p.add_argument("--limit", type=int)
    eval_p.add_argument("--resume", action="store_true")
    serve_p = sub.add_parser("serve")
    agent_options(serve_p)
    serve_p.add_argument("--port", type=int, default=7860)
    serve_p.add_argument("--questions")
    serve_p.add_argument("--run-dir", default="runs/demo")
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            from .media import file_fingerprint, probe_video
            video = str(Path(args.video).resolve())
            metadata = probe_video(video)
            segments = read_subtitles(args.subtitles) if args.subtitles else []
            if args.captions:
                captions = json.loads(Path(args.captions).read_text(encoding="utf-8"))
                segments.extend(Segment(**{k: c[k] for k in ("id", "start", "end", "text", "source")}) for c in captions)
            if not segments:
                raise ValueError("Provide --subtitles or --captions; use transcribe/caption first")
            # Caption timing may overhang by a fraction of a frame; never silently clip a materially wrong file.
            for s in segments:
                if s.end > metadata["duration"]:
                    raise ValueError(f"Cue {s.id} ends beyond this video's duration; check subtitle alignment")
            index = VideoIndex(metadata["duration"], segments, video, file_fingerprint(video), Path(video).stem, args.chunk_seconds)
            index.save(args.output)
            print(json.dumps({"index": args.output, "segments": len(segments), **metadata}, indent=2))
        elif args.command == "caption":
            from .media import build_caption_segments, file_fingerprint
            backend = make_backend(args)
            records = []
            start = time.perf_counter()
            for record in build_caption_segments(args.video, backend, args.interval_seconds, args.frames_per_segment):
                records.append(record)
                save_json(args.output, records)
                print(f"Captioned {record['start']:.0f}–{record['end']:.0f}s", flush=True)
            save_json(str(args.output) + ".manifest.json", {"backend": backend.name, "usage": backend.usage,
                      "video_sha256": file_fingerprint(args.video), "complete": True,
                      "latency_seconds": time.perf_counter()-start, "frames_per_segment": args.frames_per_segment,
                      "interval_seconds": args.interval_seconds, "caption_prompt_version": "concise-visual-v1",
                      "note": "Preprocessing cost excluded from query cost; sparse captions are not exhaustive."})
        elif args.command == "transcribe":
            from .transcribe import transcribe
            transcribe(args.video, args.output, model=args.model, language=args.language, device=args.device)
        elif args.command == "ask":
            result = make_agent(args).run(args.question, mode=args.mode)
            if args.output:
                save_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            from .evaluation import evaluate, retrieval_eval
            agent = make_agent(args)
            questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
            questions = [q for q in questions if args.split == "all" or q["split"] == args.split]
            if args.limit is not None:
                if args.limit < 1:
                    raise ValueError("--limit must be positive")
                questions = questions[:args.limit]
            if not questions:
                raise ValueError("No questions in selected split")
            out = Path(args.output)
            if not args.resume and ((out / "manifest.json").exists() or any(out.glob("*.predictions.json"))):
                raise ValueError("Experiment output already exists; choose a new directory or explicitly --resume")
            if args.resume and not (out / "manifest.json").exists() and any(out.glob("*.predictions.json")):
                raise ValueError("Cannot resume predictions without their experiment manifest")
            manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "backend": agent.backend.name,
                        "index_fingerprint": agent.index.fingerprint(), "question_ids": [q["id"] for q in questions],
                        "modes": args.modes, "vision": not args.text_only, "frames_per_clip": args.frames_per_clip,
                        "max_frames": args.max_frames, "image_size": args.image_size, "max_new_tokens": args.max_new_tokens,
                        "max_tool_calls": 4, "cache_enabled": bool(args.cache),
                        "prompt_version": PROMPT_VERSION,
                        "split": args.split, "python": sys.version, "question_file_sha256": hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
                        "provenance": experiment_provenance()}
            previous = out / "manifest.json"
            if args.resume and previous.exists():
                old = json.loads(previous.read_text(encoding="utf-8"))
                for k in ("backend", "index_fingerprint", "question_ids", "vision", "frames_per_clip", "max_frames", "question_file_sha256", "image_size", "max_new_tokens", "cache_enabled", "prompt_version"):
                    if old.get(k) != manifest.get(k):
                        raise ValueError(f"Cannot resume changed experiment: {k}")
                if old.get("provenance") != manifest["provenance"]:
                    raise ValueError("Cannot resume with changed source code or runtime")
            # All model load and a real warm-up generation happen before question timers.
            warmup_started = time.perf_counter()
            warmup_images = None
            if not args.text_only and agent.index.video_path:
                from .media import extract_frames
                warmup_images = extract_frames(agent.index.video_path, 0, min(4.0, agent.index.duration),
                                               count=args.frames_per_clip, max_side=args.image_size)
            agent.backend.generate("Reply with the single word READY.", "Initialize inference.", images=warmup_images)
            manifest["warmup_seconds"] = time.perf_counter() - warmup_started
            manifest["warmup_frames"] = len(warmup_images or [])
            manifest["warmup_usage"] = dict(agent.backend.usage)
            save_json(previous, manifest)
            save_json(out / "retrieval.json", retrieval_eval(questions, lambda q, k: agent.index.search(q[:500], top_k=k), k=5))
            for mode in args.modes:
                predictions_file = out / f"{mode}.predictions.json"
                predictions = json.loads(predictions_file.read_text(encoding="utf-8")) if args.resume and predictions_file.exists() else []
                evaluate(questions, predictions, max_frames=args.max_frames, bootstrap_samples=1)
                done = {p["question_id"] for p in predictions}
                for q in questions:
                    if q["id"] in done:
                        continue
                    print(f"{mode} / {q['id']}: {q['question']}", flush=True)
                    try:
                        pred = agent.run(q["question"], q["id"], mode)
                    except Exception as exc:
                        pred = {"question_id": q["id"], "status": "error", "answer": "", "citations": [], "tool_calls": 0, "frames_used": 0, "latency_seconds": 0, "error": f"{type(exc).__name__}: {exc}"}
                    predictions.append(pred)
                    save_json(predictions_file, predictions)
                    print(f"  {pred['status']} / {pred['tool_calls']} calls / {pred['latency_seconds']:.1f}s", flush=True)
                save_json(out / f"{mode}.metrics.json", evaluate(questions, predictions, max_frames=args.max_frames))
        elif args.command == "serve":
            from .server import serve
            serve(make_agent(args), port=args.port, run_dir=args.run_dir, questions_path=args.questions)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"clipscout: {exc}\n")


if __name__ == "__main__":
    main()
