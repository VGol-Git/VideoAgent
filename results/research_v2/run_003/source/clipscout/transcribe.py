"""Optional CPU ASR: keeps CUDA memory available for the video language model."""
from pathlib import Path
import json
import time


def decode_audio(video, sampling_rate=16000):
    """Decode via public PyAV APIs and retain audio PTS on the video timeline.

    Passing the resulting array also avoids faster-whisper 1.2's removed
    PyAV-19 `metadata_errors` keyword. No third-party code is patched.
    """
    import av
    import numpy as np
    chunks, cursor = [], 0
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    def append(frame):
        nonlocal cursor
        values = frame.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
        start = round(float(frame.pts * frame.time_base) * sampling_rate) if frame.pts is not None else cursor
        if start > cursor:
            if start - cursor > 60 * sampling_rate:
                raise ValueError("Audio has a gap >60 seconds; provide aligned subtitles instead")
            chunks.append(np.zeros(start - cursor, dtype=np.float32))
            cursor = start
        elif start < cursor:
            values = values[min(len(values), cursor - start):]
        if len(values):
            chunks.append(values)
            cursor += len(values)
    with av.open(str(video), mode="r") as container:
        if not container.streams.audio:
            raise ValueError("Video has no audio stream; build a visual caption index")
        for frame in container.decode(audio=0):
            for output in resampler.resample(frame):
                append(output)
        for output in resampler.resample(None):
            append(output)
    if not chunks:
        raise ValueError("Audio stream decoded no samples")
    return np.concatenate(chunks)


def transcribe(video, output, model="base.en", language="en", device="cpu"):
    from faster_whisper import WhisperModel
    from .media import file_fingerprint
    root = Path(__file__).resolve().parents[1]
    began = time.perf_counter()
    dll_handles = []
    if device == "cuda":
        import os
        if os.name == "nt":
            import torch
            dll_dir = Path(torch.__file__).parent / "lib"
            dll_handles.append(os.add_dll_directory(str(dll_dir)))
    compute_type = "float16" if device == "cuda" else "int8"
    recognizer = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=6,
                              download_root=str(root / ".cache" / "whisper"))
    audio = decode_audio(video)
    segments, info = recognizer.transcribe(audio, language=language, beam_size=5,
                                           vad_filter=True, condition_on_previous_text=False)
    records = []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for s in segments:
        records.append({"id": f"T{len(records)+1:05}", "start": s.start, "end": s.end, "text": s.text.strip(), "source": "transcript"})
        if len(records) % 20 == 0:
            from .cli import save_json
            save_json(output.with_suffix(".partial.json"), records)
            print(f"ASR {s.end:.1f}/{info.duration:.1f}s", flush=True)
    def stamp(t):
        milliseconds = round(t * 1000)
        return f"{milliseconds//3600000:02}:{milliseconds//60000%60:02}:{milliseconds//1000%60:02}.{milliseconds%1000:03}"
    output.write_text("WEBVTT\n\n" + "\n\n".join(f"{stamp(s['start'])} --> {stamp(s['end'])}\n{s['text']}" for s in records) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".manifest.json").write_text(json.dumps({"model": model, "language": info.language, "duration": info.duration,
                                "video_sha256": file_fingerprint(video), "complete": True,
                                "device": device, "compute_type": compute_type, "latency_seconds": time.perf_counter()-began,
                                "note": "Automatic speech recognition; manually check question evidence and timing."}, indent=2), encoding="utf-8")
    partial = output.with_suffix(".partial.json")
    if partial.exists():
        partial.unlink()
