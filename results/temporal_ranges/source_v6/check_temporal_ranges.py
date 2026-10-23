"""Run only the three user-supplied temporal regressions against the live demo."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from clipscout.index import VideoIndex
from clipscout.media import extract_frames

QUESTIONS = [
    ("20_to_21", "What is displayed on the screen from the 20th to the 21st minute", 1200, 1260),
    ("after_15", "What happens in the video after 15 minutes?", 900, None),
    ("15_to_20", "What happens in the video from the 15th to the 20th minute?", 900, 1200),
]


def review_frames(result, index, destination):
    """Save all actual sampled frames, plus a contact sheet for diagnostic review."""
    from PIL import Image, ImageDraw
    times = result["unique_sampled_timestamps"]
    sheet = Image.new("RGB", (1280, ((len(times) + 1) // 2) * 390), "white")
    draw = ImageDraw.Draw(sheet)
    for i, timestamp in enumerate(times):
        actual, frame = extract_frames(index.video_path, max(0, timestamp - .1), min(index.duration, timestamp + .1),
                                       count=1, max_side=1280, target_timestamp=timestamp)[0]
        frame.save(destination / f"frame_{i:02}_{actual:.3f}.png")
        frame.thumbnail((640, 360))
        x, y = i % 2 * 640, i // 2 * 390
        sheet.paste(frame, (x, y + 25))
        draw.text((x + 8, y + 6), f"{int(actual)//60:02}:{actual%60:06.3f} | {actual:.3f} seconds", fill="black")
    sheet.save(destination / "sampled_frames.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7871")
    parser.add_argument("--output", default="results/temporal_ranges/live")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    index = VideoIndex.load("data/indexes/user_video.json")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("clipscout").glob("*.py")}
    records = []
    for name, question, start, end in QUESTIONS:
        end = index.duration if end is None else end
        destination = root / name
        destination.mkdir(exist_ok=True)
        path = destination / "response.json"
        if path.exists():
            raise FileExistsError(f"Do not overwrite a completed response: {path}")
        print(f"Asking: {question}", flush=True)
        request = Request(args.url + "/api/ask", data=json.dumps({"question": question, "mode": "adaptive"}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"case": name, "status": result["status"], "answer": result["answer"],
                          "citations": result["citations"], "latency_seconds": result["latency_seconds"]}), flush=True)
        errors = []
        if result["status"] != "answered":
            errors.append("No substantive answer")
        if result.get("temporal_scope", {}).get("start") != start or result.get("temporal_scope", {}).get("end") != end:
            errors.append("Wrong requested scope")
        if not 2 <= result["tool_calls"] <= 4 or not 1 <= result["frames_used"] <= 16:
            errors.append("Missing distributed inspection or exceeded budget")
        for event in result["trace"]:
            if event["tool"] == "inspect_clip" and not start <= event["args"]["start"] < event["args"]["end"] <= end:
                errors.append("Inspection escaped requested scope")
        for evidence in result["evidence"].values():
            if any(not start <= t < end for t in evidence.get("sampled_timestamps", [])):
                errors.append("A sampled frame escaped requested scope")
            if any(not start <= t["start"] < t["end"] <= end for t in evidence.get("transcript_segments", [])):
                errors.append("Transcript cue escaped requested scope")
        if any(not start <= cite["start"] < cite["end"] <= end for cite in result["citations"]):
            errors.append("Citation escaped requested scope")
        if not result.get("coverage", {}).get("note") or result["coverage"]["complete_visual_coverage"]:
            errors.append("Missing sampling disclosure")
        if result["unique_sampled_timestamps"]:
            review_frames(result, index, destination)
        records.append({"case": name, "question": question, "run_id": result["run_id"],
                        "status": result["status"], "response_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "automated_checks_passed": not errors, "errors": errors})
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "cases": records,
              "source_sha256": source_hashes, "scope": "Only three user questions; no benchmark rerun.",
              "semantic_check": "Compare answers with saved actual sampled frames; structural checks do not grade factual accuracy."}
    (root / "checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if any(record["errors"] for record in records):
        raise SystemExit("One or more targeted checks failed; raw responses retained.")


if __name__ == "__main__":
    main()
