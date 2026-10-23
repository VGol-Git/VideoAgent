"""Two targeted live-service checks, separate from the completed research run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from clipscout.index import VideoIndex
from clipscout.media import extract_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7871")
    parser.add_argument("--output", default="results/moment_fix")
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    questions = [(610, "Describe what was on the slide at 10 minutes 10 seconds"),
                 (800, "Describe what was on the slide at 800 seconds")]
    # Read-only reference frames for visual auditing; these are not a new index
    # or model input. The service independently decodes its own target images.
    index = VideoIndex.load("data/indexes/user_video.json")
    for target, _ in questions:
        actual, image = extract_frames(index.video_path, target - .5, target + .5,
                                       target_timestamp=target, max_side=1280)[0]
        image.save(directory / f"frame_{target}.png")
        print(f"Reference frame: requested={target}, actual={actual:.6f}", flush=True)
    records = []
    for target, question in questions:
        path = directory / f"{target}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite a completed check: {path}")
        print(f"Asking live service: {question}", flush=True)
        request = Request(args.url + "/api/ask", data=json.dumps({"question": question, "mode": "adaptive"}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"target": target, "status": result["status"], "answer": result["answer"],
                          "timestamps": result["unique_sampled_timestamps"], "latency": result["latency_seconds"]}), flush=True)
        assert result["status"] == "answered", result.get("error", result["answer"])
        assert result["requested_timestamps"] == [target]
        assert result["tool_calls"] == 1 and result["frames_used"] == 1
        assert [event["tool"] for event in result["trace"]] == ["inspect_clip"]
        for citation in result["citations"]:
            evidence = result["evidence"][citation["evidence_id"]]
            assert evidence["requested_timestamp"] == target and evidence["target_frame_verified"]
            assert citation["start"] <= target < citation["end"]
            assert abs(evidence["target_frame_timestamp"] - target) <= 0.034
        records.append({"target": target, "question": question, "run_id": result["run_id"],
                        "result": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "actual_timestamps": result["unique_sampled_timestamps"], "status": result["status"]})
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "checks": records,
              "scope": "Two targeted service regressions only; no research experiment rerun.",
              "temporal_checks_passed": True, "semantic_correctness": "Requires inspection of saved reference frames and answer text."}
    (directory / "checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
