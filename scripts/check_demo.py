"""Read-only checks of the running local demo; no model generation is triggered."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen


def check(base_url, output):
    def get(path, **kwargs):
        return urlopen(Request(base_url.rstrip("/") + path, **kwargs), timeout=15)

    with get("/api/info") as response:
        info = json.load(response)
    assert info["video_available"] and info["limits"]["tool_calls"] == 4
    with get("/api/examples") as response:
        examples = json.load(response)["questions"]
    assert examples and all("question" in q for q in examples)
    forbidden = {"reference_answer", "answer", "evidence", "required_terms"}
    assert all(not forbidden.intersection(q) for q in examples)
    with get("/video", method="HEAD") as response:
        size = int(response.headers["Content-Length"])
        assert response.status == 200 and size > 0
    with get("/video", headers={"Range": "bytes=0-1023"}) as response:
        assert response.status == 206 and len(response.read()) == 1024
        assert response.headers["Content-Range"] == f"bytes 0-1023/{size}"
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "base_url": base_url,
              "info": info, "example_count": len(examples), "examples_exclude_reference_labels": True,
              "video_head_bytes": size, "video_range_206_verified": True,
              "note": "HTTP delivery checks only; browser interaction and real model answer are checked separately."}
    Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7871")
    parser.add_argument("--output", default="results/demo_http_check.json")
    args = parser.parse_args()
    check(args.url, args.output)
