"""Prepare/resume a local CPU CLIP and optional OCR frame sidecar."""
import argparse
import json

from clipscout.index import VideoIndex
from clipscout.visual_index import DEFAULT_CLIP_MODEL, prepare_visual_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, help="Existing ClipScout transcript/caption index JSON")
    parser.add_argument("--output", required=True, help="Sidecar output directory; same configuration resumes")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="Cached repository or local snapshot directory")
    parser.add_argument("--ocr", action="store_true", help="Recognize sampled screen text using CPU RapidOCR")
    parser.add_argument("--cadence", type=float, default=5., help="Requested frame cadence in seconds")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args = parser.parse_args()
    index = VideoIndex.load(args.index)
    result = prepare_visual_index(index, args.output, clip_model=args.clip_model,
        cadence=args.cadence, device=args.device, ocr=args.ocr,
        on_progress=lambda progress: print(json.dumps(progress), flush=True))
    print(json.dumps({"status": "complete", "frames": len(result.frames),
                      "output": args.output, "usage": result.metadata["usage"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
