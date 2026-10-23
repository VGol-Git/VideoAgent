"""Reproduce the visual_01 diagnostic figure from decoded source frames (CPU)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sources = [
    ROOT / ".cache/annotation/refine_0020.jpg",
    ROOT / ".cache/visual-audit/adaptive_visual01_1481.246.png",
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
figure = plt.figure(figsize=(15, 7.1), facecolor="white")
figure.text(0.04, 0.952, "Visual question 01: a wrong-scene inspection", fontsize=20, fontweight="bold", color="#182330")
figure.text(0.04, 0.902, "Question: which black object stands on the desk during the opening introduction?", fontsize=12, color="#465465")
titles = ["A  Reference context: 20.053 s (00:20.053)", "B  Agent sample: 1481.246 s (24:41.246)"]
notes = ["A black microphone on a desk stand is visible.", "Application UI is visible; no physical desk is shown."]
for index, path in enumerate(sources):
    axis = figure.add_axes([0.04 + index * 0.49, 0.31, 0.43, 0.53])
    axis.imshow(Image.open(path), interpolation="nearest")
    axis.set_axis_off()
    axis.set_title(titles[index], loc="left", fontsize=12, fontweight="bold", pad=13)
    figure.text(0.04 + index * 0.49, 0.292, notes[index], fontsize=11, color="#243447")
figure.text(0.04, 0.205, "Observed failure chain", fontsize=12, fontweight="bold", color="#182330")
figure.text(0.04, 0.154,
            "Retrieval → wrong scene → VLM invents a desk monitor → final answer rejected (missing citations)",
            fontsize=12, color="#913B2D", bbox={"boxstyle": "round,pad=0.7", "facecolor": "#FAEEEB", "edgecolor": "#F0D4CC"})
figure.text(0.04, 0.075,
            "User-provided source video. Decoded frame content is unchanged; no generated imagery.\n"
            "Left: reference frame at 1280 × 720. Right: the actual inspection sample size, 560 × 315.\n"
            "Assistant diagnostic review, not human/blinded evaluation. The reference frame was not supplied to this agent run.",
            fontsize=9, color="#596877", va="top")
for extension in ("png", "svg"):
    figure.savefig(OUT / f"visual01_failure.{extension}", dpi=180, facecolor="white")
plt.close(figure)
print(OUT / "visual01_failure.png")
print(OUT / "visual01_failure.svg")
