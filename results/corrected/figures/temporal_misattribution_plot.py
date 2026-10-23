"""Reproduce the corrected visual_06 diagnostic from original decoded frames (CPU)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sources = [ROOT / ".cache/visual-audit/adaptive_visual04_1372.504.png",
           ROOT / ".cache/annotation/boundary_1615.jpg"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
fig = plt.figure(figsize=(15, 7.4), facecolor="white")
fig.text(.04, .953, "Correct event order can hide a fabricated timestamp", fontsize=20,
         fontweight="bold", color="#182330")
fig.text(.04, .906, "Corrected adaptive run · visual_06: table first, task-graph diagram later", fontsize=12,
         color="#465465")
titles = ["A  Agent sample: 1372.504 s (22:52.504)",
          "B  Reference context: 1615.047 s (26:55.047)"]
notes = ["Still the comparison table; the answer labels it a diagram.",
         "The actual diagram; this frame was not supplied to the agent."]
for n, source in enumerate(sources):
    x = .04 + n * .49
    ax = fig.add_axes([x, .325, .43, .51])
    ax.imshow(Image.open(source), interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(titles[n], loc="left", fontsize=12, fontweight="bold", pad=13)
    fig.text(x, .3, notes[n], fontsize=10.5, color="#243447")
fig.text(.04, .223, "Delivered claim: table at 1357.489 s; diagram at 1372.504 s.",
         fontsize=12, fontweight="bold", color="#913B2D")
fig.text(.04, .161,
         "Citation E001 [1320, 1380] passes ID/bounds checks, but the graph was never inspected.",
         fontsize=12, color="#913B2D",
         bbox={"boxstyle": "round,pad=0.7", "facecolor": "#FAEEEB", "edgecolor": "#F0D4CC"})
fig.text(.04, .087,
         "User-provided source video; decoded frame content is unchanged. Left: actual 560 × 315 inspection sample.\n"
         "Right: 1280 × 720 reference image added for this audit, not an input to the run. No generated imagery.\n"
         "Assistant review, not human or blind validation. Posthoc protocol-corrected repeat; not independent heldout evidence.",
         fontsize=9, color="#596877", va="top")
for extension in ("png", "svg"):
    fig.savefig(OUT / f"temporal_misattribution.{extension}", dpi=180, facecolor="white")
plt.close(fig)
print(OUT / "temporal_misattribution.png")
