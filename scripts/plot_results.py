"""Export scientific comparison figures from completed per-mode JSON metrics."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(directory):
    directory = Path(directory)
    modes = ["adaptive", "retrieval-only", "static", "uniform"]
    labels = ["Adaptive", "Retrieval only", "Static top-3", "Uniform"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest_modes = manifest.get("modes", [])
    if len(manifest_modes) != len(modes) or set(manifest_modes) != set(modes):
        raise ValueError("Comparison plot requires exactly the four modes recorded in the manifest")
    question_ids = manifest.get("question_ids", [])
    if (not question_ids or not all(isinstance(identifier, str) and identifier for identifier in question_ids)
            or len(question_ids) != len(set(question_ids))):
        raise ValueError("Manifest must contain a nonempty list of unique question IDs")
    expected_ids = set(question_ids)
    split = manifest.get("split")
    if split not in ("dev", "test", "all"):
        raise ValueError("Manifest must declare a supported split: dev, test or all")
    summaries, predictions = [], []
    for mode in modes:
        report = json.loads((directory / f"{mode}.metrics.json").read_text(encoding="utf-8"))
        summary = report["summary"]
        rows = json.loads((directory / f"{mode}.predictions.json").read_text(encoding="utf-8"))
        for name, entries in (("predictions", rows), ("metrics", report["per_question"])):
            ids = [entry.get("question_id") for entry in entries]
            if len(ids) != len(question_ids) or set(ids) != expected_ids:
                raise ValueError(f"{mode} {name} question IDs differ from the manifest or contain duplicates")
        if any(row.get("mode") != mode for row in rows):
            raise ValueError(f"Prediction mode differs from its manifest entry: {mode}")
        if (summary["missing_prediction_count"] or summary["example_count"] != len(question_ids)
                or summary["prediction_count"] != len(question_ids)):
            raise ValueError("Only complete manifest-matched evaluations may be plotted")
        summaries.append(summary)
        predictions.append(rows)
    counts = {s["example_count"] for s in summaries}
    if len(counts) != 1:
        raise ValueError("Evaluation counts differ across modes")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "svg.fonttype": "none"})
    colors = ["#245cc5", "#bc7825", "#248575", "#8997a7"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.9), layout="constrained")
    x = np.arange(len(modes))
    for axis, metric, title in zip(axes[:2], ["token_f1", "temporal_iou"],
                                   ["Answer overlap (lexical proxy)", "Citation localization (mean tIoU)"]):
        values = [s[metric] for s in summaries]
        intervals = [s["confidence_intervals"][metric] for s in summaries]
        axis.bar(x, values, color=colors, width=.64, zorder=3)
        axis.errorbar(x, values, yerr=[[max(0, v-ci["low"]) for v, ci in zip(values, intervals)],
                                     [max(0, ci["high"]-v) for v, ci in zip(values, intervals)]],
                      fmt="none", color="#263548", capsize=4, linewidth=1, zorder=4)
        axis.set_ylim(0, 1)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("Score [0, 1]")
        axis.set_yticks(np.arange(0, 1.01, .2))
    times = [statistics.median(p["latency_seconds"] for p in rows) for rows in predictions]
    axes[2].bar(x, times, color=colors, width=.64, zorder=3)
    axes[2].set_title("Query latency (median across questions)", fontsize=11)
    axes[2].set_ylabel("Seconds; uncontrolled shared GPU")
    for i, value in enumerate(times):
        axes[2].annotate(f"{value:.1f}s", (i, value), xytext=(0, 5), textcoords="offset points",
                         ha="center", fontsize=9)
    axes[2].set_ylim(0, max(times) * 1.2 if max(times) else 1)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=.2, zorder=0)
    fig.suptitle(f"ClipScout · {next(iter(counts))} questions ({split} split) on one 42:55 video", fontsize=15, fontweight="bold")
    answer_count = summaries[0]["denominators"]["answer_metrics"]
    fig.supxlabel(f"Answer/grounding metrics: n={answer_count} answerable questions; latency: n={next(iter(counts))} completed results.\n"
                  "Assistant/ASR references; not human-validated. Error bars: exploratory 95% question-bootstrap intervals.\n"
                  "Same cap: 4 tools / 16 frames; actual costs differ. Query time excludes preprocessing/warm-up and is NOT a controlled speed comparison.", fontsize=9)
    output = directory / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    print(output / "comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    plot(parser.parse_args().directory)
