"""Plot full native MC E2E and actual PIM stage time from the audited CSV."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = {(row["case"], row["model"]): row for row in csv.DictReader(stream)}
    cases = "bcdefg"
    positions = range(len(cases))
    models = ("instant", "four-batch")
    colors = ("#9FB1C6", "#086C70")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#162D43", "axes.labelcolor": "#162D43"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1))
    fig.subplots_adjust(left=.07, right=.98, top=.695, bottom=.15, wspace=.2)
    fig.text(.07, .93, "Rank 1 ACTAB timing on the full workload", fontsize=21, weight="bold")
    fig.text(.07, .865, "18 layers × 10 AE steps  ·  2,357 VLM units  ·  Native MC  ·  MAC 2×", color="#425771")
    handles = [Patch(color=color, label=label) for color, label in zip(colors,
               ("16 banks at once", "4 batches × 4 banks"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .827), frameon=False, ncols=2)
    for axis, metric, title in zip(axes, ("e2e_ms", "pim_ms"), ("End-to-end latency", "PIM stage time")):
        values = [[float(rows[(case, model)][metric]) for case in cases] for model in models]
        for index, (series, color) in enumerate(zip(values, colors)):
            axis.bar([x + (-.17 if index == 0 else .17) for x in positions], series, .31, color=color, zorder=3)
        top = max(map(max, values))
        for x, before, after in zip(positions, *values):
            axis.text(x, max(before, after) + top * .020, f"{after-before:+.2f}",
                      ha="center", va="bottom", fontsize=10)
        axis.set_title(title, loc="left", weight="bold", pad=9)
        axis.set_ylim(0, top * 1.17)
        axis.set_ylabel("ms")
        axis.set_xticks(positions, cases)
        axis.tick_params(length=0)
        axis.grid(axis="y", color="#E0E7F0", linewidth=.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
    fig.text(.07, .045, "Both groups use masked PIM columns. Labels show change in ms. Case b is serial.", color="#425771")
    fig.savefig(args.output, dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
