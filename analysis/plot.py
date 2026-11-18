"""Plot native evaluation CSV accounting; never read or retime command traces.

The legacy plot_comparison.py supplied the horizontal breakdown presentation.
Its trace partitions and external replay audit are intentionally not inputs.
Only the new native evaluation columns below have defined additive semantics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

COMPONENTS = (
    ("AE_solo_ms", "AE standalone", "#4477AA"),
    ("PIM_interference_ms", "PIM interference (signed)", "#EE6677"),
    ("AE_GPU_wait_ms", "AE GPU wait", "#CCBB44"),
    ("VLM_remaining_ms", "VLM remaining after AE", "#228833"),
)
REQUIRED = ("case", "total_ms", *(field for field, _, _ in COMPONENTS))
TOLERANCE_MS = 0.000002  # Same 0.002 us accounting bound as report.py.


def read_evaluation(path, cases=None):
    """Validate signed additive accounting before creating a figure."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(REQUIRED) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"native evaluation CSV is missing {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("evaluation CSV has no cases")
    requested = None if cases is None else set(cases)
    seen, validated = set(), []
    for row in rows:
        case = row["case"]
        if not case or case in seen:
            raise ValueError(f"empty or duplicated case: {case!r}")
        seen.add(case)
        values = {key: float(row[key]) for key in REQUIRED if key != "case"}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"{case}: non-finite latency")
        for key in ("AE_solo_ms", "AE_GPU_wait_ms", "VLM_remaining_ms", "total_ms"):
            if values[key] < 0:
                raise ValueError(f"{case}: {key} must be nonnegative")
        components_sum = sum(values[field] for field, _, _ in COMPONENTS)
        error = components_sum - values["total_ms"]
        if abs(error) > TOLERANCE_MS:
            raise ValueError(f"{case}: native accounting error {error:.12g} ms")
        if row.get("AE_parallel_ms"):
            ae_error = (values["AE_solo_ms"] + values["PIM_interference_ms"]
                        + values["AE_GPU_wait_ms"] - float(row["AE_parallel_ms"]))
            if not math.isfinite(ae_error) or abs(ae_error) > TOLERANCE_MS:
                raise ValueError(f"{case}: AE accounting identity failed")
        if requested is None or case in requested:
            validated.append(dict(case=case, **values, accounting_error_ms=error))
    if requested is not None and requested - seen:
        raise ValueError(f"unknown requested cases: {sorted(requested - seen)}")
    if not validated:
        raise ValueError("no cases selected")
    return validated


def plot_evaluation(path, output, *, cases=None, formats=("png", "svg", "pdf"),
                    title="Native Ramulator2 latency accounting"):
    rows = read_evaluation(path, cases)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "pim-vla-native-evaluation"}):
        figure, axis = plt.subplots(figsize=(10, max(4, len(rows) * .48 + 2.2)))
        positive = [0.0] * len(rows)
        negative = [0.0] * len(rows)
        for field, _, color in COMPONENTS:
            for y, row in enumerate(rows):
                value = row[field]
                left = positive[y] if value >= 0 else negative[y]
                axis.barh(y, value, left=left, height=.6, color=color,
                          edgecolor="white" if value >= 0 else "#7A2432",
                          linewidth=.6, hatch=None if value >= 0 else "////", zorder=3)
                if value >= 0:
                    positive[y] += value
                else:
                    negative[y] += value

        span = max(max(positive), 1e-6) - min(negative)
        pad = span * .012
        for y, row in enumerate(rows):
            axis.plot(row["total_ms"], y, marker="D", color="#222222",
                      markersize=4.5, zorder=5)
            axis.text(positive[y] + pad, y, f"{row['total_ms']:.3f}",
                      va="center", fontsize=9)
            if negative[y] < 0:
                axis.annotate(f"{negative[y]:+.6g}", (negative[y], y),
                              xytext=(-5, 0), textcoords="offset points",
                              ha="right", va="center", fontsize=8, color="#9D2235")
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels([row["case"] for row in rows])
        axis.invert_yaxis()
        axis.axvline(0, color="#777777", linewidth=.7, zorder=2)
        axis.set_xlim(min(negative) - span * (.13 if min(negative) < 0 else .01),
                      max(positive) + span * .13)
        axis.set_xlabel("Milliseconds (signed additive components)")
        axis.set_title(title, loc="left", pad=14, weight="bold")
        axis.set_axisbelow(True)
        axis.xaxis.grid(True, color="#E5E5E5", linewidth=.6)
        handles = [Patch(facecolor=color, label=label) for _, label, color in COMPONENTS]
        handles.append(Line2D([], [], color="#222222", marker="D", linestyle="None",
                              markersize=4.5, label="Native total (algebraic sum)"))
        figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .075),
                      ncol=3, frameon=False, fontsize=9)
        note = ("Negative interference is retained left of zero; the diamond is the signed total."
                if min(negative) < 0 else
                "Total = AE standalone + PIM interference + AE GPU wait + VLM remaining.")
        figure.text(.06, .025, note + "  Model estimates; no latency correction.",
                    fontsize=8, color="#555555")
        figure.tight_layout(rect=(.025, .17, .99, .99))
        paths = []
        for fmt in formats:
            if fmt not in {"png", "svg", "pdf"}:
                raise ValueError(f"unsupported figure format {fmt}")
            destination = Path(f"{output}.{fmt}")
            metadata = {"CreationDate": None, "ModDate": None} if fmt == "pdf" else (
                {"Date": None} if fmt == "svg" else None)
            figure.savefig(destination, dpi=220, facecolor="white", metadata=metadata)
            paths.append(str(destination))
        plt.close(figure)
    manifest = {
        "schema_version": 1,
        "source_csv": str(Path(path).resolve()),
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "components": [field for field, _, _ in COMPONENTS],
        "negative_values_clamped": False,
        "accounting_tolerance_ms": TOLERANCE_MS,
        "rows": rows,
        "figures": paths,
    }
    Path(f"{output}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_csv", type=Path)
    parser.add_argument("--output", type=Path, help="output filename stem without extension")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--formats", nargs="+", choices=("png", "svg", "pdf"),
                        default=("png", "svg", "pdf"))
    parser.add_argument("--title", default="Native Ramulator2 latency accounting")
    args = parser.parse_args()
    output = args.output or args.evaluation_csv.with_name("latency-breakdown")
    for path in plot_evaluation(args.evaluation_csv, output, cases=args.cases,
                                formats=args.formats, title=args.title):
        print(path)


if __name__ == "__main__":
    main()
