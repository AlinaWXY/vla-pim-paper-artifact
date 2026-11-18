"""Audit native bus traces and plot PIM-only DQ and CA occupancy."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import read_json, sha256, write_csv
from tools.capture_gap_traces import COMPARISON_SLICES, REPEATS


def collect(directory, bin_us):
    on = directory / "on"
    verification = read_json(directory / "verification.json")
    for name in ("raw.json", "ca-commands.csv", "dq-bursts.csv", "dq-timeline.json"):
        expected = (verification["raw_sha256"] if name == "raw.json"
                    else verification["trace_sha256"][name])
        if sha256(on / name) != expected:
            raise ValueError(f"trace checksum mismatch: {on / name}")
    timeline = read_json(on / "dq-timeline.json")
    raw = read_json(on / "raw.json")
    count, clock_ps, end = (timeline[k] for k in ("channels", "clock_ps", "ae_end_ck"))
    width = max(1, round(bin_us * 1e6 / clock_ps))
    bins = {name: [0] * ((end + width - 1) // width) for name in ("dq", "pim_ca", "refresh_ca")}
    events = timeline["events"]
    if (not events or events[0]["start_ck"] != 0 or events[-1]["end_ck"] != end
            or any(a["end_ck"] != b["start_ck"] for a, b in zip(events, events[1:]))):
        raise ValueError(f"non-contiguous AE timeline: {directory}")
    event_ends = [event["end_ck"] for event in events]
    stages = {event["stage_id"]: event for event in events if event["kind"] == "pim"}
    counters = defaultdict(Counter)

    def account(start, stop, channel, metric):
        for i in range(start // width, (stop - 1) // width + 1):
            bins[metric][i] += min(stop, (i + 1) * width) - max(start, i * width)
        i = bisect_right(event_ends, start)
        while i < len(events) and events[i]["start_ck"] < stop:
            event = events[i]
            duration = min(stop, event["end_ck"]) - max(start, event["start_ck"])
            if duration > 0 and event["kind"] == "pim":
                counters[event["stage_id"], channel][metric] += duration
            i += 1

    for filename in ("ca-commands.csv", "dq-bursts.csv"):
        last_end = [0] * count
        with (on / filename).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                channel, start, stop = (int(row[k]) for k in ("channel", "start_ck", "end_ck"))
                if not (0 <= channel < count and last_end[channel] <= start < stop):
                    raise ValueError(f"overlapping or malformed bus interval: {filename}, {row}")
                last_end[channel] = stop
                if filename == "ca-commands.csv":
                    if row["rank"] not in ("0", "1"):
                        raise ValueError(f"unexpected rank: {row}")
                    metric = "pim_ca" if row["rank"] == "1" else "refresh_ca"
                else:
                    metric = "dq"
                if start < end:
                    account(start, min(stop, end), channel, metric)

    for row in raw["bus"]:
        sid, channel = row["stage_id"], row["channel"]
        if sid not in stages:
            continue
        measured = counters[sid, channel]
        if (measured["pim_ca"] + measured["refresh_ca"] != row["ca_cycles"]
                or measured["dq"] != row["dq_cycles"]):
            raise ValueError(f"bus counters differ: {directory}, stage {sid}, channel {channel}")
    stage_rows = []
    for sid, event in stages.items():
        duration = event["end_ck"] - event["start_ck"]
        bus = sum((counters[sid, c] for c in range(count)), Counter())
        stage_rows.append(dict(stage_id=sid, stage=event["name"], duration_ck=duration,
                               capacity_channel_ck=count * duration, dq_channel_ck=bus["dq"],
                               rank1_ca_channel_ck=bus["pim_ca"],
                               rank0_refresh_ca_channel_ck=bus["refresh_ca"]))
    window_rows = []
    for index, start in enumerate(range(0, end, width)):
        stop = min(start + width, end)
        capacity = count * (stop - start)
        dq = bins["dq"][index]
        rank1 = bins["pim_ca"][index]
        rank0 = bins["refresh_ca"][index]
        if dq > capacity or rank1 + rank0 > capacity:
            raise ValueError(f"bus capacity exceeded: {directory}, bin {index}")
        window_rows.append(dict(start_us=start * clock_ps / 1e6, end_us=stop * clock_ps / 1e6,
                                dq_channel_ck=dq, rank1_ca_channel_ck=rank1,
                                rank0_refresh_ca_channel_ck=rank0, capacity_channel_ck=capacity,
                                dq_pct=100 * dq / capacity, ca_pct=100 * (rank1 + rank0) / capacity,
                                rank1_ca_pct=100 * rank1 / capacity))
    capacity = sum(row["capacity_channel_ck"] for row in stage_rows)
    dq = sum(row["dq_channel_ck"] for row in stage_rows)
    rank1 = sum(row["rank1_ca_channel_ck"] for row in stage_rows)
    rank0 = sum(row["rank0_refresh_ca_channel_ck"] for row in stage_rows)
    summary = dict(slice_ae_us=end * clock_ps / 1e6, ae_180x_ms=end * clock_ps * REPEATS / 1e9,
                   pim_stage_us=capacity / count * clock_ps / 1e6, capacity_channel_ck=capacity,
                   dq_channel_ck=dq, rank1_ca_channel_ck=rank1, rank0_refresh_ca_channel_ck=rank0,
                   dq_pct=100 * dq / capacity, rank1_ca_pct=100 * rank1 / capacity,
                   ca_pct=100 * (rank1 + rank0) / capacity)
    return summary, stage_rows, window_rows, timeline


def draw(traces, step, layer, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(traces), 1, figsize=(13, 2.1 * len(traces) + 1.4),
                             sharex=True, sharey=True, layout="constrained", squeeze=False)
    for index, (ax, (summary, windows, timeline)) in enumerate(zip(axes[:, 0], traces, strict=True)):
        edges = [row["start_us"] for row in windows] + [windows[-1]["end_us"]]
        for event in timeline["events"]:
            if event["kind"] == "gpu":
                clock_us = timeline["clock_ps"] / 1e6
                ax.axvspan(event["start_ck"] * clock_us, event["end_ck"] * clock_us,
                           color="#e6e9ee", zorder=0)
        ax.stairs([row["ca_pct"] for row in windows], edges, color="#184f78",
                  linewidth=0.9, label="CA (incl. rank 0 refresh)")
        ax.stairs([row["dq_pct"] for row in windows], edges, color="#ba4f17",
                  linewidth=0.9, label="DQ")
        ax.axvline(summary["slice_ae_us"], color="#303641", linewidth=0.8, linestyle="--")
        ax.set_title(f'{summary["case"]}   |   AE {summary["slice_ae_us"]:.2f} µs'
                     f'   |   PIM-stage DQ {summary["dq_pct"]:.2f}% / CA {summary["ca_pct"]:.2f}%',
                     loc="left", fontsize=10)
        ax.set(ylim=(0, 104), yticks=(0, 50, 100))
        ax.grid(axis="y", alpha=0.25)
        if index == 0:
            ax.legend(loc="upper right", framealpha=1)
    axes[-1, 0].set_xlabel("AE inference time in sampled slice (µs)  ·  shaded: AE GPU blocks")
    fig.supylabel("Occupied channel-cycles (%)")
    fig.suptitle(f"{title}  ·  step {step} / layer {layer}")
    stem = output / f"pim-bus-s{step}-l{layer}"
    for ext in ("png", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=165)
    plt.close(fig)


def draw_repeated(traces, step, layer, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(traces), 1, figsize=(13, 2.0 * len(traces) + 1.2),
                             sharex=True, sharey=True, layout="constrained", squeeze=False)
    for ax, (summary, windows, _) in zip(axes[:, 0], traces, strict=True):
        source_edges = [row["start_us"] for row in windows] + [windows[-1]["end_us"]]
        slice_us = source_edges[-1]
        bin_count = 1600
        target_edges = [i * slice_us * REPEATS / bin_count for i in range(bin_count + 1)]
        target_ms = [edge / 1000 for edge in target_edges]
        for metric, color, label in (("ca_pct", "#184f78", "CA"), ("dq_pct", "#ba4f17", "DQ")):
            rates = [row[metric] for row in windows]
            integral = [0.0]
            for i, rate in enumerate(rates):
                integral.append(integral[-1] + rate * (source_edges[i + 1] - source_edges[i]))

            def accumulated(time):
                repeats = int(time / slice_us)
                phase = max(0.0, min(time - repeats * slice_us, slice_us))
                i = min(bisect_right(source_edges, phase) - 1, len(rates) - 1)
                return repeats * integral[-1] + integral[i] + rates[i] * (phase - source_edges[i])

            cumulative = [accumulated(edge) for edge in target_edges]
            average = [(b - a) / (t1 - t0) for a, b, t0, t1 in
                       zip(cumulative, cumulative[1:], target_edges, target_edges[1:])]
            ax.stairs(average, target_ms, color=color, linewidth=0.8, label=label)
        ax.axvline(summary["ae_180x_ms"], color="#303641", linewidth=0.8, linestyle="--")
        ax.set_title(f'{summary["case"]}   |   projected AE {summary["ae_180x_ms"]:.2f} ms'
                     f'   |   bins {target_edges[1]:.0f} µs', loc="left", fontsize=10)
        ax.set(ylim=(0, 104), yticks=(0, 50, 100))
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="upper right", framealpha=1)
    axes[-1, 0].set_xlabel("Projected AE inference time (ms)  ·  matched slice repeated 180×")
    fig.supylabel("Occupied channel-cycles (%)")
    fig.suptitle(f"{title}  ·  repeated step {step} / layer {layer}")
    stem = output / f"pim-bus-180x-s{step}-l{layer}"
    for ext in ("png", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--bin-us", type=float, default=0.5)
    parser.add_argument("--cases", choices=list("bcdefg"), nargs="+", default=list("cdefg"))
    args = parser.parse_args()
    if not args.bin_us > 0:
        parser.error("--bin-us must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for step, layer in COMPARISON_SLICES:
        traces = []
        for case in args.cases:
            label = f"{case}-s{step}-l{layer}"
            summary, stages, windows, timeline = collect(args.traces / label, args.bin_us)
            summary = dict(case=case, step=step, layer=layer, **summary)
            summaries.append(summary)
            traces.append((summary, windows, timeline))
            write_csv(args.output / f"{label}-stages.csv", stages)
            write_csv(args.output / f"{label}-timeline.csv", windows)
        draw(traces, step, layer, args.output, args.title)
        draw_repeated(traces, step, layer, args.output, args.title)
    write_csv(args.output / "pim-bus-summary.csv", summaries)


if __name__ == "__main__":
    main()
