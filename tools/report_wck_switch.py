"""Summarize native occupancy and fixed-trace GPU read capacity during π0.5 PIM."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


LINEAR = {"qkv", "qkt", "pv", "o", "gate_up", "down"}
CK_MS = 938 / 1e9
GUARD = {
    "always_on": {"RR": 2, "RW": 2, "WR": 11, "WW": 1},
    "resync": {"RR": 19, "RW": 11, "WR": 19, "WW": 11},
}


def occupancy(path):
    data = path.read_bytes()
    raw = json.loads(data)
    assert raw["clock_ps"] == 938
    ids = {s["stage_id"] for s in raw["stages"] if s["name"] in LINEAR}
    assert Counter(s["name"] for s in raw["stages"] if s["stage_id"] in ids) == dict.fromkeys(LINEAR, 180)
    rows = [row for row in raw["bus"] if row["stage_id"] in ids]
    assert len(rows) == 16 * len(ids)
    assert len({(r["stage_id"], r["channel"]) for r in rows}) == len(rows)
    duration = sum(r["duration_cycles"] for r in rows)
    pim = sum(r["pim_read_bytes"] + r["pim_write_bytes"] for r in rows) // 16
    busy = sum(r["dq_cycles"] for r in rows)
    assert 0 <= pim <= busy <= duration
    all_bus = raw["bus"]
    assert sum(r["dq_cycles"] for r in all_bus) * 16 == sum(
        r["read_bytes"] + r["write_bytes"] for r in all_bus
    )
    return dict(
        path=str(path), raw_sha256=hashlib.sha256(data).hexdigest(),
        duration_channel_ck=duration, pim_channel_ck=pim, gpu_channel_ck=busy-pim,
        pim_linear_wall_ms=duration * CK_MS / 16,
        gpu_actual_dq_pct=100 * (busy-pim) / duration,
        dq_not_occupied_by_pim_pct=100 * (duration-pim) / duration,
        metrics=raw["metrics"],
    )


def read_capacity(root, mode, channel=0, *, exclude_io_epochs=False):
    """Pack 2 CK GPU reads into fixed rank1 gaps, respecting both switch edges."""
    trace = json.loads((root / "dq-timeline.json").read_bytes())
    assert trace["clock_ps"] == 938
    windows = [(r["start_ck"], r["end_ck"]) for r in trace["events"]
               if r["kind"] == "pim" and r["name"] in LINEAR]
    assert len(windows) == 1080
    elapsed = sum(end-start for start,end in windows)
    horizon = windows[-1][1]
    if exclude_io_epochs:
        epochs = {}
        with (root / "dq-epochs.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if int(row["channel"]) == channel:
                    bounds = (int(row["start_ck"]), int(row["end_ck"]))
                    epochs.setdefault(int(row["stage_id"]), []).append(bounds)
        windows = []
        for event in trace["events"]:
            if event["kind"] != "pim" or event["name"] not in LINEAR:
                continue
            start, end = event["start_ck"], event["end_ck"]
            for lo, hi in sorted(epochs.get(event["stage_id"], [])):
                if start < min(lo, end):
                    windows.append((start, min(lo, end)))
                start = max(start, hi)
            if start < end:
                windows.append((start, end))
    guards = GUARD[mode]
    window = 0
    read_ck = 0
    prior_end = 0
    prior_direction = ""

    def pack(start, end):
        nonlocal window, read_ck
        while window < len(windows):
            lo, hi = windows[window]
            if lo >= end:
                break
            read_ck += 2 * (max(0, min(end, hi) - max(start, lo)) // 2)
            if hi > end:
                break
            window += 1

    with (root / "dq-bursts.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["channel"]) != channel:
                continue
            start, end = int(row["start_ck"]), int(row["end_ck"])
            direction = row["direction"]
            assert start >= prior_end and end-start == 2
            setup = guards[prior_direction + "R"] if prior_direction else 0
            pack(prior_end + setup, start - guards["R" + direction])
            prior_end, prior_direction = end, direction
    assert prior_direction, "missing PIM DQ trace"
    pack(prior_end + guards[prior_direction + "R"], horizon)
    return dict(channel=channel, timing=mode, read_capacity_ck=read_ck,
                exclude_io_epochs=exclude_io_epochs, duration_ck=elapsed,
                gpu_read_dq_capacity_pct=100 * read_ck / elapsed)


def summarize_switches(path):
    total = {direction: Counter() for direction in GUARD["resync"]}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            direction = row["direction"]
            counts = {k: int(row[k]) for k in ("switches", "dq_gap_ck", "required_resync_gap_ck")}
            assert counts["required_resync_gap_ck"] == GUARD["resync"][direction] * counts["switches"]
            assert counts["dq_gap_ck"] >= counts["required_resync_gap_ck"]
            total[direction].update(counts)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical", type=Path, help="2026-11-16 results directory")
    parser.add_argument("--case", action="append", nargs=2, default=[], metavar=("CASE", "RUN_DIR"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = {case: args.historical / f"combined-{case}" / "repeat-1" for case in "cg"}
    result = {"always_on_c_g": {case: occupancy(root / "raw.json") for case,root in old.items()},
              "fixed_prior_pim_trace_channel_0": {
                  case: {mode: read_capacity(root,mode) for mode in GUARD} for case,root in old.items()},
              "fixed_prior_pim_trace_excluding_io_epochs_channel_0": {
                  case: {mode: read_capacity(root,mode,exclude_io_epochs=True) for mode in GUARD}
                  for case,root in old.items()},
              "resync_runs": {}}
    for case, path in args.case:
        root = Path(path)
        row = occupancy(root / "raw.json")
        row["cross_rank_switches"] = summarize_switches(root / "wck-switches.csv")
        if (root / "dq-timeline.json").is_file():
            row["timing_only_gpu_read_capacity"] = read_capacity(root, "resync")
        result["resync_runs"][case] = row
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
