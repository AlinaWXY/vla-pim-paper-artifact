"""Replay b–g on the measured AE DQ windows with whole-kernel GPU handoff."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, read_json, sha256, write_csv, write_json
from tools.capture_gap_traces import COMPARISON_SLICES, REPEATS
from tools.estimate_dq_gaps import (
    FUSED_COMPUTE_STAGES, FUSED_SHARED_IO, MATRIX, N, PROFILE, edge_costs, load_trace, units_for,
)

CASES = "bcdefg"


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    return merged


def usable_windows(gaps, blocked, begin, end):
    """Clip candidates without discarding the actual neighboring burst."""
    blocked_ends = [stop for _, stop in blocked]
    for a, b, left, right in gaps:
        lo, hi = max(a, begin), min(b, end)
        i = bisect_right(blocked_ends, lo)
        while i < len(blocked) and blocked[i][0] < hi:
            start, stop = blocked[i]
            if start - lo >= 2:
                yield a - begin, b - begin, lo - begin, start - begin, left, right
            lo = max(lo, stop)
            i += 1
        if hi - lo >= 2:
            yield a - begin, b - begin, lo - begin, hi - begin, left, right


def fused_epochs(path):
    epochs = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            epochs[int(row["stage_id"])].append((int(row["start_ck"]), int(row["end_ck"])))
    return {key: union(values) for key, values in epochs.items()}


def write_ca_trace(source, dest, pim_stages, latencies):
    by_channel = [[] for _ in range(N)]
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            channel, start, end = (int(row[k]) for k in ("channel", "start_ck", "end_ck"))
            if not 0 <= channel < N or not 0 <= start < end:
                raise ValueError(f"invalid CA interval in {source}: {row}")
            by_channel[channel].append((start, end))
    by_channel = [union(intervals) for intervals in by_channel]
    stops = [[b for _, b in intervals] for intervals in by_channel]
    with dest.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{latencies[0]} {latencies[1]}\n")
        for origin, end in pim_stages:
            # A data burst can start in this stage after a column command
            # issued in the preceding stage. Keep that CA prefix as well.
            first_issue = origin - max(latencies)
            for intervals, ends in zip(by_channel, stops, strict=True):
                first = bisect_right(ends, first_issue)
                last = bisect_left(intervals, (end, -1))
                stream.write(f"{last - first}\n")
                stream.writelines(f"{a - origin} {b - origin}\n" for a, b in intervals[first:last])


def prepare(case, step, layer, traces, output, units, edges, spec_digest, clock_ps, latencies):
    name = f"{case}-s{step}-l{layer}"
    path = traces / name
    rules = FUSED_SHARED_IO.get(case)
    timeline, geometry, gaps, verification = load_trace(path, step, layer, shared_io_rules=rules)
    if verification["spec_sha256"] != spec_digest or timeline["clock_ps"] != clock_ps:
        raise ValueError(f"source spec differs for {name}")
    rows = {row["event_index"]: row for row in geometry}
    epochs = fused_epochs(path / "on/dq-epochs.csv") if rules else {}
    stages, pim_stages, shared_ck = [], [], {}
    for index, event in enumerate(timeline["events"]):
        start, stop = event["start_ck"], event["end_ck"]
        duration = stop - start
        pim = event["kind"] == "pim"
        if not pim:
            stages.append((event["name"], duration, None))
            continue
        begin, end = 0, duration
        shared = rows[index].get("shared_tensor_io")
        if shared:
            a, b = shared["start_ck"], shared["end_ck"]
            if a == 0:
                begin = b
            elif b == duration:
                end = a
            else:
                raise ValueError(f"shared I/O is inside the stage: {name}/{event['name']}")
            shared_ck[event["name"]] = b - a
        if begin == end:
            continue
        blocked = [(a-start, b-start) for a, b in epochs.get(event["stage_id"], ())]
        if rules and event["name"] not in FUSED_COMPUTE_STAGES:
            blocked = [(0, duration)]
        channels = [list(usable_windows(c, blocked, begin, end)) for c in gaps[index]]
        if event["name"].startswith("q_exchange") and case in "fg" and any(channels):
            raise ValueError("fused Q exchange must reserve DQ")
        stages.append((event["name"], end - begin, channels))
        pim_stages.append((start + begin, start + end))
    ae_ck = sum(duration for _, duration, _ in stages)
    if ae_ck + sum(shared_ck.values()) != timeline["ae_end_ck"]:
        raise ValueError(f"AE shared-I/O accounting does not reconcile for {name}")

    dest = output / f"{name}.txt"
    with dest.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(" ".join(str(edges[a + b]) for a in "RWO" for b in "RWO") + "\n")
        stream.write(f"{len(units)} {len(stages)}\n")
        stream.writelines(f"{r} {w} {duration:.17g}\n" for r, w, duration in units)
        for stage, duration, channels in stages:
            stream.write(f"{json.dumps(stage)} {duration} {int(channels is not None)}\n")
            for channel in channels or ():
                stream.write(f"{len(channel)}\n")
                stream.writelines(" ".join(map(str, gap)) + "\n" for gap in channel)
    ca_path = None
    if latencies:
        ca_path = dest.with_suffix(".ca.txt")
        write_ca_trace(path / "on/ca-commands.csv", ca_path, pim_stages, latencies)
    return dest, ca_path, ae_ck, shared_ck


def run(engine, case, step, layer, traces, output, units, edges, spec_digest, clock_ps, prior, latencies,
        preserve_pace, four_batch_actab):
    source, ca_path, ae_ck, shared = prepare(case, step, layer, traces, output / "inputs", units,
                                            edges, spec_digest, clock_ps, latencies)
    command = [str(engine), str(source), str(REPEATS), str(int(case == "b"))]
    variants = {"off": None, "on": ca_path} if ca_path else {"off": None}
    source_hash = sha256(source)
    results = []
    for mode, path in variants.items():
        ca_args = [str(path)] if path else []
        if path and preserve_pace:
            ca_args.append("--preserve-pace")
        raw = json.loads(subprocess.check_output(command + ca_args,
                                                 text=True, encoding="utf-8"))
        results.append(summarize(raw, case, step, layer, output, units, clock_ps, prior,
                                 source.stem, ae_ck, shared, source_hash, mode, path, bool(ca_path),
                                 four_batch_actab))
    return results


def summarize(raw, case, step, layer, output, units, clock_ps, prior, name, ae_ck, shared,
              source_hash, mode, ca_path, comparison, four_batch_actab):
    if raw["ae_solo_ck"] != ae_ck * REPEATS:
        raise ValueError(f"unexpected AE time for {name}")
    total = [sum(unit[i] for unit in units) for i in range(2)]
    if raw["total_requests"] != total or [
        sum(raw[k][i] for k in ("pim_requests", "join_requests", "tail_requests")) for i in range(2)
    ] != total:
        raise ValueError(f"request counts do not reconcile for {name}")
    if [sum(row["requests"][i] for row in raw["stages"]) for i in range(2)] != raw["pim_requests"]:
        raise ValueError(f"PIM request counts do not reconcile for {name}")
    ck_ms = clock_ps / 1e9
    old = prior[case, step, layer]
    if not four_batch_actab and not math.isclose(
            ae_ck * REPEATS * ck_ms, float(old["accounted_ae_ms"]), abs_tol=1e-8):
        raise ValueError(f"shared I/O differs from previous accounting for {name}")
    raw = dict(raw, shared_io_ck_per_slice=shared, input_sha256=source_hash)
    if comparison:
        raw["ca_input_sha256"] = sha256(ca_path) if ca_path else None
    filename = f"{name}-ca-{mode}.json" if comparison else f"{name}.json"
    write_json(output / filename, raw)
    summary = dict(case=case, step=step, layer=layer, e2e_ms=raw["total_ck"] * ck_ms,
                   prior_e2e_ms=float(old["e2e_ms"]), ae_solo_ms=ae_ck * REPEATS * ck_ms,
                   ae_gpu_wait_ms=raw["join_ck"] * ck_ms, ae_elapsed_ms=raw["ae_ck"] * ck_ms,
                   vlm_tail_ms=raw["tail_ck"] * ck_ms, vlm_solo_ms=raw["vlm_solo_ck"] * ck_ms,
                   vlm_done_at_ms=raw["vlm_end_ck"] * ck_ms, pim_vlm_overlap_ms=raw["overlap_ck"] * ck_ms,
                   shared_io_profile_ms=sum(shared.values()) * REPEATS * ck_ms,
                   gpu_requests_in_pim=sum(raw["pim_requests"]),
                   gpu_requests_during_handoff=sum(raw["join_requests"]),
                   gpu_requests_in_tail=sum(raw["tail_requests"]),
                   gpu_requests_total=sum(total), units_at_ae_end=raw["units_at_ae_end"])
    print(f"{name} CA {mode}: E2E {summary['e2e_ms']:.6f} ms", flush=True)
    stages = [dict(case=case, step=step, layer=layer, stage=row["name"],
                   gpu_reads=row["requests"][0], gpu_writes=row["requests"][1],
                   vlm_overlap_ms=row["overlap_ck"] * ck_ms) for row in raw["stages"]]
    if comparison:
        summary["ca"] = mode
        for row in stages:
            row["ca"] = mode
    return summary, stages


def compare(results):
    rows, stages = [], []
    for (off, stage_off), (on, stage_on) in results:
        key = {k: off[k] for k in ("case", "step", "layer")}
        if any(off[k] != on[k] for k in ("ae_solo_ms", "shared_io_profile_ms", "vlm_solo_ms")):
            raise ValueError(f"CA changed the baseline timing for {key}")
        if off["case"] == "b" and any(off[k] != on[k] for k in off if k != "ca"):
            raise ValueError(f"CA changed the serial control for {key}")
        off_count, on_count = off["gpu_requests_in_pim"], on["gpu_requests_in_pim"]
        lost = off_count - on_count
        rows.append(dict(key, e2e_off_ms=off["e2e_ms"], e2e_on_ms=on["e2e_ms"],
                         e2e_delta_ms=on["e2e_ms"] - off["e2e_ms"],
                         gpu_requests_off=off_count, gpu_requests_on=on_count,
                         gpu_requests_lost=lost, gpu_requests_lost_pct=100 * lost / off_count if off_count else 0))
        stage_lost = 0
        for a, b in zip(stage_off, stage_on, strict=True):
            if a["stage"] != b["stage"]:
                raise ValueError(f"CA changed the stage sequence for {key}")
            count_off, count_on = (x["gpu_reads"] + x["gpu_writes"] for x in (a, b))
            stage_lost += count_off - count_on
            stages.append(dict(key, stage=a["stage"], gpu_requests_off=count_off,
                               gpu_requests_on=count_on, gpu_requests_lost=count_off-count_on))
        if stage_lost != lost:
            raise ValueError(f"CA per-stage request counts do not reconcile for {key}")
    return rows, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--slice", nargs=2, type=int, metavar=("STEP", "LAYER"),
                        help="replay just one captured slice")
    parser.add_argument("--compare-ca", action="store_true", help="run both modes on identical DQ input")
    parser.add_argument("--preserve-pace", action="store_true",
                        help="retain later requests' original pacing when a CA command is delayed")
    args = parser.parse_args()
    if args.preserve_pace and not args.compare_ca:
        parser.error("--preserve-pace requires --compare-ca")
    slices = (tuple(args.slice),) if args.slice else COMPARISON_SLICES
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "inputs").mkdir(exist_ok=True)
    spec_path = args.traces / "spec-k2.json"
    spec = read_json(spec_path)
    clock_ps = spec["clock_ps"]
    four_batch_actab = spec["metadata"].get("four_batch_actab", False)
    latencies = None
    if args.compare_ca:
        columns = [spec["metadata"]["commands"][cmd] for cmd in ("RD", "WR")]
        if any(c["external_ca"] != 1 or c["burst"] != 2 for c in columns):
            raise ValueError("CA replay requires one CK per column command and two CK per burst")
        latencies = [c["latency"] for c in columns]
    source_units = units_for(1.0, clock_ps)
    units = [(round(r * N), round(w * N), t) for r, w, t in source_units]
    edges = edge_costs(MATRIX)
    spec_digest = sha256(spec_path)
    with args.prior_summary.open(newline="", encoding="utf-8") as stream:
        prior = {(r["case"], int(r["step"]), int(r["layer"])): r for r in csv.DictReader(stream)}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, args.engine.resolve(), case, step, layer, args.traces, args.output,
                               units, edges, spec_digest, clock_ps, prior, latencies, args.preserve_pace,
                               four_batch_actab)
                   for step, layer in slices for case in CASES]
        results = [future.result() for future in futures]
    if args.compare_ca:
        comparison, stage_comparison = compare(results)
        write_csv(args.output / "ca-comparison.csv", comparison)
        write_csv(args.output / "stage-ca-comparison.csv", stage_comparison)
    runs = [run for case in results for run in case]
    rows = [row for row, _ in runs]
    write_csv(args.output / "summary.csv", rows)
    write_csv(args.output / "stage-summary.csv", [stage for _, stages in runs for stage in stages])
    write_json(args.output / "summary.json", dict(
        clock_ps=clock_ps, repeats=REPEATS, alpha=1.0, burst_ck=2, scheduling="same_first",
        profile_ms=math.fsum(t for _, _, t in units) * clock_ps / 1e9,
        spec_sha256=spec_digest, engine_sha256=sha256(args.engine), profile_sha256=sha256(PROFILE),
        turnaround_ck=edges, ca_column_latency_ck=latencies,
        ca_scope="VLM column command conflicts against fixed AE CA trace during PIM replay" if latencies else None,
        ca_preserve_pace=args.preserve_pace,
        **({"four_batch_actab": True} if four_batch_actab else {}),
        rows=rows))


if __name__ == "__main__":
    main()
