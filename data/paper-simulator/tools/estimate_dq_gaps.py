"""Check native AE traces and report offline DQ-gap interference for c/d/e."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import read_json, sha256, write_json
from tools.audit_fused_kernels import audit_timeline
from tools.capture_gap_traces import COMPARISON_SLICES, REPEATS, SLICES

N = 16
POLICIES = ("same_first", "best", "random")
ALPHAS = (1.0, 0.5)
MATRIX = ROOT / "reference/spec/command-pair-matrix.json"
PROFILE = ROOT / "input-data/vlm.json"
FUSED = ROOT / "reference/fused-kernel-continuous-e2e/summary.json"
SCALED_ABLATION = ROOT / "results/ablation-scaled-w128/ablation.json"
FUSED_COMPUTE_STAGES = {"qkv", "qkt", "pv", "o", "gate_up", "down", "rope"}
FUSED_TRANSFER_STAGES = {"rope_export", "rope_import", "q_exchange_export", "q_exchange_import"}
# Match the early shared-tensor phase rules to the current fused programs.
# Inputs are at the stage head and the score consumer is at the stage tail.
FUSED_INPUT_AND_SCORE_IO = {
    "qkv": ("qkv_input", "WR", "head", "projection_gpu_input_stage"),
    "qkt": ("score_export", "RD", "tail", "qk_scores_dram_export"),
    "pv": ("pv_input", "WR", "head", "pv_gpu_input_stage"),
    "gate_up": ("gate_up_input", "WR", "head", "gate_up_gpu_input_stage"),
    "down": ("down_input", "WR", "head", "down_gpu_input_stage"),
}
FUSED_SHARED_IO = {
    "f": dict(FUSED_INPUT_AND_SCORE_IO, rope_export=("rope_export", "RD", "whole", "gpu_rope_export"),
              rope_import=("rope_import", "WR", "whole", "gpu_rope_import")),
    "g": FUSED_INPUT_AND_SCORE_IO,
}
PAIR_REFERENCE = dict(RRR=(6, 4), RRW=(6, 4), RWR=(15, 13), RWW=(5, 3),
                      WRR=(15, 13), WRW=(15, 13), WWR=(14, 12), WWW=(4, 2))


def edge_costs(path):
    commands = {"RD": "R", "WR": "W"}
    edges = {}
    for row in read_json(path):
        a, b = row["first"], row["second"]
        if row["relation"] == "different_rank" and a in commands and b in commands:
            pair = commands[a] + commands[b]
            if pair in edges:
                raise ValueError(f"duplicate edge {pair} in {path}")
            edges[pair] = row["dq_gap_ck"]
    if edges != dict(RR=2, RW=2, WR=11, WW=1):
        raise ValueError(f"unexpected cross-rank DQ edges: {edges}")
    edges.update({a + b: 0 for a in "RWO" for b in "RWO" if "O" in a + b})
    for trio, (required, deduction) in PAIR_REFERENCE.items():
        measured = edges[trio[:2]] + edges[trio[1:]]
        if (measured + 2, measured) != (required, deduction):
            raise ValueError(f"gap lookup failed for {trio}")
    return edges


def units_for(alpha, clock_ps):
    source = read_json(PROFILE)
    rows = sorted(source["units"], key=lambda r: r["ordinal"])
    if [r["ordinal"] for r in rows] != list(range(2357)):
        raise ValueError("expected 2357 contiguous VLM ordinals")
    if abs(math.fsum(r["us"] for r in rows) - source["measured_duration_sum_us"]) > 1e-6:
        raise ValueError("measured VLM profile sum differs from its units")
    units = []
    for r in rows:
        reads = r["read_sectors"]
        writes = r["write_sectors"]
        if r["write_is_proxy"]:
            writes = math.ceil(alpha * writes)
        if min(reads, writes) < 0 or reads + writes == 0 or r["us"] <= 0:
            raise ValueError(f"invalid VLM unit {r['ordinal']}")
        units.append((reads / N, writes / N, r["us"] * 1e6 / clock_ps))
    return units


def interval_union_ck(intervals):
    end = total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def load_trace(directory, step, layer, *, shared_io_rules=None):
    shared_io_rules = shared_io_rules or {}
    on, off = (directory / mode for mode in ("on", "off"))
    verification = read_json(directory / "verification.json")
    if ((on / "raw.json").read_bytes() != (off / "raw.json").read_bytes()
            or sha256(on / "raw.json") != verification["raw_sha256"]):
        raise ValueError(f"raw.json verification failed in {directory}")
    for name, digest in verification["trace_sha256"].items():
        if sha256(on / name) != digest:
            raise ValueError(f"trace checksum mismatch in {directory}: {name}")
    timeline = read_json(on / "dq-timeline.json")
    raw = read_json(on / "raw.json")
    events = timeline["events"]
    if timeline["channels"] != N or raw["clock_ps"] != timeline["clock_ps"]:
        raise ValueError(f"unexpected channel count or clock in {directory}")
    last = 0
    for event in events:
        if ((event["step"], event["layer"], event["start_ck"]) != (step, layer, last)
                or event["end_ck"] < last):
            raise ValueError(f"broken timeline in {directory}: {event}")
        last = event["end_ck"]
    if (timeline["ae_end_ck"] != last
            or abs(last * timeline["clock_ps"] / 1e6 - raw["metrics"]["AE_elapsed_us"]) > 1e-7):
        raise ValueError(f"AE elapsed time differs from trace in {directory}")
    pim = {e["stage_id"]: e for e in events if e["kind"] == "pim"}
    bus = {(r["stage_id"], r["channel"]): r for r in raw["bus"]}
    bursts, epochs = defaultdict(list), defaultdict(list)
    with (on / "dq-bursts.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sid, channel = int(row["stage_id"]), int(row["channel"])
            if (row["stage"] != pim[sid]["name"] or not 0 <= channel < N
                    or row["direction"] not in ("R", "W") or row["fused_io_epoch"] not in ("0", "1")):
                raise ValueError(f"invalid burst in {directory}: {row}")
            bursts[sid, channel].append((int(row["start_ck"]), int(row["end_ck"]),
                                         row["direction"], row["command"], row["fused_io_epoch"] == "1"))
    with (on / "dq-epochs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sid, channel = int(row["stage_id"]), int(row["channel"])
            if sid not in pim or not 0 <= channel < N:
                raise ValueError(f"invalid epoch in {directory}: {row}")
            epochs[sid, channel].append((int(row["start_ck"]), int(row["end_ck"])))
    geometry, stage_gaps = [], {}
    for index, event in enumerate(events):
        if event["kind"] != "pim":
            continue
        start, end, sid = (event[k] for k in ("start_ck", "end_ck", "stage_id"))
        duration = end - start
        pairs, lengths = Counter(), Counter()
        dq = 0
        blocked_count = blocked_full = blocked_ck = 0
        channels = []
        shared_rule = shared_io_rules.get(event["name"])
        shared_span = None
        for channel in range(N):
            key = sid, channel
            intervals = epochs[key]
            last = start
            for a, b in intervals:
                if not last <= a < b <= end:
                    raise ValueError(f"overlapping or out-of-stage epochs: {directory}, {key}")
                last = b
            epoch_index = 0
            epoch_commands = [set() for _ in intervals] if shared_rule else []

            def overlap(a, b):
                nonlocal epoch_index
                while epoch_index < len(intervals) and intervals[epoch_index][1] <= a:
                    epoch_index += 1
                used = 0
                for i in range(epoch_index, len(intervals)):
                    left, right = intervals[i]
                    if left >= b:
                        break
                    used += min(right, b) - max(left, a)
                return used

            gaps = []

            def add_gap(a, b, left, right):
                nonlocal blocked_count, blocked_full, blocked_ck
                if b <= a:
                    return
                excluded = overlap(a, b)
                blocked_ck += excluded
                blocked_count += excluded > 0
                blocked_full += excluded == b - a
                pairs[left + right] += 1
                lengths[left + right] += b - a
                gaps.append((a - start, b - start, "RWO".index(left), "RWO".index(right)))

            last, left = start, "O"
            occupied = Counter()
            byte_counts = Counter()
            for a, b, direction, command, fused in bursts[key]:
                if not last <= a < b <= end:
                    raise ValueError(f"overlapping or out-of-stage bursts: {directory}, {key}")
                add_gap(last, a, left, direction)
                if overlap(a, b) != (b - a if fused else 0):
                    raise ValueError(f"burst and I/O epoch disagree: {directory}, {key}")
                if shared_rule and fused:
                    left_ck, right_ck = intervals[epoch_index]
                    if not left_ck <= a < b <= right_ck:
                        raise ValueError(f"burst crosses I/O epoch boundaries: {directory}, {key}")
                    epoch_commands[epoch_index].add(command)
                occupied[direction] += b - a
                byte_counts[direction] += 32
                last, left = b, direction
            add_gap(last, end, left, "O")
            r = bus[key]
            if (occupied.total(), byte_counts["R"], byte_counts["W"]) != (
                    r["dq_cycles"], r["pim_read_bytes"], r["pim_write_bytes"]):
                raise ValueError(f"burst and native DQ counters disagree: {directory}, {key}")
            dq += occupied.total()
            channels.append(gaps)
            if shared_rule:
                _, command, boundary, _ = shared_rule
                matches = [(a, b) for (a, b), commands in zip(intervals, epoch_commands, strict=True)
                           if commands == {command}]
                if len(matches) != 1:
                    raise ValueError(f"cannot identify a shared-tensor epoch: {directory}, {key}")
                candidate = matches[0]
                if ((boundary in ("head", "whole") and candidate[0] != start)
                        or (boundary in ("tail", "whole") and candidate[1] != end)
                        or (shared_span is not None and candidate != shared_span)):
                    raise ValueError(f"shared-tensor epoch not aligned across the stage/channels: {directory}, {key}")
                shared_span = candidate
        if dq + lengths.total() != N * duration:
            raise ValueError(f"stage DQ plus idle differs from duration: {directory}, {sid}")
        stage_gaps[index] = channels
        epoch_union_ck = interval_union_ck(
            interval for channel in range(N) for interval in epochs[sid, channel])
        geometry.append(dict(event_index=index, name=event["name"], t_ck=duration,
                             dq_channel_ck=dq, dq_over_t=dq / (N * duration) if duration else 0,
                             gap_count=pairs.total(), gap_channel_ck=lengths.total(),
                             pairs={k: dict(count=pairs[k], length_ck=lengths[k]) for k in pairs},
                             epoch_union_ck=epoch_union_ck,
                             epoch_touching_gap_count=blocked_count, epoch_internal_gap_count=blocked_full,
                             epoch_blocked_idle_channel_ck=blocked_ck,
                             idle_outside_epoch_channel_ck=lengths.total() - blocked_ck))
        if shared_span is not None:
            epoch, command, _, historical_phase = shared_rule
            a, b = shared_span
            geometry[-1]["shared_tensor_io"] = dict(epoch=epoch, command=command,
                                                   historical_phase=historical_phase,
                                                   start_ck=a - start, end_ck=b - start)
    return timeline, geometry, stage_gaps, verification


def write_replay_input(path, clock_ps, edges, units, events=(), gaps=None):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{clock_ps}\n")
        handle.write(" ".join(str(edges[a + b]) for a in "RWO" for b in "RWO") + "\n")
        handle.write(f"{len(units)} {len(events)}\n")
        handle.writelines(f"{r:.17g} {w:.17g} {d:.17g}\n" for r, w, d in units)
        for index, event in enumerate(events):
            duration = event["end_ck"] - event["start_ck"]
            handle.write(f"{json.dumps(event['name'])} {duration} {int(index in gaps)}\n")
            for channel in gaps.get(index, ()):
                handle.write(f"{len(channel)}\n")
                handle.writelines(f"{a} {b} {left} {right}\n" for a, b, left, right in channel)


def replay(engine, path, policy, seed, detail=None, detail_repeats=1):
    command = [str(engine), str(path), str(REPEATS), str(POLICIES.index(policy)), str(seed)]
    if detail:
        command.extend((str(detail), str(detail_repeats)))
    return json.loads(subprocess.check_output(command, text=True, encoding="utf-8"))


def expectation(values):
    present = [v for v in values if v is not None]
    sample = present[0]
    if isinstance(sample, dict):
        return {k: expectation([v.get(k) if v is not None else None for v in values])
                for k in sorted(set().union(*present))}
    if isinstance(sample, list):
        return [expectation([v[i] if v is not None else None for v in values]) for i in range(len(sample))]
    if isinstance(sample, str):
        if any(v != sample for v in present):
            raise ValueError("random samples have different stage identities")
        return sample
    return statistics.fmean(0 if v is None else v for v in values)


def add_units(result, ck_ms, b_ms):
    for name in ("ae", "tail", "total", "vlm_loss"):
        result[f"{name}_ms"] = result[f"{name}_ck"] * ck_ms
    result["speedup_vs_b"] = b_ms / result["total_ms"]
    return result


def validate_replay(result, geometry, ae_slice_ck):
    if result["ae_ck"] != ae_slice_ck * REPEATS:
        raise ValueError("replay AE duration differs from the fixed solo timeline")
    loss = 0
    for row, trace in zip(result["stages"], geometry, strict=True):
        active = row["active_repetitions"]
        if (row["event_index"] != trace["event_index"]
                or row["count"] != trace["gap_count"] * active
                or row["length_ck"] != trace["gap_channel_ck"] * active
                or not 0 <= row["sum_u_channel_ck"] <= row["length_ck"]):
            raise ValueError(f"replay stage coverage/capacity differs for {trace['name']}")
        for key in ("count", "length_ck", "infeasible_count", "infeasible_ck", "sum_u_channel_ck"):
            if sum(group[key] for group in row["combinations"].values()) != row[key]:
                raise ValueError(f"replay combination accounting differs for {trace['name']}: {key}")
        loss += trace["t_ck"] * active - row["sum_u_ck"]
    if loss != result["vlm_loss_ck"] or result["total_ck"] != result["ae_ck"] + result["tail_ck"]:
        raise ValueError("replay loss/latency accounting differs")


@lru_cache(maxsize=1)
def load_fused_program(path):
    return read_json(path)


def fused_vlm_in_ae_by_phase(raw, ae_end):
    kernels = [(row["start_ck"], row["end_ck"]) for row in raw["kernels"]]
    ends = [end for _, end in kernels]
    phases = raw["phases"] + [dict(row, phase="ae_gpu") for row in raw["ae_gpu"]]
    active = Counter()
    for row in phases:
        start, end = row["start_ck"], row["end_ck"]
        phase = row["phase"]
        active[phase] += 0
        index = bisect_right(ends, start)
        while index < len(kernels) and kernels[index][0] < end:
            kernel_start, kernel_end = kernels[index]
            active[phase] += min(end, kernel_end) - max(start, kernel_start)
            index += 1
    expected = sum(max(0, min(ae_end, end) - start) for start, end in kernels)
    if sum(active.values()) != expected:
        raise ValueError("fused phase VLM time differs from VLM time before AE completion")
    return active


def fused_result(path, case, ck_ms, expected_stages, b_ms):
    raw = read_json(path)
    metrics = raw["metrics"]
    if (raw["window_policy"] != "continuous_pim_descriptors" or len(raw["kernels"]) != 2357
            or len(raw["stages"]) != expected_stages
            or metrics["PIM_descriptor_continuations"] != {"f": 180, "g": 360}[case]
            or metrics["gpu_read_bytes"] or metrics["gpu_write_bytes"]):
        raise ValueError(f"unexpected fused-window results: {path}")
    cfg = read_json(path.parent / "input-config.json")
    ledger = read_json(cfg["calibrated_vlm_path"])
    audit = audit_timeline(raw, read_json(cfg["ae_schedule_path"]), ledger,
                           load_fused_program(cfg["program_path"]))
    vlm_in_ae = fused_vlm_in_ae_by_phase(raw, audit["AE_cycles"])
    total = metrics["total_us"] / 1000
    return dict(total_ms=total, AE_elapsed_ms=metrics["AE_elapsed_us"] / 1000,
                AE_VLM_join_ms=metrics["AE_VLM_join_us"] / 1000,
                PIM_native_ms=audit["native_PIM_cycles"] * ck_ms,
                AE_GPU_compute_ms=audit["AE_GPU_cycles"] * ck_ms,
                VLM_tail_ms=metrics["VLM_remaining_us"] / 1000,
                VLM_calibrated_sum_ms=math.fsum(k["duration_cycles"] for k in raw["kernels"]) * ck_ms,
                VLM_in_AE_ms=sum(vlm_in_ae.values()) * ck_ms,
                VLM_in_AE_by_phase_ms={phase: ck * ck_ms for phase, ck in vlm_in_ae.items()},
                PIM_descriptor_continuations=metrics["PIM_descriptor_continuations"],
                raw_sha256=sha256(path), calibrated_vlm_sha256=sha256(cfg["calibrated_vlm_path"]),
                timeline_audit_passed=True, speedup_vs_b=b_ms / total,
                source="本轮原生融合窗口")


def fused_summary(traces, ck_ms, b_ms):
    archive = read_json(FUSED)
    if not archive["passed"]:
        raise ValueError("archived fused-window audit failed")
    rows = {}
    for case, reference in zip("fg", archive["results"], strict=True):
        path = traces / f"{case}-fused/raw.json"
        if path.is_file():
            row = fused_result(path, case, ck_ms, reference["PIM_stages"], b_ms)
            row["archived_total_ms"] = reference["total_ms"]
            row["delta_from_archive_ms"] = row["total_ms"] - reference["total_ms"]
        else:
            row = dict(reference, source=f"已验证归档 `{archive['source_commit'][:7]}`")
        row["speedup_vs_b"] = b_ms / row["total_ms"]
        rows[case] = row
    return rows


def fused_ae_solo_overlap(timeline, geometry, vlm_ms, b_ms, *, shared_io_rules=None):
    shared_io_rules = shared_io_rules or {}
    stages = []
    expected = set(shared_io_rules)
    if {row["name"] for row in geometry if "shared_tensor_io" in row} != expected:
        raise ValueError("fused AE solo shared-tensor phase coverage differs")
    if "rope_export" in expected:
        positions = {row["name"]: row["event_index"] for row in geometry}
        rope_events = timeline["events"][positions["rope_export"] + 1:positions["rope_import"]]
        if not any(row["kind"] == "gpu" and row["name"] == "rope_apply" for row in rope_events):
            raise ValueError("shared Q/K I/O requires retained GPU RoPE between export and import")
    for row in geometry:
        name, duration, epoch = (row[key] for key in ("name", "t_ck", "epoch_union_ck"))
        if not 0 <= epoch <= duration or name not in FUSED_COMPUTE_STAGES | FUSED_TRANSFER_STAGES:
            raise ValueError(f"unexpected fused AE solo stage: {name}")
        shared = row.get("shared_tensor_io")
        deduplicated = shared["end_ck"] - shared["start_ck"] if shared else 0
        if not 0 <= deduplicated <= epoch:
            raise ValueError(f"invalid shared-tensor time for {name}")
        accounted = duration - deduplicated
        retained_io = epoch - deduplicated
        overlap = accounted - retained_io if name in FUSED_COMPUTE_STAGES else 0
        stages.append(dict(name=name, t_ck=duration, io_epoch_union_ck=epoch,
                           deduplicated_io_ck=deduplicated, accounted_ck=accounted, overlap_ck=overlap))

    ae_ck = timeline["ae_end_ck"]
    ae_gpu_ck = sum(event["end_ck"] - event["start_ck"]
                    for event in timeline["events"] if event["kind"] == "gpu")
    if ae_ck != ae_gpu_ck + sum(row["t_ck"] for row in stages):
        raise ValueError("fused AE solo stages and GPU blocks do not cover the timeline")
    scale = timeline["clock_ps"] * REPEATS / 1e9
    deduplicated_ck = sum(row["deduplicated_io_ck"] for row in stages)
    q_exchange_ck = sum(row["accounted_ck"] for row in stages
                        if row["name"] in ("q_exchange_export", "q_exchange_import"))
    ae_ms = (ae_ck - deduplicated_ck) * scale
    epoch_free_ms = sum(row["overlap_ck"] for row in stages) * scale
    overlap_ms = min(vlm_ms, epoch_free_ms)
    total_ms = ae_ms + vlm_ms - overlap_ms
    return dict(ae_solo_ms=ae_ms, ae_trace_ms=ae_ck * scale, ae_gpu_ms=ae_gpu_ck * scale,
                ae_accounting="legacy_shared_tensor_io" if shared_io_rules else "explicit_native_io",
                deduplicated_io_ms=deduplicated_ck * scale, vlm_profile_ms=vlm_ms,
                q_exchange_ms=q_exchange_ck * scale,
                epoch_free_compute_stage_ms=epoch_free_ms, overlap_ms=overlap_ms,
                vlm_remaining_ms=vlm_ms - overlap_ms, total_ms=total_ms,
                explicit_native_io_total_ms=ae_ck * scale + vlm_ms - overlap_ms,
                repetitions=REPEATS, b_ms=b_ms, speedup_vs_b=b_ms / total_ms, stages=stages)


def g_model_comparison(estimate):
    archive = read_json(SCALED_ABLATION)
    old = next(row for row in archive["results"] if row["case"] == "g")
    old_parts = {key: old[key] for key in ("ae_solo_ms", "pim_stage_delta_ms", "ae_gpu_wait_ms", "vlm_tail_ms")}
    old_parts["ae_elapsed_ms"] = math.fsum(old_parts[key] for key in
                                          ("ae_solo_ms", "pim_stage_delta_ms", "ae_gpu_wait_ms"))
    new_parts = dict(ae_solo_ms=estimate["ae_solo_ms"], pim_stage_delta_ms=0, ae_gpu_wait_ms=0,
                     ae_elapsed_ms=estimate["ae_solo_ms"], vlm_tail_ms=estimate["vlm_remaining_ms"])
    return dict(closed_loop_ms=old_parts, overlap_estimate_ms=new_parts,
                closed_loop_total_ms=old["total_ms"], overlap_estimate_total_ms=estimate["total_ms"],
                closed_loop_vlm_solo_ms=archive["calibration"]["total_us"] / 1000,
                estimate_vlm_profile_ms=estimate["vlm_profile_ms"],
                closed_loop_raw_sha256=old["raw_sha256"])


def same_slice_comparison(data):
    profile_ms = data["solo"]["1.0"]["profile_ms"]
    candidates = []
    for step, layer in COMPARISON_SLICES:
        suffix = f"s{step}-l{layer}"
        rows = {}
        for case in "cdefg":
            label = f"{case}-{suffix}"
            if case in "cde":
                replay = data["replay"][label]["1.0"]["same_first"]
                ae, tail, total = (replay[key] for key in ("ae_ms", "tail_ms", "total_ms"))
                overlap = profile_ms - tail
            else:
                replay = data["fused_ae_solo_overlap"][label]
                ae, tail, total, overlap = (replay[key] for key in
                                             ("ae_solo_ms", "vlm_remaining_ms", "total_ms", "overlap_ms"))
            if (not math.isclose(ae + tail, total, rel_tol=0, abs_tol=1e-9)
                    or not math.isclose(overlap + tail, profile_ms, rel_tol=0, abs_tol=1e-9)):
                raise ValueError(f"same-slice AE/VLM accounting differs: {label}")
            rows[case] = dict(ae_ms=ae, overlap_ms=overlap, vlm_tail_ms=tail, total_ms=total)
        g = data["fused_ae_solo_overlap"][f"g-{suffix}"]
        e_total, g_total = rows["e"]["total_ms"], rows["g"]["total_ms"]
        explicit_g = g["explicit_native_io_total_ms"]
        if not math.isclose(explicit_g, g_total + g["deduplicated_io_ms"], rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"g explicit and shared-tensor accounting differ: {suffix}")
        candidates.append(dict(slice=suffix, step=step, layer=layer, cases=rows,
                               e_over_g=e_total / g_total, e_minus_g_ms=e_total - g_total,
                               g_explicit_io_ms=explicit_g, e_minus_g_explicit_io_ms=e_total - explicit_g))
    chosen = max(candidates, key=lambda row: row["e_over_g"])
    return dict(alpha=1.0, streaming_policy="same_first", vlm_profile_ms=profile_ms,
                selected_slice=chosen["slice"], candidates=candidates)


def e_idle_breakdown(data, gaps, edges, burst_ck):
    comparison = data["same_slice_comparison"]
    selected = next(row for row in comparison["candidates"] if row["slice"] == comparison["selected_slice"])
    label = f"e-{selected['slice']}"
    replay = data["replay"][label]["1.0"]
    directions = "RW"
    min_left = {p: min(edges[p + d] for d in directions) for p in "RWO"}
    min_right = {p: min(edges[d + p] for d in directions) for p in "RWO"}
    phases, transitions = [], Counter()
    totals = Counter()
    for row in replay["same_first"]["stages"]:
        histogram = Counter((b - a, "RWO"[left], "RWO"[right])
                            for channel in gaps[row["event_index"]] for a, b, left, right in channel)
        if row["active_repetitions"] != REPEATS or sum(histogram.values()) * REPEATS != row["count"]:
            raise ValueError("e idle audit requires VLM active for every repetition")
        short_ck = hard_ck = short_count = hard_count = 0
        for (width, left, right), count in histogram.items():
            if width < burst_ck:
                short_ck += width * count * REPEATS
                short_count += count * REPEATS
            elif width < burst_ck + min_left[left] + min_right[right]:
                hard_ck += width * count * REPEATS
                hard_count += count * REPEATS
        edge_ck = edge_out_ck = 0
        for combo, group in row["combinations"].items():
            if combo[1] == "-":
                if group["count"]:
                    raise ValueError("e idle audit found a gap the replay did not attempt")
                continue
            feasible = group["count"] - group["infeasible_count"]
            into, out = combo[:2], combo[-2:]
            edge_ck += feasible * edges[into]
            edge_out_ck += feasible * edges[out]
            transitions[into] += feasible * edges[into]
            transitions[out] += feasible * edges[out]
        idle = row["length_ck"]
        if idle != sum(width * count * REPEATS for (width, _, _), count in histogram.items()):
            raise ValueError("e idle geometry and replay length differ")
        used = int(row["sum_u_channel_ck"])
        if used != row["sum_u_channel_ck"]:
            raise ValueError("e idle guard capacity must be an integer number of cycles")
        guard = idle - row["infeasible_ck"] - used
        conservative = guard - edge_ck - edge_out_ck
        chosen_direction_ck = row["infeasible_ck"] - short_ck - hard_ck
        chosen_direction_count = row["infeasible_count"] - short_count - hard_count
        if min(conservative, chosen_direction_ck, chosen_direction_count) < 0:
            raise ValueError(f"e idle loss classification differs: {row['name']}")
        phase = dict(idle_ck=idle, filled_ck=used, guard_ck=guard,
                     left_guard_ck=edge_ck, right_guard_ck=edge_out_ck,
                     conservative_guard_ck=conservative, short_ck=short_ck, short_count=short_count,
                     hard_rank_ck=hard_ck, hard_rank_count=hard_count,
                     chosen_direction_ck=chosen_direction_ck, chosen_direction_count=chosen_direction_count,
                     infeasible_ck=row["infeasible_ck"], infeasible_count=row["infeasible_count"],
                     gap_count=row["count"])
        totals.update(phase)
        phases.append(dict(stage=row["name"], **phase))

    ck_ms = data["clock_ps"] / 1e9
    mean_ck_ms = ck_ms / N
    e_overlap = selected["cases"]["e"]["overlap_ms"]
    g_overlap = selected["cases"]["g"]["overlap_ms"]
    geometry = data["geometry"][label]
    pim_stage_channel_ck = sum(row["t_ck"] for row in geometry) * N * REPEATS
    pim_dq_channel_ck = sum(row["dq_channel_ck"] for row in geometry) * REPEATS
    if pim_stage_channel_ck - pim_dq_channel_ck != totals["idle_ck"]:
        raise ValueError("e native PIM DQ and idle do not cover the PIM stages")
    # Each channel's use is bounded by its accepted gap capacity, and the
    # slowest-channel progress is bounded by mean use. Equality certifies that
    # every accepted gap was consumed in this e same_first run.
    if not math.isclose(totals["filled_ck"] * mean_ck_ms, e_overlap, rel_tol=0, abs_tol=1e-9):
        raise ValueError("e VLM progress does not certify full use of the accepted capacity")
    idle_ms = totals["idle_ck"] * mean_ck_ms
    best_overlap_ms = comparison["vlm_profile_ms"] - replay["best"]["tail_ms"]
    payload_ck = burst_ck * math.fsum(sum(row["requests"]) for row in replay["same_first"]["stages"])
    return dict(slice=selected["slice"], totals=dict(totals), stages=phases,
                guard_by_transition_ck=dict(transitions), fill_fraction=totals["filled_ck"] / totals["idle_ck"],
                pim_stage_channel_ck=pim_stage_channel_ck, pim_dq_channel_ck=pim_dq_channel_ck,
                idle_ms=idle_ms, overlap_ms=e_overlap, unused_idle_ms=idle_ms - e_overlap,
                best_overlap_ms=best_overlap_ms, best_recovered_ms=best_overlap_ms - e_overlap,
                payload_equivalent_ck=payload_ck, burst_ck=burst_ck,
                g_overlap_ms=g_overlap, g_vs_e_raw_idle_ms=g_overlap - idle_ms,
                g_overlap_gain_ms=g_overlap - e_overlap,
                g_ae_delta_ms=selected["cases"]["g"]["ae_ms"] - selected["cases"]["e"]["ae_ms"],
                g_end_to_end_gain_ms=selected["e_minus_g_ms"])


def report_md(data):
    baseline = data["b_ms"]
    baseline_source = "b 的 AE solo 首层切片" if data.get("serial_slices") else "c 的 AE solo 首层切片"
    lines = [
        "# 基于 PIM DQ 空隙的 VLM 干扰估算", "",
        f"基准 b 为 {baseline_source}重复 180 次加 profile VLM，合计 **{baseline:.6f} ms**。"
        f"MAC {data['mac_freq_mult']}×，16 通道，{data['clock_ps']} ps/CK。"
        f"{len(data['provenance'])} 组 trace 开关对照的 raw.json 均逐字节相同。", "",
        "| α | VLM 单元 | solo (ms) | profile (ms) | 单元最大绝对误差 (%) | 总时长误差 (%) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha, row in data["solo"].items():
        lines.append(f"| {alpha} | {row['units']} | {row['total_ms']:.9f} | {row['profile_ms']:.9f} | "
                     f"{row['max_unit_error_percent']:.3g} | {row['total_error_percent']:.3g} |")
    if "serial_slices" in data:
        lines += ["", "| b 串行切片 | AE ×180 (ms) | VLM (ms) | E2E (ms) |",
                  "|---|---:|---:|---:|"]
        for label, row in data["serial_slices"].items():
            lines.append(f"| {label} | {row['ae_ms']:.6f} | {row['vlm_ms']:.6f} | {row['total_ms']:.6f} |")
    comparison = data["same_slice_comparison"]
    selected = next(row for row in comparison["candidates"] if row["slice"] == comparison["selected_slice"])
    lines += ["", "## c–g 同切片对照", "",
              f"在实跑的两个候选切片中，step {selected['step']} / layer {selected['layer']} 的 e/g "
              "端到端比值最高。五个变体均使用该切片重复 180 次。c–e 取 α=1.0、same_first，"
              "f/g 取融合阶段的重叠估算，并统一使用旧版共享张量 I/O 去重。", "",
              "重叠指 AE 结束前已经推进的 VLM profile 等效时间。它与 VLM 尾部之和为 "
              f"{comparison['vlm_profile_ms']:.6f} ms，计费 AE 与尾部之和为端到端估算。", "",
              "| 变体 | 计费 AE (ms) | AE 期间 VLM 重叠 (ms) | AE 后 VLM 尾部 (ms) | 端到端估算 (ms) |",
              "|---|---:|---:|---:|---:|"]
    for case, r in selected["cases"].items():
        lines.append(f"| {case} | {r['ae_ms']:.6f} | {r['overlap_ms']:.6f} | "
                     f"{r['vlm_tail_ms']:.6f} | {r['total_ms']:.6f} |")
    lines += ["", "| 候选切片 | e 端到端 (ms) | g 去重后端到端 (ms) | e/g | e − g (ms) | g 保留全部 I/O (ms) |",
              "|---|---:|---:|---:|---:|---:|"]
    for candidate in comparison["candidates"]:
        cases = candidate["cases"]
        lines.append(f"| {candidate['slice']} | {cases['e']['total_ms']:.6f} | {cases['g']['total_ms']:.6f} | "
                     f"{candidate['e_over_g']:.6f} | {candidate['e_minus_g_ms']:.6f} | {candidate['g_explicit_io_ms']:.6f} |")
    usage = data["e_idle_breakdown"]
    if usage.get("status") == "vlm_completed_within_ae":
        lines += ["", "VLM 在 AE 重复期间完成，逐阶段重放保留实际推进量。"
                  "全程有 VLM 需求时的 DQ 空闲分解不适用。"]
    else:
        totals = usage["totals"]
        mean_ck_ms = data["clock_ps"] / 1e9 / N
        lines += ["", "### e 的 DQ 空隙去向", "",
                  f"以下按 {usage['slice']}、α=1.0、same_first 统计。通道 CK 为 16 通道 ×180 次之和，"
                  "均摊时长为通道 CK 除以 16 后换算。范围为全部 PIM stage，包含 c–e 原模型允许填充的 Q 交换空隙。", "",
                  "| DQ 空闲分项 | 通道 CK | 均摊时长 (ms) |",
                  "|---|---:|---:|"]
        for key, title in (("idle_ck", "纯 PIM 的 DQ 空闲"), ("filled_ck", "VLM 实际消耗的 profile 推进预算"),
                           ("guard_ck", "可用空隙扣除的跨 rank 边界开销"),
                           ("short_ck", "空隙原长不足 2 CK"),
                           ("hard_rank_ck", "任意边界方向均放不下 2 CK"),
                           ("chosen_direction_ck", "当前方向选择下放不下 2 CK")):
            ck = totals[key]
            lines.append(f"| {title} | {ck:,} | {ck * mean_ck_ms:.6f} |")
        lines += ["", f"有 {totals['infeasible_count']:,} 个整段不可用空隙，共 {totals['infeasible_ck']:,} 通道 CK。"
                  f"same_first 消耗了全部接受的预算，占原始空闲的 {usage['fill_fraction']:.2%}。"
                  f"直接边界开销中 W→R 占 {usage['guard_by_transition_ck']['WR']:,} 通道 CK。"
                  f"best 实际比 same_first 多推进 {usage['best_recovered_ms']:.6f} ms。", "",
                  f"按 {usage['burst_ck']} CK/burst 折算，VLM 实际发送的流体请求量约相当于 "
                  f"{usage['payload_equivalent_ck']:,.0f} 通道 CK 的 DQ 数据载荷。"
                  "该值来自请求量折算，profile 推进预算还涵盖 kernel 其余服务时间，原生 GPU DQ 时序未在本模型中重建。", "",
                  f"g 的理想重叠比 e 多 {usage['g_overlap_gain_ms']:.6f} ms。其中 "
                  f"{usage['g_vs_e_raw_idle_ms']:.6f} ms 来自 g 的重叠窗口比 e 的原始 DQ 空闲更长，"
                  f"{usage['unused_idle_ms']:.6f} ms 来自 e 未能利用的空闲。"
                  f"扣除 g 增加的 {usage['g_ae_delta_ms']:.6f} ms 计费 AE，端到端净收益为 "
                  f"{usage['g_end_to_end_gain_ms']:.6f} ms。", "",
                  "| e 的 stage | 原始空闲 CK | 已用 CK | 边界扣减 CK | 任意方向都不可用 CK | 当前方向下不可用 CK |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in usage["stages"]:
            lines.append(f"| {row['stage']} | {row['idle_ck']:,} | {row['filled_ck']:,} | {row['guard_ck']:,} | "
                         f"{row['short_ck'] + row['hard_rank_ck']:,} | {row['chosen_direction_ck']:,} |")
    lines += ["", "## c、d、e 的估算", "",
              "| 切片 × 180 | α | policy | AE (ms) | VLM 尾部 (ms) | total (ms) | VLM 损失 (ms) | b / total |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for label, case in data["replay"].items():
        for alpha, policies in case.items():
            for policy, row in policies.items():
                lines.append(f"| {label} | {alpha} | {policy} | {row['ae_ms']:.6f} | {row['tail_ms']:.6f} | "
                             f"{row['total_ms']:.6f} | {row['vlm_loss_ms']:.6f} | {row['speedup_vs_b']:.6f} |")
    lines += ["", "## f、g 的 AE solo 重叠估算", "",
              "端到端时间等于计费 AE 加 c–e 使用的同一份 VLM profile 时长，再减去 VLM 可在 AE "
              "期间完成的时间。两个 AE solo 切片各重复 180 次。"
              "f/g 均按旧版共享张量口径去重，同时沿用当前 trace 中的 AE GPU 块时长。", "",
              ("顶层 Memory Controller 使用共享 CA 发送 PIM 命令。本节沿用旧估算，未建模与 VLM 的 CA 争用。"
               if data.get("mc_command_control") else
               "私有 PIM 计算指描述符内走私有 CA 的本地访问与运算。描述符发令占用共享 CA。") +
              "AE solo trace 记录的是 stage 和 I/O epoch 边界。估算取含 PIM 计算的 stage 在所有通道"
              "均处于 I/O epoch 外的时长，按理想重叠计入。该窗口也可能包含屏障及等待。"
              "Q 交换及其余非去重传输全程计入 AE。"
              "去重阶段原本就在 I/O epoch 内，扣除后 epoch 外重叠时长相同。", "",
              "| 切片 × 180 | 原始 AE (ms) | 去重 I/O (ms) | 计费 AE (ms) | VLM profile (ms) | epoch 外重叠 (ms) | 端到端估算 (ms) | b / 端到端 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r in data["fused_ae_solo_overlap"].items():
        lines.append(f"| {label} | {r['ae_trace_ms']:.6f} | {r['deduplicated_io_ms']:.6f} | {r['ae_solo_ms']:.6f} | "
                     f"{r['vlm_profile_ms']:.6f} | {r['overlap_ms']:.6f} | {r['total_ms']:.6f} | {r['speedup_vs_b']:.6f} |")
    g_reference = data["fused_ae_solo_overlap"]["g-s0-l0"]
    lines += ["", f"g 首层切片按同一份 VLM profile、保留全部原生 I/O 时为 "
              f"{g_reference['explicit_native_io_total_ms']:.6f} ms。", ""]
    g_e2e = data["fused_windows"]["g"]
    lines += ["### g 的模型口径对照", ""]
    if "archived_total_ms" in g_e2e:
        lines += [f"整 kernel 融合窗口的归档与回退版本重跑分别为 {g_e2e['archived_total_ms']:.6f} ms、"
                  f"{g_e2e['total_ms']:.6f} ms，同口径差值 {g_e2e['delta_from_archive_ms']:+.6f} ms。", ""]
    comparison = data["g_model_comparison"]
    lines += ["逐请求闭环在 AE 时间里计入原生 PIM I/O。g 的重叠估算在 AE 中计入按旧版规则去重后保留的 PIM I/O。"
              "VLM 的实测 kernel 时长来自 Thor 自然缓存 trace，包含该次执行实际经历的访存等待。"
              "L2 读 miss 和写流量代理来自单独采集的计数器。", "",
              "逐请求闭环在 I/O epoch 期间限制新 GPU DRAM 命令，已发请求仍可返回。"
              "kernel 完成时刻取实测时长下限与 DRAM 请求完成时刻中的较晚者。"
              "重叠估算把整段 I/O epoch 作为 VLM 暂停时间，影响的是 AE 结束后的 VLM 剩余量。", "",
              f"| g 的时间项（ms） | 4× 归档逐请求闭环 | {data['mac_freq_mult']}× AE solo 重叠估算 |",
              "|---|---:|---:|"]
    for key, label in (("ae_solo_ms", "计费 AE solo（含保留的 PIM I/O）"), ("pim_stage_delta_ms", "并发 PIM stage 增量"),
                       ("ae_gpu_wait_ms", "AE GPU 等待"), ("ae_elapsed_ms", "AE 完成时刻"),
                       ("vlm_tail_ms", "AE 完成后的 VLM 尾部")):
        lines.append(f"| {label} | {comparison['closed_loop_ms'][key]:.6f} | {comparison['overlap_estimate_ms'][key]:.6f} |")
    lines += [f"| 端到端 | {comparison['closed_loop_total_ms']:.6f} | {comparison['overlap_estimate_total_ms']:.6f} |", "",
              f"闭环使用 {comparison['closed_loop_vlm_solo_ms']:.6f} ms 的 VLM solo 校准，"
              f"重叠估算使用 {comparison['estimate_vlm_profile_ms']:.6f} ms 的 VLM profile。"]
    lines += ["", "### AE GPU 计时与历史共享张量 I/O 去重", "",
              "早期账本（`09234a9`）用冷缓存的 AE GPU kernel 时长，保留的 GPU producer 或 consumer "
              "如果已承担同一共享张量的存取，便在合成 AE 时间线中去除对应的 native phase。"
              "当时的规则包括融合输入暂存写入、QKT score 批量读出及 GPU RoPE 的 Q/K 导入导出，"
              "原始 DQ 流量和时间线完整保留。", "",
              "当前 AE 输入对普通保留 kernel 使用自然缓存 trace，对另测的边界 helper 使用 32 MiB "
              "清缓存后的设备 kernel 中位数。两类时长都包含各自运行时经历的访存。"
              "f/g 沿用旧版共享分配假设，将对应的暂存写入及 score 读出归入现有 GPU producer 或 consumer。"
              "f 额外去重 GPU RoPE 已覆盖的 Q/K 导出和导入。跨 rank Q 交换和其他结果导出仍计入 AE。", "",
              f"| {selected['slice']} × 180 的去重阶段 | 当前 epoch | f (ms) | g (ms) |", "|---|---|---:|---:|"]
    scale = data["clock_ps"] * REPEATS / 1e9
    fused = data["fused_ae_solo_overlap"]
    f, g = (fused[f"{case}-{selected['slice']}"] for case in "fg")
    f_stages, g_stages = ({row["name"]: row["deduplicated_io_ck"] for row in case["stages"]}
                         for case in (f, g))
    for stage, (epoch, _, _, _) in FUSED_SHARED_IO["f"].items():
        g_value = f"{g_stages[stage] * scale:.6f}" if stage in g_stages else "—"
        lines.append(f"| {stage} | {epoch} | {f_stages[stage] * scale:.6f} | {g_value} |")
    lines += ["", f"同切片中完整计入 AE 的 Q 交换时间为 f {f['q_exchange_ms']:.6f} ms、"
              f"g {g['q_exchange_ms']:.6f} ms。"]
    if "VLM_in_AE_by_phase_ms" in g_e2e:
        phases = g_e2e["VLM_in_AE_by_phase_ms"]
        windows_origin = "本轮" if "archived_total_ms" in g_e2e else "历史归档"
        lines += ["", f"### {windows_origin} g 原生融合窗口 E2E 中，AE 结束前的 VLM 执行时间", "",
                  "| AE 阶段 | VLM 实际执行（ms） |", "|---|---:|"]
        for phase, label in (("compute", "PIM 私有计算"), ("launch", "连续描述符启动"),
                             ("join", "等待当前 VLM kernel 完成"), ("upload", "上传"),
                             ("download", "导出"), ("transfer", "纯传输"), ("ae_gpu", "AE GPU 块")):
            lines.append(f"| {label} | {phases[phase]:.6f} |")
        lines += [f"| 合计 | {g_e2e['VLM_in_AE_ms']:.6f} |", "",
                  f"这次 E2E 的 AE 在 {g_e2e['AE_elapsed_ms']:.6f} ms 完成，"
                  f"之后 VLM 再执行 {g_e2e['VLM_tail_ms']:.6f} ms。"
                  "AE solo 的重叠估算取自独立的纯 AE trace，整 kernel 交接等待只出现在并发 E2E 时间线上。"]
    lines += ["", "## 切片代表性", "",
              "| 变体 | 指标 | α | policy | step 0 / layer 0 | step 9 / layer 17 | 差值 | 相对偏差 (%) |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    for case, rows in data["slice_comparison"].items():
        for row in rows:
            lines.append(f"| {case} | {row['metric']} | {row.get('alpha', '—')} | {row.get('policy', '—')} | "
                         f"{row['base']:.6f} | {row['alternate']:.6f} | {row['difference']:+.6f} | {row['percent']:+.4f} |")
    lines += ["", "## 分阶段数据", "",
              "T 和 DQ 占用来自单个原生切片。空隙数量、长度及 ΣU 来自 α=1.0、same_first 的 180 次重放，"
              "f、g 列为单切片几何值。DQ、空隙长度是 16 通道之和，ΣU 是通道平均后累加的 CK。"
              "JSON 含各 policy、α 和 stage 的组合明细，O 表示 open，四方向字符表示中间切换，"
              "省略号表示多次切换，短横线表示未尝试填充。", "",
              "| 切片 | stage | T (CK) | PIM DQ (通道 CK) | DQ/T | 空隙数 | 空隙 (通道 CK) | 不可行数 | 不可行 (通道 CK) | ΣU (CK) | U/T |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, geometry in data["geometry"].items():
        rows = {r["event_index"]: r for r in data["replay"].get(label, {}).get("1.0", {})
                .get("same_first", {}).get("stages", [])}
        for r in geometry:
            p = rows.get(r["event_index"])
            suffix = (f"{p['infeasible_count']} | {p['infeasible_ck']} | {p['sum_u_ck']:.2f} | {p['u_over_t']:.4%}"
                      if p else "— | — | — | —")
            lines.append(f"| {label} | {r['name']} | {r['t_ck']} | {r['dq_channel_ck']} | {r['dq_over_t']:.4%} | "
                         f"{p['count'] if p else r['gap_count']} | {p['length_ck'] if p else r['gap_channel_ck']} | {suffix} |")
    lines += ["", "### g 的融合 I/O epoch", "",
              "| 切片 | stage | 接触 epoch 空隙数 | 完全在 epoch 内空隙数 | 排除 (通道 CK) | 保留 (通道 CK) |",
              "|---|---|---:|---:|---:|---:|"]
    for label in ("g-s0-l0", "g-s9-l17"):
        for r in data["geometry"][label]:
            lines.append(f"| {label} | {r['name']} | {r['epoch_touching_gap_count']} | "
                         f"{r['epoch_internal_gap_count']} | {r['epoch_blocked_idle_channel_ck']} | "
                         f"{r['idle_outside_epoch_channel_ck']} |")
        r = data["g_epoch_filter"][label]
        lines.append(f"| {label} | 合计 | {r['epoch_touching_gap_count']} | {r['epoch_internal_gap_count']} | "
                     f"{r['epoch_blocked_idle_channel_ck']} | {r['idle_outside_epoch_channel_ck']} |")
    lines += ["", "## 口径与局限", "",
              "- 只计 DQ，忽略 CA 争用和 rank0 行冲突。GPU 仅使用空隙，PIM 时间线固定，相当于理想的 pim_head。",
              "- 每单元均采用 memory-bound 流体速率。PIM 占用时间按 1:1 计入损失。VLM 损失在活跃 PIM stage "
              "内按 T 减 16 通道平均 ΣU 计算，AE GPU 块暂停 VLM，等待时间为零。",
              "- 写不能超前于读。实际 kernel 的读写交替形状未建模。单元在最慢通道完成时结束，"
              "其它通道在同一空隙内可以等到该时刻继续下一单元。",
              "- α 只缩放代理写请求，按原生输入约定向上取整到 sector，随后将请求平均分给 16 通道。"
              "每个 α 都从 profile 时长重新计算每条请求的 CK。",
              "- same_first 无历史时优先读，best 的入口成本相同时沿用 same_first。random 按剩余读写比例抽样，"
              "条件是两个方向当前均可发。表中 random 为固定种子样本均值，JSON 记录种子、样本和 total 的标准误。",
              "- 每个空隙先假设尾方向等于入口方向，最多重新计算两次右边开销。仍不收敛时预留两方向中较大的右边开销，"
              "JSON 按阶段记录次数。ΣU 是可用容量，逐空隙表另列实际发射的读写数，起止 CK 相对各自 stage 的起点。",
              "- DQ 估算把独立启动的单层 AE solo 时间线重复 180 次。层外 AE GPU 块和跨层 DRAM 状态延续未进入该模型。"
              "g 的代表性对照用两个实跑 AE solo 切片分别重复 180 次。",
              ""]
    return "\n".join(lines)


def compare(base, alternate, metric, **labels):
    return dict(metric=metric, base=base, alternate=alternate, difference=alternate - base,
                percent=100 * (alternate / base - 1), **labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=ROOT / "reference/spec/spec.json",
                        help="native spec used to capture the traces (default: archived 4× spec)")
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--random-samples", type=int, default=8)
    parser.add_argument("--details", nargs="?", const="first", choices=("first", "all"),
                        help="write per-gap CSV for the first or all same_first repetitions")
    args = parser.parse_args()
    if args.random_samples < 2:
        parser.error("--random-samples requires at least two samples")
    output, engine = args.output.resolve(), args.engine.resolve()
    output.mkdir(parents=True, exist_ok=True)
    working = output / "replay-input"
    working.mkdir(exist_ok=True)
    spec = read_json(args.spec)
    spec_digest = sha256(args.spec)
    edges = edge_costs(MATRIX)
    clock_ps = spec["clock_ps"]
    ck_ms = clock_ps / 1e9
    units = {a: units_for(a, clock_ps) for a in ALPHAS}
    data = dict(schema_version=1, clock_ps=clock_ps, channels=N, repetitions=REPEATS,
                mac_freq_mult=spec["mac_freq_mult"], spec_sha256=spec_digest,
                matrix_sha256=sha256(MATRIX), vlm_sha256=sha256(PROFILE), edge_ck=edges,
                replay_engine_sha256=sha256(engine),
                single_direction_reference=PAIR_REFERENCE, solo={}, geometry={}, provenance={},
                replay={}, slice_comparison={}, hand_checks_passed=True)
    if spec["metadata"].get("mc_command_control"):
        data["mc_command_control"] = True
    for alpha in ALPHAS:
        path = working / f"solo-{alpha}.txt"
        write_replay_input(path, clock_ps, edges, units[alpha])
        data["solo"][str(alpha)] = replay(engine, path, "same_first", 0)["solo"]
    seeds = [20261110 + n for n in range(args.random_samples)]
    profile_ms = data["solo"]["1.0"]["profile_ms"]
    jobs, fused_slices = {}, {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for case, slices in SLICES.items():
            for step, layer in slices:
                label = f"{case}-s{step}-l{layer}"
                timeline, geometry, gaps, verification = load_trace(
                    args.traces / label, step, layer, shared_io_rules=FUSED_SHARED_IO.get(case))
                if verification["spec_sha256"] != spec_digest:
                    raise ValueError(f"trace spec differs for {label}")
                if timeline["clock_ps"] != clock_ps:
                    raise ValueError(f"trace clock differs for {label}")
                data["geometry"][label] = geometry
                data["provenance"][label] = verification
                verification["ae_slice_ck"] = timeline["ae_end_ck"]
                if case in "fg":
                    fused_slices[label] = timeline, geometry
                    continue
                for alpha in ALPHAS:
                    path = working / f"{label}-{alpha}.txt"
                    write_replay_input(path, clock_ps, edges, units[alpha], timeline["events"], gaps)
                    for policy in POLICIES:
                        detail = (output / f"{label}-a{alpha}-gaps.csv"
                                  if args.details and policy == "same_first" else None)
                        jobs[label, str(alpha), policy] = [
                            pool.submit(replay, engine, path, policy, seed, detail,
                                        REPEATS if args.details == "all" else 1)
                            for seed in (seeds if policy == "random" else [0])]
        serial_slices = {}
        for step, layer in COMPARISON_SLICES:
            label = f"b-s{step}-l{layer}"
            directory = args.traces / label
            if directory.exists():
                timeline, geometry, _, verification = load_trace(directory, step, layer)
                if (verification["spec_sha256"] != spec_digest or timeline["clock_ps"] != clock_ps
                        or verification["program_sha256"] !=
                        data["provenance"][f"c-s{step}-l{layer}"]["program_sha256"]):
                    raise ValueError(f"serial AE trace does not match the feature-wise inputs: {label}")
                data["provenance"][label] = dict(verification, ae_slice_ck=timeline["ae_end_ck"])
                data["geometry"][label] = geometry
                ae_ms = timeline["ae_end_ck"] * REPEATS * ck_ms
                serial_slices[label] = dict(ae_ms=ae_ms, vlm_ms=profile_ms, total_ms=ae_ms+profile_ms)
        if serial_slices:
            data["serial_slices"] = serial_slices
        baseline = serial_slices.get("b-s0-l0", {}).get(
            "total_ms", data["provenance"]["c-s0-l0"]["ae_slice_ck"] * REPEATS * ck_ms + profile_ms)
        data["b_ms"] = baseline
        for (label, alpha, policy), futures in jobs.items():
            samples = [future.result() for future in futures]
            for row in samples:
                if row.pop("solo") != data["solo"][alpha]:
                    raise ValueError("replay solo validation differs")
                validate_replay(row, data["geometry"][label], data["provenance"][label]["ae_slice_ck"])
            result = expectation(samples) if policy == "random" else samples[0]
            if policy == "random":
                totals = [sample["total_ck"] * ck_ms for sample in samples]
                result.update(seeds=seeds, total_samples_ms=totals,
                              total_standard_error_ms=statistics.stdev(totals) / math.sqrt(len(totals)))
            data["replay"].setdefault(label, {}).setdefault(alpha, {})[policy] = add_units(result, ck_ms, baseline)
            print(f"{label} alpha={alpha} {policy}: {result['total_ms']:.6f} ms", flush=True)
    data["fused_windows"] = fused_summary(args.traces, ck_ms, baseline)
    data["fused_ae_solo_overlap"] = {}
    for label, (timeline, geometry) in fused_slices.items():
        data["fused_ae_solo_overlap"][label] = fused_ae_solo_overlap(
            timeline, geometry, profile_ms, baseline, shared_io_rules=FUSED_SHARED_IO[label[0]])
    data["g_model_comparison"] = g_model_comparison(data["fused_ae_solo_overlap"]["g-s0-l0"])
    data["slice_comparison"]["d"] = [
        compare(row[key], data["replay"]["d-s9-l17"][alpha][policy][key],
                metric, alpha=alpha, policy=policy)
        for alpha, policies in data["replay"]["d-s0-l0"].items() for policy, row in policies.items()
        for key, metric in (("total_ms", "180× total (ms)"), ("vlm_loss_ms", "180× VLM 损失 (ms)"))]
    g_labels = ("g-s0-l0", "g-s9-l17")
    data["g_epoch_filter"] = {
        label: {key: sum(row[key] for row in data["geometry"][label]) for key in
                ("gap_count", "gap_channel_ck", "epoch_touching_gap_count", "epoch_internal_gap_count",
                 "epoch_blocked_idle_channel_ck", "idle_outside_epoch_channel_ck")}
        for label in g_labels
    }
    data["slice_comparison"]["g"] = [
        compare(*(data["provenance"][label]["ae_slice_ck"] * REPEATS * ck_ms
                  for label in g_labels), "180× 原始 AE trace (ms)"),
        compare(*(sum(r["idle_outside_epoch_channel_ck"] for r in data["geometry"][label])
                  for label in g_labels), "epoch 外 idle (通道 CK)"),
    ]
    g_paths = {label: args.traces / f"{label}-fused-repeated/raw.json" for label in g_labels}
    if all(path.is_file() for path in g_paths.values()):
        repeated = {label: fused_result(path, "g", ck_ms, len(data["geometry"][label]) * REPEATS, baseline)
                    for label, path in g_paths.items()}
        data["fused_g_repeated"] = repeated
    data["slice_comparison"]["g"] += [
        compare(*(data["fused_ae_solo_overlap"][label][key] for label in g_labels), metric)
        for key, metric in (("deduplicated_io_ms", "180× 去重 I/O (ms)"),
                            ("ae_solo_ms", "180× 计费 AE (ms)"),
                            ("overlap_ms", "180× epoch 外重叠 (ms)"),
                            ("total_ms", "180× 端到端 (ms)"))]
    data["same_slice_comparison"] = same_slice_comparison(data)
    selected = next(row for row in data["same_slice_comparison"]["candidates"]
                    if row["slice"] == data["same_slice_comparison"]["selected_slice"])
    _, _, e_gaps, _ = load_trace(args.traces / f"e-{selected['slice']}", selected["step"], selected["layer"])
    commands = spec["commands"]
    if {commands[name]["dq_cycles"] for name in ("RD", "WR")} != {2}:
        raise ValueError("e idle audit requires the replay's 2 CK minimum DQ burst")
    e_replay = data["replay"][f"e-{selected['slice']}"]["1.0"]["same_first"]
    if data.get("mc_command_control") and e_replay["tail_ms"] == 0:
        data["e_idle_breakdown"] = {"status": "vlm_completed_within_ae"}
    else:
        data["e_idle_breakdown"] = e_idle_breakdown(data, e_gaps, edges, commands["RD"]["dq_cycles"])
    write_json(output / "gap-estimate.json", data)
    (output / "gap-estimate.md").write_text(report_md(data), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
