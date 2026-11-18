"""Audit saved full f/g results using calibrated VLM kernel windows; never simulate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

MODEL = "fused_kernel_windows"
WINDOW_POLICY = "continuous_pim_descriptors"
GPU_COUNTERS = ("gpu_read_bytes", "gpu_write_bytes", "GPU_reads", "GPU_writes", "GPU_act1", "GPU_act2",
                "GPU_prepb", "GPU_preab", "GPU_row_hits", "GPU_row_misses")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def integer(value, label):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{label}: invalid cycle/count")
    return value


def walk(items):
    for op in items:
        yield op
        yield from walk(op.get("children", []))


def disjoint(left, right, label):
    """Half-open interval check; zero-duration intervals occupy no time."""
    left, right = sorted((a, b) for a, b in left if a < b), sorted((a, b) for a, b in right if a < b)
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        require(b <= c or d <= a, label)
        if b <= d:
            i += 1
        else:
            j += 1


def safe_continuations(schedule, program):
    """Derive safe outgoing stage IDs from actual adjacent events and all 16 programs."""
    templates = {stage["name"]: stage for stage in program["stages"]}
    edges = {}
    for name, stage in templates.items():
        require(len(stage["channels"]) == 16, "program does not contain 16 channels")
        head_only, tail_only = [], []
        for ops in stage["channels"]:
            indices = [i for i, op in enumerate(ops) if op["op"] in {"MAC_RUN", "ROPE_RUN"}]
            require(len(indices) <= 1, "multiple descriptors within one stage are unsupported")
            head_only.append(bool(indices) and all(op["op"] == "BARRIER" for op in ops[:indices[0]]))
            tail_only.append(bool(indices) and all(op["op"] == "BARRIER" for op in ops[indices[0]+1:]))
        edges[name] = all(head_only), all(tail_only)
    events = schedule["events"]
    outgoing, stage_id = set(), -1
    for index, event in enumerate(events[:-1]):
        if event["kind"] != "pim":
            continue
        stage_id += 1
        next_event = events[index+1]
        if (next_event["kind"] == "pim" and event["step"] == next_event["step"] and
                event["layer"] == next_event["layer"] and edges[event["stage"]][1] and edges[next_event["stage"]][0]):
            outgoing.add(stage_id)
    return outgoing


def audit_timeline(raw, schedule, ledger, program):
    """Validate generic saved timelines; full-workload scope is checked by audit_case."""
    clock = integer(raw["clock_ps"], "clock")
    require(clock > 0 and ledger["clock_ps"] == clock, "ledger/native clock mismatch")

    def to_cycles(value, label):
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0,
                f"{label}: invalid time")
        cycles = round(value * 1000000 / clock)
        require(abs(value - cycles * clock / 1000000) < 2e-6, f"{label}: not an integer native timestamp")
        return cycles

    def interval(row, label):
        a, b = integer(row["start_ck"], label), integer(row["end_ck"], label)
        require(a <= b and to_cycles(row["start_us"], label) == a and to_cycles(row["end_us"], label) == b,
                f"{label}: inconsistent interval")
        return a, b

    require(raw["execution_model"] == program["execution_model"] == MODEL, "execution model mismatch")
    require(ledger["kind"] == "calibrated_vlm_durations", "invalid calibrated ledger kind")
    continuous = raw.get("window_policy") == WINDOW_POLICY
    if continuous:
        require(raw.get("schema_version") == 3, "continuous descriptors require native schema 3")
    else:
        require(raw.get("schema_version", 2) == 2 and "window_policy" not in raw,
                "unsupported native window policy")
    outgoing = safe_continuations(schedule, program) if continuous else set()
    incoming = {sid+1 for sid in outgoing}
    stages, phases, kernels, ae_gpu, metrics = (raw[key] for key in ("stages", "phases", "kernels", "ae_gpu", "metrics"))
    templates = {stage["name"]: stage for stage in program["stages"]}
    require(len(templates) == len(program["stages"]), "duplicate program stage")
    require([s["stage_id"] for s in stages] == list(range(len(stages))), "invalid stage identifiers")
    by_stage, compute, exclusive, joins = defaultdict(list), [], [], []
    phase_rows = []
    for phase in phases:
        sid = phase["stage_id"]
        require(isinstance(sid, int) and 0 <= sid < len(stages), "unknown phase stage")
        a, b = interval(phase, "phase")
        ready = phase["channel_ready_ck"]
        require(len(ready) == 16 and all(integer(x, "channel ready") <= b for x in ready), "phase crossed a channel barrier early")
        if phase["phase"] in {"join", "continue"}:
            require(all(x == b for x in ready), "join/continue ready clocks differ from completion")
            if phase["phase"] == "continue":
                require(a == b, "descriptor continuation must have zero duration")
            else:
                joins.append((a, b))
        else:
            # Empty downloads can be immediately ready after a long join.
            require(max(a, max(ready)) == b, "phase completion differs from all-channel ready barrier")
        if phase["phase"] == "compute":
            compute.append((a, b))
        elif phase["phase"] in {"upload", "download", "transfer"}:
            exclusive.append((a, b))
        by_stage[sid].append(phase)
        phase_rows.append(dict(stage_id=sid, stage=stages[sid]["name"], phase=phase["phase"], duration_cycles=b-a))
    require([p["stage_id"] for p in phases] == sorted(p["stage_id"] for p in phases), "phase records are out of stage order")
    stage_cycles = 0
    for stage in stages:
        sid, name = stage["stage_id"], stage["name"]
        channels = templates[name]["channels"]
        require(len(channels) == 16, "program does not contain 16 channels")
        counts = [sum(op["op"] in {"MAC_RUN", "ROPE_RUN"} for op in ops) for ops in channels]
        require(len(set(counts)) == 1 and counts[0] in {0, 1}, "unsupported descriptor grouping across channels")
        expected = (["launch" if sid in incoming else "upload", "compute",
                     "continue" if sid in outgoing else "join", "download"] if counts[0] else ["transfer"])
        group = by_stage[sid]
        require([p["phase"] for p in group] == expected, f"{name}: phase sequence differs")
        a, b = to_cycles(stage["start_us"], name), to_cycles(stage["end_us"], name)
        require(a <= b and to_cycles(stage["duration_us"], name) == b-a, "inconsistent PIM stage duration")
        require(group[0]["start_ck"] == a and group[-1]["end_ck"] == b, "phases do not cover their entire stage")
        require(all(x["end_ck"] == y["start_ck"] for x, y in zip(group, group[1:])), "noncontiguous stage phases")
        if sid in outgoing:
            require(group[-1]["start_ck"] == group[-1]["end_ck"], "continuous descriptor boundary has a nonzero download")
        stage_cycles += b-a
    continuation_count = sum(phase["phase"] == "continue" for phase in phases)
    require(continuation_count == len(outgoing), "continuation records differ from derived safe boundaries")
    if continuous:
        require(integer(metrics["PIM_descriptor_continuations"], "continuations") == continuation_count,
                "native continuation count differs from derived safe boundaries")
    require(len(kernels) == len(ledger["units"]), "VLM ledger coverage mismatch")
    kernel_intervals, previous_end, calibrated_sum = [], 0, 0
    for kernel, unit in zip(kernels, ledger["units"]):
        require(all(kernel[key] == unit[key] for key in ("ordinal", "name", "kind")), "VLM identity/order mismatch")
        duration = integer(unit["duration_cycles"], "calibrated duration")
        a, b = interval(kernel, "VLM kernel")
        require(duration > 0 and kernel["duration_cycles"] == duration == b-a and
                to_cycles(kernel["duration_us"], "VLM duration") == duration,
                "VLM duration changed or kernel was preempted")
        require(a >= previous_end, "VLM kernels overlap")
        previous_end = b
        calibrated_sum += duration
        kernel_intervals.append((a, b))
    require(calibrated_sum == ledger["total_cycles"] == to_cycles(metrics["vlm_calibrated_sum_us"], "VLM sum"),
            "calibrated VLM sum differs")
    gpu_intervals, stage_index, gpu_index, now, ae_gpu_cycles = [], 0, 0, 0, 0
    for event in schedule["events"]:
        if event["kind"] == "pim":
            require(stage_index < len(stages), "missing scheduled PIM stage")
            row = stages[stage_index]
            require((row["name"], row["step"], row["layer"]) == (event["stage"], event["step"], event["layer"]),
                    "PIM schedule coverage/order differs")
            a, b = to_cycles(row["start_us"], "PIM start"), to_cycles(row["end_us"], "PIM end")
            stage_index += 1
        else:
            require(event["kind"] == "gpu" and gpu_index < len(ae_gpu), "invalid/missing AE GPU event")
            row = ae_gpu[gpu_index]
            require((row["name"], row["step"], row["layer"]) == (event["name"], event["step"], event["layer"]),
                    "AE GPU event identity differs")
            a, b = interval(row, "AE GPU")
            duration = math.ceil(event["us"] * 1000000 / clock - 1e-10)
            require(b-a == duration, "AE measured GPU duration changed")
            gpu_intervals.append((a, b))
            ae_gpu_cycles += duration
            gpu_index += 1
        require(a == now, "AE event timeline contains a gap or overlap")
        now = b
    require(stage_index == len(stages) and gpu_index == len(ae_gpu), "extra native AE events")
    ae_end = to_cycles(metrics["AE_elapsed_us"], "AE end")
    require(ae_end == now == stage_cycles + ae_gpu_cycles, "AE time identity failed")
    disjoint(kernel_intervals, exclusive + gpu_intervals, "VLM kernel overlaps exclusive transfer or AE GPU work")
    available = sorted(compute) + [(ae_end, 10**30)]
    window, last_end = 0, 0
    for a, b in kernel_intervals:
        while available[window][1] <= max(last_end, available[window][0]):
            window += 1
        require(a == max(last_end, available[window][0]), "kernel did not start at the next eligible compute/tail opportunity")
        last_end = b
    for a, b in joins:
        active_ends = [end for start, end in kernel_intervals if start < a < end]
        require(b == max([a] + active_ends), "join does not end with the already-running VLM kernel")
    join_cycles = sum(b-a for a, b in joins)
    require(join_cycles == to_cycles(metrics["AE_VLM_join_us"], "join total"), "join accounting differs")
    total = to_cycles(metrics["total_us"], "total")
    require(total == max(ae_end, last_end) and total-ae_end == to_cycles(metrics["VLM_remaining_us"], "VLM tail"),
            "final completion/tail identity failed")
    require(metrics["AE_GPU_wait_us"] == 0 and not metrics["serial_epoch_reset"], "legacy GPU wait or serial reset reported")
    return dict(AE_cycles=ae_end, total_cycles=total, stage_cycles=stage_cycles, join_cycles=join_cycles,
                native_PIM_cycles=stage_cycles-join_cycles, AE_GPU_cycles=ae_gpu_cycles,
                descriptor_continuations=continuation_count, phases=phase_rows)


def audit_case(raw, cfg, program, schedule, vlm, ledger, execution):
    require(cfg["execution_model"] == MODEL and cfg["dataflow"] == "fused" and cfg["layout"] == "token" and cfg["mode"] == "concurrent" and
            cfg["run_kind"] == "combined" and "arbitration" not in cfg, "wrong full-run configuration")
    require(raw["clock_ps"] == cfg["spec"]["clock_ps"], "native clock differs from configured spec")
    require(schedule["layers"] == 18 and schedule["denoising_steps"] == 10, "expected 18 layers x 10 steps")
    names = {"qkv", "q_exchange_export", "q_exchange_import", "qkt", "pv", "o", "gate_up", "down"}
    names |= {"rope_export", "rope_import"} if cfg["rope"] == "gpu" else {"rope"}
    stages, metrics = raw["stages"], raw["metrics"]
    require(Counter(s["name"] for s in stages) == Counter({name: 180 for name in names}), "full stage coverage differs")
    for name in names:
        require({(s["step"], s["layer"]) for s in stages if s["name"] == name} ==
                {(step, layer) for step in range(10) for layer in range(18)}, "layer/step coverage differs")
    require(len(vlm["units"]) == len(ledger["units"]) == len(raw["kernels"]) == 2357, "expected all 2357 VLM units")
    require(len({u["ordinal"] for u in vlm["units"]}) == 2357, "duplicate VLM ordinals")
    timing = audit_timeline(raw, schedule, ledger, program)
    require(raw.get("window_policy") == WINDOW_POLICY, "full-run audit requires the continuous-descriptor policy")
    require(timing["descriptor_continuations"] == (180 if cfg["rope"] == "gpu" else 360),
            "full workload continuation count differs from f=180 / g=360")
    require(all(metrics[key] == 0 for key in GPU_COUNTERS), "native GPU requests/commands were issued")
    require(metrics["PIM_out_of_order_issues"] == metrics["PIM_prefetched_prerequisites"] == 0, "FIFO reordered/prefetched")
    reads = writes = 0
    for unit, observed in zip(vlm["units"], raw["kernels"]):
        require(all(unit[key] == observed[key] for key in ("ordinal", "name", "kind")), "source VLM identity differs")
        rd = unit["read_sectors"] * 32
        wr = (math.ceil(unit["write_sectors"] * cfg["calibration"]["alpha"]) if unit.get("write_is_proxy", True)
              else unit["write_sectors"]) * 32
        require((observed["read_bytes"], observed["write_bytes"]) == (rd, wr), "VLM metadata bytes differ")
        reads += rd
        writes += wr
    require((metrics["vlm_accounted_read_bytes"], metrics["vlm_accounted_write_bytes"]) == (reads, writes), "VLM byte accounting differs")
    expected = {}
    for stage in program["stages"]:
        for channel, operations in enumerate(stage["channels"]):
            for op in operations:
                if op.get("children"):
                    require(all(child["op"] not in {"RD", "WR", "WRGB", "RDRES"} and not child.get("epoch")
                                for child in walk(op["children"])), "external I/O appeared inside compute descriptor")
            counts = Counter(op["op"] for op in walk(operations))
            expected[stage["name"], channel] = ((counts["RD"]+counts["RDRES"])*32, (counts["WR"]+counts["WRGB"])*32)
    template_r, template_w = (sum(pair[i] for pair in expected.values()) for i in (0, 1))
    require(template_r+template_w == program["expected_dq_bytes_per_layer"], "program byte contract differs")
    require((metrics["pim_read_bytes"], metrics["pim_write_bytes"]) == (template_r*180, template_w*180), "full PIM bytes differ")
    buses, totals = defaultdict(Counter), Counter()
    for bus in raw["bus"]:
        require(isinstance(bus["channel"], int) and 0 <= bus["channel"] < 16 and
                isinstance(bus["stage_id"], int) and -1 <= bus["stage_id"] < len(stages), "invalid bus channel/stage")
        require(bus["gpu_read_bytes"] == bus["gpu_write_bytes"] == 0, "native GPU bus traffic exists")
        require(0 <= bus["ca_cycles"] <= bus["duration_cycles"] and 0 <= bus["dq_cycles"] <= bus["duration_cycles"], "invalid bus occupancy")
        require(bus["read_bytes"] == bus["pim_read_bytes"] and bus["write_bytes"] == bus["pim_write_bytes"], "bus source accounting differs")
        for key in ("pim_read_bytes", "pim_write_bytes", "dq_cycles"):
            buses[bus["stage_id"], bus["channel"]][key] += bus[key]
            totals[key] += bus[key]
        if bus["stage_id"] < 0:
            require(bus["pim_read_bytes"] == bus["pim_write_bytes"] == 0, "PIM bytes escaped stages")
    for stage in stages:
        for channel in range(16):
            bus = buses[stage["stage_id"], channel]
            require((bus["pim_read_bytes"], bus["pim_write_bytes"]) == expected[stage["name"], channel], "per-stage/channel byte conservation failed")
    require(totals["pim_read_bytes"] == template_r*180 and totals["pim_write_bytes"] == template_w*180 and
            totals["dq_cycles"]*16 == (template_r+template_w)*180, "aggregate native DQ conservation failed")
    require(raw["refresh_policy"] == {"rank0": "REFpb", "rank1": "REFab"}, "refresh policy differs")
    for rank, allowed, forbidden in (("rank0", "REFpb", "REFab"), ("rank1", "REFab", "REFpb")):
        require(raw["refresh_commands"][rank][allowed] > 0 and raw["refresh_commands"][rank][forbidden] == 0, "refresh commands differ")
    require(integer(metrics["checked_commands"], "checked commands") >= (template_r+template_w)*180//32 and
            integer(metrics["frontend_iterations"], "frontend iterations") > 0, "invalid native work counters")
    require(execution["total_us"] == metrics["total_us"] and math.isfinite(execution["wall_seconds"]) and
            execution["wall_seconds"] > 0, "execution timing record differs")
    scale = raw["clock_ps"] / 1e9
    result = dict(variant=program["variant"], total_ms=metrics["total_us"]/1000, AE_elapsed_ms=metrics["AE_elapsed_us"]/1000,
                  AE_VLM_join_ms=timing["join_cycles"]*scale, PIM_stage_sum_ms=timing["stage_cycles"]*scale,
                  PIM_native_ms=timing["native_PIM_cycles"]*scale, AE_GPU_compute_ms=timing["AE_GPU_cycles"]*scale,
                  VLM_tail_ms=metrics["VLM_remaining_us"]/1000, VLM_calibrated_sum_ms=metrics["vlm_calibrated_sum_us"]/1000,
                  wall_seconds=execution["wall_seconds"], checked_commands=metrics["checked_commands"],
                  frontend_iterations=metrics["frontend_iterations"], PIM_stages=len(stages), VLM_units=2357,
                  PIM_descriptor_continuations=timing["descriptor_continuations"],
                  pim_read_bytes=template_r*180, pim_write_bytes=template_w*180,
                  vlm_accounted_read_bytes=reads, vlm_accounted_write_bytes=writes, native_GPU_transactions=0)
    return result, timing["phases"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / "frozen/manifest.json"
    manifest = read(manifest_path)
    require(manifest["window_policy"] == WINDOW_POLICY, "frozen window policy differs")
    require(sha(manifest["binary_path"]) == manifest["binary_sha256"], "frozen binary changed")
    for library in manifest["runtime_libraries"].values():
        require(sha(library["path"]) == library["sha256"], "runtime library changed")
    for path, digest in manifest["inputs"].items():
        require(sha(path) == digest, f"input changed: {path}")
    results, evidence, all_phases = [], {}, []
    for case, rope in (("f", "gpu"), ("g", "local")):
        destination, config_path = run / "native" / case, run / "configs" / f"{case}.json"
        cfg, provenance = read(config_path), read(destination / "provenance.json")
        require(cfg["rope"] == rope and sha(config_path) == manifest["configs"][case], "frozen case configuration differs")
        require(read(destination / "input-config.json") == cfg and provenance["config_sha256"] == sha(destination / "input-config.json"), "executed config binding differs")
        require(provenance["raw_sha256"] == sha(destination / "raw.json"), "raw hash differs")
        require(provenance["source_commit"] == manifest["source_commit"] and provenance["binary_sha256"] == manifest["binary_sha256"] and
                provenance["runtime_libraries"] == manifest["runtime_libraries"], "native source/binary/library binding differs")
        require(provenance["spec_sha256"] == object_sha(cfg["spec"]) and provenance["calibration"] == cfg["calibration"], "spec/calibration binding differs")
        for key in ("program_path", "ae_schedule_path", "vlm_path", "calibrated_vlm_path"):
            require(provenance["inputs"][key]["sha256"] == manifest["inputs"][cfg[key]] == sha(cfg[key]), f"input binding differs: {key}")
        program, schedule, vlm, ledger = (read(cfg[key]) for key in ("program_path", "ae_schedule_path", "vlm_path", "calibrated_vlm_path"))
        require(ledger["source"]["vlm_sha256"] == sha(cfg["vlm_path"]) and ledger["source"]["spec_sha256"] == object_sha(cfg["spec"]), "ledger source VLM/spec binding differs")
        require(ledger["calibration_parameters"] == cfg["calibration"], "ledger calibration differs")
        row, phase_rows = audit_case(read(destination / "raw.json"), cfg, program, schedule, vlm, ledger, read(destination / "execution.json"))
        row["case"] = case
        require(row["PIM_descriptor_continuations"] == manifest["expected_descriptor_continuations"][case],
                "frozen continuation contract differs")
        results.append(row)
        all_phases.extend(dict(phase, case=case) for phase in phase_rows)
        evidence[case] = {name: sha(destination / name) for name in ("raw.json", "provenance.json", "execution.json", "input-config.json")}
    summary = dict(passed=True, execution_model=MODEL, window_policy=WINDOW_POLICY,
                   source_commit=manifest["source_commit"], results=results,
                   scope="Full 18 layers x 10 steps; all 2357 calibrated VLM units; f/g once each",
                   native_GPU_requests_simulated=False, recalibration_executed=False, repeat_consistency_claimed=False,
                   PIM_native_definition="Native PIM stage time minus explicit VLM join time; no AE-solo estimate",
                   manifest_sha256=sha(manifest_path), evidence=evidence)
    (run / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    for name, rows in (("summary.csv", results), ("phase-durations.csv", all_phases)):
        with (run / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
