"""Prepare and audit c–g with GPU output reads charged in the Thor timing."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from case_b_single_read import compile_schedule, write


CASES = "cdefg"
EXPORTS = {
    "rope_export": {"rope_export"}, "qkt": {"score_stream", "score_export"}, "pv": {"pv_export"},
    "o": {"o_tile0_export", "o_tile1_export", "o_export"}, "gate_up": {"gate_up_export"},
    "down": {"down_export"},
}
RETAINED_READS = {"c": 0, "d": 0, "e": 11796480, "f": 0, "g": 11796480}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, **kwargs):
    return json.loads(path.read_bytes(), **kwargs)


def prepare_case(case, base, out, times):
    old_dir = base / "ae-priority-2us" / f"full-{case}"
    config = read(old_dir / "config.json")
    assert config["case"] == case and config["vlm_gpu_quantum_us"] == 2
    source = Path(config["program_path"])
    program = read(source)
    dst = out / case
    dst.mkdir(parents=True, exist_ok=True)
    omitted = Counter()
    kept = Counter()
    for stage in program["stages"]:
        name = stage["name"]
        for channel in stage["channels"]:
            for index, op in enumerate(channel):
                assert op["op"] != "RDRES" and all(child["op"] not in {"RD", "RDRES"}
                                                      for child in op.get("children", ()))
                if op["op"] != "RD":
                    continue
                if name == "q_exchange_export":
                    assert op["epoch"] in {"q_exchange_export", "q_exchange_exclusive"}
                    kept[name] += 1
                    continue
                assert op["epoch"] in EXPORTS[name], (case, name, op["epoch"])
                barrier = {field: op[field] for field in ("id", "addr", "deps", "issue_deps") if field in op}
                channel[index] = dict(barrier, op="BARRIER")
                omitted[name] += 1
        stage["expected_dq_bytes"] -= omitted[name] * 32
    removed_count = sum(omitted.values())
    program["counts"]["RD"] -= removed_count
    program["counts"]["BARRIER"] += removed_count
    for field in ("expected_dq_bytes", "expected_dq_bytes_per_layer"):
        program[field] -= removed_count * 32
        assert program[field] == sum(s["expected_dq_bytes"] for s in program["stages"])
    old_metrics = read(old_dir / "raw.json")["metrics"]
    kept_read_bytes = sum(kept.values()) * 32 * 180
    assert kept_read_bytes == RETAINED_READS[case]
    assert kept_read_bytes + removed_count * 32 * 180 == old_metrics["pim_read_bytes"]
    new_program = dst / "pim-program.json"
    write(new_program, program)

    schedule = read(Path(config["ae_schedule_path"]))
    original_budget = math.fsum(e["us"] for e in schedule["events"] if e["kind"] == "gpu")
    restored = math.fsum(e.get("shared_tensor_debit_us", 0) for e in schedule["events"])
    assert (restored > 0) == (case in "fg")
    assert math.isclose(restored, math.fsum(schedule.pop("shared_tensor_debits_us", {}).values()), abs_tol=1e-7)
    for e in schedule["events"]:
        if "shared_tensor_debit_us" in e:
            e["us"] += e.pop("shared_tensor_debit_us")
    schedule["events"] = compile_schedule(schedule["events"], times, gpu_rope=config["rope"] == "gpu")
    schedule["gpu_us"] = math.fsum(e["us"] for e in schedule["events"] if e["kind"] == "gpu")
    ae_path = dst / "ae.json"
    write(ae_path, schedule)
    cfg = dict(config, ae_schedule_path=str(ae_path), program_path=str(new_program))
    write(dst / "config.json", cfg)
    write(dst / "scope.json", dict(case=case, source_program_sha256=digest(source),
                                   program_sha256=digest(new_program), schedule_sha256=digest(ae_path),
                                   restored_gpu_budget_us=restored, prev_gpu_budget_us=original_budget,
                                   retained_read_bytes=kept_read_bytes, excluded_read_bytes=removed_count * 32 * 180,
                                   expected_write_bytes=old_metrics["pim_write_bytes"],
                                   read_bursts_removed_per_layer=omitted, read_bursts_kept_per_layer=kept))


def audit_case(case, directory, prior, solo):
    raw = read(directory / "raw.json", parse_float=Decimal)
    baseline = read(prior / f"full-{case}" / "raw.json")
    inputs = read(directory.parent.parent / case / "scope.json")
    metrics, kernels, stages = raw["metrics"], raw["kernels"], raw["stages"]
    assert len(stages) == (1800 if case in "cdf" else 1620)
    assert metrics["pim_read_bytes"] == inputs["retained_read_bytes"]
    assert metrics["pim_write_bytes"] == inputs["expected_write_bytes"]
    assert [(s["name"], s["step"], s["layer"]) for s in stages] == [
        (s["name"], s["step"], s["layer"]) for s in baseline["stages"]]
    def stage_traffic(trace, direction):
        return {(r["stage_id"], r["channel"]): r[direction] for r in trace["bus"] if r["stage_id"] >= 0}

    assert stage_traffic(raw, "pim_write_bytes") == stage_traffic(baseline, "pim_write_bytes")
    config = read(directory / "config.json")
    schedule = read(Path(config["ae_schedule_path"]))
    profile = read(Path(config["vlm_path"]))
    alpha = config["calibration"]["alpha"]
    assert config["vlm_gpu_quantum_us"] == 2 and metrics["VLM_GPU_yields"] > 0
    assert len(kernels) == len(profile["units"]) == len(solo)
    for k, u in zip(kernels, profile["units"]):
        assert k["ordinal"] == u["ordinal"]
        assert k["read_bytes"] == u["read_sectors"] * 32
        assert k["write_bytes"] == (math.ceil(u["write_sectors"] * alpha)
                                     if u.get("write_is_proxy", True) else u["write_sectors"]) * 32
    tick_us = Decimal(raw["clock_ps"]) / 10**6
    gpu_ck = sum(math.ceil(e["us"] / float(tick_us) - 1e-10)
                 for e in schedule["events"] if e["kind"] == "gpu")
    gpu_us = tick_us * gpu_ck
    pim = sum(Decimal(s["duration_us"]) for s in stages)
    ae, wait = metrics["AE_elapsed_us"], metrics["AE_GPU_wait_us"]
    assert ae == gpu_us + pim + wait
    vlm = kernels[-1]["end_us"]
    assert metrics["total_us"] == max(ae, vlm)
    assert all(not k["paused_us"] for k in kernels if k["start_us"] >= ae)
    active = sum(min(k["end_us"], ae) - k["start_us"] - k["paused_us"]
                 for k in kernels if k["start_us"] < ae)
    finished = [k for k in kernels if k["end_us"] <= ae]
    solo_work = sum(solo[k["ordinal"]] for k in finished)
    return dict(case=case, ae_ms=ae / 1000, pim_ms=pim / 1000, ae_gpu_ms=gpu_us / 1000,
                ae_gpu_wait_ms=wait / 1000, vlm_during_ae_ms=active / 1000, vlm_finished_solo_ms=solo_work / 1000,
                vlm_finished_units=len(finished), vlm_units=len(kernels), vlm_total_ms=vlm / 1000,
                e2e_ms=metrics["total_us"] / 1000, vlm_requests=(metrics["gpu_read_bytes"] +
                                                                 metrics["gpu_write_bytes"]) // 32,
                excluded_read_bytes=inputs["excluded_read_bytes"], retained_read_bytes=metrics["pim_read_bytes"],
                pim_write_bytes=metrics["pim_write_bytes"], raw_sha256=digest(directory / "raw.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("base", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workload", default="legacy")
    args = parser.parse_args()
    if args.mode == "prepare":
        with (args.out / "comparison.csv").open(newline="") as handle:
            times = {row["case"]: float(row["compile_device_p50_us"]) for row in csv.DictReader(handle)}
        for case in CASES:
            prepare_case(case, args.base, args.out, times)
    else:
        runs = args.out / args.workload
        solo_path = (args.base / "runs/four-batch-b/raw.json" if args.workload == "legacy"
                     else runs / "solo/raw.json")
        solo = {k["ordinal"]: Decimal(k["duration_us"]) for k in read(solo_path, parse_float=Decimal)["kernels"]}
        results = [audit_case(c, runs / c, args.base / "ae-priority-2us", solo) for c in CASES]
        with (runs / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
