"""Run repeated native b–g experiments with a passed closed-loop MLP ledger."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

from analysis.report import duration, kernel_active_duration, summarize
from experiments.common import ROOT, read_json, sha256, write_csv, write_json
from experiments.config import variant_name
from experiments.native import execute, library_hashes, runtime_libraries, total_us


def load_calibration(binary, config, path, *, allow_scaled_unit_errors=False):
    ledger = read_json(path)
    scaled = ledger.get("calibration_model") == "scaled_ablation"
    if ledger.get("kind") != "kernel_mlp" or not (
        ledger["validation"]["passed"] or (scaled and allow_scaled_unit_errors)
    ):
        raise ValueError("closed-loop MLP calibration has not passed")
    provenance = ledger["provenance"]
    if (provenance["binary_sha256"] != sha256(binary) or
        library_hashes(provenance) != library_hashes({"runtime_libraries": runtime_libraries(binary)})):
        raise ValueError("native binary/library differs from closed-loop calibration")
    if provenance["inputs"]["vlm"]["sha256"] != sha256(config["vlm_path"]):
        raise ValueError("VLM input differs from closed-loop calibration")
    spec_file = provenance["inputs"]["spec"]
    if sha256(spec_file["path"]) != spec_file["sha256"] or read_json(spec_file["path"]) != config["spec"]:
        raise ValueError("spec differs from closed-loop calibration")
    validation = ledger["validation"]
    if scaled:
        [first] = validation["raw_paths"]
        if ([sha256(first)] != validation["raw_sha256"] or
            (validation["unit_violations"] and not allow_scaled_unit_errors) or
            abs(validation["relative_error"]) > .01):
            raise ValueError("scaled solo provenance or required timing validation failed")
    else:
        first, second = validation["raw_paths"]
        hashes = [sha256(p) for p in (first, second)]
        if hashes != validation["raw_sha256"] or len(set(hashes)) != 1:
            raise ValueError("calibrated solo repetitions differ")
    solo = read_json(first)
    if (solo["vlm_release"] != "closed_loop" or solo["w_max"] != ledger["w_max"] or
        total_us(solo) != validation["total_us"] or
        [(k["ordinal"], k["w_k"]) for k in solo["kernels"]] !=
        [(k["ordinal"], k["w_k"]) for k in ledger["units"]]):
        raise ValueError("MLP ledger windows or timing differ from the repeated solo")
    return ledger, solo


def repeat(binary, config, output, *, resume):
    paths = [output / f"repeat-{i}" for i in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, _ = pool.map(lambda path: execute(binary, config, path, resume=resume), paths)
    raw_files = [path / "raw.json" for path in paths]
    if raw_files[0].read_bytes() != raw_files[1].read_bytes():
        raise ValueError(f"repeated raw.json differs byte-for-byte in {output}")
    return first, sha256(raw_files[0])


def phase_utilization(raw):
    totals = defaultdict(Counter)
    for bus in raw["bus"]:
        if bus["stage_id"] < 0:
            continue
        phases = bus["phase_cycles"]
        if sum(p["duration"] for p in phases.values()) != bus["duration_cycles"]:
            raise ValueError("bus phases did not partition stage time")
        if sum(p["gpu_dq"] + p["pim_dq"] for p in phases.values()) != bus["dq_cycles"]:
            raise ValueError("DQ attribution did not conserve occupied cycles")
        if sum(p["gpu_ca"] + p["pim_ca"] for p in phases.values()) != bus["ca_cycles"]:
            raise ValueError("CA attribution did not conserve occupied cycles")
        for phase, cycles in phases.items():
            elapsed = cycles["duration"]
            if (cycles["gpu_ca"] + cycles["pim_ca"] > elapsed or
                cycles["gpu_dq"] + cycles["pim_dq"] > elapsed or cycles["pim_private_ca"] > elapsed):
                raise ValueError(f"impossible bus utilization in {phase}")
            totals[phase].update(cycles)
    return {
        phase: dict(channel_cycles=c["duration"], **{k: v / c["duration"] if c["duration"] else None
                    for k, v in c.items() if k != "duration"})
        for phase, c in totals.items()
    }


def run_closed_loop(binary, configs, output, calibration_path, cases, *, resume=False,
                    allow_scaled_unit_errors=False):
    output = Path(output)
    calibration_path = Path(calibration_path).resolve()
    ledger, solo = load_calibration(binary, configs["b"], calibration_path,
                                    allow_scaled_unit_errors=allow_scaled_unit_errors)
    historical = {r["case"]: r for r in read_json(
        ROOT / "reference/e2e-gio-bcast-20261109/data/completed-cases.json")["evaluation"]}
    solo_kernels = {k["ordinal"]: k for k in solo["kernels"]}
    units = {u["ordinal"]: u for u in ledger["units"]}
    solos = {}
    results = []
    for case in cases:
        cfg = dict(configs[case], w_max=ledger["w_max"], mlp_path=str(calibration_path))
        variant = variant_name(cfg)
        if variant not in solos:
            ae_cfg = dict(cfg, run_kind="ae_solo")
            ae_cfg.pop("mlp_path")
            ae_cfg.pop("vlm_gpu_quantum_us", None)
            print(f"AE solo {variant}, two repetitions", flush=True)
            solos[variant] = repeat(binary, ae_cfg, output / "ae-solo" / variant, resume=resume)
        ae_solo, ae_hash = solos[variant]
        print(f"Closed-loop {case}, two repetitions", flush=True)
        raw, raw_hash = repeat(binary, cfg, output / "native" / case, resume=resume)
        if raw["execution_model"] != "native_dram_requests" or raw["vlm_release"] != "closed_loop":
            raise ValueError("case did not use the common native closed-loop model")
        report = summarize(case, raw, ae_solo, solo, output, program=read_json(cfg["program_path"]),
                           calibration={"parameters": {"alpha": 1.0}}, vlm_input=read_json(cfg["vlm_path"]))
        residual = 1000 * (report["total_ms"] - sum(report[k] for k in (
            "AE_solo_ms", "PIM_interference_ms", "AE_GPU_wait_ms", "VLM_remaining_ms")))
        if abs(residual) > 0.002:
            raise ValueError(f"{case}: total timestamp identity failed ({residual} µs)")
        stalls = defaultdict(list)
        for row in raw["kernels"]:
            ordinal = row["ordinal"]
            if row["w_k"] != units[ordinal]["w_k"]:
                raise ValueError(f"{case}: used a different W_k for ordinal {ordinal}")
            stalls[units[ordinal]["classification"]].append(kernel_active_duration(row) - duration(solo_kernels[ordinal]))
        delta = {k: math.fsum(stalls[k]) for k in (
            "compute_limited", "memory_limited", "simulator_slower_than_thor")}
        phases = phase_utilization(raw)
        result = dict(
            **report, w_max=ledger["w_max"], total_identity_residual_us=residual,
            vlm_compute_limited_delta_us=delta["compute_limited"],
            vlm_memory_side_delta_us=delta["memory_limited"] + delta["simulator_slower_than_thor"],
            vlm_slower_than_thor_subset_delta_us=delta["simulator_slower_than_thor"],
            raw_sha256=raw_hash, ae_solo_raw_sha256=ae_hash, raw_repeats_byte_identical=True,
            uniform_total_ms=historical.get(case, {}).get("total_ms"),
            **{f"{phase}_{k}": v for phase, fields in phases.items() for k, v in fields.items()},
        )
        results.append(result)
        write_json(output / "closed-loop-summary.json", {
            "execution_model": "native_dram_requests", "vlm_release": "closed_loop",
            "mlp_sha256": sha256(calibration_path), "evaluation": results,
            "allow_scaled_unit_errors": allow_scaled_unit_errors,
        })
        write_csv(output / "closed-loop-evaluation.csv", results)
    return results
