"""Calibrate closed-loop VLM memory-level parallelism with native DRAM probes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, percentile, read_json, sha256, source_revision, write_json
from experiments.native import execute, runtime_libraries, total_us


@contextmanager
def native_probe(binary, config_path):
    with subprocess.Popen([str(binary), "--probe-mlp", str(config_path)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        def memory_cycles(ordinal, window):
            proc.stdin.write(f"{ordinal} {window}\n")
            proc.stdin.flush()
            response = proc.stdout.readline()
            if not response:
                raise RuntimeError(f"native probe failed at ordinal {ordinal}, W={window}")
            return int(response)

        try:
            yield memory_cycles
        finally:
            proc.stdin.close()
        if proc.wait():
            raise RuntimeError("native probe exited unsuccessfully")


def minimum_window(probe, low, high, limit):
    """Bisect a bracket whose upper endpoint meets the isolated deadline."""
    while low < high:
        mid = (low + high) // 2
        if probe(mid) <= limit:
            high = mid
        else:
            low = mid + 1
    return low


def calibrate_unit(probe, unit, w_max, tick_us):
    measured_cycles = unit["us"] / tick_us
    fast = probe(w_max)
    if fast <= 0.8 * measured_cycles:
        classification, w_k = "compute_limited", w_max
    elif fast > measured_cycles:
        classification, w_k = "simulator_slower_than_thor", w_max
    else:
        classification = "memory_limited"
        w_k = minimum_window(probe, 1, w_max, measured_cycles)
    return dict(ordinal=unit["ordinal"], kind=unit["kind"], name=unit["name"],
                measured_us=unit["us"], classification=classification, w_k=w_k,
                w_max_isolated_memory_ck=fast, isolated_memory_ck=probe(w_k))


def refine_unit(probe, row, observed, w_max, tick_us):
    # Estimate the isolated budget by removing the startup/refresh penalty
    # observed in the continuous run. Only memory-limited units can change W.
    measured = row["measured_us"]
    row["initial_w_k"] = row["w_k"]
    row["initial_solo_duration_us"] = observed["duration_us"]
    if row["classification"] != "memory_limited" or observed["duration_us"] <= 1.05 * measured:
        return
    penalty = observed["memory_ck"] - row["isolated_memory_ck"]
    limit = measured / tick_us - max(0, penalty)
    row["refinement_isolated_budget_ck"] = limit
    if probe(w_max) > limit:
        row["w_k"] = w_max
    else:
        row["w_k"] = minimum_window(probe, row["w_k"], w_max, limit)
    row["isolated_memory_ck"] = probe(row["w_k"])


def probe_units(binary, config_path, units, work):
    # Use the CPUs granted by the scheduler. This changes only host throughput.
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
    workers = min(cpu_count, len(units))
    ordered = sorted(units, key=lambda u: u["read_sectors"] + u["write_sectors"], reverse=True)

    def batch(items):
        result = {}
        with native_probe(binary, config_path) as native:
            for unit in items:
                ordinal = unit["ordinal"]
                samples = {}

                def probe(window):
                    if window not in samples:
                        samples[window] = native(ordinal, window)
                    return samples[window]

                row = work(probe, unit)
                row["probe_memory_ck"].update({str(w): ck for w, ck in samples.items()})
                result[ordinal] = row
        return result

    if not workers:
        return {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return {ordinal: row for shard in pool.map(batch, (ordered[i::workers] for i in range(workers)))
                for ordinal, row in shard.items()}


def source_sha256():
    if subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=ROOT).strip():
        raise ValueError("commit source changes before calibrating")
    digest = hashlib.sha256()
    with subprocess.Popen(["git", "archive", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE) as proc:
        for block in iter(lambda: proc.stdout.read(1024 * 1024), b""):
            digest.update(block)
        if proc.wait():
            raise RuntimeError("git archive failed")
    return digest.hexdigest()


def calibrate(binary, vlm_path, spec_path, output, w_max):
    workload = read_json(vlm_path)
    units = workload["units"]
    if Counter(u["kind"] for u in units) != {"kernel": 2353, "memory": 4}:
        raise ValueError("calibration requires 2353 kernels and four memory copies")
    if len({u["ordinal"] for u in units}) != len(units):
        raise ValueError("VLM input has duplicate ordinals")
    spec = read_json(spec_path)
    tick_us = spec["clock_ps"] / 1e6
    output.mkdir(parents=True, exist_ok=False)
    provenance = dict(
        source_commit=source_revision(), source_sha256=source_sha256(),
        source_hash_format="SHA-256 of git archive HEAD (tar)",
        binary_sha256=sha256(binary), runtime_libraries=runtime_libraries(binary),
        inputs={"vlm": {"path": str(vlm_path), "sha256": sha256(vlm_path)},
                "spec": {"path": str(spec_path), "sha256": sha256(spec_path)}},
        isolated_initial_state={
            "start_ck": 0, "banks": "native power-on closed state",
            "rank0_refresh": "REFpb, first due at nREFIpb, pair index 0",
            "rank1_refresh": "REFab, first due at nREFI",
            "policy": "frfcfs",
        },
    )
    config = dict(run_kind="vlm_solo", layout="feature", rope="gpu", dataflow="streaming", mode="serial",
                  execution_model="native_dram_requests", vlm_release="closed_loop", w_max=w_max,
                  vlm_path=str(vlm_path), spec=spec)
    probe_config = output / "probe-config.json"
    write_json(probe_config, config)
    provenance["probe_config_sha256"] = sha256(probe_config)

    def first_probe(probe, unit):
        return dict(calibrate_unit(probe, unit, w_max, tick_us), probe_memory_ck={})

    print(f"Isolated probes for {len(units)} work units, W_max={w_max}", flush=True)
    by_ordinal = probe_units(binary, probe_config, units, first_probe)
    rows = [by_ordinal[u["ordinal"]] for u in units]
    ledger = dict(schema_version=1, kind="kernel_mlp", w_max=w_max, alpha=1.0, clock_ps=spec["clock_ps"],
                  write_high=3*w_max//4, write_low=w_max//4, units=rows, provenance=provenance)
    initial_path = output / "initial-mlp.json"
    write_json(initial_path, ledger)
    print("First continuous VLM solo", flush=True)
    initial_dir = output / "initial-solo"
    initial = execute(binary, dict(config, mlp_path=str(initial_path)), initial_dir)
    if [k["ordinal"] for k in initial["kernels"]] != [u["ordinal"] for u in units]:
        raise ValueError("continuous solo did not execute the input work-unit order")
    observed = {k["ordinal"]: k for k in initial["kernels"]}
    original_windows = [row["w_k"] for row in rows]

    def refine_probe(probe, unit):
        row = by_ordinal[unit["ordinal"]]
        refine_unit(probe, row, observed[row["ordinal"]], w_max, tick_us)
        return row

    refine = []
    for unit in units:
        row = by_ordinal[unit["ordinal"]]
        first = observed[unit["ordinal"]]
        row.update(initial_w_k=row["w_k"], initial_solo_duration_us=first["duration_us"],
                   initial_solo_memory_ck=first["memory_ck"])
        if row["classification"] == "memory_limited" and first["duration_us"] > 1.05 * unit["us"]:
            refine.append(unit)
    print(f"One local refinement for {len(refine)} memory-limited units", flush=True)
    probe_units(binary, probe_config, refine, refine_probe)

    selection_path = output / "selected-mlp.json"
    write_json(selection_path, ledger)
    selected = dict(config, mlp_path=str(selection_path))
    final_dirs = [output / "solo-repeat-0", output / "solo-repeat-1"]
    if original_windows == [row["w_k"] for row in rows]:
        # This execution already used the exact final window vector.
        final_dirs[0] = initial_dir
        final = initial
        execute(binary, selected, final_dirs[1])
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            final, _ = pool.map(lambda path: execute(binary, selected, path), final_dirs)
    hashes = [sha256(directory / "raw.json") for directory in final_dirs]
    if len(set(hashes)) != 1:
        raise ValueError(f"continuous VLM solo repetitions differ: {hashes}")

    histogram = Counter()
    by_name = defaultdict(Counter)
    slow = []
    nonmonotonic = []
    errors = []
    total = total_us(final)
    compute_us = 0.0
    for row, observed in zip(rows, final["kernels"], strict=True):
        if (row["ordinal"], row["w_k"]) != (observed["ordinal"], observed["w_k"]):
            raise ValueError("final raw does not match the selected kernel window vector")
        measured = row["measured_us"]
        error = (observed["duration_us"] - measured) / measured if measured else None
        isolated_us = row["isolated_memory_ck"] * tick_us
        row.update(isolated_memory_us=isolated_us,
                   isolated_duration_us=max(isolated_us, observed["floor_ck"] * tick_us),
                   solo_memory_ck=observed["memory_ck"], solo_duration_us=observed["duration_us"],
                   relative_error=error)
        if row["kind"] == "kernel" and error is not None:
            errors.append(error)
        histogram[row["w_k"]] += 1
        by_name[row["name"]][row["w_k"]] += 1
        samples = sorted((int(w), ck) for w, ck in row["probe_memory_ck"].items())
        if any(a[1] < b[1] for a, b in zip(samples, samples[1:])):
            nonmonotonic.append(row["ordinal"])
        if row["classification"] == "compute_limited":
            compute_us += row["solo_duration_us"]
        if row["classification"] == "simulator_slower_than_thor":
            slow.append(dict(ordinal=row["ordinal"], name=row["name"], kind=row["kind"],
                             measured_us=measured, isolated_memory_us=isolated_us,
                             isolated_excess_us=isolated_us-measured,
                             solo_excess_us=row["solo_duration_us"]-measured))
    target = 104950.496
    error = total / target - 1
    ledger["validation"] = dict(
        passed=abs(error) <= 0.05, target_us=target, total_us=total, relative_error=error,
        kernel_error_median=percentile(errors, .5), kernel_error_p90=percentile(errors, .9),
        kernel_error_max=max(errors), kernel_errors_over_5_pct=sum(x > .05 for x in errors),
        classifications=dict(Counter(r["classification"] for r in rows)),
        w_k_histogram=dict(histogram), w_k_by_name=dict(by_name),
        compute_limited_time_share=compute_us / total, compute_limited_us=compute_us,
        slower_than_thor=slow, observed_nonmonotonic_ordinals=nonmonotonic,
        raw_paths=[str(d / "raw.json") for d in final_dirs], raw_sha256=hashes,
        byte_identical=True, refined_ordinals=[u["ordinal"] for u in refine],
    )
    if (sha256(binary) != provenance["binary_sha256"] or
        runtime_libraries(binary) != provenance["runtime_libraries"] or
        sha256(vlm_path) != provenance["inputs"]["vlm"]["sha256"] or
        sha256(spec_path) != provenance["inputs"]["spec"]["sha256"]):
        raise ValueError("calibration inputs or runtime changed")
    write_json(output / "calibration-mlp.json", ledger)
    print(f"VLM solo {total/1000:.9f} ms, error {error:+.3%}, identical SHA-256 {hashes[0]}", flush=True)
    return ledger["validation"]["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/pim_vla_sim")
    parser.add_argument("--vlm", type=Path, default=ROOT / "input-data/vlm.json")
    parser.add_argument("--spec", type=Path, default=ROOT / "reference/spec/spec.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--w-max", type=int, choices=(128, 256), default=128)
    args = parser.parse_args()
    return 0 if calibrate(args.binary.resolve(), args.vlm.resolve(), args.spec.resolve(),
                          args.output.resolve(), args.w_max) else 2


if __name__ == "__main__":
    sys.exit(main())
