"""Fit the Stage A slow units' traffic to the captured GPU profile."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, read_json, sha256, source_revision, write_json
from experiments.native import execute, library_hashes, runtime_libraries
from tools.calibrate_kernel_mlp import native_probe, source_sha256

PROFILE = ROOT / "input-data/vlm.json"
SPEC = ROOT / "reference/spec/spec.json"
STAGE_A = ROOT / "reference/closed-loop-vlm/stage-a/calibration-mlp.json"
TARGET_US = 105220.578
SLOW = "simulator_slower_than_thor"
HISTOGRAM_EDGES = (0, 0.25, 0.5, 0.75, 0.9, 1)


def traffic(unit, numerator, denominator):
    """Round each stream up to sectors using the same exact scale."""
    return {key: (unit[key] * numerator + denominator - 1) // denominator
            for key in ("read_sectors", "write_sectors")}


def scaled_unit(unit, numerator, denominator):
    return dict(unit, read_address_extent_sectors=unit["read_sectors"],
                **traffic(unit, numerator, denominator))


@dataclass
class ScaleSearch:
    unit: dict
    denominator: int
    limit: Fraction
    high: int
    low: int = 0
    memory_ck: int = 0

    def query(self):
        candidate = (self.low + self.high) // 2
        return scaled_unit(self.unit, candidate, self.denominator)

    def observe(self, request, cycles):
        # Jump over scales with exactly the same integer request counts.
        # Each ceil(count * s) changes only at an integer sector boundary.
        starts, ends = [], []
        for key in ("read_sectors", "write_sectors"):
            count = self.unit[key]
            if count:
                starts.append((request[key] - 1) * self.denominator // count + 1)
                ends.append(request[key] * self.denominator // count)
        if cycles <= self.limit:
            self.low = min(self.high - 1, min(ends))
            self.memory_ck = cycles
        else:
            self.high = max(self.low + 1, max(starts))


def probe_requests(binary, directory, requests):
    vlm_path = directory / "vlm.json"
    cfg = directory / "probe-config.json"
    write_json(vlm_path, {"units": requests})
    write_json(cfg, dict(spec_path=str(SPEC), vlm_path=str(vlm_path), w_max=128))
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
    cpus = min(cpus, int(os.environ.get("SLURM_CPUS_PER_TASK", cpus)), len(requests))
    ordered = sorted(requests, key=lambda u: u["read_sectors"] + u["write_sectors"], reverse=True)

    def batch(units):
        with native_probe(binary, cfg) as probe:
            return {unit["ordinal"]: probe(unit["ordinal"], 128) for unit in units}

    with ThreadPoolExecutor(max_workers=cpus) as pool:
        return {key: ck for shard in pool.map(batch, (ordered[i::cpus] for i in range(cpus)))
                for key, ck in shard.items()}


def solve(binary, output, searches):
    with TemporaryDirectory(prefix="scale-probes-", dir=output) as tmp:
        active = [search for search in searches if search.high > search.low + 1]
        while active:
            requests = [search.query() for search in active]
            clocks = probe_requests(binary, Path(tmp), requests)
            for search, request in zip(active, requests, strict=True):
                search.observe(request, clocks[request["ordinal"]])
            active = [search for search in active if search.high > search.low + 1]
            print(f"Isolated scale search: {len(active)}/{len(searches)} remaining", flush=True)
    return {search.unit["ordinal"]: search for search in searches}


def duration_budget(unit, clock_ps):
    return Fraction(str(unit["us"])) * 1_000_000 / clock_ps


def check_stage_a(source, provenance):
    stage_a = read_json(STAGE_A)
    binding = stage_a["provenance"]
    if (stage_a["alpha"], stage_a["w_max"], stage_a["write_high"], stage_a["write_low"]) != (1, 128, 96, 32):
        raise ValueError("Stage A hardware configuration differs")
    for key, path in (("vlm", PROFILE), ("spec", SPEC)):
        if sha256(path) != binding["inputs"][key]["sha256"]:
            raise ValueError(f"Stage A {key} input differs")
    if library_hashes(provenance) != library_hashes(binding):
        raise ValueError("reuse the verified Stage A runtime library")
    if [(r["ordinal"], r["measured_us"]) for r in stage_a["units"]] != [
        (u["ordinal"], u["us"]) for u in source["units"]
    ]:
        raise ValueError("Stage A unit sequence differs")
    if sum(r["classification"] == SLOW for r in stage_a["units"]) != 874:
        raise ValueError("Stage A slow unit count differs")
    return stage_a


def apply_search(row, search):
    if search.low == 0:
        raise ValueError(f"ordinal {row['ordinal']} has no feasible positive traffic scale")
    row.update(s_k=search.low / search.denominator, s_numerator=search.low,
               s_denominator=search.denominator, isolated_memory_ck=search.memory_ck)


def solo(binary, output, tag, rows, workload, ledger, config):
    vlm_path = output / f"{tag}-vlm.json"
    mlp_path = output / f"{tag}-mlp.json"
    units = [scaled_unit(unit, row["s_numerator"], row["s_denominator"])
             for unit, row in zip(workload["units"], rows, strict=True)]
    scaled = dict(workload, units=units,
                  read_bytes=sum(u["read_sectors"] for u in units)*32,
                  write_proxy_bytes=sum(u["write_sectors"] for u in units if u["write_is_proxy"])*32)
    write_json(vlm_path, scaled)
    write_json(mlp_path, ledger)
    destination = output / f"{tag}-solo"
    raw = execute(binary, dict(config, mlp_path=str(mlp_path), vlm_path=str(vlm_path)), destination)
    expected = [(u["ordinal"], r["w_k"], u["read_sectors"]*32, u["write_sectors"]*32)
                for u, r in zip(units, rows, strict=True)]
    actual = [(k["ordinal"], k["w_k"], k["read_bytes"], k["write_bytes"]) for k in raw["kernels"]]
    if actual != expected:
        raise ValueError("continuous solo differs from the window and traffic ledger")
    return raw, vlm_path, destination / "raw.json"


def summarize(ledger, raw, source, profile_path, raw_path):
    rows = ledger["units"]
    tick_us = ledger["clock_ps"] / 1e6
    over, histogram = [], {}
    scaled_measured_us, scaled_solo_us, deleted_read, deleted_write = 0.0, 0.0, 0, 0
    for row, unit, observed in zip(rows, source["units"], raw["kernels"], strict=True):
        measured = row["measured_us"]
        elapsed = observed["duration_us"]
        error = elapsed / measured - 1
        row.update(isolated_memory_us=row["isolated_memory_ck"]*tick_us,
                   isolated_duration_us=max(row["isolated_memory_ck"], observed["floor_ck"])*tick_us,
                   solo_duration_us=elapsed, solo_memory_us=observed["memory_ck"]*tick_us, relative_error=error)
        if abs(error) > .01:
            over.append(dict(ordinal=row["ordinal"], name=row["name"], classification=row["classification"],
                             w_k=row["w_k"], s_k=row["s_k"], measured_us=measured, solo_us=elapsed,
                             relative_error=error))
        s = row["s_k"]
        if s < 1:
            scaled_measured_us += measured
            scaled_solo_us += elapsed
        label = next((f"({lo:g}, {hi:g}]" for lo, hi in zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:])
                      if lo < s <= hi), None) if s < 1 else "1"
        histogram[label] = histogram.get(label, 0) + 1
        deleted_read += unit["read_sectors"] * 32 - observed["read_bytes"]
        deleted_write += unit["write_sectors"] * 32 - observed["write_bytes"]
    total = raw["metrics"]["total_us"]
    error = total / TARGET_US - 1
    ledger["provenance"]["inputs"]["vlm"] = dict(path=str(profile_path), sha256=sha256(profile_path))
    ledger["validation"] = dict(passed=not over and abs(error) <= .01,
                                target_us=TARGET_US, total_us=total, relative_error=error,
                                max_absolute_unit_error=max(abs(r["relative_error"]) for r in rows),
                                unit_violations=sorted(over, key=lambda r: -abs(r["relative_error"])),
                                raw_paths=[str(raw_path)], raw_sha256=[sha256(raw_path)],
                                s_k_histogram=histogram, scaled_units=sum(r["s_k"] < 1 for r in rows),
                                scaled_profile_time_share=scaled_measured_us / source["measured_duration_sum_us"],
                                scaled_solo_time_share=scaled_solo_us / total,
                                removed_read_bytes=deleted_read, removed_write_bytes=deleted_write)


def calibrate(binary, output):
    workload = read_json(PROFILE)
    spec = read_json(SPEC)
    clock_ps = spec["clock_ps"]
    provenance = dict(source_commit=source_revision(), source_archive_sha256=source_sha256(),
                      binary_sha256=sha256(binary), runtime_libraries=runtime_libraries(binary),
                      stage_a_sha256=sha256(STAGE_A),
                      inputs={"original_vlm": {"path": str(PROFILE), "sha256": sha256(PROFILE)},
                              "spec": {"path": str(SPEC), "sha256": sha256(SPEC)}})
    stage_a = check_stage_a(workload, provenance)
    provenance["stage_a_binary_sha256"] = stage_a["provenance"]["binary_sha256"]
    provenance["isolated_initial_state"] = stage_a["provenance"]["isolated_initial_state"]
    output.mkdir(parents=True, exist_ok=False)
    rows, searches = [], []
    for unit, reference in zip(workload["units"], stage_a["units"], strict=True):
        row = {key: reference[key] for key in ("ordinal", "name", "kind", "measured_us",
                                               "classification", "w_k", "isolated_memory_ck")}
        row.update(s_k=1.0, s_numerator=1, s_denominator=1)
        if row["classification"] == SLOW:
            if row["w_k"] != 128:
                raise ValueError("Stage A slow unit must have W_k=128")
            den = math.lcm(unit["read_sectors"] or 1, unit["write_sectors"] or 1)
            searches.append(ScaleSearch(unit, den, duration_budget(unit, clock_ps), den))
        rows.append(row)

    found = solve(binary, output, searches)
    for row in rows:
        if row["ordinal"] in found:
            apply_search(row, found[row["ordinal"]])
    ledger = dict(schema_version=1, kind="kernel_mlp", calibration_model="scaled_ablation",
                  w_max=128, alpha=1.0, write_high=96, write_low=32, clock_ps=clock_ps,
                  units=rows, provenance=provenance)
    config = dict(run_kind="vlm_solo", mode="serial", layout="feature", rope="gpu",
                  dataflow="streaming", vlm_release="closed_loop", w_max=128, spec=spec)
    print("Continuous solo on the initial scaled vector", flush=True)
    raw, profile_path, raw_path = solo(binary, output, "initial", rows, workload, ledger, config)
    local_searches = []
    for row, unit, observed in zip(rows, workload["units"], raw["kernels"], strict=True):
        row.update(initial_s_k=row["s_k"], initial_solo_duration_us=observed["duration_us"])
        if row["classification"] == SLOW and observed["duration_us"] > 1.01 * row["measured_us"]:
            penalty = max(0, observed["memory_ck"] - row["isolated_memory_ck"])
            limit = duration_budget(unit, clock_ps) - penalty
            local_searches.append(ScaleSearch(unit, row["s_denominator"], limit, row["s_numerator"]))
    if local_searches:
        print(f"One local scale refinement for {len(local_searches)} units", flush=True)
        refined = solve(binary, output, local_searches)
        for row in rows:
            if row["ordinal"] in refined:
                apply_search(row, refined[row["ordinal"]])
        print("Continuous solo on the refined vector", flush=True)
        raw, profile_path, raw_path = solo(binary, output, "final", rows, workload, ledger, config)
    summarize(ledger, raw, workload, profile_path, raw_path)
    write_json(output / "calibration-scaled.json", ledger)
    v = ledger["validation"]
    print(f"Scaled solo {v['total_us']/1000:.9f} ms, {len(v['unit_violations'])} units outside 1%", flush=True)
    return v["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/pim_vla_sim")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ablation-scaled-w128")
    args = parser.parse_args()
    return 0 if calibrate(args.binary.resolve(), args.output.resolve()) else 2


if __name__ == "__main__":
    sys.exit(main())
