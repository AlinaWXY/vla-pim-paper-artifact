"""Prepare and audit full native MC ACTAB comparisons."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ramulator2/python"))
from experiments.common import read_json, sha256, write_csv, write_json
from experiments.config import case_config, variant_name
from ramulator.dram.lpddr5x_pim import build_spec
from tools.capture_gap_traces import generate, native
from tools.estimate_dq_gaps import FUSED_SHARED_IO, load_trace

CASES = "bcdefg"
MODELS = ("instant", "four-batch")
CALIBRATION = dict(alpha=1.0, window=1024, write_high=768, write_low=256)


def io_budget(traces, case):
    """Freeze the old shared-tensor overlap convention from the instant trace."""
    budgets = []
    for step, layer in ((0, 0), (9, 17)):
        path = traces / f"{case}-s{step}-l{layer}"
        timeline, geometry, _, _ = load_trace(path, step, layer, shared_io_rules=FUSED_SHARED_IO[case])
        budgets.append({row["name"]: (row["shared_tensor_io"]["end_ck"] -
                                        row["shared_tensor_io"]["start_ck"])
                        * timeline["clock_ps"] / 1e6
                        for row in geometry if "shared_tensor_io" in row})
    return {name: sum(item[name] for item in budgets) / len(budgets) for name in budgets[0]}


def debit_profile(schedule, budgets):
    """Keep explicit PIM I/O and debit the shared portion of each GPU block."""
    events = schedule["events"]
    debits = Counter()
    for position, event in enumerate(events):
        stage = event.get("stage")
        if stage not in budgets:
            continue
        backward = stage in {"qkv", "pv", "gate_up", "down", "rope_import"}
        indices = range(position - 1, -1, -1) if backward else range(position + 1, len(events))
        block = []
        for index in indices:
            other = events[index]
            if (other["step"], other["layer"]) != (event["step"], event["layer"]):
                break
            if other["kind"] == "gpu":
                block.append(other)
            elif block:
                break
        # The Down input budget exceeds GELU + multiply in some blocks.
        # Match at layer granularity so the measured nonlinearity keeps
        # its kernel boundaries and the original layer budget is debited once.
        if stage in {"down", "gate_up", "qkv"}:
            block = [other for other in events if other["kind"] == "gpu"
                     and (other["step"], other["layer"]) == (event["step"], event["layer"])]
        total = math.fsum(other["us"] for other in block)
        if total <= budgets[stage]:
            raise ValueError(f"Shared tensor debit exceeds GPU block at {event}")
        for other in block:
            share = (other["us"] / total) * budgets[stage]
            other["us"] -= share
            other["shared_tensor_debit_us"] = other.get("shared_tensor_debit_us", 0) + share
        debits[stage] += budgets[stage]
    schedule["gpu_us"] = math.fsum(e["us"] for e in events if e["kind"] == "gpu")
    schedule["shared_tensor_debits_us"] = dict(debits)
    return schedule


def prepare(root, traces, binary, programs):
    inputs = root / "inputs"
    inputs.mkdir()
    shutil.copyfile(ROOT / "input-data/vlm.json", inputs / "vlm.json")
    for model in MODELS:
        spec = build_spec(2, mc_control=True, four_batch_actab=model == "four-batch")
        spec["metadata"]["batch_pim_columns"] = True
        write_json(inputs / f"{model}-spec.json", spec)
    budgets = {case: io_budget(traces, case) for case in "fg"}
    write_json(inputs / "shared-io-budgets-us.json", budgets)
    variants = sorted({variant_name(case_config(case)) for case in CASES})
    if programs:
        (inputs / "programs").mkdir()
        for variant in variants:
            shutil.copyfile(programs / f"{variant}.json", inputs / "programs" / f"{variant}.json")
    else:
        with ThreadPoolExecutor(max_workers=len(variants)) as pool:
            jobs = [pool.submit(generate, variant, inputs / "programs", "native_dram_requests", mc_control=True)
                    for variant in variants]
            for job in jobs:
                job.result()
    for case in CASES:
        config = case_config(case, mac_freq_mult=2, vlm_release="uniform", calibration=CALIBRATION)
        variant = variant_name(config)
        program = inputs / "programs" / f"{variant}.json"
        schedule = read_json(ROOT / f"input-data/ae-{variant}.json")
        if case in budgets:
            debit_profile(schedule, budgets[case])
        schedule_path = inputs / f"{case}-ae.json"
        write_json(schedule_path, schedule)
        for model in MODELS:
            write_json(inputs / f"{model}-{case}.json", dict(config, program_path=str(program),
                       ae_schedule_path=str(schedule_path), vlm_path=str(inputs / "vlm.json"),
                       spec_path=str(inputs / f"{model}-spec.json")))
    write_json(inputs / "manifest.json", {
        "binary_sha256": sha256(binary),
        "inputs": {str(path.relative_to(inputs)): sha256(path)
                   for path in sorted(inputs.rglob("*.json"))},
    })


def run_full(root, binary, jobs, labels):
    manifest = read_json(root / "inputs/manifest.json")
    if manifest["binary_sha256"] != sha256(binary):
        raise ValueError("Native binary changed after input preparation")
    for name, digest in manifest["inputs"].items():
        if sha256(root / "inputs" / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        runs = [pool.submit(native, binary, read_json(root / "inputs" / f"{label}.json"),
                            root / "runs" / label) for label in labels]
        for run in runs:
            run.result()


def audit(root):
    inputs = root / "inputs"
    vlm = read_json(inputs / "vlm.json")["units"]
    totals = [32 * sum(unit[f"{direction}_sectors"] for unit in vlm) for direction in ("read", "write")]
    rows, stages = [], []
    for case in CASES:
        pair = [read_json(inputs / f"{model}-{case}.json") for model in MODELS]
        for config in pair:
            config.pop("spec_path")
        if pair[0] != pair[1]:
            raise ValueError(f"Inputs differ outside the ACTAB spec for {case}")
        schedule = read_json(pair[0]["ae_schedule_path"])
        expected = Counter((e["step"], e["layer"], e["stage"]) for e in schedule["events"] if e["kind"] == "pim")
        if len({(step, layer) for step, layer, _ in expected}) != 180 or set(expected.values()) != {1}:
            raise ValueError(f"Incomplete AE schedule for {case}")
        pim_bytes = []
        for model in MODELS:
            raw = read_json(root / "runs" / f"{model}-{case}" / "raw.json")
            metrics = raw["metrics"]
            actual = Counter((s["step"], s["layer"], s["name"]) for s in raw["stages"])
            if actual != expected or len(raw["kernels"]) != len(vlm):
                raise ValueError(f"Incomplete AE or VLM execution for {model}/{case}")
            for kernel, unit in zip(raw["kernels"], vlm, strict=True):
                if (kernel["ordinal"] != unit["ordinal"]
                        or kernel["read_bytes"] != 32 * unit["read_sectors"]
                        or kernel["write_bytes"] != 32 * unit["write_sectors"]):
                    raise ValueError(f"VLM kernel traffic differs for {model}/{case}")
            if [metrics[f"gpu_{direction}_bytes"] for direction in ("read", "write")] != totals:
                raise ValueError(f"VLM traffic changed for {model}/{case}")
            if not math.isclose(metrics["total_us"], metrics["AE_elapsed_us"] + metrics["VLM_remaining_us"], abs_tol=1e-6):
                raise ValueError(f"E2E does not reconcile for {model}/{case}")
            if metrics["serial_epoch_reset"] != (case == "b"):
                raise ValueError(f"Unexpected serial epoch boundary for {model}/{case}")
            pim_bytes.append([metrics[f"pim_{direction}_bytes"] for direction in ("read", "write")])
            stage_us = Counter()
            for stage in raw["stages"]:
                stage_us[stage["name"]] += stage["duration_us"]
            stages.extend(dict(case=case, model=model, stage=stage, total_ms=us / 1000)
                          for stage, us in sorted(stage_us.items()))
            rows.append(dict(case=case, model=model, e2e_ms=metrics["total_us"] / 1000,
                             ae_ms=metrics["AE_elapsed_us"] / 1000, pim_ms=stage_us.total() / 1000,
                             gpu_wait_ms=metrics["AE_GPU_wait_us"] / 1000,
                             vlm_tail_ms=metrics["VLM_remaining_us"] / 1000,
                             pim_read_bytes=pim_bytes[-1][0], pim_write_bytes=pim_bytes[-1][1],
                             vlm_requests=sum(totals) // 32, layers_steps=180, vlm_units=len(vlm)))
        if pim_bytes[0] != pim_bytes[1]:
            raise ValueError(f"PIM external traffic changed for {case}")
    write_csv(root / "full-summary.csv", rows)
    write_csv(root / "full-stage-summary.csv", stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--programs", type=Path, help="reuse the frozen MC-controlled programs")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--labels", nargs="+")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(output, args.traces.resolve(), args.binary.resolve(), args.programs)
    elif args.action == "run":
        labels = args.labels or [f"{model}-{case}" for model in MODELS for case in CASES]
        run_full(output, args.binary.resolve(), args.jobs, labels)
    else:
        audit(output)


if __name__ == "__main__":
    main()
