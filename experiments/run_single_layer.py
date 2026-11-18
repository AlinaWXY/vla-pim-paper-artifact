"""Run only step 0 / layer 0 of the five AE designs, with no VLM workload."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, object_sha256, read_json, sha256, source_revision, write_csv, write_json
from experiments.config import CASES, GR00T_CASES, validate_config, variant_name
from experiments.native import execute, runtime_libraries
from experiments.prepare_inputs import ae_schedule, gpu_measurements, single_layer_schedule
from workloads.models import MODELS, PI05

VARIANTS = {variant_name(CASES[c]): c for c in ("b", "d", "e", "f", "g")}


def prepare(output, *, input_dir, program_dir, spec):
    output, input_dir, program_dir = Path(output), Path(input_dir), Path(program_dir)
    ledger = read_json(input_dir / "measured/natural-ae-kernel-ledger.json")
    helpers, transpose = gpu_measurements(input_dir)
    configs = {}
    for variant, case in VARIANTS.items():
        full = read_json(input_dir / f"ae-{variant}.json")
        schedule = single_layer_schedule(full, ledger)
        expected = single_layer_schedule(ae_schedule(ledger, CASES[case], helpers=helpers, transpose_us=transpose), ledger)
        if schedule["events"] != expected["events"]:
            raise ValueError(f"{variant}: single-layer inputs differ from the measured-source reconstruction")
        schedule["source_full_schedule_sha256"] = sha256(input_dir / f"ae-{variant}.json")
        schedule_path = (output / "inputs" / f"ae-{variant}-step0-layer0.json").resolve()
        write_json(schedule_path, schedule)
        program_path = (program_dir / f"{variant}.json").resolve()
        if not program_path.is_file():
            raise FileNotFoundError(program_path)
        cfg = {k: v for k, v in CASES[case].items() if k != "arbitration"}
        cfg.update(case=case, mode="serial", run_kind="ae_solo", spec=spec,
                   mac_freq_mult=spec["mac_freq_mult"], program_path=str(program_path),
                   ae_schedule_path=str(schedule_path))
        # There is deliberately no VLM path or calibration entry. Native AE
        # solo never constructs GPU memory work and never runs the fit pipeline.
        validate_config(cfg)
        write_json(output / "configs" / f"{variant}.json", cfg)
        configs[variant] = cfg
    return configs


def walk(operations):
    for op in operations:
        yield op
        yield from walk(op.get("children", []))


def refresh_counts(raw):
    policy = {"rank0": "REFpb", "rank1": "REFab"}
    if raw.get("refresh_policy") != policy:
        raise ValueError("single-layer refresh policy must be rank0 REFpb / rank1 REFab")
    commands = raw.get("refresh_commands")
    if not isinstance(commands, dict) or set(commands) != set(policy):
        raise ValueError("single-layer refresh counters must cover both ranks")
    result = {}
    for rank, required in policy.items():
        counts = commands[rank]
        if not isinstance(counts, dict) or set(counts) != {"REFpb", "REFab"}:
            raise ValueError(f"{rank} must report REFpb and REFab counters")
        for command, count in counts.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{rank} {command} count must be a nonnegative integer")
            if (command == required and count == 0) or (command != required and count != 0):
                raise ValueError(f"{rank} must issue {required} and no other refresh command in a complete single layer")
            result[f"{rank}_{command}"] = count
    return result


def summarize(variant, raw, schedule, program, spec):
    refresh = refresh_counts(raw)
    metrics, stages = raw["metrics"], raw["stages"]
    tick_us = raw["clock_ps"] / 1_000_000
    expected_names = [e["stage"] for e in schedule["events"] if e["kind"] == "pim"]
    if [(s["name"], s["step"], s["layer"]) for s in stages] != [(name, 0, 0) for name in expected_names]:
        raise ValueError("native stage coverage is not exactly one step-0/layer-0 template")
    if raw["kernels"] or metrics["gpu_read_bytes"] or metrics["gpu_write_bytes"]:
        raise ValueError("single-layer AE run unexpectedly executed VLM work")
    if metrics["AE_GPU_wait_us"] or metrics["VLM_remaining_us"] or metrics["AE_elapsed_us"] != metrics["total_us"]:
        raise ValueError("single-layer AE has unexpected overlap or tail accounting")
    if any(metrics[k] for k in ("GPU_act1", "GPU_act2", "GPU_prepb", "GPU_preab", "GPU_reads", "GPU_writes", "GPU_row_hits", "GPU_row_misses")):
        raise ValueError("AE solo reports GPU memory commands")
    expected = {}
    for stage in program["stages"]:
        for channel, ops in enumerate(stage["channels"]):
            counts = Counter(op["op"] for op in walk(ops))
            expected[stage["name"], channel] = ((counts["RD"] + counts["RDRES"]) * 32,
                                                (counts["WR"] + counts["WRGB"]) * 32)
    read_bytes = sum(value[0] for value in expected.values())
    write_bytes = sum(value[1] for value in expected.values())
    if read_bytes + write_bytes != program["expected_dq_bytes_per_layer"]:
        raise ValueError("program byte contract differs from actual operations")
    if (read_bytes, write_bytes) != (metrics["pim_read_bytes"], metrics["pim_write_bytes"]):
        raise ValueError("single-layer PIM byte conservation failed")
    buses = defaultdict(lambda: dict(read_bytes=0, write_bytes=0, ca_cycles=0, dq_cycles=0))
    for row in raw["bus"]:
        if not (0 <= row["ca_cycles"] <= row["duration_cycles"] and 0 <= row["dq_cycles"] <= row["duration_cycles"]):
            raise ValueError("native bus occupancy exceeds its interval")
        if row["gpu_read_bytes"] or row["gpu_write_bytes"]:
            raise ValueError("AE solo bus reports rank0 bytes")
        entry = buses[row["stage_id"], row["channel"]]
        for key in entry:
            entry[key] += row[key]
    stage_rows = []
    for stage in stages:
        totals = dict(read_bytes=0, write_bytes=0, ca_cycles=0, dq_cycles=0)
        for channel in range(16):
            bus = buses[stage["stage_id"], channel]
            if (bus["read_bytes"], bus["write_bytes"]) != expected[stage["name"], channel]:
                raise ValueError("per-stage/channel PIM byte conservation failed")
            for key in totals:
                totals[key] += bus[key]
        stage_rows.append(dict(variant=variant, **stage, **totals))
    if any(b["read_bytes"] or b["write_bytes"] for (sid, channel), b in buses.items() if sid < 0):
        raise ValueError("PIM traffic escaped the stage boundaries")
    stage_iter, gpu_rows = iter(stages), []
    now = 0.0
    for ordinal, event in enumerate(schedule["events"]):
        if event["kind"] == "pim":
            stage = next(stage_iter)
            if abs(stage["start_us"] - now) > 1e-6:
                raise ValueError("native PIM start does not match the complete measured GPU sequence")
            now = stage["end_us"]
        else:
            rounded = math.ceil(event["us"] / tick_us - 1e-10) * tick_us
            gpu_rows.append(dict(variant=variant, event=ordinal, name=event["name"],
                                 source_ordinal=event["source_ordinal"], measured_us=event["us"],
                                 duration_us=rounded, start_us=now, end_us=now + rounded))
            now += rounded
    pim_us = sum(s["duration_us"] for s in stages)
    gpu_us = sum(row["duration_us"] for row in gpu_rows)
    if abs(metrics["total_us"] - now) > 1e-6 or abs(metrics["total_us"] - pim_us - gpu_us) > 1e-6:
        raise ValueError("single-layer total does not equal native PIM stages plus rounded measured GPU work")
    if metrics["checked_commands"] < (read_bytes + write_bytes) // 32:
        raise ValueError("bus checker did not observe every external request")
    result = dict(variant=variant, total_us=metrics["total_us"], PIM_us=pim_us, GPU_us=gpu_us,
                  GPU_measured_us=sum(r["measured_us"] for r in gpu_rows),
                  transpose_measured_us=sum(r["measured_us"] for r in gpu_rows if r["name"] == "pv_to_o_transpose"),
                  pim_read_bytes=read_bytes, pim_write_bytes=write_bytes,
                  checked_commands=metrics["checked_commands"], PIM_stages=len(stages), GPU_blocks=len(gpu_rows))
    result.update(PIM_out_of_order_issues=metrics["PIM_out_of_order_issues"],
                  PIM_prefetched_prerequisites=metrics["PIM_prefetched_prerequisites"])
    result.update(refresh)
    return result, stage_rows, gpu_rows


def run(binary, configs, output, *, repeats=2, input_dir=ROOT / "input-data", requirements=None):
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    output = Path(output)
    rows, stages, gpu_blocks, bindings = [], [], [], {}
    for variant, cfg in configs.items():
        schedule, program = read_json(cfg["ae_schedule_path"]), read_json(cfg["program_path"])
        hashes, timings, checked = [], [], []
        for repetition in range(repeats):
            destination = output / "native" / variant / f"repeat-{repetition}"
            started = time.monotonic()
            raw = execute(binary, cfg, destination)
            timings.append(time.monotonic() - started)
            row, stage_rows, gpu_rows = summarize(variant, raw, schedule, program, cfg["spec"])
            checked.append(dict(repetition=repetition, checks_passed=True, checked_commands=raw["metrics"]["checked_commands"]))
            hashes.append(sha256(destination / "raw.json"))
        if len(set(hashes)) != 1:
            raise ValueError(f"{variant}: repeated native output is not byte-identical")
        row.update(repeats=repeats, repeat_byte_identical=True if repeats >= 2 else None, wall_seconds=sum(timings))
        rows.append(row)
        stages.extend(stage_rows)
        gpu_blocks.extend(gpu_rows)
        bindings[variant] = dict(raw_sha256=hashes, wall_seconds=timings, checks=checked,
                                 program_sha256=sha256(cfg["program_path"]), schedule_sha256=sha256(cfg["ae_schedule_path"]),
                                 boundary=schedule["boundary"])
        write_json(output / f"{variant}-checks.json", bindings[variant])
        print(f"{variant}: total={row['total_us']:.6f} us PIM={row['PIM_us']:.6f} us GPU={row['GPU_us']:.6f} us", flush=True)
    write_csv(output / "single-layer-latency.csv", rows)
    write_csv(output / "stage-latencies.csv", stages)
    write_csv(output / "gpu-blocks.csv", gpu_blocks)
    write_json(output / "summary.json", dict(scope="One decoder layer (layer 0), one denoise step (step 0), AE only",
               results=rows, checks_passed=True, no_vlm=True, no_full_campaign=True,
               comparison="PIM_us is comparable to the sum of old native one-layer PIM stages. total_us additionally includes the complete measured step-0 block GPU work; neither is a full AE or end-to-end estimate.",
               preconditions="Mask/RoPE coefficients already available; action prefix and final norm/output projection excluded"))
    write_json(output / "provenance.json", dict(schema_version=1, source_commit=source_revision(),
               binary_sha256=sha256(binary), runtime_libraries=runtime_libraries(binary),
               spec_sha256=object_sha256(next(iter(configs.values()))["spec"]), variants=bindings,
               measured_sources={p.name: sha256(p) for p in sorted((Path(input_dir) / "measured").glob("*.json"))
                                 if p.name in {"natural-ae-kernel-ledger.json", "arithmetic-helper-measurements.json", "direct-pv-handoff.json"}},
               requirements_sha256=sha256(requirements) if requirements else None,
               document_published=False, calibration_executed=False, full_matrix_executed=False))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/pim_vla_sim")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--model", choices=MODELS, default=PI05.name)
    parser.add_argument("--layer-kind", choices=("cross", "self"), default="cross")
    parser.add_argument("--program-dir", type=Path, default=ROOT / "programs/generated")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--requirements", type=Path, help="hash the requirements text; never copy or publish the original document")
    parser.add_argument("--mac-freq-mult", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(p.name not in {"configs", "inputs"} for p in args.output.iterdir()):
        raise FileExistsError("use a fresh output directory for each single-layer experiment")
    if args.spec:
        spec = read_json(args.spec)
    else:
        sys.path.insert(0, str(ROOT / "ramulator2/python"))
        from ramulator.dram.lpddr5x_pim import build_spec
        spec = build_spec(mac_freq_mult=args.mac_freq_mult)
    input_dir = args.input_dir or ROOT / "input-data"
    if args.model != PI05.name:
        from experiments.gr00t_n16_inputs import single_layer
        if args.input_dir is None:
            input_dir /= args.model
        variants = {variant_name(c): c for c in GR00T_CASES.values()}
        for variant, case in variants.items():
            full = read_json(input_dir / f"ae-{variant}.json")
            schedule = single_layer(full, args.layer_kind)
            schedule_path = (args.output / "inputs" / f"{variant}-{args.layer_kind}.json").resolve()
            write_json(schedule_path, schedule)
            cfg = {k: v for k, v in case.items() if k != "arbitration"}
            cfg.update(mode="serial", run_kind="ae_solo", spec=spec, mac_freq_mult=args.mac_freq_mult,
                       ae_schedule_path=str(schedule_path),
                       program_path=str((args.program_dir / f"{variant}.json").resolve()))
            write_json(args.output / "configs" / f"{variant}.json", validate_config(cfg))
            if not args.prepare_only:
                hashes = []
                for i in range(args.repeats):
                    destination = args.output / variant / f"repeat-{i}"
                    raw = execute(args.binary, cfg, destination)
                    if len(raw["stages"]) != sum(schedule["stage_occurrences"].values()):
                        raise ValueError("single-layer native stage count differs from the schedule")
                    hashes.append(sha256(destination / "raw.json"))
                if len(set(hashes)) != 1:
                    raise ValueError(f"{variant} single-layer repetitions differ")
        return
    configs = prepare(args.output, input_dir=input_dir, program_dir=args.program_dir, spec=spec)
    if not args.prepare_only:
        run(args.binary, configs, args.output, repeats=args.repeats, input_dir=input_dir, requirements=args.requirements)


if __name__ == "__main__":
    main()
