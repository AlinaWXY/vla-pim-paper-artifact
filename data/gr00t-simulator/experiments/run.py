"""Prepare, calibrate, and run the native b-g and arbitration ablation matrix."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.report import summarize, write_report
from calib.calibrate_vlm import assess, calibrate
from experiments.common import ROOT, object_sha256, read_json, sha256, source_revision, write_json
from experiments.config import CASES, GR00T_CASES, case_config, model_cases, validate_config, variant_name
from experiments.native import execute, runtime_libraries, library_hashes
from workloads.models import PI05, MODELS


def prepare_configs(output, *, mac_freq_mult=4, program_dir=None, input_dir=None, spec_path=None,
                    vlm_release="closed_loop", model=PI05.name):
    program_dir = Path(program_dir or ROOT / "programs/generated").resolve()
    default_input = ROOT / "input-data"
    if model != PI05.name:
        default_input /= model
    input_dir = Path(input_dir or default_input).resolve()
    if spec_path:
        spec = read_json(spec_path)
    else:
        sys.path.insert(0, str(ROOT / "ramulator2/python"))
        from ramulator.dram.lpddr5x_pim import build_spec
        spec = build_spec(mac_freq_mult=mac_freq_mult)
    output = Path(output)
    write_json(output / f"spec-k{mac_freq_mult}.json", spec)
    cfgs = {}
    for case in model_cases(model):
        c = case_config(case, model=model, mac_freq_mult=mac_freq_mult)
        variant = variant_name(c)
        c.update(spec=deepcopy(spec), program_path=str(program_dir / f"{variant}.json"),
                 vlm_path=str(input_dir / "vlm.json"),
                 ae_schedule_path=str(input_dir / f"ae-{variant}.json"),
                 execution_model="native_dram_requests", vlm_release=vlm_release)
        if vlm_release == "uniform":
            c["calibration"] = dict(alpha=1.0, window=64, write_high=48, write_low=16)
        for key in ("program_path", "vlm_path", "ae_schedule_path"):
            if not Path(c[key]).is_file():
                raise FileNotFoundError(c[key])
        cfgs[case] = c
        write_json(output / f"{case}.json", c)
    return cfgs


def run_solos(binary, configs, output, *, resume=False, reuse_solos=None, cases=None):
    ae_results = {}
    for case in cases or (("c", "d", "e", "f", "g") if "b" in configs else tuple(configs)):
        cfg = dict(configs[case], run_kind="ae_solo")
        cfg.pop("vlm_gpu_quantum_us", None)
        variant = variant_name(cfg)
        if variant in ae_results:
            continue
        if reuse_solos:
            directory = Path(reuse_solos) / "ae-solo" / variant
            previous = read_json(directory / "input-config.json")
            provenance = read_json(directory / "provenance.json")
            if provenance["binary_sha256"] != sha256(binary) or provenance["raw_sha256"] != sha256(directory / "raw.json"):
                raise ValueError(f"standalone {variant} binary/result differs")
            if library_hashes(provenance) != library_hashes({"runtime_libraries": runtime_libraries(binary)}):
                raise ValueError(f"standalone {variant} runtime library differs")
            for key in ("spec", "layout", "rope", "dataflow", "mac_freq_mult", "run_kind", "model"):
                default = PI05.name if key == "model" else None
                if previous.get(key, default) != cfg.get(key, default):
                    raise ValueError(f"standalone {variant} config {key} differs")
            for key in ("program_path", "ae_schedule_path"):
                if provenance["inputs"][key]["sha256"] != sha256(cfg[key]):
                    raise ValueError(f"standalone {variant} source {key} differs")
            ae_results[variant] = read_json(directory / "raw.json")
        else:
            print(f"AE SOLO {variant}", flush=True)
            ae_results[variant] = execute(binary, cfg, Path(output) / "ae-solo" / variant, resume=resume)
    return ae_results


def run_matrix(binary, configs, output, *, calibration_path=None, cases=None, repeats=2, resume=False, reuse_solos=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = list(cases or configs)
    if any(c not in configs for c in cases):
        raise ValueError("matrix includes an unknown case")
    if repeats < 2:
        raise ValueError("full matrix acceptance requires at least two native repetitions")
    # Gate 4 runs before VLM calibration. Every unique AE design is exercised,
    # even when a caller is running only an arbitration subset.
    serial = configs["b"] if "b" in configs else configs["n16-b"]
    ae_results = run_solos(binary, configs, output, resume=resume, reuse_solos=reuse_solos,
                           cases=None if "b" in configs else cases)
    if calibration_path:
        calibration = read_json(calibration_path)
        if not calibration.get("validation", {}).get("passed"):
            raise ValueError("calibration file has not passed the standalone acceptance gate")
        if calibration["vlm_sha256"] != sha256(serial["vlm_path"]):
            raise ValueError("calibration VLM source hash differs")
        if calibration["binary_sha256"] != sha256(binary):
            raise ValueError("calibration binary differs; recalibration is required")
        if library_hashes(calibration) != library_hashes({"runtime_libraries": runtime_libraries(binary)}):
            raise ValueError("calibration runtime library differs; recalibration is required")
        if calibration["spec"] != serial["spec"]:
            raise ValueError("calibration spec differs; recalibration is required")
        vlm = read_json(calibration["raw_path"])
        if sha256(calibration["raw_path"]) != calibration["raw_sha256"]:
            raise ValueError("calibration native result hash differs")
        assess(vlm, read_json(serial["vlm_path"]))
    else:
        calibration = calibrate(binary, serial, output / "calibration", resume=resume)
        vlm = read_json(calibration["raw_path"])
    rows, repeat_checks = [], []
    for case in cases:
        cfg = dict(configs[case], calibration=calibration["parameters"])
        variant = variant_name(cfg)
        runs = []
        for repetition in range(repeats):
            destination = output / "native" / case / f"repeat-{repetition}"
            print(f"CASE {case} REPEAT {repetition}", flush=True)
            runs.append(execute(binary, cfg, destination, resume=resume))
        hashes = [sha256(output / "native" / case / f"repeat-{r}" / "raw.json") for r in range(repeats)]
        check = {"case": case, "raw_sha256": hashes, "byte_identical": len(set(hashes)) == 1}
        repeat_checks.append(check)
        if not check["byte_identical"]:
            write_json(output / "repeat-failure.json", check)
            raise ValueError(f"{case}: repeated native output differs byte-for-byte")
        rows.append(summarize(case, runs[0], ae_results[variant], vlm, output,
                              program=read_json(cfg["program_path"]), calibration=calibration,
                              vlm_input=read_json(cfg["vlm_path"]),
                              schedule=read_json(cfg["ae_schedule_path"]) if case in GR00T_CASES else None))
        write_report(output, rows, calibration, repeat_checks)
    write_json(output / "provenance.json", {
        "schema_version": 1, "source_commit": source_revision(), "binary_sha256": sha256(binary),
        "runtime_libraries": runtime_libraries(binary),
        "calibration": calibration, "cases": cases,
        "spec_sha256": object_sha256(serial["spec"]),
        "programs": {variant_name(cfg): sha256(cfg["program_path"]) for cfg in configs.values()},
        "measured_inputs": {"vlm": sha256(serial["vlm_path"]),
                            **{variant_name(cfg): sha256(cfg["ae_schedule_path"]) for cfg in configs.values()}},
        "reproducibility": repeat_checks, "no_external_trace_replay": True, "no_constant_correction": True,
    })
    return rows


def aggregate(parts, output, *, expected_cases=None):
    """Merge independently run case jobs only after their native checks passed."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, checks, sources = {}, {}, []
    calibration = None
    for directory in map(Path, parts):
        summary = read_json(directory / "summary.json")
        manifest = read_json(directory / "provenance.json")
        if calibration is None:
            calibration = summary["calibration"]
        elif summary["calibration"] != calibration:
            raise ValueError("case jobs did not use the identical frozen calibration")
        if manifest["binary_sha256"] != calibration["binary_sha256"]:
            raise ValueError("case job native binary differs from the calibrated binary")
        if library_hashes(manifest) != library_hashes(calibration):
            raise ValueError("case job runtime library differs from calibration")
        for row in summary["evaluation"]:
            case = row["case"]
            if case in rows:
                raise ValueError(f"duplicate case {case} in aggregation inputs")
            rows[case] = row
            for suffix in ("PIM-and-AE-waits.csv", "VLM-placement.csv", "bus.csv", "checks.json"):
                source = directory / f"{case}-{suffix}"
                shutil.copyfile(source, output / source.name)
        for check in summary["repeat_checks"]:
            if not check["byte_identical"]:
                raise ValueError("case repetition is not byte-identical")
            checks[check["case"]] = check
        sources.append(dict(directory=str(directory.resolve()), provenance_sha256=sha256(directory / "provenance.json")))
    wanted = list(expected_cases or CASES)
    if set(rows) != set(wanted) or set(checks) != set(wanted):
        raise ValueError(f"incomplete aggregate: received {sorted(rows)}, expected {sorted(wanted)}")
    ordered = [rows[c] for c in wanted]
    write_report(output, ordered, calibration, [checks[c] for c in wanted])
    write_json(output / "provenance.json", {"schema_version": 1, "parts": sources,
               "calibration": calibration, "cases": wanted, "source_commit": source_revision(),
               "binary_sha256": calibration["binary_sha256"],
               "runtime_libraries": calibration["runtime_libraries"],
               "spec_sha256": object_sha256(calibration["spec"]),
               "no_external_trace_replay": True, "no_constant_correction": True})
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/pim_vla_sim")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program-dir", type=Path)
    parser.add_argument("--model", choices=MODELS, default=PI05.name)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--mac-freq-mult", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--cases", nargs="+", choices=[*CASES, *GR00T_CASES])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--vlm-release", choices=("closed_loop", "uniform"))
    parser.add_argument("--vlm-gpu-quantum-us", type=float,
                        help="let AE take the GPU at VLM execution checkpoints in concurrent cases")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--solo-only", action="store_true", help="run the five AE standalone designs, then stop")
    parser.add_argument("--reuse-solos", type=Path, help="reuse verified standalone outputs under this directory/ae-solo")
    parser.add_argument("--aggregate-from", type=Path, nargs="+", help="collect completed separate case jobs instead of running native work")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    available_cases = model_cases(args.model)
    args.cases = args.cases or list(available_cases)
    if any(case not in available_cases for case in args.cases):
        parser.error("--cases must belong to --model")
    args.vlm_release = args.vlm_release or ("closed_loop" if args.model == PI05.name else "uniform")
    if args.aggregate_from:
        aggregate(args.aggregate_from, args.output, expected_cases=args.cases)
        return
    configs = prepare_configs(args.output / "configs", mac_freq_mult=args.mac_freq_mult,
                              program_dir=args.program_dir, input_dir=args.input_dir, spec_path=args.spec,
                              vlm_release=args.vlm_release, model=args.model)
    if args.vlm_gpu_quantum_us is not None:
        for case in args.cases:
            if configs[case]["mode"] == "concurrent":
                try:
                    configs[case] = validate_config(dict(configs[case], vlm_gpu_quantum_us=args.vlm_gpu_quantum_us))
                except ValueError as exc:
                    parser.error(str(exc))
                write_json(args.output / "configs" / f"{case}.json", configs[case])
    if args.prepare_only:
        return
    if args.solo_only:
        run_solos(args.binary, configs, args.output, resume=args.resume, reuse_solos=args.reuse_solos, cases=args.cases)
        return
    if args.vlm_release == "closed_loop":
        from experiments.run_closed_loop import run_closed_loop
        if args.calibration is None:
            parser.error("closed_loop requires --calibration from tools/calibrate_kernel_mlp.py")
        run_closed_loop(args.binary, configs, args.output, args.calibration, args.cases, resume=args.resume)
        return
    run_matrix(args.binary, configs, args.output, calibration_path=args.calibration,
               cases=args.cases, repeats=args.repeats, resume=args.resume, reuse_solos=args.reuse_solos)


if __name__ == "__main__":
    main()
