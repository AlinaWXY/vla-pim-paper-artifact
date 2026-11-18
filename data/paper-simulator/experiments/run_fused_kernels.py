"""Freeze and run f/g with calibrated VLM kernels and native PIM windows."""
from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import read_json, write_json, sha256, source_revision, object_sha256
from experiments.config import validate_config
from experiments.native import execute, runtime_libraries
from tools.audit_fused_kernels import WINDOW_POLICY, safe_continuations


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(output, baseline, ledger_path, program_dir, binary, library):
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing campaign: {output}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT):
        raise ValueError("Commit source changes before freezing the campaign")
    ledger = read_json(ledger_path)
    if ledger["kind"] != "calibrated_vlm_durations" or ledger["work_units"] != 2357:
        raise ValueError("Full validated VLM ledger is required")
    for sub in ("frozen/bin", "frozen/lib", "frozen/inputs", "configs", "logs"):
        (output / sub).mkdir(parents=True)
    frozen_binary = output / "frozen/bin/pim_vla_sim"
    shutil.copy2(binary, frozen_binary)
    shutil.copy2(library, output / "frozen/lib/libramulator.so")
    os.environ["LD_LIBRARY_PATH"] = str(output / "frozen/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    inputs, source_inputs, configs, continuation_counts = {}, {}, {}, {}
    for case in ("f", "g"):
        cfg = read_json(baseline / f"configs/{case}.json")
        cfg.update(execution_model="fused_kernel_windows", calibration=ledger["calibration_parameters"])
        originals = {key: Path(cfg[key]) for key in ("program_path", "ae_schedule_path", "vlm_path")}
        originals["program_path"] = program_dir / originals["program_path"].name
        originals["calibrated_vlm_path"] = ledger_path
        for key, original in originals.items():
            destination = output / "frozen/inputs" / original.name
            digest = sha256(original)
            if destination.exists():
                if sha256(destination) != digest:
                    raise ValueError(f"Frozen filename collision: {destination}")
            else:
                shutil.copy2(original, destination)
            if sha256(destination) != digest:
                raise ValueError(f"Copy verification failed: {destination}")
            inputs[str(destination)] = digest
            source_inputs[str(destination)] = str(original)
            cfg[key] = str(destination)
        if ledger["source"]["vlm_sha256"] != sha256(cfg["vlm_path"]) or ledger["source"]["spec_sha256"] != object_sha256(cfg["spec"]):
            raise ValueError("Calibrated ledger does not match frozen VLM/spec")
        continuation_counts[case] = len(safe_continuations(read_json(cfg["ae_schedule_path"]), read_json(cfg["program_path"])))
        if continuation_counts[case] != {"f": 180, "g": 360}[case]:
            raise ValueError(f"Unexpected full-workload descriptor continuation count: {case}={continuation_counts[case]}")
        validate_config(cfg)
        config_path = output / f"configs/{case}.json"
        write_json(config_path, cfg)
        configs[case] = sha256(config_path)
    commit = source_revision(ROOT)
    (output / "frozen/SOURCE_COMMIT").write_text(commit + "\n", encoding="utf-8")
    shutil.copy2(ROOT / "tools/audit_fused_kernels.py", output / "frozen/audit_fused_kernels.py")
    write_json(output / "frozen/manifest.json", dict(
        schema_version=1, execution_model="fused_kernel_windows", created_utc=utc(), source_commit=commit,
        binary_path=str(frozen_binary), binary_sha256=sha256(frozen_binary), runtime_libraries=runtime_libraries(frozen_binary),
        inputs=inputs, source_inputs=source_inputs, configs=configs,
        calibrated_vlm_source=ledger["source"], calibration_parameters=ledger["calibration_parameters"],
        recalibrated=False, window_policy=WINDOW_POLICY, expected_descriptor_continuations=continuation_counts,
        window_policy_description="Skip VLM joins only at adjacent same-layer/step PIM descriptors with barrier-only outgoing suffix and incoming prefix on all 16 channels; allow a running kernel through descriptor launch, but admit new kernels only in compute",
        scope="f/g once each; complete 18 AE layers x 10 denoising steps and 2357 calibrated VLM units"))
    print(f"Prepared {output}: {commit}", flush=True)


def run_case(output, case):
    manifest = read_json(output / "frozen/manifest.json")
    if manifest.get("window_policy") != WINDOW_POLICY:
        raise ValueError("Campaign does not use the continuous PIM descriptor policy")
    os.environ["PIM_VLA_SOURCE_COMMIT"] = manifest["source_commit"]
    os.environ["LD_LIBRARY_PATH"] = str(output / "frozen/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    if sha256(manifest["binary_path"]) != manifest["binary_sha256"] or runtime_libraries(manifest["binary_path"]) != manifest["runtime_libraries"]:
        raise ValueError("Frozen executable or runtime library changed")
    for path, digest in manifest["inputs"].items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    config_path = output / f"configs/{case}.json"
    if sha256(config_path) != manifest["configs"][case]:
        raise ValueError("Frozen configuration changed")
    started = utc()
    print(f"Starting {case} at {started}", flush=True)
    start = time.monotonic()
    raw = execute(manifest["binary_path"], read_json(config_path), output / f"native/{case}")
    if raw.get("window_policy") != WINDOW_POLICY or raw.get("schema_version") != 3:
        raise ValueError("Native result did not use the frozen continuous-descriptor policy")
    if raw["metrics"]["PIM_descriptor_continuations"] != manifest["expected_descriptor_continuations"][case]:
        raise ValueError("Native continuation count differs from the frozen workload contract")
    execution = dict(case=case, started_utc=started, finished_utc=utc(), wall_seconds=time.monotonic()-start,
                     hostname=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"), total_us=raw["metrics"]["total_us"])
    write_json(output / f"native/{case}/execution.json", execution)
    print(execution, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--baseline-campaign", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--program-dir", type=Path, default=ROOT / "programs/generated/fused-kernel")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/pim_vla_sim")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--case", choices=("f", "g"))
    args = parser.parse_args()
    if args.prepare_only:
        if not all((args.baseline_campaign, args.ledger, args.library)):
            parser.error("prepare requires --baseline-campaign, --ledger and --library")
        prepare(args.output.resolve(), args.baseline_campaign.resolve(), args.ledger.resolve(),
                args.program_dir.resolve(), args.binary.resolve(), args.library.resolve())
    else:
        if not args.case:
            parser.error("run requires --case")
        run_case(args.output.resolve(), args.case)


if __name__ == "__main__":
    main()
