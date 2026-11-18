"""Fit only writeback alpha, outstanding window, and write-drain watermarks.

The fit must pass on the complete 2353-kernel capture. A failed fit is a failed
acceptance gate: no offset, duration rescaling, or post-hoc latency correction.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, object_sha256, percentile, read_json, sha256, source_revision, write_csv, write_json
from experiments.config import validate_calibration
from experiments.native import execute, total_us, runtime_libraries, library_hashes


def candidate_parameters():
    # The positive floor is an explicit search bound, not a measured physical
    # lower bound. Kernel L2 write sectors do not reveal actual DRAM writebacks.
    # One small-window diagnostic precedes the wide window: 64 consecutive
    # sectors occupy one bank, so a 64-request window cannot expose all BGs.
    # Widening a queue changes outstanding work, never the DRAM timing table.
    yield validate_calibration(dict(alpha=1.0, window=64, write_high=48, write_low=16))
    for window, high, low in ((1024, 768, 256), (256, 192, 64), (256, 224, 128)):
        for alpha in (1.0, 0.5, 0.2, 0.1, 0.05):
            params = dict(alpha=alpha, window=window, write_high=high, write_low=low)
            yield validate_calibration(params)


def assess(raw, vlm):
    expected = {u["ordinal"]: u for u in vlm["units"]}
    observed = {u["ordinal"]: u for u in raw["kernels"]}
    if set(observed) != set(expected):
        raise ValueError("calibration did not execute every captured VLM work unit")
    errors = []
    for ordinal, source in expected.items():
        row = observed[ordinal]
        duration = row.get("duration_us", row["end_us"] - row["start_us"])
        if duration + 1e-9 < source["us"]:
            raise ValueError(f"kernel {ordinal} completed before its measured compute duration")
        if source["kind"] == "kernel" and source["us"] > 0:
            errors.append((duration - source["us"]) / source["us"])
    duration = total_us(raw)
    error = (duration - vlm["target_us"]) / vlm["target_us"]
    return dict(total_us=duration, target_us=vlm["target_us"], relative_error=error,
                kernel_relative_error_median=percentile(errors, 0.5),
                kernel_relative_error_p90=percentile(errors, 0.9),
                work_units=len(expected), kernel_count=len(errors), passed=abs(error) <= 0.05)


def require_complete_capture(vlm):
    units = vlm["units"]
    if vlm.get("model", "pi05") == "pi05":
        complete = len(units) == 2357
    else:
        complete = (vlm.get("model") == "gr00t_n16" and vlm.get("capture_complete")
                    and sum(u["kind"] == "kernel" for u in units) == vlm["kernel_count"] > 0
                    and [u["ordinal"] for u in units] == list(range(len(units))))
    if not complete:
        raise ValueError("calibration requires the full captured VLM workload")


def evaluate_candidate(binary, base_config, output, index, *, resume=False, parameters=None):
    parameters = parameters or list(candidate_parameters())[index]
    output = Path(output)
    vlm = read_json(base_config["vlm_path"])
    require_complete_capture(vlm)
    cfg = dict(base_config, run_kind="vlm_solo", vlm_release="uniform", calibration=parameters)
    print(f"CALIBRATION {index}: {parameters}", flush=True)
    raw = execute(binary, cfg, output / f"candidate-{index:02d}", resume=resume)
    result = assess(raw, vlm)
    row = dict(candidate=index, **parameters, **result)
    write_json(output / f"candidate-{index:02d}" / "assessment.json", row)
    return row


def freeze_candidate(binary, base_config, output, row, *, selection="first passing candidate in declared order"):
    output = Path(output)
    index = row["candidate"]
    parameters = {key: row[key] for key in ("alpha", "window", "write_high", "write_low")}
    result = {k: v for k, v in row.items() if k not in parameters and k != "candidate"}
    if not result["passed"]:
        raise ValueError("cannot freeze a failed calibration candidate")
    libraries = runtime_libraries(binary)
    measured = read_json(output / f"candidate-{index:02d}" / "provenance.json")
    if measured["binary_sha256"] != sha256(binary):
        raise ValueError("calibration executable changed before freezing")
    if library_hashes(measured) != library_hashes({"runtime_libraries": libraries}):
        raise ValueError("calibration runtime library changed before freezing")
    if measured["raw_sha256"] != sha256(output / f"candidate-{index:02d}" / "raw.json"):
        raise ValueError("calibration raw output changed before freezing")
    if measured["inputs"]["vlm_path"]["sha256"] != sha256(base_config["vlm_path"]):
        raise ValueError("calibration VLM input changed before freezing")
    selected = dict(schema_version=1, parameters=parameters, validation=result,
                    selected_candidate=index, selection=selection,
                    source_commit=source_revision(), vlm_sha256=sha256(base_config["vlm_path"]),
                    binary_sha256=sha256(binary), spec=base_config.get("spec"),
                    spec_sha256=object_sha256(base_config.get("spec")),
                    runtime_libraries=libraries,
                    raw_sha256=sha256(output / f"candidate-{index:02d}" / "raw.json"),
                    raw_path=str((output / f"candidate-{index:02d}" / "raw.json").resolve()),
                    search_bounds={"alpha": [0.05, 1.0], "window": [64, 256, 1024],
                                   "alpha_floor_basis": "Explicit positive search floor; actual writeback fraction is unmeasured"},
                    limitations=["alpha is a fitted writeback fraction, not a DRAM counter measurement",
                                 "Uniform arrivals and sequential synthetic tensor addresses remain assumptions",
                                 "GPU SM cost of transfers and GPU/PIM handoff control delay are not modeled"],
                    no_latency_offsets=True)
    write_json(output / "calibration.json", selected)
    return selected


def collect_candidates(binary, base_config, output):
    """Gather Slurm candidate jobs without racing the deterministic selection."""
    output = Path(output)
    rows = []
    for index, parameters in enumerate(candidate_parameters()):
        directory = output / f"candidate-{index:02d}"
        if not (directory / "assessment.json").is_file():
            continue
        expected_config = dict(base_config, run_kind="vlm_solo", vlm_release="uniform", calibration=parameters)
        if read_json(directory / "input-config.json") != expected_config:
            raise ValueError(f"candidate {index} configuration is stale")
        provenance = read_json(directory / "provenance.json")
        if provenance["binary_sha256"] != sha256(binary) or provenance["raw_sha256"] != sha256(directory / "raw.json"):
            raise ValueError(f"candidate {index} binary or raw output hash differs")
        if library_hashes(provenance) != library_hashes({"runtime_libraries": runtime_libraries(binary)}):
            raise ValueError(f"candidate {index} runtime library differs")
        if provenance["inputs"]["vlm_path"]["sha256"] != sha256(base_config["vlm_path"]):
            raise ValueError(f"candidate {index} VLM input differs")
        result = assess(read_json(directory / "raw.json"), read_json(base_config["vlm_path"]))
        rows.append(dict(candidate=index, **parameters, **result))
    write_csv(output / "calibration-sweep.csv", rows)
    write_json(output / "calibration-sweep.json", rows)
    complete = {r["candidate"] for r in rows}
    passing = [r for r in rows if r["passed"]]
    if passing:
        best = min(passing, key=lambda r: r["candidate"])
        pending = sorted(set(range(best["candidate"])) - complete)
        if pending:
            raise RuntimeError(f"earlier calibration candidates still pending: {pending}")
        return freeze_candidate(binary, base_config, output, best)
    pending = sorted(set(range(len(list(candidate_parameters())))) - complete)
    if pending:
        raise RuntimeError(f"no passing completed candidate; still pending: {pending}")
    raise RuntimeError("all calibration candidates failed the 5% gate; do not run concurrent cases")


def calibrate(binary, base_config, output, *, resume=False, candidates=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vlm = read_json(base_config["vlm_path"])
    require_complete_capture(vlm)
    rows = []
    selected = None
    for index, parameters in enumerate(candidates or candidate_parameters()):
        row = evaluate_candidate(binary, base_config, output, index, resume=resume, parameters=parameters)
        rows.append(row)
        write_csv(output / "calibration-sweep.csv", rows)
        write_json(output / "calibration-sweep.json", rows)
        if row["passed"]:
            selected = freeze_candidate(binary, base_config, output, row)
            break
    if selected is None:
        write_json(output / "calibration-failed.json", {
            "passed": False, "attempts": len(rows), "target_us": vlm["target_us"],
            "tolerance": 0.05, "alpha_floor": 0.05,
            "action": "Stop before concurrent evaluation; review request/address model and measured writebacks",
            "constant_correction_used": False,
        })
        raise RuntimeError("VLM calibration failed the 5% acceptance gate; concurrent matrix was not run")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/pim_vla_sim")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--candidate", type=int, choices=range(len(list(candidate_parameters()))),
                      help="run only this candidate; safe for a Slurm array sharing --output")
    mode.add_argument("--collect", action="store_true", help="collect candidate jobs and freeze the first passing configuration")
    mode.add_argument("--list-candidates", action="store_true")
    args = parser.parse_args()
    if args.list_candidates:
        for i, p in enumerate(candidate_parameters()):
            print(i, p)
        return
    if args.candidate is not None:
        result = evaluate_candidate(args.binary, read_json(args.config), args.output, args.candidate, resume=args.resume)
        print(result, flush=True)
        return
    if args.collect:
        result = collect_candidates(args.binary, read_json(args.config), args.output)
        print(result["parameters"], result["validation"], flush=True)
        return
    result = calibrate(args.binary, read_json(args.config), args.output, resume=args.resume)
    print(result["parameters"], result["validation"], flush=True)


if __name__ == "__main__":
    main()
