"""Prepare and assess native VLM-only replays against the new Thor sample."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


CANDIDATES = (
    (1, 1024, 768, 256), (0.5, 1024, 768, 256), (0.2, 1024, 768, 256),
    (0.5, 256, 192, 64),
)


def load(path):
    return json.loads(path.read_bytes())


def prepare(config, root):
    profile = root / "vlm.json"
    workload = load(profile)
    assert workload["kernel_count"] > 0 and workload["phase"].startswith("ViT +")
    assert workload["target_us"] > 0 and workload["read_bytes"] > 0
    for i, fields in enumerate(CANDIDATES):
        dest = root / f"candidate-{i}"
        dest.mkdir(exist_ok=True)
        params = dict(zip(("alpha", "window", "write_high", "write_low"), fields))
        cfg = dict(config, vlm_path=str(profile), calibration=params, run_kind="vlm_solo")
        cfg.pop("vlm_gpu_quantum_us", None)
        (dest / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def assess(config, raw, workload):
    alpha = config["calibration"]["alpha"]
    kernels, units = raw["kernels"], workload["units"]
    assert len(kernels) == len(units)
    assert raw["metrics"]["AE_elapsed_us"] == 0 and raw["metrics"]["total_us"] == kernels[-1]["end_us"]
    kernel_errors = []
    for got, unit in zip(kernels, units):
        assert got["ordinal"] == unit["ordinal"]
        assert got["read_bytes"] == unit["read_sectors"] * 32
        write = math.ceil(unit["write_sectors"] * alpha) if unit["write_is_proxy"] else unit["write_sectors"]
        assert got["write_bytes"] == write * 32
        assert got["duration_us"] >= unit["us"] - 1e-8
        if unit["kind"] == "kernel" and unit["us"] > 0:
            kernel_errors.append(got["duration_us"] / unit["us"] - 1)
    kernel_errors.sort()
    total = raw["metrics"]["total_us"]
    target = workload["target_us"]
    return dict(target_ms=target / 1000, solo_ms=total / 1000, relative_error=total / target - 1,
                kernel_error_p50=kernel_errors[math.ceil(0.50 * len(kernel_errors)) - 1],
                kernel_error_p90=kernel_errors[math.ceil(0.90 * len(kernel_errors)) - 1],
                read_bytes=raw["metrics"]["gpu_read_bytes"], write_bytes=raw["metrics"]["gpu_write_bytes"],
                within_five_percent=abs(total / target - 1) <= 0.05)


def select(root, preferred):
    profile = root / "vlm.json"
    workload = load(profile)
    results = []
    for i in range(len(CANDIDATES)):
        path = root / f"candidate-{i}"
        if not (path / "raw.json").is_file():
            continue
        config = load(path / "config.json")
        raw = load(path / "raw.json")
        row = dict(candidate=i, **config["calibration"], **assess(config, raw, workload))
        results.append(row)
    with (root / "calibration-sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    passing_preferred = next((row for row in results if row["candidate"] == preferred and row["within_five_percent"]), None)
    chosen = passing_preferred or min(results, key=lambda row: (abs(row["relative_error"]), row["kernel_error_p90"]))
    winner = root / f"candidate-{chosen['candidate']}"
    calibration = dict(parameters=load(winner / "config.json")["calibration"], result=chosen,
                       native_solo_sha256=hashlib.sha256((winner / "raw.json").read_bytes()).hexdigest(),
                       vlm_sha256=hashlib.sha256(profile.read_bytes()).hexdigest())
    (root / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (root / "solo").symlink_to(winner.name, target_is_directory=True)
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "select"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preferred", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(load(args.config), args.directory)
    else:
        select(args.directory, args.preferred)
