"""One Slurm array task, consuming prepared immutable campaign inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import read_json
from experiments.config import CASES
from experiments.run import run_solos, run_matrix, aggregate
from calib.calibrate_vlm import evaluate_candidate, collect_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("solo", "calibrate", "collect", "case", "aggregate"), required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    run = args.run.resolve()
    binary = run / "frozen/bin/pim_vla_sim"
    configs = {case: read_json(run / "configs" / f"{case}.json") for case in CASES}
    if args.phase == "solo":
        case = ("b", "d", "e", "f", "g")[args.index]
        run_solos(binary, configs, run / "solos", cases=[case])
    elif args.phase == "calibrate":
        evaluate_candidate(binary, configs["b"], run / "calibration", args.index)
    elif args.phase == "collect":
        collect_candidates(binary, configs["b"], run / "calibration")
    elif args.phase == "case":
        case = list(CASES)[args.index]
        run_matrix(binary, configs, run / "cases" / case, cases=[case], repeats=2,
                   calibration_path=run / "calibration/calibration.json", reuse_solos=run / "solos")
    else:
        aggregate([run / "cases" / case for case in CASES], run / "matrix")


if __name__ == "__main__":
    main()
