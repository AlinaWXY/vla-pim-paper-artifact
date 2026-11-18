"""Verify and summarize the four completed WCK rank-switch replays on dev0."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def load(path):
    return json.loads(path.read_bytes())


def collect(work, old, job, cases):
    sys.path.insert(0, str(work / "source/tools"))
    from report_wck_switch import occupancy, read_capacity, summarize_switches

    old_b = old / "evidence/case-b-single-read.bauEXp/10/raw.json"
    vlm_solo = load(old / "results/vlm-solo/repeat-1/raw.json")["metrics"]["total_us"]
    result = dict(
        job_id=job,
        source_commit=(work / "source-commit.txt").read_text().strip(),
        calibration_policy="same compiled Thor MHA8 inputs and calibration as the 1115 replay",
        b_vlm_solo_us=vlm_solo,
        historical_c_g=load(work / "results/historical-dq-comparison.json"),
        cases={},
    )
    invariant_bytes = ("gpu_read_bytes", "gpu_write_bytes", "pim_read_bytes", "pim_write_bytes")
    rows = []
    for case in cases:
        directory = work / "results" / case / f"job-{job}"
        old_path = old_b if case == "b" else old / f"results/combined-{case}/repeat-1/raw.json"
        subprocess.run(["sha256sum", "--check", str(directory / "outputs.sha256")],
                       cwd=work, check=True, stdout=subprocess.DEVNULL)
        run_hashes = dict(reversed(line.split()) for line in (directory / "inputs.sha256").read_text().splitlines())
        config = work / "configs" / f"{case}.json"
        assert hashlib.sha256(config.read_bytes()).hexdigest() == run_hashes[f"configs/{case}.json"]
        assert hashlib.sha256((work / "build/pim_vla_sim").read_bytes()).hexdigest() == run_hashes["build/pim_vla_sim"]
        assert load(config)["wck_rank_switch"] == "resync"
        previous = occupancy(old_path)
        current = occupancy(directory / "raw.json")
        prev_metrics, cur_metrics = previous["metrics"], current["metrics"]
        for key in invariant_bytes:
            assert cur_metrics[key] == prev_metrics[key], (case, key, cur_metrics[key], prev_metrics[key])
        assert sorted((s["name"], s["step"], s["layer"]) for s in load(old_path)["stages"]) == sorted(
            (s["name"], s["step"], s["layer"]) for s in load(directory / "raw.json")["stages"]
        )
        if case != "b":
            assert abs(cur_metrics["total_us"] - cur_metrics["AE_elapsed_us"] - cur_metrics["VLM_remaining_us"]) < 0.002
        switches = summarize_switches(directory / "wck-switches.csv")
        switch_count = sum(row["switches"] for row in switches.values())
        assert (switch_count == 0) == (case == "b")
        diag = load(directory / "bus-diagnostics.json")
        assert diag, directory
        serial_vlm = vlm_solo if case == "b" else 0
        previous_ms = (prev_metrics["total_us"] + serial_vlm) / 1000
        new_ms = (cur_metrics["total_us"] + serial_vlm) / 1000
        row = dict(
            case=case, always_on_e2e_ms=previous_ms, resync_e2e_ms=new_ms,
            e2e_delta_ms=new_ms-previous_ms, e2e_delta_pct=100*(new_ms/previous_ms-1),
            resync_ae_ms=cur_metrics["AE_elapsed_us"]/1000,
            resync_vlm_tail_ms=(cur_metrics["VLM_remaining_us"]+serial_vlm)/1000,
            always_on_gpu_dq_pct=previous["gpu_actual_dq_pct"],
            resync_gpu_dq_pct=current["gpu_actual_dq_pct"],
            always_on_dq_not_pim_pct=previous["dq_not_occupied_by_pim_pct"],
            resync_dq_not_pim_pct=current["dq_not_occupied_by_pim_pct"],
            rank_switches=switch_count,
        )
        rows.append(row)
        result["cases"][case] = dict(
            comparison=row, always_on=previous, resync=current,
            switches_by_direction=switches, input_hashes=run_hashes,
        )
        if (directory / "dq-timeline.json").is_file():
            result["cases"][case]["trace_channel_0_capacity"] = {
                "dq_only": read_capacity(directory, "resync"),
                "mask_io_epochs": read_capacity(directory, "resync", exclude_io_epochs=True),
            }
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("historical", type=Path)
    parser.add_argument("job")
    parser.add_argument("--cases", choices=("bcde", "fg"), default="bcde")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result, rows = collect(args.work, args.historical, args.job, args.cases)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"wck-{args.cases}-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / f"wck-{args.cases}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
