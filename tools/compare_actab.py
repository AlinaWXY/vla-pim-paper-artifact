"""Compare matched MC-controlled AE slices and their request replays."""
import argparse
import csv
from collections import Counter, deque
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import read_json, write_csv, write_json
from tools.capture_gap_traces import COMPARISON_SLICES, REPEATS


def inspect_activations(path, spec, four_batch):
    timing = spec["timings"]
    row_selector = spec["organization"]["row"] - 3
    faw = timing["nFAW"]
    offsets = range(0, 4 * faw, faw) if four_batch else (0,)
    history = [deque() for _ in range(16)]
    active = [None] * 16
    first_read = Counter()
    activations = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["rank"] != "1":
                continue
            ch, clk, address = (int(row[key]) for key in ("channel", "start_ck", "row"))
            cmd = row["command"]
            if cmd == "ACTAB" and address != row_selector:
                activations += 1
                active[ch] = (address, clk)
                pulses = [(clk + offset, 4 if four_batch else 1) for offset in offsets]
            elif cmd in {"ACT2", "REFpb"}:
                pulses = [(clk, 1)]
            else:
                pulses = []
            for stamp, count in pulses:
                window = history[ch]
                if window and stamp < window[-1]:
                    raise ValueError(f"activation overlaps an ACTAB batch in {path}: {row}")
                while window and window[0] <= stamp - faw:
                    window.popleft()
                window.extend([stamp] * count)
                if len(window) > 4:
                    raise ValueError(f"activation exceeds the model's tFAW budget in {path}: {row}")
            if cmd == "PREab":
                active[ch] = None
            if cmd == "RD" and int(row["opcode"]) >= 0 and active[ch] and active[ch][0] == address:
                first_read[clk - active[ch][1]] += 1
                active[ch] = None
    expected = offsets[-1] + timing["nRCD"]
    if not first_read or min(first_read) != expected:
        raise ValueError(f"first all-bank RD observed {first_read}, expected minimum {expected}: {path}")
    return {"array_actab_count": activations, "first_all_bank_rd_min_ck": min(first_read)}


def replay_rows(root, model):
    with (root / f"{model}-replay" / "summary.csv").open(newline="", encoding="utf-8") as stream:
        return {(row["case"], int(row["step"]), int(row["layer"]), row["ca"]): row
                for row in csv.DictReader(stream)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.traces_root
    models = ("instant", "four-batch")
    specs = [read_json(root / model / "spec-k2.json") for model in models]
    if (specs[0]["timings"] != specs[1]["timings"]
            or specs[0]["organization"] != specs[1]["organization"]
            or not specs[1]["metadata"].get("four_batch_actab")):
        raise ValueError("ACTAB comparison specs differ in base timing or geometry")
    replay = [replay_rows(root, model) for model in models]
    summary, stages, audit = [], [], {}
    for case in "bcdefg":
        for step, layer in COMPARISON_SLICES:
            label = f"{case}-s{step}-l{layer}"
            paths = [root / model / label for model in models]
            verification = [read_json(path / "verification.json") for path in paths]
            for key in ("schedule_sha256", "program_sha256", "binary_sha256"):
                if verification[0][key] != verification[1][key]:
                    raise ValueError(f"mismatched {key} for {label}")
            raw = [read_json(path / "on/raw.json") for path in paths]
            metrics = [result["metrics"] for result in raw]
            for key in ("pim_read_bytes", "pim_write_bytes"):
                if metrics[0][key] != metrics[1][key]:
                    raise ValueError(f"mismatched {key} for {label}")
            before, after = (m["AE_elapsed_us"] for m in metrics)
            result = dict(case=case, step=step, layer=layer, instant_ae_us=before, four_batch_ae_us=after,
                          delta_ae_us=after - before, delta_ae_pct=100 * (after / before - 1))
            for ca in ("off", "on"):
                left, right = (rows[case, step, layer, ca] for rows in replay)
                for key in ("vlm_solo_ms", "gpu_requests_total"):
                    if left[key] != right[key]:
                        raise ValueError(f"mismatched replay {key} for {label}")
                for model, row in zip(("instant", "four_batch"), (left, right), strict=True):
                    result[f"{model}_e2e_ca_{ca}_ms"] = float(row["e2e_ms"])
                result[f"delta_e2e_ca_{ca}_ms"] = float(right["e2e_ms"]) - float(left["e2e_ms"])
            summary.append(result)
            for left, right in zip(raw[0]["stages"], raw[1]["stages"], strict=True):
                if left["name"] != right["name"] or left["stage_id"] != right["stage_id"]:
                    raise ValueError(f"mismatched stage sequence for {label}")
                start, stop = left["duration_us"], right["duration_us"]
                stages.append(dict(case=case, step=step, layer=layer, stage=left["name"],
                                   instant_us=start, four_batch_us=stop, delta_us=stop - start))
            audit[label] = {model: inspect_activations(path / "on/ca-commands.csv", spec, index == 1)
                            for index, (model, path, spec) in enumerate(zip(models, paths, specs, strict=True))}
    write_csv(args.output / "comparison.csv", summary)
    write_csv(args.output / "stage-comparison.csv", stages)
    write_json(args.output / "activation-audit.json", {
        "repeated_slices_for_e2e": REPEATS, "slices": audit,
    })


if __name__ == "__main__":
    main()
