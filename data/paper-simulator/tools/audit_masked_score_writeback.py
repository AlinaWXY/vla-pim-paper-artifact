"""Compare score writeback before and after applying the rank-1 bank mask."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path


def score_writeback_histogram(path, qkt_stage, selector_row):
    recent_act = {}
    result = Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            if record["rank"] != "1" or int(record["stage_id"]) != qkt_stage:
                continue
            channel, clk, row = (int(record[k]) for k in ("channel", "start_ck", "row"))
            command = record["command"]
            if command == "ACTAB" and row != selector_row:
                recent_act[channel] = (clk, row)
            elif command in {"PREab", "REFab"}:
                recent_act.pop(channel, None)
            elif command == "WR" and int(record["opcode"]) == 3 and int(record["bank_mask"]) == 0x11:
                start, opened_row = recent_act[channel]
                if opened_row != row:
                    raise ValueError(f"Score write does not target the open ACTAB row: {record}")
                result[clk - start] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--updated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = []
    for model in ("instant", "four-batch"):
        spec = json.loads((args.baseline / model / "spec-k2.json").read_bytes())
        selector = spec["organization"]["row"] - 3
        earliest = spec["timings"]["nRCDWR"]
        old_earliest = earliest + (3 * spec["timings"]["nFAW"] if model == "four-batch" else 0)
        for case in "bdefg":
            before = args.baseline / model / f"{case}-s9-l17" / "on"
            after = args.updated / model / case
            left, right = (json.loads((path / "raw.json").read_bytes()) for path in (before, after))
            old_qkt, new_qkt = (next(stage for stage in raw["stages"] if stage["name"] == "qkt")
                                for raw in (left, right))
            old_hist, new_hist = (score_writeback_histogram(path / "ca-commands.csv", stage["stage_id"], selector)
                                  for path, stage in ((before, old_qkt), (after, new_qkt)))
            if old_hist.total() != new_hist.total() or min(old_hist) != old_earliest or min(new_hist) != earliest:
                raise ValueError(f"Unexpected score write count or earliest issue for {model}/{case}")
            if any(left["metrics"][key] != right["metrics"][key] for key in ("pim_read_bytes", "pim_write_bytes")):
                raise ValueError(f"PIM traffic changed for {model}/{case}")
            old_time, new_time = (raw["metrics"]["AE_elapsed_us"] for raw in (left, right))
            result.append(dict(model=model, case=case, step=9, layer=17, bank_mask="0x0011",
                               score_writes=new_hist.total(), earliest_before_ck=min(old_hist),
                               earliest_after_ck=min(new_hist), writes_at_earliest_before=old_hist[min(old_hist)],
                               writes_at_earliest_after=new_hist[min(new_hist)],
                               qkt_before_us=old_qkt["duration_us"], qkt_after_us=new_qkt["duration_us"],
                               ae_before_us=old_time, ae_after_us=new_time))
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=result[0])
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()
