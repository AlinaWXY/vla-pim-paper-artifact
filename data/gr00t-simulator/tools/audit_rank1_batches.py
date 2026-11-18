"""Audit the rank-1 wire schedule of a four-batch AE solo run."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

BATCH_MASKS = (0x1111, 0x2222, 0x4444, 0x8888)
SCORE_MASK = 0x0011


def audit(path, spec):
    timing = spec["timings"]
    selector = spec["organization"]["row"] - 3
    mode = [False] * 16
    selected = [False] * 16
    open_rows = [[None] * 16 for _ in mode]
    opened_at = [[None] * 16 for _ in mode]
    faw_last = [None] * 16
    masks = [None] * 16
    earliest = {}
    counts = Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            if record["rank"] != "1":
                if int(record["opcode"]) >= 0 or int(record["bank_mask"]):
                    raise ValueError(f"Unexpected PIM opcode on rank 0: {record}")
                continue
            ch, clk, end, bg, ba, row, col, opcode, mask = (
                int(record[key]) for key in ("channel", "start_ck", "end_ck", "bank_group",
                                            "bank", "row", "column", "opcode", "bank_mask"))
            cmd = record["command"]
            if end - clk != (2 if cmd == "ACTAB" else 1):
                raise ValueError(f"Incorrect CA length: {record}")
            if cmd == "ACT2":
                bank = 4 * bg + ba
                if faw_last[ch] is not None and clk < faw_last[ch] + timing["nFAW"]:
                    raise ValueError(f"ACT2 entered a reserved ACTAB window: {record}")
                open_rows[ch][bank] = row
                opened_at[ch][bank] = clk
            elif cmd == "ACTAB":
                if not mode[ch]:
                    raise ValueError(f"ACTAB outside PIM mode: {record}")
                if row == selector:
                    selected[ch] = True
                    continue
                if any(open_row is not None for open_row in open_rows[ch]):
                    raise ValueError(f"ACTAB on an open rank: {record}")
                open_rows[ch] = [row] * 16
                opened_at[ch] = [clk + (bank % 4) * timing["nFAW"] for bank in range(16)]
                faw_last[ch] = clk + 3 * timing["nFAW"]
            elif cmd == "PREpb":
                bank = 4 * bg + ba
                active = open_rows[ch][bank]
                if bank in (0, 1) and active == selector + 2 - bank:
                    mode[ch] = bank == 0
                    selected[ch] = False
                open_rows[ch][bank] = None
                opened_at[ch][bank] = None
            elif cmd in {"PREab", "REFab"}:
                open_rows[ch] = [None] * 16
                opened_at[ch] = [None] * 16
            elif cmd in {"RD", "WR"} and opcode >= 0:
                allowed = BATCH_MASKS + ((SCORE_MASK,) if cmd == "WR" and opcode == 3 else ())
                if not mode[ch] or mask not in allowed or (bg, ba) != (-1, -1):
                    raise ValueError(f"Invalid rank-1 masked column command: {record}")
                targets = [bank for bank in range(16) if mask & (1 << bank)]
                if any(open_rows[ch][bank] != row for bank in targets):
                    raise ValueError(f"Selected bank lacks the requested open row: {record}")
                gap = clk - max(opened_at[ch][bank] for bank in targets)
                if gap < timing["nRCD" if cmd == "RD" else "nRCDWR"]:
                    raise ValueError(f"Selected bank column command precedes activation delay: {record}")
                first_act = min(value for value in opened_at[ch] if value is not None)
                key = f"{cmd}:{opcode}:0x{mask:04x}"
                earliest[key] = min(earliest.get(key, clk - first_act), clk - first_act)
                counts[key] += 1
                if mask != SCORE_MASK:
                    batch = BATCH_MASKS.index(mask)
                    op = (cmd, opcode, row, col, record["stage_id"])
                    if batch == 0 and masks[ch] is None:
                        masks[ch] = (op, 1)
                    elif masks[ch] == (op, batch):
                        masks[ch] = None if batch == 3 else (op, batch + 1)
                    else:
                        raise ValueError(f"Logical FIFO operation issued out of batch order: {record}")
            elif cmd in {"RD", "WR"} and row == selector and not selected[ch]:
                raise ValueError(f"SRF column command without selector: {record}")
    if any(entry is not None for entry in masks):
        raise ValueError("A rank-1 operation ended with missing bank batches")
    if earliest.get("RD:0:0x1111", 100) >= timing["nRCD"] + 3 * timing["nFAW"]:
        raise ValueError("No MAC issued before the final activation batch was readable")
    return {"earliest_from_actab_ck": earliest, "wire_command_counts": dict(counts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("smoke", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_bytes())
    results = {case: audit(args.smoke / case / "ca-commands.csv", spec) for case in "bdefg"}
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
