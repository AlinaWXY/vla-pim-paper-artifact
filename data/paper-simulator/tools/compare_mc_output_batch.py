"""Compare per-packet and batched output with the same streaming inputs."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.common import read_json, write_csv
from programs.gen_pim_program import walk
from tools.capture_gap_traces import COMPARISON_SLICES
from tools.estimate_dq_gaps import MATRIX, REPEATS, edge_costs, load_trace, replay, units_for
from tools.estimate_dq_gaps import validate_replay, write_replay_input


def assert_same_inputs(original, batched):
    before = read_json(original)
    after = read_json(batched)
    if before["tensor_contract"] != after["tensor_contract"]:
        raise ValueError("output batching changed the tensor contract")
    if before["expected_dq_bytes"] != after["expected_dq_bytes"]:
        raise ValueError("output batching changed expected DQ bytes")
    packets = 0
    for left, right in zip(before["stages"], after["stages"], strict=True):
        if left["name"] != right["name"]:
            raise ValueError("output batching changed stage order")
        for x, y in zip(left["channels"], right["channels"], strict=True):
            before_ops, after_ops = list(walk(x)), list(walk(y))

            def inputs(ops):
                return [(op["op"], op["addr"], op.get("slot"), op.get("fragment"),
                         op.get("kv_token"), op.get("weight_lane_offset"))
                        for op in ops if op["op"] in {"WRGB", "MACAB", "MACQK"}]
            if inputs(before_ops) != inputs(after_ops):
                raise ValueError(f"output batching changed input/MAC issue order in {left['name']}")
            spills = Counter(tuple(op["spill_addr"]) for op in before_ops if op["op"] == "RDRES")
            reads = Counter(tuple(op["addr"]) for op in before_ops if op["op"] == "RD")
            batched_reads = Counter(tuple(op["addr"]) for op in after_ops if op["op"] == "RD")
            batched_spills = Counter(tuple(op["addr"]) for op in after_ops
                                     if op["op"] == "LOCALWR" and
                                     op.get("source") in {"reduction_output", "stream_result"})
            if batched_reads != reads + spills or batched_spills != spills:
                raise ValueError(f"output batching changed Host read or spill addresses in {left['name']}")
            packets += spills.total()
    return packets


def count_modes(directory, rows):
    counts = dict(mode_switches=0, register_activations=0)
    selectors = set()
    with (directory / "on/ca-commands.csv").open(newline="", encoding="utf-8") as file:
        for r in csv.DictReader(file):
            if r["rank"] != "1":
                continue
            channel, bank = r["channel"], r["bank"]
            if r["command"] == "ACT2" and r["bank_group"] == "0" and bank in {"0", "1"}:
                if int(r["row"]) == rows - 1 - int(bank):
                    selectors.add((channel, bank))
            elif r["command"] == "PREpb" and r["bank_group"] == "0" and (channel, bank) in selectors:
                counts["mode_switches"] += 1
                selectors.remove((channel, bank))
            elif r["command"] == "PREab":
                selectors.difference_update(((channel, "0"), (channel, "1")))
            elif r["command"] == "ACTAB" and r["row"] == str(rows-3):
                counts["register_activations"] += 1
    return counts


def evaluate(case, step, layer, reference, per_packet, rows, single, batched, engine, work):
    label = f"{case}-s{step}-l{layer}"
    old, new = single / label, batched / label
    a, ag, _, av = load_trace(old, step, layer)
    b, bg, gaps, bv = load_trace(new, step, layer)
    if any(av[k] != bv[k] for k in ("spec_sha256", "binary_sha256", "schedule_sha256")):
        raise ValueError(f"spec, binary or AE schedule differs for {label}")
    old_dq = sum(x["dq_channel_ck"] for x in ag)
    new_dq = sum(x["dq_channel_ck"] for x in bg)
    if old_dq != new_dq:
        raise ValueError(f"physical DQ count differs for {label}")
    clock_ps = b["clock_ps"]
    scale = clock_ps / 1e9
    source = work / f"{label}.txt"
    write_replay_input(source, clock_ps, edge_costs(MATRIX), units_for(1.0, clock_ps), b["events"], gaps)
    result = replay(engine, source, "same_first", 0)
    validate_replay(result, bg, b["ae_end_ck"])
    ae_old = reference["provenance"][label]["ae_slice_ck"] * REPEATS * scale
    ae_single, ae_batch = (t["ae_end_ck"] * REPEATS * scale for t in (a, b))
    e2e_old = reference["replay"][label]["1.0"]["same_first"]["total_ms"]
    e2e_single = per_packet["replay"][label]["1.0"]["same_first"]["total_ms"]
    counts_old, counts_batch = count_modes(old, rows), count_modes(new, rows)
    row = dict(case=case, step=step, layer=layer, baseline_ae_ms=ae_old, per_packet_ae_ms=ae_single,
               batch_ae_ms=ae_batch, ae_saved_ms=ae_single-ae_batch,
               fraction_of_ae_regression_recovered=(ae_single-ae_batch)/(ae_single-ae_old),
               baseline_e2e_ms=e2e_old, per_packet_e2e_ms=e2e_single,
               batch_e2e_ms=result["total_ck"]*scale,
               batch_vlm_tail_ms=result["tail_ck"]*scale, dq_channel_ck=old_dq)
    row.update({f"per_packet_{k}":v for k,v in counts_old.items()})
    row.update({f"batch_{k}":v for k,v in counts_batch.items()})
    stage_a = {item["name"]:item["t_ck"] for item in ag}
    stage_b = {item["name"]:item["t_ck"] for item in bg}
    if stage_a.keys() != stage_b.keys():
        raise ValueError(f"output batching changed PIM stages for {label}")
    stage_rows = [dict(case=case, step=step, layer=layer, stage=name,
                       per_packet_ms=stage_a[name]*REPEATS*scale,
                       batch_ms=stage_b[name]*REPEATS*scale,
                       saved_ms=(stage_a[name]-stage_b[name])*REPEATS*scale) for name in stage_a]
    return row, stage_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("single", "batched", "single-replay", "engine", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    reference = read_json(ROOT / "reference/gap-estimate/mac2/gap-estimate.json")
    per_packet = read_json(args.single_replay)
    rows = read_json(args.single / "spec-k2.json")["organization"]["row"]
    work = args.output / "replay-input"
    work.mkdir(parents=True, exist_ok=True)
    for path in sorted((args.batched / "programs").glob("*.json")):
        assert_same_inputs(args.single / "programs" / path.name, path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(evaluate, case, step, layer, reference, per_packet, rows,
                            args.single, args.batched, args.engine, work)
                for step, layer in COMPARISON_SLICES for case in "cde"]
        pairs = [job.result() for job in jobs]
    write_csv(args.output / "output-batch-comparison.csv", [row for row, _ in pairs])
    write_csv(args.output / "output-batch-stages.csv", [row for _, stages in pairs for row in stages])


if __name__ == "__main__":
    main()
