"""Pair cold Thor consumer timings with deletion of matching PIM output reads."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, read_json, sha256, write_json
from experiments.config import GR00T_CASES, variant_name
from workloads.layout import BURST_BYTES
from workloads.models import GR00T_N16_H15 as SPEC


def export_group(stage):
    if "_qkt_" in stage or "_pv_" in stage:
        return stage.rsplit("_", 1)[0]
    if stage.endswith(("_qkv", "_o")) or stage in {"up", "down"}:
        return stage
    return None


def consumer_for(event, layout):
    if event["step"] < 0 or not 0 <= event["layer"] < SPEC.layers:
        return None
    layer, name = event["layer"], event["name"]
    kind = SPEC.layer_kinds[layer]
    if name == "kv_pack":
        if kind != "self":
            raise ValueError("unexpected per-step cross K/V pack")
        return "self_kv_pack", "self_qkv"
    if name in {"q_pack", "q_pack_feature"}:
        return name, f"{kind}_qkv"
    if name in {"softmax", "softmax_feature"}:
        # The pinned checkpoint defaults to one text cross layer for every two cross layers.
        suffix = "self" if kind == "self" else "cross_text" if layer % 4 == 0 else "cross_image"
        return f"{name}_{suffix}", f"{kind}_qkt"
    if name == "pv_pack":
        return ("pv_pack_feature" if layout == "feature" else name), f"{kind}_pv"
    if name == "o_residual":
        return name, f"{kind}_o"
    if name in {"gelu", "ff_residual"}:
        return name, ("up" if name == "gelu" else "down")
    return None


def replace_export_reads(program):
    stage_reads = Counter()
    write_count = Counter()
    for stage in program["stages"]:
        name = stage["name"]
        for channel in stage["channels"]:
            for i, op in enumerate(channel):
                if op["op"] in {"WR", "WRGB"}:
                    write_count[name] += 1
                if any(c["op"] in {"RD", "RDRES", "WR", "WRGB"} for c in op.get("children", ())):
                    raise ValueError(f"{name} has an unexpected nested external transfer")
                if op["op"] == "RDRES":
                    raise ValueError(f"{name} has an unmapped register export")
                if op["op"] != "RD":
                    continue
                if export_group(name) is None or op.get("epoch") != name + "_export":
                    raise ValueError(f"{name} has a read outside its matched GPU output export")
                barrier = {k: op[k] for k in ("id", "addr", "deps", "issue_deps") if k in op}
                channel[i] = dict(barrier, op="BARRIER")
                stage_reads[name] += 1
        stage["expected_dq_bytes"] -= stage_reads[name] * BURST_BYTES
        if stage["expected_dq_bytes"] != write_count[name] * BURST_BYTES:
            raise ValueError(f"{name} remaining DQ demand differs from its original writes")
    removed = sum(stage_reads.values())
    if removed != program["counts"].pop("RD", 0):
        raise ValueError("program output-read count differs from its command totals")
    program["counts"]["BARRIER"] = program["counts"].get("BARRIER", 0) + removed
    program["io_accounting"] = "gpu_cold_single_read"
    return stage_reads, write_count


def patch_schedule(schedule, program, profile, stage_reads):
    if (schedule["model"], schedule["variant"], schedule["vl_tokens"]) != (
            SPEC.name, program["variant"], profile["vl_tokens"]):
        raise ValueError("program, schedule, and cold profile must use the same GR00T fixture")
    replaced = Counter()
    removed = Counter()
    measured = Counter()
    for event in schedule["events"]:
        step, layer = event["step"], event["layer"]
        if event["kind"] == "pim":
            stage = event["stage"]
            if stage_reads[stage]:
                removed[step, layer, export_group(stage)] += stage_reads[stage] * BURST_BYTES
            continue
        target = consumer_for(event, program["layout"])
        if target is None:
            continue
        name, group = target
        timing = profile["consumers"][name]
        us = timing["compile"]["kernel"]["p50_us"]
        if not math.isfinite(us) or us <= 0:
            raise ValueError(f"invalid cold Thor kernel time for {name}")
        measured[step, layer, group] += timing["pim_export_bytes"]
        event["us"] = us
        event["source"] = profile["timing"]
        event["measurement_operator"] = name
        replaced[name] += 1
    if removed != measured:
        raise ValueError(f"readback/consumer byte mismatch: unread={removed-measured}, unserved={measured-removed}")
    schedule["io_accounting"] = program["io_accounting"]
    schedule["gpu_us"] = math.fsum(e["us"] for e in schedule["events"] if e["kind"] == "gpu")
    return replaced, sum(removed.values())


def prepare(baseline, source_programs, profile_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    profile = read_json(profile_path)
    vlm = read_json(baseline / "vlm.json")
    if (profile["model"], profile["horizon"], profile["queries"], profile["vl_tokens"]) != (
            SPEC.name, SPEC.horizon, SPEC.queries, vlm["vl_tokens"]):
        raise ValueError("cold profile and frozen VLM fixture differ")
    if profile["flush_mib"] * 2**20 < 4 * profile["l2_bytes"]:
        raise ValueError("measurement lacks the required cold-cache sweep")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".single-read-") as tmp:
        staged = Path(tmp) / "inputs"
        programs = staged / "programs"
        data = staged / "input-data"
        programs.mkdir(parents=True)
        data.mkdir()
        shutil.copyfile(baseline / "vlm.json", data / "vlm.json")
        manifest = dict(cold_profile_sha256=sha256(profile_path), vlm_sha256=sha256(data / "vlm.json"), variants={})
        for variant in dict.fromkeys(variant_name(config) for config in GR00T_CASES.values()):
            source = source_programs / f"{variant}.json"
            schedule_path = baseline / f"ae-{variant}.json"
            program, schedule = read_json(source), read_json(schedule_path)
            if program["tensor_contract"]["shape"]["valid_kv"] != profile["vl_tokens"]:
                raise ValueError(f"{variant} program uses a different VLM sequence length")
            stage_reads, writes = replace_export_reads(program)
            old_gpu_us = schedule["gpu_us"]
            replaced, removed_bytes = patch_schedule(schedule, program, profile, stage_reads)
            program["source_sha256"][str(Path(__file__).resolve().relative_to(ROOT))] = sha256(__file__)
            p = programs / source.name
            with p.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(program, stream, separators=(",", ":"))
                stream.write("\n")
            ae = data / schedule_path.name
            write_json(ae, schedule)
            occurrences = schedule["stage_occurrences"]
            written = sum(writes[stage] * count for stage, count in occurrences.items()) * BURST_BYTES
            manifest["variants"][variant] = dict(
                baseline_program_sha256=sha256(source), baseline_schedule_sha256=sha256(schedule_path),
                program_sha256=sha256(p), ae_schedule_sha256=sha256(ae),
                removed_rd_bytes=removed_bytes, preserved_write_bytes=written,
                cross_kv_write_bytes=writes["cross_kv_store"] * occurrences["cross_kv_store"] * BURST_BYTES,
                replaced_gpu_calls=dict(replaced), old_gpu_us=old_gpu_us, cold_gpu_us=schedule["gpu_us"],
            )
            print(variant, manifest["variants"][variant], flush=True)
            del program
        write_json(staged / "provenance.json", manifest)
        staged.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-profile", type=Path, required=True)
    parser.add_argument("--baseline-inputs", type=Path, default=ROOT / "input-data/gr00t_n16")
    parser.add_argument("--baseline-programs", type=Path, default=ROOT / "programs/generated")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.baseline_inputs, args.baseline_programs, args.cold_profile, args.output)
