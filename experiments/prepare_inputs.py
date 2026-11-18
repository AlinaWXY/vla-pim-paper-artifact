"""Migrate measured Thor inputs and generate timing-free AE stage schedules.

Only source measurement ledgers are read. No old PIM trace, bus interval,
arbitration result, or latency correction is imported.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, read_json, sha256
from experiments.config import CASES, variant_name
from workloads.models import MODELS, PI05

REMOVED = {
    "q_projection", "k_projection", "v_projection", "attention_qkt", "attention_sv",
    "ffn_gate", "ffn_up", "ffn_down", "rope_apply", "kv_concat", "o_projection",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def layer_boundaries(ledger):
    units = ledger["units"]
    rms = [u["ordinal"] for u in units if u["category"] == "rms_pow"]
    if len(units) != 1033 or len(rms) != 37 or [u["ordinal"] for u in units] != list(range(1033)):
        raise ValueError("unexpected AE ledger; cannot infer 18 layer boundaries")
    starts = [rms[2 * layer] - 1 for layer in range(18)] + [rms[-1] - 1]
    for start in starts:
        if units[start]["category"] != "copies_casts" or units[start]["operator"] != "aten::to":
            raise ValueError("layer entry is not the measured pre-normalization cast")
    for end in starts[1:]:
        if units[end - 1]["category"] != "residual_gate_add":
            raise ValueError("layer exit is not the measured FFN residual add")
    return [(starts[layer], starts[layer + 1]) for layer in range(18)]


def gpu_measurements(input_dir):
    directory = Path(input_dir) / "measured"
    measured = read_json(directory / "arithmetic-helper-measurements.json")["operators"]
    helpers = {name: measured[name]["cold32MiB"]["device_kernel_median_us"] for name in
               ("feature_score_rank_sum", "common_O_rank_sum_cast", "common_GateUp_fp32_to_bf16", "common_Down_fp32_to_bf16")}
    direct = read_json(directory / "direct-pv-handoff.json")["operators"]
    transpose = {layout: direct[f"{layout}_direct_PV_to_O_pack"]["cold32MiB"]["device_kernel_median_us"]
                 for layout in ("feature", "token")}
    return helpers, transpose


def normalize_vlm(demand, warm):
    phases = warm["baseline"]["phase_ms"]
    target_us = (phases["vit"]["mean"] + phases["vlm"]["mean"]) * 1000
    cursor = 0
    units = []
    for i, source in enumerate(demand["units"]):
        if source["ordinal"] != i:
            raise ValueError("VLM measurement ordinals must be contiguous")
        reads, writes = int(source["read_sectors"]), int(source["write_sectors"])
        if reads < 0 or writes < 0 or source["us"] < 0:
            raise ValueError("negative measured demand or duration")
        # Logical tensor regions are disjoint. The native controller maps these
        # into a 16 GiB rank0 ring, reusing space across sequential kernels.
        cursor = ((cursor + 1023) // 1024) * 1024
        units.append(dict(ordinal=i, kind=source["kind"], name=source["kernel"],
                          us=source["us"], read_sectors=reads, write_sectors=writes,
                          write_is_proxy=source["kind"] == "kernel", base_sector=cursor))
        cursor += max(1, reads + writes)
        if (reads + writes + 16) * 32 > 16 * 1024**3:
            raise ValueError("a single kernel tensor region exceeds rank0 capacity")
    if collections.Counter(x["kind"] for x in units) != {"kernel": 2353, "memory": 4}:
        raise ValueError("expected complete 2353-kernel plus four-copy Thor capture")
    return dict(schema_version=1, target_us=target_us, units=units,
                measured_duration_sum_us=sum(x["us"] for x in units),
                read_bytes=sum(x["read_sectors"] for x in units) * 32,
                write_proxy_bytes=sum(x["write_sectors"] for x in units if x["write_is_proxy"]) * 32,
                exact_copy_write_bytes_rounded_to_sectors=sum(x["write_sectors"] for x in units if not x["write_is_proxy"]) * 32,
                logical_address_extent_bytes=cursor * 32,
                physical_rank0_capacity_bytes=16 * 1024**3,
                assumptions={"addresses": "Sequential logical tensor regions, 32 B sectors, channel interleaving, physical row modulo spec row count; rank0 ring reused across nonoverlapping kernels; real GPU addresses unavailable",
                             "writeback": "alpha applies to kernel L2 write sectors only; exact memcpy sectors are not scaled",
                             "arrivals": "Uniform per kernel over its measured duration, closed-loop native backpressure",
                             "normalization": "none; measured per-unit durations are unchanged"})


def ae_schedule(ledger, config, *, helpers=None, transpose_us=None):
    units = ledger["units"]
    boundaries = layer_boundaries(ledger)
    starts = {start: layer for layer, (start, end) in enumerate(boundaries)}
    starts[boundaries[-1][1]] = 18
    if config["dataflow"] == "streaming" and (transpose_us is None or config["layout"] not in transpose_us):
        raise ValueError("streaming requires the measured direct PV-to-O transpose duration")
    events = []
    covered = collections.Counter()
    kept = collections.Counter()
    gpu_rope = config["rope"] == "gpu"
    helpers = helpers or {}
    for step in range(10):
        layer = -1
        seen = set()

        def pim(stage):
            events.append(dict(kind="pim", stage=stage, step=step, layer=layer, source_ordinal=ordinal))
            covered[stage] += 1

        def helper(name):
            if name in helpers:
                events.append(dict(kind="gpu", name=name, us=helpers[name], step=step, layer=layer,
                                   source_ordinal=-1, source="measured arithmetic helper"))

        for source in units:
            ordinal, category = source["ordinal"], source["category"]
            if ordinal in starts:
                layer = starts[ordinal]
            trigger = (layer, category)
            if category in {"q_projection", "attention_qkt", "attention_sv", "ffn_gate", "ffn_down"} and trigger not in seen:
                seen.add(trigger)
                if category == "q_projection":
                    pim("qkv")
                    if gpu_rope:
                        pim("rope_export")
                    else:
                        pim("rope")
                        pim("q_exchange_export")
                        pim("q_exchange_import")
                elif category == "attention_qkt":
                    if gpu_rope:
                        pim("rope_import")
                        pim("q_exchange_export")
                        pim("q_exchange_import")
                    pim("qkt")
                    if config["layout"] == "feature":
                        helper("feature_score_rank_sum")
                elif category == "attention_sv":
                    pim("pv")
                    if config["dataflow"] == "streaming":
                        events.append(dict(kind="gpu", name="pv_to_o_transpose", us=transpose_us[config["layout"]],
                                           step=step, layer=layer, source_ordinal=-1, anchor_source_ordinal=ordinal,
                                           source="Thor direct-pv-handoff.json cold32MiB device kernel median",
                                           measurement_operator=f"{config['layout']}_direct_PV_to_O_pack"))
                    pim("o")
                    helper("common_O_rank_sum_cast")
                elif category == "ffn_gate":
                    pim("gate_up")
                    helper("common_GateUp_fp32_to_bf16")
                elif category == "ffn_down":
                    pim("down")
                    helper("common_Down_fp32_to_bf16")
            keep = category not in REMOVED or (gpu_rope and category == "rope_apply")
            # RoPE coefficients are offline-generated in local mode, including
            # the per-request GPU trigonometric preparation, not just apply.
            if not gpu_rope and category == "rope_generation":
                keep = False
            if category == "copies_casts" and source["operator"] == "aten::contiguous" and source["shapes"] == [[1, 16, 8, 256]]:
                keep = False  # Replaced by the direct packing measurement in streaming; local in fused.
            if keep:
                us = source["step_mean_us"][step]
                events.append(dict(kind="gpu", name=category, us=us, step=step, layer=layer,
                                   source_ordinal=ordinal, source="natural-cache Thor kernel"))
                kept[category] += 1
        if len(seen) != 18 * 5:
            raise ValueError(f"incomplete PIM trigger coverage in step {step}: {len(seen)}")
    expected = {"qkv", "q_exchange_export", "q_exchange_import", "qkt", "pv", "o", "gate_up", "down"}
    expected |= {"rope_export", "rope_import"} if gpu_rope else {"rope"}
    if dict(covered) != {name: 180 for name in expected}:
        raise ValueError(f"stage coverage mismatch: {covered}")
    return dict(schema_version=1, variant=variant_name(config), events=events,
                layers=18, denoising_steps=10,
                ledger_layer_boundaries=[dict(layer=i, source_start=a, source_end_exclusive=b)
                                         for i, (a, b) in enumerate(boundaries)],
                gpu_us=sum(e["us"] for e in events if e["kind"] == "gpu"),
                stage_occurrences=dict(covered), retained_gpu_categories=dict(kept),
                assumptions={"GPU_execution": "Nonpreemptive measured kernels, no AE/VLM overlap on GPU",
                             "handoff_control_us": 0,
                             "streaming_pv_to_o": "Measured direct packing kernel between PV export and O input; no added control delay",
                             "arithmetic_helpers": list(helpers),
                             "ledger": "Natural-cache profiled per-kernel durations; launch gaps excluded",
                             "local_rope_coefficients": "offline; GPU coefficient-generation kernels omitted in local mode"})


def single_layer_schedule(schedule, ledger):
    """Extract exactly the step-0 decoder block, including both residuals."""
    start, end = layer_boundaries(ledger)[0]
    events = [dict(e) for e in schedule["events"] if e["step"] == 0 and e["layer"] == 0]
    gpu_rope = "_gpu_" in schedule["variant"]
    expected = []
    for unit in ledger["units"][start:end]:
        keep = unit["category"] not in REMOVED or (gpu_rope and unit["category"] == "rope_apply")
        if unit["category"] == "copies_casts" and unit["operator"] == "aten::contiguous" and unit["shapes"] == [[1, 16, 8, 256]]:
            keep = False
        if keep:
            expected.append(unit["ordinal"])
    actual = [e["source_ordinal"] for e in events if e["kind"] == "gpu" and e["source_ordinal"] >= 0]
    if actual != expected:
        raise ValueError("single-layer GPU coverage crosses a boundary or omits a retained measured kernel")
    for event in events:
        if event["kind"] == "gpu" and event["source_ordinal"] >= 0:
            if event["us"] != ledger["units"][event["source_ordinal"]]["step_mean_us"][0]:
                raise ValueError("single-layer GPU duration differs from the measured step-0 kernel")
    stages = [e["stage"] for e in events if e["kind"] == "pim"]
    wanted = ["qkv"] + (["rope_export", "rope_import"] if gpu_rope else ["rope"])
    wanted += ["q_exchange_export", "q_exchange_import", "qkt", "pv", "o", "gate_up", "down"]
    if stages != wanted:
        raise ValueError("single-layer PIM coverage is not exactly one complete template")
    transposes = [e for e in events if e.get("name") == "pv_to_o_transpose"]
    if len(transposes) != int(schedule["variant"].endswith("before")):
        raise ValueError("single-layer PV-to-O transpose does not match its dataflow")
    return dict(schema_version=1, variant=schedule["variant"], events=events, layers=1, denoising_steps=1,
                selected_step=0, selected_layer=0, gpu_us=sum(e["us"] for e in events if e["kind"] == "gpu"),
                stage_occurrences=dict(collections.Counter(stages)),
                boundary=dict(source_start=start, source_end_exclusive=end, first="attention pre-normalization cast",
                              last="FFN residual add", retained_natural_gpu_ordinals=expected),
                scope="One complete decoder block at denoise step 0; no full-request extrapolation",
                excluded_global_work="Action input projection, mask/setup, shared RoPE coefficient generation, final norm/action output projection",
                preconditions="Mask and GPU RoPE coefficients already available; weights/KV and local coefficients preloaded",
                assumptions=schedule["assumptions"])


def prepare(source, output, *, include_helpers=False):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = {"vlm-memory-demand.json": source / "gpu/uniform-memory-demand.json",
             "warm-summary.json": source / "gpu/warm-summary.json",
             "direct-pv-handoff.json": source / "gpu/direct-pv-handoff.json",
             "natural-ae-kernel-ledger.json": source / "online-arbitration-preparation/natural-ae-kernel-ledger.json"}
    if include_helpers:
        files["arithmetic-helper-measurements.json"] = source / "gpu/handoff-measurements.json"
    sources = {}
    for name, path in files.items():
        dst = output / "measured" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != dst.resolve():
            shutil.copyfile(path, dst)
        captured = sha256(dst)
        dst.write_bytes(dst.read_bytes().replace(b"\r\n", b"\n"))
        sources[name] = {"source_relative_path": path.relative_to(source).as_posix(), "sha256": sha256(dst)}
        if captured != sources[name]["sha256"]:
            sources[name].update(captured_sha256=captured, byte_conversion="CRLF to LF only")
    demand = read_json(files["vlm-memory-demand.json"])
    warm = read_json(files["warm-summary.json"])
    ledger = read_json(files["natural-ae-kernel-ledger.json"])
    write_json(output / "vlm.json", normalize_vlm(demand, warm))
    helpers = {}
    if include_helpers:
        measured = read_json(files["arithmetic-helper-measurements.json"])["operators"]
        for name in ("feature_score_rank_sum", "common_O_rank_sum_cast", "common_GateUp_fp32_to_bf16", "common_Down_fp32_to_bf16"):
            helpers[name] = measured[name]["cold32MiB"]["device_kernel_median_us"]
    direct = read_json(files["direct-pv-handoff.json"])["operators"]
    transpose = {layout: direct[f"{layout}_direct_PV_to_O_pack"]["cold32MiB"]["device_kernel_median_us"]
                 for layout in ("feature", "token")}
    variants = {variant_name(c): c for c in CASES.values()}
    for name, config in variants.items():
        write_json(output / f"ae-{name}.json", ae_schedule(ledger, config, helpers=helpers, transpose_us=transpose))
    write_json(output / "provenance.json", {"schema_version": 1, "sources": sources,
               "included": "Thor measured durations and traffic counts only",
               "excluded": "Old PIM traces, external arbitration, DQ postprocessing and all latency offsets",
               "arithmetic_helpers_included": include_helpers,
               "generated": {p.name: sha256(p) for p in sorted(output.glob("*.json")) if p.name != "provenance.json"}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=MODELS, default=PI05.name)
    parser.add_argument("--include-arithmetic-helpers", action="store_true")
    args = parser.parse_args()
    if args.model == PI05.name:
        prepare(args.source, args.output or ROOT / "input-data", include_helpers=args.include_arithmetic_helpers)
    else:
        from experiments.gr00t_n16_inputs import prepare as prepare_gr00t
        prepare_gr00t(args.source, args.output or ROOT / "input-data" / args.model)


if __name__ == "__main__":
    main()
