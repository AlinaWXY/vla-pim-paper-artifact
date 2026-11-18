"""Build native GR00T inputs from the compiled Thor capture."""
from collections import Counter
import ast
import math
from pathlib import Path
import sys

from experiments.common import ROOT, read_json, sha256
from experiments.config import GR00T_CASES, variant_name
from workloads.models import GR00T_N16_H15 as SPEC


def ae_schedule(ledger, config):
    if (ledger.get("model"), ledger.get("horizon"), ledger.get("queries"), ledger.get("steps"),
        ledger.get("layers"), tuple(ledger.get("layer_kinds", ()))) != (
            SPEC.name, SPEC.horizon, SPEC.queries, SPEC.steps, SPEC.layers, SPEC.layer_kinds):
        raise ValueError("Thor DiT capture does not match GR00T N1.6 H=15")
    units = {}
    token_layout = config["layout"] == "token"
    fused = config["dataflow"] == "fused"
    slots = SPEC.heads // 16 * (1 if token_layout else 2)
    for unit in ledger["units"]:
        key = unit["step"], unit["layer"], unit["name"]
        if key in units or not math.isfinite(unit["us"]) or unit["us"] < 0:
            raise ValueError(f"invalid measured DiT operator {key}")
        units[key] = unit["us"]

    events = []

    def gpu(name, step=-1, layer=-1):
        events.append(dict(kind="gpu", name=name, step=step, layer=layer,
                           us=units[step, layer, name], source=ledger.get("timing", "Thor compiled operator timing")))

    def pim(stage, step, layer):
        events.append(dict(kind="pim", stage=stage, step=step, layer=layer))

    gpu("state_and_vl_norm")
    for layer, kind in enumerate(SPEC.layer_kinds):
        if kind == "cross":
            gpu("cross_kv", layer=layer)
            gpu("kv_pack", layer=layer)
            pim("cross_kv_store", -1, layer)
    boundaries = []
    for step in range(SPEC.steps):
        gpu("step_input", step)
        for layer, kind in enumerate(SPEC.layer_kinds):
            start = len(events)
            gpu("adaln", step, layer)
            pim(f"{kind}_qkv", step, layer)
            if kind == "self":
                gpu("kv_pack", step, layer)
                pim("self_kv_store", step, layer)
            gpu("q_pack" if token_layout else "q_pack_feature", step, layer)
            for slot in range(slots):
                pim(f"{kind}_qkt_{slot}", step, layer)
            gpu("softmax" if token_layout else "softmax_feature", step, layer)
            for slot in range(slots):
                pim(f"{kind}_pv_{slot}", step, layer)
            if not fused:
                gpu("pv_pack", step, layer)
            pim(f"{kind}_o", step, layer)
            gpu("o_residual", step, layer)
            pim("up", step, layer)
            gpu("gelu", step, layer)
            pim("down", step, layer)
            gpu("ff_residual", step, layer)
            boundaries.append(dict(step=step, layer=layer, kind=kind, start=start, stop=len(events)))
        gpu("step_output", step)
    return dict(schema_version=1, model=SPEC.name, variant=variant_name(config),
                layers=SPEC.layers, denoising_steps=SPEC.steps, layer_kinds=SPEC.layer_kinds,
                vl_tokens=ledger["vl_tokens"], boundaries=boundaries, events=events,
                stage_occurrences=dict(Counter(e["stage"] for e in events if e["kind"] == "pim")),
                gpu_us=math.fsum(e["us"] for e in events if e["kind"] == "gpu"))


def single_layer(schedule, kind):
    boundary = next((b for b in schedule["boundaries"] if b["kind"] == kind), None)
    if boundary is None:
        raise ValueError(f"GR00T DiT has no {kind!r} layer")
    events = [dict(e, layer=0) for e in schedule["events"][boundary["start"]:boundary["stop"]]]
    return dict(schema_version=1, model=SPEC.name, variant=schedule["variant"],
                layers=1, denoising_steps=1, selected_layer=boundary["layer"], layer_kind=kind,
                events=events, stage_occurrences=dict(Counter(e["stage"] for e in events if e["kind"] == "pim")),
                gpu_us=math.fsum(e["us"] for e in events if e["kind"] == "gpu"))


def build_vlm(source, summary):
    # Reuse the Pi0.5 CUPTI/NCU ordinal alignment and 32B demand conversion.
    benchmark = ROOT / "reference/2026-11-15/benchmarks"
    sys.path.insert(0, str(benchmark))
    from build_compiled_vlm import build, counter_name

    def eagle_counter_name(name):
        # NCU's demangler drops ATen internal namespace qualifiers.
        return counter_name(name).replace("at::native::", "at::").replace(
            "at::upsample_antialias::", "upsample_antialias::")

    if summary["model"] != SPEC.name:
        raise ValueError("expected the GR00T N1.6 VLM capture")
    def eagle_launch_variant(event, row):
        # This fused SigLIP reduction autotunes one or two patch rows per CTA.
        # Keep the counter launch as an explicit proxy; retain CUPTI timing.
        if summary.get("vit_input_shapes") != [[1, 3, 252, 336]] * 3:
            return False
        return (event["name"] == "triton_red_fused_add_native_layer_norm_view_8"
                and event["args"]["grid"] == [1296, 1, 1]
                and list(ast.literal_eval(row["Grid Size"])) == [648, 1, 1]
                and event["args"]["block"] == [256, 1, 1]
                and list(ast.literal_eval(row["Block Size"])) == [256, 1, 1])

    workload = build(source, counter_identity=eagle_counter_name,
                     launch_variant_validator=eagle_launch_variant)
    workload.update(model=SPEC.name, vl_tokens=summary["vl_tokens"], capture_complete=True)
    return workload


def prepare(source, output):
    from experiments.prepare_inputs import write_json

    source, output = Path(source), Path(output)
    summary = read_json(source / "capture-summary.json")
    ledger = read_json(source / "ae-ledger.json")
    if summary["model"] != SPEC.name or any(summary[key] != ledger[key] for key in (
            "vl_tokens", "image_tokens", "text_tokens", "cameras", "prompt", "checkpoint_revision",
            "source_revision", "horizon", "gpu", "torch", "compile_options")):
        raise ValueError("VLM and DiT captures must use the same checkpoint, fixture, and Thor runtime")
    for key in ("image_size", "processed_image_sizes", "preprocessing", "vit_input_shapes"):
        if summary.get(key) != ledger.get(key):
            raise ValueError(f"VLM and DiT captures differ in {key}")
    workload = build_vlm(source, summary)
    write_json(output / "vlm.json", workload)
    variants = {variant_name(config): config for config in GR00T_CASES.values()}
    for name, config in variants.items():
        write_json(output / f"ae-{name}.json", ae_schedule(ledger, config))
    write_json(output / "provenance.json", dict(model=SPEC.name, vl_tokens=summary["vl_tokens"],
               ae_ledger_sha256=sha256(source / "ae-ledger.json"), vlm_sha256=sha256(output / "vlm.json")))
