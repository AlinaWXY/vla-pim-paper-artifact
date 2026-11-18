"""Measure the GR00T H=15 GPU consumers of explicit PIM exports on Thor."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference/2026-11-15/benchmarks"))
from bench import compare_outputs, event_samples, profile_samples
from tools.capture_gr00t_ae import (
    adaln, feature_query_pack, ff_residual, gelu, o_residual, pv_pack, query_pack, self_kv_pack,
    softmax, softmax_feature,
)
from workloads.models import GR00T_N16_H15 as SPEC


def feature_pv_pack(x):
    # Four 32-feature shards per channel, including the physical padding.
    return x.permute(0, 2, 1, 3).contiguous().flatten(2)


def normalization(x, weight, bias, eps):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def additional_nonlinear_cases(fixture):
    query = torch.randn((1, SPEC.queries, SPEC.hidden), device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn((1, SPEC.hidden), device="cuda", dtype=torch.bfloat16)
    cases = {
        "adaln": (adaln, (query, modulation, modulation, fixture["nonlinear_norms"]["adaln"]["eps"]), (0,)),
        "residual_add": (torch.add, (query, query), (0,)),
        "timestep_silu": (F.silu, (torch.randn((1, fixture["timestep_embedding_width"]),
                                              device="cuda", dtype=torch.bfloat16),), ()),
    }
    for name, norm in fixture["nonlinear_norms"].items():
        tokens = fixture["vl_tokens"] if name == "vl" else SPEC.queries
        width = norm["shape"][0]
        x = torch.randn((1, tokens, width), device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(width, device="cuda", dtype=x.dtype) if norm["weight"] else None
        bias = torch.zeros(width, device="cuda", dtype=x.dtype) if norm["bias"] else None
        cases[f"layer_norm_{name}"] = (normalization, (x, weight, bias, norm["eps"]), (0,))
    for kind, valid in (("text", fixture["text_tokens"]), ("image", fixture["image_tokens"])):
        scores = torch.randn((1, SPEC.heads, SPEC.queries, fixture["vl_tokens"]),
                             device="cuda", dtype=torch.float32)
        mask = (torch.arange(fixture["vl_tokens"], device="cuda") < valid)[None, :]
        cases[f"softmax_cross_{kind}_unpad"] = (softmax, (scores, mask), (0,))
    return cases


def consumers(vl_tokens, text_tokens):
    bf16, fp32 = torch.bfloat16, torch.float32
    q, h, heads = SPEC.queries, SPEC.hidden, SPEC.heads
    _, padded = SPEC.kv_shape("cross", vl_tokens)

    def rand(shape, dtype=bf16):
        return torch.randn(shape, dtype=dtype, device="cuda")

    q_input = rand((1, q, h))
    bias = rand((h,))
    queries = (q_input, bias)
    cases = {
        "q_pack": (query_pack, queries, (0,)),
        "q_pack_feature": (feature_query_pack, queries, (0,)),
        "self_kv_pack": (self_kv_pack, (rand((1, q, h)), rand((1, q, h)), bias, bias), (0, 1)),
        "pv_pack": (pv_pack, (rand((1, heads, q, SPEC.head_dim)),), (0,)),
        "pv_pack_feature": (feature_pv_pack, (rand((16, 4, q, 32)),), (0,)),
        "o_residual": (o_residual, (q_input, rand((16, 1, q, h), fp32), bias, 1e-5), (1,)),
        "gelu": (gelu, (rand((1, q, SPEC.ffn), fp32), rand((SPEC.ffn,))), (0,)),
        "ff_residual": (ff_residual, (q_input, rand((1, q, h), fp32), bias), (1,)),
    }
    for kind, tokens, valid in (("self", q, q), ("cross_text", padded, text_tokens),
                                ("cross_image", padded, vl_tokens - text_tokens)):
        mask = (torch.arange(tokens, device="cuda") < valid)[None, :]
        shape = (1, heads, q, tokens)
        cases[f"softmax_{kind}"] = (softmax, (rand(shape, fp32), mask), (0,))
        cases[f"softmax_feature_{kind}"] = (softmax_feature, (rand((2, *shape), fp32), mask), (0,))
    return cases


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if (fixture["model"], fixture["horizon"], fixture["queries"]) != (SPEC.name, SPEC.horizon, SPEC.queries):
        raise ValueError("capture fixture is not GR00T H=15")
    props = torch.cuda.get_device_properties(0)
    if "Thor" not in props.name or torch.__version__ != fixture["torch"]:
        raise ValueError("use the Thor and PyTorch runtime from the original capture")
    flush_bytes = args.flush_mib * 2**20
    if flush_bytes < 4 * props.L2_cache_size:
        raise ValueError("L2 eviction sweep is smaller than four cache capacities")
    buffer = torch.zeros(flush_bytes // 4, dtype=torch.int32, device="cuda")

    def evict():
        buffer.add_(1)

    options = fixture["compile_options"]
    output = dict(model=SPEC.name, horizon=SPEC.horizon, queries=SPEC.queries,
                  vl_tokens=fixture["vl_tokens"], image_tokens=fixture["image_tokens"],
                  text_tokens=fixture["text_tokens"], gpu=props.name, sm=[props.major, props.minor],
                  l2_bytes=props.L2_cache_size, torch=torch.__version__, compile_options=options,
                  flush_mib=args.flush_mib, samples=args.samples, profile_samples=args.profile_samples,
                  timing="Thor cold-L2 CUPTI device time, median of per-call kernel sums",
                  fixture_policy="Seeded shape-matched BF16/FP32 tensors for PIM exports and GPU operands",
                  cold_policy="Pi0.5-style read/write eviction sweep and stream sync, outside timing",
                  consumers={})
    for key in ("image_size", "image_size_order", "preprocessing", "processed_image_sizes", "vit_input_shapes"):
        if key in fixture:
            output[key] = fixture[key]
    cases = consumers(fixture["vl_tokens"], fixture["text_tokens"])
    if args.extended_nonlinear:
        cases.update(additional_nonlinear_cases(fixture))
    for name, (eager, inputs, exports) in cases.items():
        compiled = torch.compile(eager, fullgraph=True, dynamic=False, options=options)
        numerics = compare_outputs(eager(*inputs), compiled(*inputs))
        functions = dict(eager=eager, compile=compiled)
        for _ in range(12):
            for fn in functions.values():
                fn(*inputs)
        torch.cuda.synchronize()
        before = counters["stats"]["unique_graphs"]
        events = event_samples(functions, inputs, evict, args.samples)
        prof = profile_samples(name, functions, inputs, evict, args.profile_samples)
        if counters["stats"]["unique_graphs"] != before:
            raise ValueError(f"{name} changed its compiled graph during measurement")
        export_bytes = sum(inputs[i].numel() * inputs[i].element_size() for i in exports)
        output["consumers"][name] = dict(
            pim_export_bytes=export_bytes, pim_input_indices=exports,
            inputs=[dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype))
                    if isinstance(t, torch.Tensor) else t for t in inputs],
            numerics=numerics, eager=prof["eager"], compile=prof["compile"], event_samples=events,
        )
        if args.include_warm:
            warm_events = event_samples(functions, inputs, lambda: None, args.samples)
            warm_profile = profile_samples(name, functions, inputs, lambda: None, args.profile_samples)
            output["consumers"][name]["warm"] = dict(
                eager=warm_profile["eager"], compile=warm_profile["compile"], event_samples=warm_events,
            )
        print(name, f"{prof['compile']['kernel']['p50_us']:.3f} us", export_bytes, "export B", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flush-mib", type=int, default=192)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--profile-samples", type=int, default=60)
    parser.add_argument("--extended-nonlinear", action="store_true")
    parser.add_argument("--include-warm", action="store_true")
    run(parser.parse_args())
