"""Cold-L2 eager/Inductor microbenchmarks for the pi0.5 AE GPU boundaries."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import torch
import torch.nn.functional as F
from torch._dynamo.utils import counters


def gelu(gate):
    return F.gelu(gate, approximate="tanh")


def gelu_gate(gate, up):
    return gelu(gate) * up


def adarms(x, scale, shift, gate):
    var = x.float().square().mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + 1e-6)
    return (normed * (1 + scale.float()) + shift.float()).to(x.dtype), gate.to(x.dtype)


def gated_residual(x, projected, gate):
    return x + projected * gate


def residual_adarms(x, projected, prev_gate, scale, shift, gate):
    residual = gated_residual(x, projected, prev_gate)
    normed, next_gate = adarms(residual, scale, shift, gate)
    return residual, normed, next_gate


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_qk(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def softmax(logits):
    return torch.softmax(logits, dim=-1, dtype=torch.float32)


def score_to_probability(score, mask):
    logits = score * (256**-0.5) + mask
    return softmax(logits).to(score.dtype)


def ae_cases():
    bf16, fp32 = torch.bfloat16, torch.float32

    def rand(shape, dtype=bf16):
        return torch.randn(shape, dtype=dtype, device="cuda")

    gate_ffn, up = (rand((1, 16, 4096)) for _ in range(2))
    x, proj = (rand((1, 16, 1024)) for _ in range(2))
    scale, shift, gate = (rand((1, 1, 3072), fp32) * 0.25).chunk(3, dim=-1)
    prev_gate = (rand((1, 1, 1024), fp32) * 0.25).to(bf16)
    # Match the Gemma projection view: [batch, token, head, dim] -> [batch, head, token, dim].
    q, k = (rand((1, 16, 8, 256)).transpose(1, 2) for _ in range(2))
    positions = torch.arange(968, 984, dtype=fp32, device="cuda").view(1, 16, 1)
    inv_freq = 10000 ** (-torch.arange(0, 256, 2, dtype=fp32, device="cuda") / 256)
    phase = positions * inv_freq
    phase = torch.cat((phase, phase), dim=-1)
    cos, sin = phase.cos().to(bf16), phase.sin().to(bf16)
    score = rand((1, 8, 16, 984)) * 16
    mask = torch.zeros((1, 1, 16, 984), dtype=fp32, device="cuda")
    mask[..., 910:968] = -2.3819763e38
    logits = score * (256**-0.5) + mask

    return {
        "gelu": (gelu, (gate_ffn,), 180),
        "gelu_gate": (gelu_gate, (gate_ffn, up), 180),
        "softmax": (softmax, (logits,), 180),
        "scale_mask_softmax_cast": (score_to_probability, (score, mask), 180),
        "adarmsnorm": (adarms, (x, scale, shift, gate), 370),
        "gated_residual": (gated_residual, (x, proj, prev_gate), 360),
        "residual_adarmsnorm": (residual_adarms, (x, proj, prev_gate, scale, shift, gate), 360),
        "rope_qk": (rope_qk, (q, k, cos, sin), 180),
    }


def statistics(samples):
    ranked = sorted(samples)
    return {
        "n": len(samples),
        "p50_us": median(samples),
        "p95_us": ranked[math.ceil(0.95 * len(samples)) - 1],
        "min_us": ranked[0],
        "samples_us": samples,
    }


def compare_outputs(eager, compiled):
    eager = eager if isinstance(eager, tuple) else (eager,)
    compiled = compiled if isinstance(compiled, tuple) else (compiled,)
    assert len(eager) == len(compiled)
    summary = []
    for reference, actual in zip(eager, compiled):
        atol, rtol = (0.0039, 0.012) if reference.dtype == torch.bfloat16 else (1e-6, 5e-5)
        torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
        summary.append({
            "dtype": str(reference.dtype), "atol": atol, "rtol": rtol,
            "max_abs": float((actual.float() - reference.float()).abs().max()),
            "exact_fraction": float((actual == reference).float().mean()),
        })
    return summary


def sample_order(index):
    return ("eager", "compile") if index % 2 == 0 else ("compile", "eager")


def event_samples(functions, args, evict, n):
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    stop.record()
    stop.synchronize()
    samples = {mode: {"event": [], "host": []} for mode in functions}
    for i in range(n):
        for mode in sample_order(i):
            evict()
            torch.cuda.current_stream().synchronize()
            start.record()
            host_start = perf_counter_ns()
            result = functions[mode](*args)
            stop.record()
            stop.synchronize()
            samples[mode]["host"].append((perf_counter_ns() - host_start) / 1000)
            samples[mode]["event"].append(start.elapsed_time(stop) * 1000)
            del result
    return samples


def kernels_under(event):
    return [*event.kernels, *(k for child in event.cpu_children for k in kernels_under(child))]


def profile_samples(name, functions, args, evict, n):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        for i in range(n):
            for mode in sample_order(i):
                evict()
                torch.cuda.current_stream().synchronize()
                with torch.profiler.record_function(f"ae/{name}/{mode}"):
                    result = functions[mode](*args)
                torch.cuda.current_stream().synchronize()
                del result

    results = {}
    for mode in functions:
        events = [e for e in profile.events() if e.name == f"ae/{name}/{mode}"
                  and e.device_type == torch.autograd.DeviceType.CPU]
        durations = [e.device_time_total for e in events]
        kernels = [kernels_under(e) for e in events]
        assert len(events) == n and min(durations) > 0 and all(kernels), (name, mode, durations)
        names = Counter(k.name for sample in kernels for k in sample)
        assert not any(any(x in k.lower() for x in ("gemm", "matmul", "nvjet", "cublas")) for k in names), names
        for duration, sample in zip(durations, kernels):
            assert math.isclose(duration, sum(k.duration for k in sample), abs_tol=0.002)
        results[mode] = {"kernel": statistics(durations), "kernel_count_each_call": sorted({len(k) for k in kernels}),
                         "kernel_names": dict(names),
                         "kernel_launches": [[dict(name=k.name, duration_us=k.duration) for k in sample]
                                             for sample in kernels]}
    return results


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    props = torch.cuda.get_device_properties(0)
    assert "Thor" in props.name and (props.major, props.minor) == (11, 0), props
    assert args.flush_mib * 2**20 >= 4 * props.L2_cache_size
    evict_buffer = torch.zeros(args.flush_mib * 2**20 // 4, dtype=torch.int32, device="cuda")

    def evict():
        evict_buffer.add_(1)

    rows = []
    out = {"gpu": props.name, "sm": [props.major, props.minor], "sms": props.multi_processor_count,
           "l2_bytes": props.L2_cache_size, "torch": torch.__version__, "cuda": torch.version.cuda,
           "cold_policy": "GPU L2 eviction by read/write sweep before each sample, outside timing",
           "flush_mib": args.flush_mib,
           "compile_options": {"triton.cudagraphs": False, "emulate_precision_casts": True},
           "fullgraph": True, "dynamic": False, "seed": 17, "cases": rows}
    cases = ae_cases()
    for name in args.only or cases:
        fn, inputs, ae_calls = cases[name]
        print(f"{name}: preparing eager and compile", flush=True)
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, options=out["compile_options"])
        ref = fn(*inputs)
        result = compiled(*inputs)
        torch.cuda.synchronize()
        accuracy = compare_outputs(ref, result)
        del ref, result
        functions = {"eager": fn, "compile": compiled}
        for _ in range(12):
            for f in functions.values():
                f(*inputs)
        torch.cuda.synchronize()

        before = counters["stats"]["unique_graphs"]
        print(f"{name}: timing cold-L2 calls", flush=True)
        timing = event_samples(functions, inputs, evict, args.samples)
        profile = profile_samples(name, functions, inputs, evict, args.profile_samples)
        assert counters["stats"]["unique_graphs"] == before, (name, before, counters["stats"])
        row = {"name": name, "calls_per_ae": ae_calls, "numerics": accuracy,
               "inputs": [{"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype)} for t in inputs]}
        for mode in functions:
            row[mode] = {**profile[mode], "event": statistics(timing[mode]["event"]),
                         "host": statistics(timing[mode]["host"])}
        rows.append(row)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "results.json").write_text(json.dumps(out, indent=2) + "\n")
        with (args.output / "comparison.csv").open("w", newline="") as f:
            fields = ["case", "calls_per_ae", "eager_kernels", "compile_kernels", "eager_device_p50_us",
                      "compile_device_p50_us", "device_speedup", "eager_event_p50_us", "compile_event_p50_us",
                      "eager_device_p95_us", "compile_device_p95_us", "eager_event_p95_us", "compile_event_p95_us"]
            writer = csv.writer(f)
            writer.writerow(fields)
            for item in rows:
                eager, opt = item["eager"], item["compile"]
                e, c = eager["kernel"]["p50_us"], opt["kernel"]["p50_us"]
                writer.writerow([item["name"], item["calls_per_ae"], eager["kernel_count_each_call"],
                                 opt["kernel_count_each_call"], e, c, e / c, eager["event"]["p50_us"],
                                 opt["event"]["p50_us"], eager["kernel"]["p95_us"], opt["kernel"]["p95_us"],
                                 eager["event"]["p95_us"], opt["event"]["p95_us"]])
        print(f"{name}: {row['eager']['kernel']['p50_us']:.3f} -> {row['compile']['kernel']['p50_us']:.3f} us (device)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--profile-samples", type=int, default=60)
    parser.add_argument("--flush-mib", type=int, default=192)
    parser.add_argument("--only", nargs="+")
    run(parser.parse_args())
