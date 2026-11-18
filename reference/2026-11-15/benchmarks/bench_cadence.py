"""Measure per-frame latency and AE output cadence of the Thor pipeline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path

import torch
from torch._dynamo.utils import counters

from bench_parallel import OPTIONS, PHASES, Timing, cuda_event, make_workload
from common import gpu_guard, stats


def record_start(fn, event, arg):
    event.record()
    return fn(arg)


def collect(timer, schedule, frames, prefix, denoise, previous, frame_index, samples):
    starts = [{phase: cuda_event() for phase in PHASES} for _ in range(samples + 1)]
    ends = [{phase: cuda_event() for phase in PHASES} for _ in starts]
    for event_set in (*starts, *ends):
        for event in event_set.values():
            event.record()
    torch.cuda.synchronize()

    for i, (begin, end) in enumerate(zip(starts, ends)):
        frame = frames[(frame_index + i) % 2]
        if schedule == "parallel":
            timer.ends = end
            previous, actions, _ = timer.parallel(partial(record_start, prefix, begin["VLM"]), frame,
                                                  partial(record_start, denoise, begin["AE"]), previous)
        else:
            begin["VLM"].record()
            current = prefix(frame)
            begin["AE"].record()
            actions = denoise(current)
            end["AE"].record()
            end["AE"].synchronize()
    assert torch.isfinite(actions).all()

    ae_offset = int(schedule == "parallel")
    values = {
        "input_to_own_ae_ms": [starts[i]["VLM"].elapsed_time(ends[i + ae_offset]["AE"]) for i in range(samples)],
        "ae_completion_period_ms": [ends[i]["AE"].elapsed_time(ends[i + 1]["AE"]) for i in range(samples)],
        "ae_idle_ms": [ends[i]["AE"].elapsed_time(starts[i + 1]["AE"]) for i in range(samples)],
    }
    assert all(min(series) > 0 for series in values.values())
    return values


def summarize(values):
    metrics = {key: stats(samples) for key, samples in values.items()}
    return dict(metrics=metrics, ae_chunk_hz=1000 / metrics["ae_completion_period_ms"]["mean"], raw=values)


@torch.inference_mode()
def run(args):
    gpu_guard()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    compiled = args.mode == "compile"
    with torch.inference_mode(False), torch.no_grad():
        frames, prefix, denoise, _, workload = make_workload(args.support, compiled, args.mha8)

    print(f"{args.mode}: warmup", flush=True)
    previous = prefix(frames[0])
    denoise(previous)
    torch.cuda.synchronize()

    result = dict(mode=args.mode, workload=workload, torch=torch.__version__, cuda=torch.version.cuda,
                  compile_options=OPTIONS if compiled else None, samples=args.samples, warmup=args.warmup)
    with ThreadPoolExecutor(max_workers=2) as pool:
        timer = Timing(pool)
        for i in range(args.warmup):
            previous, _, _ = timer.parallel(prefix, frames[(i + 1) % 2], denoise, previous)
        gpu_guard()
        graphs = counters["stats"]["unique_graphs"]
        assert (graphs > 0) == compiled
        result["compiled_graphs"] = graphs
        if args.compare_serial:
            blocks = {"serial": [], "parallel": []}
            order = ("serial", "parallel", "parallel", "serial")
            # Warm the strictly same-frame path on the same stream used for the measurements.
            for i in range(4):
                denoise(prefix(frames[i % 2]))
            torch.cuda.synchronize()
            block_samples = args.samples // 2
            for i, schedule in enumerate(order):
                frame_index = (args.warmup + 1 + i) % 2
                initial_cache = prefix(frames[1 - frame_index]) if schedule == "parallel" else None
                print(f"{args.mode}: {schedule} block, {block_samples} output periods", flush=True)
                block = collect(timer, schedule, frames, prefix, denoise, initial_cache, frame_index, block_samples)
                blocks[schedule].append(block)
            for schedule, pair in blocks.items():
                pooled = {key: pair[0][key] + pair[1][key] for key in pair[0]}
                result[schedule] = summarize(pooled)
                result[schedule]["block_ae_chunk_hz"] = [summarize(block)["ae_chunk_hz"] for block in pair]
            result["block_order"] = order
            filename = f"cadence-compare-{args.mode}.json"
        else:
            print(f"{args.mode}: measuring {args.samples} consecutive output periods", flush=True)
            values = collect(timer, "parallel", frames, prefix, denoise, previous, args.warmup + 1, args.samples)
            result.update(summarize(values))
            filename = f"cadence-{args.mode}.json"

        assert counters["stats"]["unique_graphs"] == graphs
        gpu_guard()

    (args.output / filename).write_text(json.dumps(result, indent=2) + "\n")
    rates = ({name: result[name]["ae_chunk_hz"] for name in order} if args.compare_serial
             else {"parallel": result["ae_chunk_hz"]})
    print(f"{args.mode}: AE output Hz {rates}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eager", "compile"))
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--compare-serial", action="store_true")
    parser.add_argument("--mha8", action="store_true", help="Physically tile the original K/V weights into eight heads")
    args = parser.parse_args()
    if args.samples < 2 or (args.compare_serial and args.samples % 2):
        parser.error("--samples must be at least 2 and even with --compare-serial")
    run(args)
