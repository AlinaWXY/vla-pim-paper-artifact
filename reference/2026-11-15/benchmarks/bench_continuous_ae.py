"""Measure back-to-back action chunks with and without a concurrent VLM."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier, Event

import torch
from torch._dynamo.utils import counters

from bench_parallel import OPTIONS, cuda_event, make_workload
from common import gpu_guard, stats


def measure_block(pool, stream_ae, stream_vlm, frames, prefix, denoise, initial_cache, warmup, samples):
    origin = cuda_event()
    starts = [cuda_event() for _ in range(warmup + samples + 1)]
    finishes = [cuda_event() for _ in starts]
    for event in (origin, *starts, *finishes):
        event.record()
    torch.cuda.current_stream().synchronize()

    latest = (0, initial_cache)
    used_versions = []
    refreshes = []
    stop = Event()
    launch = Barrier(2) if stream_vlm is not None else None

    @torch.inference_mode()
    def feed_vlm():
        nonlocal latest
        with torch.cuda.stream(stream_vlm):
            launch.wait(timeout=30)
            while not stop.is_set():
                index = latest[0] + 1
                done = cuda_event()
                cache = prefix(frames[index % len(frames)])
                done.record()
                done.synchronize()
                latest = (index, cache)
                refreshes.append(done)

    @torch.inference_mode()
    def feed_ae():
        with torch.cuda.stream(stream_ae):
            if launch is not None:
                launch.wait(timeout=30)
            in_flight = deque()
            for start, done in zip(starts, finishes):
                version, cache = latest
                used_versions.append(version)
                start.record()
                actions = denoise(cache)
                done.record()
                # One queued chunk keeps dispatch moving. Keep its inputs and output
                # alive until its completion event, even after VLM replaces the cache.
                in_flight.append((done, cache, actions))
                if len(in_flight) == 2:
                    in_flight[0][0].synchronize()
                    in_flight.popleft()
            finishes[-1].synchronize()
            return actions

    vlm = pool.submit(feed_vlm) if launch is not None else None
    ae = pool.submit(feed_ae)
    try:
        actions = ae.result()
    finally:
        stop.set()
        if vlm is not None:
            vlm.result()
    assert torch.isfinite(actions).all()

    intervals = [finishes[i].elapsed_time(finishes[i + 1]) for i in range(warmup, warmup + samples)]
    idle = [finishes[i].elapsed_time(starts[i + 1]) for i in range(warmup, warmup + samples)]
    begin = origin.elapsed_time(finishes[warmup])
    end = origin.elapsed_time(finishes[-1])
    refresh_count = sum(begin < origin.elapsed_time(event) <= end for event in refreshes)
    versions = used_versions[warmup + 1:]
    assert len(versions) == samples and versions == sorted(versions)
    assert (len(set(versions)) > 1) == (launch is not None)
    assert (refresh_count > 0) == (launch is not None)
    assert min(intervals) > 0 and min(idle) >= 0
    return dict(period_ms=intervals, ae_idle_ms=idle, cache_versions=versions,
                vlm_refreshes=refresh_count, elapsed_ms=end - begin)


@torch.inference_mode()
def run(args):
    gpu_guard()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    compiled = args.mode == "compile"
    kwargs = {"mha8": True} if args.mha8 else {}
    vlm_profile = None
    if args.vlm_profile is not None:
        vlm_profile = json.loads(args.vlm_profile.read_text())
        kwargs["vlm_compile_options"] = vlm_profile["compile_options"]
    with torch.inference_mode(False), torch.no_grad():
        frames, prefix, denoise, _, workload = make_workload(args.support, compiled, **kwargs)
    if vlm_profile is not None:
        assert str(torch.__version__) == vlm_profile["torch"]
        assert workload["checkpoint"] == vlm_profile["checkpoint"]
        assert workload["kv_heads"] == vlm_profile["conversion"]["kv_heads"]
        assert workload["gemma_overlay_sha256"] == vlm_profile["gemma_overlay_sha256"]

    print(f"{args.mode}: warming prefix and AE", flush=True)
    initial_cache = prefix(frames[0])
    denoise(initial_cache)
    torch.cuda.synchronize()
    assert workload["kv_heads"] == (8 if args.mha8 else 1)
    stream_ae, stream_vlm = torch.cuda.Stream(), torch.cuda.Stream()
    order = ("ae_only", "with_vlm", "with_vlm", "ae_only")
    blocks = {"ae_only": [], "with_vlm": []}
    with ThreadPoolExecutor(max_workers=2) as pool:
        measure_block(pool, stream_ae, stream_vlm, frames, prefix, denoise, initial_cache, 4, 4)
        graphs = counters["stats"]["unique_graphs"]
        assert (graphs > 0) == compiled
        gpu_guard()
        for condition in order:
            print(f"{args.mode}: {condition}, {args.block_samples} periods", flush=True)
            vlm_stream = stream_vlm if condition == "with_vlm" else None
            blocks[condition].append(measure_block(pool, stream_ae, vlm_stream, frames, prefix, denoise,
                                                   initial_cache, args.warmup, args.block_samples))
        assert counters["stats"]["unique_graphs"] == graphs
        gpu_guard()

    results = {}
    for condition, runs in blocks.items():
        periods = stats([value for block in runs for value in block["period_ms"]])
        idle = stats([value for block in runs for value in block["ae_idle_ms"]])
        results[condition] = dict(ae_chunk_hz=1000 / periods["mean"], period_ms=periods, ae_idle_ms=idle,
                                  block_ae_hz=[1000 * args.block_samples / block["elapsed_ms"] for block in runs],
                                  vlm_hz=1000 * sum(b["vlm_refreshes"] for b in runs) /
                                  sum(b["elapsed_ms"] for b in runs), blocks=runs)
    result = dict(mode=args.mode, workload=workload, torch=torch.__version__, cuda=torch.version.cuda,
                  compiled_graphs=graphs, compile_options=OPTIONS if compiled else None,
                  block_order=order, block_samples=args.block_samples, warmup=args.warmup, results=results)
    filename = f"continuous-ae-{args.mode}.json"
    if vlm_profile is not None:
        result["vlm_compile_options"] = kwargs["vlm_compile_options"]
        result["vlm_profile"] = str(args.vlm_profile)
        filename = f"continuous-ae-{args.mode}-profile-options.json"
    (args.output / filename).write_text(json.dumps(result, indent=2) + "\n")
    for name, row in results.items():
        print(f"{name}: {row['ae_chunk_hz']:.4f} AE Hz, {row['vlm_hz']:.4f} VLM Hz, "
              f"{row['ae_idle_ms']['p50']:.4f} ms median gap", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eager", "compile"))
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mha8", action="store_true")
    parser.add_argument("--vlm-profile", type=Path, help="Use the VLM capture summary's compiler options")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--block-samples", type=int, default=20)
    args = parser.parse_args()
    if args.vlm_profile and args.mode != "compile":
        parser.error("--vlm-profile requires compile mode")
    run(args)
