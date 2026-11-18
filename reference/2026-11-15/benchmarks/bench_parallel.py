"""Measure pipelined pi0.5 inference on Thor using two CUDA streams."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
from threading import Barrier
from time import perf_counter_ns

import torch
from common import gpu_guard, load_weights, stats
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from torch._dynamo.utils import counters
from transformers.modeling_utils import no_init_weights
from transformers.models.gemma import modeling_gemma


CHECKPOINT = Path("/var/tmp/hf-cache/hub/models--lerobot--pi05_base/snapshots/"
                  "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors")
OPTIONS = {"triton.cudagraphs": False, "emulate_precision_casts": True}
PHASES = ("VLM", "AE")


def observation(tokens, seed):
    generator = torch.Generator().manual_seed(seed)
    images = {key: (2 * torch.rand((1, 3, 224, 224), generator=generator) - 1).cuda()
              for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    return Observation(images=images,
                       image_masks={key: torch.ones(1, dtype=torch.bool, device="cuda") for key in images},
                       state=torch.zeros((1, 32), device="cuda"),
                       tokenized_prompt=torch.tensor([tokens["ids"]], device="cuda"),
                       tokenized_prompt_mask=torch.tensor([tokens["mask"]], dtype=torch.bool, device="cuda"))


def make_workload(support, compile_enabled, mha8=False, vlm_compile_options=None):
    overlay = hashlib.sha256(Path(modeling_gemma.__file__).read_bytes()).hexdigest()
    assert overlay == "17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde", overlay
    assert torch.cuda.get_device_name() == "NVIDIA Thor"
    cfg = Pi0Config(pi05=True, dtype="bfloat16", action_horizon=16, pytorch_compile_mode=None)
    with no_init_weights():
        model = PI0Pytorch(cfg)
    core = model.paligemma_with_expert
    core.paligemma.tie_weights()
    model.eval()
    print("Loading pi05_base checkpoint", flush=True)
    audit = load_weights(model, CHECKPOINT, lambda line: print(line, flush=True))
    model.cuda()
    kv_heads = 1
    conversion = None
    if mha8:
        from mha8 import convert

        conversion = convert(model)
        kv_heads = conversion["kv_heads"]
        assert kv_heads == 8 and len(conversion["changes"]) == 72
        print("Expanded VLM and AE to physical MHA8", flush=True)
    for transformer in (core.paligemma.language_model, core.gemma_expert.model):
        config = transformer.config
        assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (8, kv_heads, 256)
        config._attn_implementation = "eager"
        for layer in transformer.layers:
            assert layer.self_attn.q_proj.weight.dtype == torch.bfloat16
            assert layer.self_attn.k_proj.out_features == layer.self_attn.v_proj.out_features == kv_heads * 256

    tokens = json.loads((support / "pi_tokens.json").read_bytes())["pi05"]
    frames = [observation(tokens, seed) for seed in (17, 18)]
    noise = torch.randn((1, cfg.action_horizon, cfg.action_dim), generator=torch.Generator().manual_seed(19)).cuda()
    steps = 10
    dt = torch.tensor(-1 / steps, dtype=torch.float32, device="cuda")
    timestep = torch.ones(1, dtype=torch.float32, device="cuda")
    timesteps = []
    for _ in range(steps):
        timesteps.append(timestep)
        timestep = timestep + dt

    def prefix(obs):
        images, image_masks, tokens, token_masks, _ = model._preprocess_observation(obs, train=False)
        embeds, pad, att = model.embed_prefix(images, image_masks, tokens, token_masks)
        mask = model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        positions = torch.cumsum(pad, dim=1) - 1
        _, kv = core.forward(attention_mask=mask, position_ids=positions, inputs_embeds=[embeds, None],
                             past_key_values=None, use_cache=True)
        return pad, kv

    def step(pad, kv, x, t):
        return x + dt * model.denoise_step(frames[0].state, pad, kv, x, t)

    prefix_fn = prefix
    if compile_enabled:
        prefix_fn = torch.compile(prefix, dynamic=False, options=vlm_compile_options or OPTIONS)
    step_fn = torch.compile(step, dynamic=False, options=OPTIONS) if compile_enabled else step

    def denoise(previous):
        pad, kv = previous
        x = noise
        for t in timesteps:
            x = step_fn(pad, kv, x, t)
        return x

    def reference_sampler():
        return model.sample_actions("cuda", frames[0], noise=noise, num_steps=steps)

    metadata = dict(checkpoint=str(CHECKPOINT), source_weights=audit["source_tensors"],
                    checkpoint_bytes=audit["bytes_read"], gemma_overlay_sha256=overlay,
                    precision=cfg.dtype, horizon=cfg.action_horizon, steps=steps, batch_size=1,
                    images_per_frame=3, text_tokens=len(tokens["ids"]),
                    q_heads=8, kv_heads=kv_heads, head_dim=256)
    if conversion is not None:
        metadata["conversion"] = conversion
    return frames, prefix_fn, denoise, reference_sampler, metadata


def cuda_event():
    return torch.cuda.Event(enable_timing=True)


class Timing:
    def __init__(self, pool):
        self.pool = pool
        self.streams = {phase: torch.cuda.Stream() for phase in PHASES}
        self.start = cuda_event()
        self.ends = {phase: cuda_event() for phase in PHASES}
        self.barrier = Barrier(3)

    @torch.inference_mode()
    def submit(self, phase, fn, arg):
        with torch.cuda.stream(self.streams[phase]):
            self.barrier.wait()
            self.streams[phase].wait_event(self.start)
            with torch.cuda.nvtx.range(phase):
                result = fn(arg)
                self.ends[phase].record()
        return result

    def parallel(self, prefix, frame, denoise, previous):
        wall_start = perf_counter_ns()
        vlm = self.pool.submit(self.submit, "VLM", prefix, frame)
        ae = self.pool.submit(self.submit, "AE", denoise, previous)
        self.start.record()
        self.barrier.wait()
        current, actions = vlm.result(), ae.result()
        for end in self.ends.values():
            end.synchronize()
        wall_ms = (perf_counter_ns() - wall_start) / 1e6
        vlm_ms = self.start.elapsed_time(self.ends["VLM"])
        ae_ms = self.start.elapsed_time(self.ends["AE"])
        return current, actions, dict(vlm_ms=vlm_ms, ae_ms=ae_ms, pair_gpu_ms=max(vlm_ms, ae_ms), pair_wall_ms=wall_ms)

    def serial(self, prefix, frame, denoise, previous):
        wall_start = perf_counter_ns()
        self.start.record()
        current = prefix(frame)
        self.ends["VLM"].record()
        actions = denoise(previous)
        self.ends["AE"].record()
        self.ends["AE"].synchronize()
        wall_ms = (perf_counter_ns() - wall_start) / 1e6
        return current, actions, dict(vlm_ms=self.start.elapsed_time(self.ends["VLM"]),
                                      ae_ms=self.ends["VLM"].elapsed_time(self.ends["AE"]),
                                      pair_gpu_ms=self.start.elapsed_time(self.ends["AE"]), pair_wall_ms=wall_ms)


@torch.inference_mode()
def run(args):
    gpu_guard()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    # Weight parameters must be ordinary tensors before entering inference mode.
    with torch.inference_mode(False), torch.no_grad():
        frames, prefix, denoise, reference_sampler, metadata = make_workload(
            args.support, args.mode == "compile", args.mha8)

    print(f"{args.mode}: warming VLM + one-step AE graph (reused for all ten steps)", flush=True)
    initial = prefix(frames[0])
    pad, kv = initial
    kv_shape = (1, metadata["kv_heads"], pad.shape[1], metadata["head_dim"])
    assert len(kv) == 18 and all(t.shape == kv_shape and t.dtype == torch.bfloat16
                                 for pair in kv for t in pair)
    initial_actions = denoise(initial)
    reference = reference_sampler()
    native_delta = float((initial_actions - reference).abs().max())
    native_rrms = float((initial_actions - reference).norm() / reference.norm())
    if args.mode == "eager":
        assert native_delta == 0.0, native_delta
    assert torch.isfinite(initial_actions).all()
    print(f"{args.mode}: warmup output ready, max delta vs native eager sampler {native_delta:.6g}", flush=True)

    results = {"serial": [], "parallel": []}
    cache = {name: initial for name in results}
    actions = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        timer = Timing(pool)
        for i in range(args.warmup):
            for name in results:
                cache[name], _, _ = getattr(timer, name)(prefix, frames[(i + 1) % 2], denoise, cache[name])
        gpu_guard()
        graphs = counters["stats"]["unique_graphs"]
        assert (graphs > 0) == (args.mode == "compile"), graphs
        print(f"{args.mode}: timing {args.samples} serial and parallel pairs, {graphs} compiled graph(s)", flush=True)
        for i in range(args.samples):
            order = ("serial", "parallel") if i % 2 == 0 else ("parallel", "serial")
            for name in order:
                cache[name], actions[name], times = getattr(timer, name)(
                    prefix, frames[(args.warmup + i + 1) % 2], denoise, cache[name])
                results[name].append(times)
        assert counters["stats"]["unique_graphs"] == graphs, counters["stats"]
        assert all(torch.isfinite(output).all() for output in actions.values())
        schedule_delta = float((actions["parallel"] - actions["serial"]).abs().max())
        gpu_guard()
        recorded = {name: {metric: stats([sample[metric] for sample in samples]) for metric in samples[0]}
                    for name, samples in results.items()}
        result = dict(mode=args.mode, host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                      torch=torch.__version__, cuda=torch.version.cuda, workload=metadata, compiled_graphs=graphs,
                      compile_options=OPTIONS if args.mode == "compile" else None,
                      native_action_max_abs_diff=native_delta, native_action_relative_rms=native_rrms,
                      parallel_vs_serial_action_max_abs_diff=schedule_delta,
                      warmup=args.warmup, samples=args.samples, results_ms=recorded, raw_samples_ms=results)
        (args.output / f"{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({name: {m: round(v["p50"], 3) for m, v in metrics.items()}
                          for name, metrics in recorded.items()}), flush=True)

        if args.trace:
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
            timer.parallel(prefix, frames[(args.warmup + args.samples + 1) % 2], denoise, cache["parallel"])
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eager", "compile"))
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--mha8", action="store_true", help="Physically tile the original K/V weights into eight heads")
    run(parser.parse_args())
