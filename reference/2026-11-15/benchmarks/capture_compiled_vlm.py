"""Measure the pinned pi0.5 ViT + language prefix on the Thor."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from common import gpu_guard, load_weights
from mha8 import convert
from openpi.models.model import Observation
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from transformers.modeling_utils import no_init_weights
from transformers.models.gemma import modeling_gemma


CHECKPOINT = Path("/var/tmp/hf-cache/hub/models--lerobot--pi05_base/snapshots/"
                  "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors")
OPTIONS = {"max_autotune": True, "triton.cudagraphs": False, "triton.unique_kernel_names": True,
           "emulate_precision_casts": True}


@torch.inference_mode(False)
@torch.no_grad()
def make_prefix(support):
    gpu_guard()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    assert torch.__version__ == "2.11.0+cu130" and torch.cuda.get_device_name() == "NVIDIA Thor"
    overlay = hashlib.sha256(Path(modeling_gemma.__file__).read_bytes()).hexdigest()
    assert overlay == "17eeb54e277939e58cd6f8a92719a275e36baf0c9f463b2754227d42f87aedde"
    cfg = Pi0Config(pi05=True, action_horizon=16, pytorch_compile_mode=None)
    with no_init_weights():
        model = PI0Pytorch(cfg)
    model.paligemma_with_expert.paligemma.tie_weights()
    model.eval()
    print("Loading retained checkpoint", flush=True)
    weights = load_weights(model, CHECKPOINT, lambda line: print(line, flush=True))
    model.cuda()
    geometry = convert(model)
    tokens = json.loads((support / "pi_tokens.json").read_bytes())["pi05"]
    generator = torch.Generator().manual_seed(17)
    images = {k: (torch.rand((1, 3, 224, 224), generator=generator) * 2 - 1).cuda()
              for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    obs = Observation(images=images,
                      image_masks={k: torch.ones(1, dtype=torch.bool, device="cuda") for k in images},
                      state=torch.zeros((1, 32), device="cuda"),
                      tokenized_prompt=torch.tensor([tokens["ids"]], device="cuda"),
                      tokenized_prompt_mask=torch.tensor([tokens["mask"]], dtype=torch.bool, device="cuda"))
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

    def run_prefix(obs):
        images, image_masks, tokens, token_masks, _ = model._preprocess_observation(obs, train=False)
        embeds, pad, att = model.embed_prefix(images, image_masks, tokens, token_masks)
        mask = model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        pos = torch.cumsum(pad, dim=1) - 1
        _, kv = model.paligemma_with_expert.forward(attention_mask=mask, position_ids=pos,
                                                    past_key_values=None, inputs_embeds=[embeds, None],
                                                    use_cache=True)
        return kv

    return run_prefix, obs, {"weights": weights, "conversion": geometry, "gemma_overlay_sha256": overlay}


def compare_cache(a, b):
    a, b = a.to_legacy_cache(), b.to_legacy_cache()
    assert len(a) == len(b) == 18
    ref_square = torch.zeros((), device="cuda", dtype=torch.float64)
    err_square = torch.zeros_like(ref_square)
    error_max = torch.zeros((), device="cuda")
    for pair_a, pair_b in zip(a, b):
        for x, y in zip(pair_a, pair_b):
            assert x.shape == y.shape and x.dtype == y.dtype
            ref, actual = x.float(), y.float()
            delta = actual - ref
            error_max = torch.maximum(error_max, delta.abs().max())
            ref_square += ref.square().sum()
            err_square += delta.square().sum()
    rms = float((err_square / ref_square).sqrt())
    assert 0 <= rms < float("inf"), rms
    return {"layers": 18, "relative_rms": rms, "max_abs": float(error_max)}


@torch.inference_mode()
def run(args):
    eager, obs, provenance = make_prefix(args.support)
    compiled = torch.compile(eager, dynamic=False, options=OPTIONS)
    print("Warming the compiled ViT + language prefix", flush=True)
    first = compiled(obs)
    torch.cuda.synchronize()
    print("First compiled prefix completed", flush=True)
    if args.mode == "trace":
        correctness = compare_cache(eager(obs), first)
        print(correctness, flush=True)
    del first
    for _ in range(4):
        compiled(obs)
    torch.cuda.synchronize()
    gpu_guard()
    if args.mode == "counter":
        torch.cuda.cudart().cudaProfilerStart()
        compiled(obs)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    from torch._dynamo.utils import counters
    graphs = counters["stats"]["unique_graphs"]
    samples = {"eager": [], "compile": []}
    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for i in range(30):
        order = [("eager", eager), ("compile", compiled)]
        if i % 2:
            order.reverse()
        for name, fn in order:
            torch.cuda.synchronize()
            a.record()
            fn(obs)
            b.record()
            b.synchronize()
            samples[name].append(a.elapsed_time(b))
    for i in range(5):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            compiled(obs)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.out / f"compiled-prefix-{i}.json"))
    assert graphs == counters["stats"]["unique_graphs"]
    provenance.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(), compile_options=OPTIONS,
                      checkpoint=str(CHECKPOINT), correctness=correctness, phase="ViT + language prefix / KV cache",
                      gpu_ms={k: {"mean": statistics.mean(v), "p50": statistics.median(v), "samples": v}
                              for k, v in samples.items()})
    (args.out / "capture-summary.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print({k: statistics.mean(v) for k, v in samples.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("trace", "counter"), required=True)
    run(parser.parse_args())
