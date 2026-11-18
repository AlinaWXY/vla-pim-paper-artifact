"""Profile eager and compiled GR00T N1.6 VLM and full H=15 action head on Thor."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_utils import no_init_weights

from capture_gr00t_thor import OPTIONS, SPEC, compare_embedding, latency, load_weights, setup, write


def measure(fn, inputs, args, label):
    for _ in range(args.warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    result = latency(fn, inputs, args.samples)
    for timings in result.values():
        timings["p90"] = statistics.quantiles(timings["samples"], n=10, method="inclusive")[8]
    print(f"{label}: CUDA p50={result['cuda_ms']['p50']:.4f} ms, "
          f"wall p50={result['wall_ms']['p50']:.4f} ms", flush=True)
    if args.trace:
        for repeat in range(args.profile_repeats):
            with torch.profiler.profile(
                activities=([torch.profiler.ProfilerActivity.CUDA] if args.lightweight_trace else
                            [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]),
                record_shapes=not args.lightweight_trace,
            ) as profiler:
                with torch.profiler.record_function(label):
                    fn(*inputs)
                torch.cuda.synchronize()
            suffix = f"-{repeat}" if args.profile_repeats > 1 else ""
            profiler.export_chrome_trace(str(args.output / f"{label}{suffix}.trace.json"))
    return result


@torch.inference_mode()
def run(args):
    config, vlm, inputs, image_mask, fixture = setup(args)
    expected_image_tokens = sum(height // 28 * (width // 28)
                                for height, width in fixture["processed_image_sizes"])
    if fixture["image_tokens"] != expected_image_tokens:
        raise ValueError(f"unexpected image token count: {fixture}")
    print("Fixture", json.dumps(fixture), flush=True)
    report = dict(
        fixture,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        host=platform.node(),
        warmup=args.warmup,
        samples=args.samples,
        profile_repeats=args.profile_repeats,
        trace_scope=("CUDA activity only, no shapes" if args.lightweight_trace else "CPU and CUDA with shapes"),
        timing_scope="Separate GPU solo phases, inputs already on device, compilation and warmup excluded",
        vlm_scope="Eagle vision encoder, projection, and 16 Qwen3 layers",
        ae_scope="Full checkpoint action head including state/VL normalization, encoders, 32-layer DiT and decoder, four denoising steps",
        ae_cross_kv="Official get_action path, K/V projected during each denoising step",
        input_kind="Seeded random uint8 images, zero state, fixed English instruction",
        source_sha256={Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        phases={},
    )
    for name in ("capture_gr00t_thor.py",):
        path = Path(__file__).with_name(name)
        report["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(args.output / "profile-summary.json", report)

    vision_inputs = []

    def record_vision_inputs(module, positional, keywords):
        values = keywords["pixel_values"] if "pixel_values" in keywords else positional[0]
        vision_inputs.append([list(value.shape) for value in values])

    with vlm.vision_model.register_forward_pre_hook(record_vision_inputs, with_kwargs=True):
        embedding = vlm(*inputs)
    if len(vision_inputs) != 1 or len(vision_inputs[0]) != args.cameras:
        raise ValueError(f"unexpected ViT input batch: {vision_inputs}")
    report["vit_input_shapes"] = vision_inputs[0]
    if args.expected_vit_size:
        width, height = args.expected_vit_size
        if any(shape[-2:] != [height, width] for shape in vision_inputs[0]):
            raise ValueError(f"ViT did not receive the required {width}x{height} images: {vision_inputs}")
        if fixture["processed_image_sizes"] != [[height, width]] * args.cameras:
            raise ValueError("Eagle preprocessing and actual ViT shapes differ")
    print("Verified ViT input shapes", vision_inputs[0], flush=True)
    if embedding.shape != (1, fixture["vl_tokens"], SPEC.backbone_width) or not torch.isfinite(embedding).all():
        raise ValueError("invalid Eagle output")
    vlm_results = {}
    for backend in args.backends:
        print(f"Preparing VLM {backend}", flush=True)
        call = vlm if backend == "eager" else torch.compile(vlm, dynamic=False, options=OPTIONS)
        result = call(*inputs)
        if not torch.isfinite(result).all():
            raise ValueError(f"nonfinite VLM {backend} output")
        if backend == "compile":
            report["eager_compile_embedding_comparison"] = compare_embedding(embedding, result)
        vlm_results[backend] = measure(call, inputs, args, f"vlm-{backend}")
        report["phases"]["vlm"] = vlm_results
        write(args.output / "profile-summary.json", report)

    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead

    with no_init_weights():
        head = Gr00tN1d6ActionHead(config)
    head = load_weights(head, args.checkpoint, "action_head.")
    if head.num_inference_timesteps != SPEC.steps:
        raise ValueError("checkpoint denoising steps differ from the workload")
    state = embedding.new_zeros((1, SPEC.state_tokens, config.max_state_dim))
    embodiment = torch.tensor([2], device="cuda")
    norms = {"vl": head.vlln, "adaln": head.model.transformer_blocks[0].norm1.norm,
             "post_attention": head.model.transformer_blocks[0].norm3, "output": head.model.norm_out}
    report["nonlinear_norms"] = {
        name: dict(shape=list(norm.normalized_shape), eps=norm.eps,
                   weight=norm.weight is not None, bias=norm.bias is not None)
        for name, norm in norms.items()
    }
    report["timestep_embedding_width"] = head.model.inner_dim

    def action(v, state, embodiment, mask, image_mask):
        backbone = BatchFeature(dict(backbone_features=v, image_mask=image_mask, backbone_attention_mask=mask))
        return head.get_action(backbone, BatchFeature(dict(state=state, embodiment_id=embodiment))).action_pred

    action_inputs = embedding, state, embodiment, inputs[1], image_mask
    query_shapes = []

    def record_queries(module, args, kwargs):
        hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        query_shapes.append(list(hidden.shape))

    with head.model.transformer_blocks[0].register_forward_pre_hook(record_queries, with_kwargs=True):
        result = action(*action_inputs)
    if query_shapes != [[1, SPEC.queries, SPEC.hidden]] * SPEC.steps:
        raise ValueError(f"unexpected DiT query geometry: {query_shapes}")
    report["dit_query_shapes"] = query_shapes
    report["action_output_shape"] = list(result.shape)
    ae_results = {}
    for backend in args.backends:
        print(f"Preparing full AE {backend}", flush=True)
        call = action if backend == "eager" else torch.compile(action, dynamic=False, options=OPTIONS)
        torch.manual_seed(17)
        result = call(*action_inputs)
        if result.shape != (1, SPEC.horizon, config.max_action_dim) or not torch.isfinite(result).all():
            raise ValueError(f"invalid AE {backend} result")
        ae_results[backend] = measure(call, action_inputs, args, f"ae-{backend}")
        report["phases"]["ae"] = ae_results
        write(args.output / "profile-summary.json", report)
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.output / "profile-summary.json", report)
    print("Completed", args.output / "profile-summary.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, choices=(256, 448), default=448)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--default-image-preprocess", action="store_true")
    parser.add_argument("--expected-vit-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--prompt", default="Pick up the object and place it on the table.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--backends", nargs="+", choices=("eager", "compile"), default=("eager", "compile"))
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--lightweight-trace", action="store_true")
    parser.add_argument("--profile-repeats", type=int, default=1)
    args = parser.parse_args()
    if args.samples < 2 or args.warmup < 1 or args.cameras < 1:
        parser.error("samples must be >= 2; warmup and cameras must be positive")
    if args.profile_repeats < 1:
        parser.error("profile-repeats must be positive")
    if (args.image_width is None) != (args.image_height is None):
        parser.error("image-width and image-height must be provided together")
    if args.image_width is not None and min(args.image_width, args.image_height) < 1:
        parser.error("image dimensions must be positive")
    args.mode = "profile"
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    run(args)
