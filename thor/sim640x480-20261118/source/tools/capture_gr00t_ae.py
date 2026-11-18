"""Capture timing of compiled DiT operators at the simulated GPU/PIM boundaries."""
from collections import defaultdict
from statistics import median

import torch
import torch.nn.functional as F

from workloads.models import GR00T_N16_H15 as SPEC


def heads(value):
    return value.reshape(1, -1, SPEC.heads, SPEC.head_dim).transpose(1, 2)


def project(x, qw, kw, vw):
    return F.linear(x, qw), F.linear(x, kw), F.linear(x, vw)


def query_pack(q, bias):
    return heads((q.float() + bias).bfloat16()).contiguous()


def feature_query_pack(q, bias):
    q = F.pad(query_pack(q, bias), (0, 64 - SPEC.head_dim))
    return q.reshape(1, 8, 4, SPEC.queries, 2, 32).permute(0, 1, 4, 2, 3, 5).contiguous()


def cross_project(x, kw, kb, vw, vb):
    return heads(F.linear(x, kw, kb)), heads(F.linear(x, vw, vb))


def kv_pack(k, v, pad):
    k = F.pad(k, (0, 0, 0, pad))
    v = F.pad(v, (0, 0, 0, pad))
    return k.transpose(-1, -2).contiguous(), v.contiguous()


def self_kv_pack(k, v, kb, vb):
    return kv_pack(query_pack(k, kb), query_pack(v, vb), 0)


def adaln(x, scale, shift, eps):
    return F.layer_norm(x, (SPEC.hidden,), eps=eps) * (1 + scale[:, None]) + shift[:, None]


def score(q, k):
    return q.float() @ k.float()


def softmax(scores, mask):
    scores = scores * (SPEC.head_dim ** -0.5)
    return scores.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(-1).bfloat16()


def softmax_feature(scores, mask):
    return softmax(scores.sum(0), mask)


def pv_pack(x):
    return x.transpose(1, 2).reshape(1, SPEC.queries, SPEC.hidden).contiguous()


def o_residual(x, partial, bias, eps):
    residual = x + (partial.sum(dim=0) + bias).to(x.dtype)
    return residual, F.layer_norm(residual, (SPEC.hidden,), eps=eps)


def ff_residual(x, y, bias):
    return x + (y + bias).to(x.dtype)


def gelu(x, bias):
    return F.gelu((x + bias).bfloat16(), approximate="tanh")


def ledger(head, vl, state, embodiment, mask, image_mask, fixture, options, samples):
    dit = head.model
    counters = defaultdict(list)
    compiled = {}

    def event(name, fn, *params, step=-1, layer=-1):
        if fn not in compiled:
            compiled[fn] = torch.compile(fn, dynamic=False, options=options)
        fn = compiled[fn]
        if recording:
            with torch.profiler.record_function(f"gr00t:{step}:{layer}:{name}"):
                return fn(*params)
        return fn(*params)

    def state_input(v, s, e):
        return head.vlln(v), head.state_encoder(s, e)

    def step_input(actions, state_features, t, e):
        actions = head.action_encoder(actions, t, e)
        pos = head.position_embedding(torch.arange(SPEC.horizon, device=actions.device))
        return torch.cat((state_features, actions + pos), dim=1), dit.timestep_encoder(t)

    def step_output(x, temb, e, actions):
        shift, scale = dit.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        x = dit.proj_out_2(dit.norm_out(x) * (1 + scale[:, None]) + shift[:, None])
        delta = head.action_decoder(x, e)[:, -SPEC.horizon:]
        return actions + delta / SPEC.steps

    def pv(probability, value):
        return probability @ value

    timesteps = [torch.full((1,), step * head.num_timestep_buckets // SPEC.steps, device=vl.device)
                 for step in range(SPEC.steps)]
    base_actions = torch.randn((1, SPEC.horizon, head.action_dim), dtype=vl.dtype, device=vl.device)
    # These modulation vectors depend only on the fixed inference timestep buckets.
    modulation = [[block.norm1.linear(F.silu(dit.timestep_encoder(t))).chunk(2, 1)
                   for block in dit.transformer_blocks] for t in timesteps]
    all_tokens = vl.shape[1]
    _, aligned = SPEC.kv_shape("cross", all_tokens)
    full = torch.ones((1, SPEC.queries), device=vl.device, dtype=torch.bool)
    masks = [(~image_mask & mask) if layer % (2 * dit.attend_text_every_n_blocks) == 0 else image_mask & mask
             for layer in range(SPEC.layers)]
    masks = [F.pad(m, (0, aligned - all_tokens)) for m in masks]
    o_scratch = vl.new_zeros((16, 1, SPEC.queries, SPEC.hidden), dtype=torch.float32)

    def sweep():
        v, state_features = event("state_and_vl_norm", state_input, vl, state, embodiment)
        kv = {}
        for layer, block in enumerate(dit.transformer_blocks):
            if SPEC.layer_kinds[layer] == "cross":
                attn = block.attn1
                k, value = event("cross_kv", cross_project, v, attn.to_k.weight, attn.to_k.bias,
                                 attn.to_v.weight, attn.to_v.bias, layer=layer)
                k, value = event("kv_pack", kv_pack, k, value, aligned - all_tokens, layer=layer)
                kv[layer] = k, value
        actions = base_actions
        for step, t in enumerate(timesteps):
            x, temb = event("step_input", step_input, actions, state_features, t, embodiment, step=step)
            if x.shape != (1, SPEC.queries, SPEC.hidden):
                raise ValueError("H=15 must yield 16 DiT queries")
            for layer, block in enumerate(dit.transformer_blocks):
                position = dict(step=step, layer=layer)
                attn = block.attn1
                scale, shift = modulation[step][layer]
                z = event("adaln", adaln, x, scale, shift, block.norm1.norm.eps, **position)
                if SPEC.layer_kinds[layer] == "self":
                    q, k, value = event("qkv", project, z, attn.to_q.weight,
                                        attn.to_k.weight, attn.to_v.weight, **position)
                    k, value = event("kv_pack", self_kv_pack, k, value, attn.to_k.bias, attn.to_v.bias, **position)
                    allowed = full
                else:
                    q = event("qkv", F.linear, z, attn.to_q.weight, None, **position)
                    k, value = kv[layer]
                    allowed = masks[layer]
                event("q_pack_feature", feature_query_pack, q, attn.to_q.bias, **position)
                q = event("q_pack", query_pack, q, attn.to_q.bias, **position)
                logits = event("qkt", score, q, k, **position)
                probability = event("softmax", softmax, logits, allowed, **position)
                components = torch.stack((logits, torch.zeros_like(logits)))
                event("softmax_feature", softmax_feature, components, allowed, **position)
                attended = event("pv", pv, probability, value, **position)
                packed = event("pv_pack", pv_pack, attended, **position)
                out = attn.to_out[0]
                projected = event("o", F.linear, packed, out.weight, None, **position)
                o_scratch[0].copy_(projected)
                x, norm = event("o_residual", o_residual, x, o_scratch, out.bias, block.norm3.eps, **position)
                up = block.ff.net[0].proj
                a = event("up", F.linear, norm, up.weight, None, **position)
                a = event("gelu", gelu, a.float(), up.bias, **position)
                down = block.ff.net[2]
                d = event("down", F.linear, a, down.weight, None, **position)
                x = event("ff_residual", ff_residual, x, d.float(), down.bias, **position)
            actions = event("step_output", step_output, x, temb, embodiment, actions, step=step)
        return actions

    print("Compiling and warming isolated DiT operator boundaries", flush=True)
    recording = False
    result = sweep()
    if not torch.isfinite(result).all():
        raise ValueError("nonfinite values in segmented DiT")
    torch.cuda.synchronize()
    recording = True
    profile_samples = min(samples, 5)
    for _ in range(profile_samples):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profile:
            sweep()
            torch.cuda.synchronize()
        observed = {}
        for row in profile.events():
            if row.device_type != torch.autograd.DeviceType.CPU or not row.name.startswith("gr00t:"):
                continue
            _, step, layer, name = row.name.split(":")
            key = int(step), int(layer), name
            if key in observed:
                raise ValueError(f"duplicate CUPTI operator boundary: {key}")
            observed[key] = row.device_time_total
        if not observed or not any(observed.values()) or counters and observed.keys() != counters.keys():
            raise ValueError("incomplete CUPTI coverage of DiT operators")
        for key, elapsed_us in observed.items():
            counters[key].append(elapsed_us)
    return dict(fixture, schema_version=1, steps=SPEC.steps, layers=SPEC.layers,
                layer_kinds=SPEC.layer_kinds, samples=profile_samples,
                units=[dict(step=step, layer=layer, name=name, us=median(values))
                       for (step, layer, name), values in counters.items()],
                timing="Median of summed CUPTI device durations inside each compiled operator boundary",
                cross_kv_policy="computed once per frame for each even DiT layer",
                modulation_policy="computed once offline for each fixed timestep bucket")
