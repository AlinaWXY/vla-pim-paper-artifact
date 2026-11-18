"""Prepare case b with GPU-side DRAM reads charged in the Thor kernel timings."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")))


def prepare_program(cfg, out):
    program = json.loads(Path(cfg["program_path"]).read_bytes())
    reads_per_channel = {
        "rope_export": (256, {"rope_export"}),
        "qkt": (1984, {"score_stream"}),
        "pv": (128, {"pv_export"}),
        "o": (1024, {"o_tile0_export", "o_tile1_export"}),
        "gate_up": (1024, {"gate_up_export"}),
        "down": (128, {"down_export"}),
    }
    excluded = Counter()
    for stage in program["stages"]:
        name = stage["name"]
        expected, epochs = reads_per_channel.get(name, (0, set()))
        for channel in stage["channels"]:
            count = 0
            for i, op in enumerate(channel):
                assert "children" not in op and op["op"] != "RDRES"
                if op["op"] != "RD":
                    continue
                assert op["epoch"] in epochs, (name, op)
                # Keep all producer dependencies and ordinal references; only the host read is timed by GPU.
                channel[i] = {k: v for k, v in op.items() if k in {"id", "addr", "deps", "issue_deps"}}
                channel[i]["op"] = "BARRIER"
                count += 1
            assert count == expected, (name, count)
            excluded[name] += count
        stage["expected_dq_bytes"] -= excluded[name] * 32
    count = sum(excluded.values())
    assert program["counts"].pop("RD") == count
    assert count * 32 * 180 == 418775040
    program["counts"]["BARRIER"] += count
    for field in ("expected_dq_bytes", "expected_dq_bytes_per_layer"):
        program[field] -= count * 32
        assert program[field] == sum(stage["expected_dq_bytes"] for stage in program["stages"])
    path = out / "pim-program.json"
    write(path, program)
    write(out / "deducted-reads.json", excluded)
    return path


def compile_schedule(events, benchmarks, *, gpu_rope=True):
    norm = ("copies_casts", "rms_pow", "rms_mean", "norm_add", "rms_rsqrt",
            "norm_mul", "norm_add", "norm_mul", "norm_add", "copies_casts", "copies_casts")
    rules = {
        "residual_adarmsnorm": ("residual_gate_mul", "residual_gate_add") + norm,
        "adarmsnorm": norm,
        "rope_qk": ("rope_apply",) * 10,
        "scale_mask_softmax_cast": ("attention_mul", "attention_add", "softmax", "copies_casts"),
        "gelu_gate": ("gelu", "ffn_gate_multiply"),
    }
    names = [e["name"] if e["kind"] == "gpu" else None for e in events]
    compiled = []
    counts = Counter()
    i = 0
    while i < len(events):
        for label, pattern in rules.items():
            n = len(pattern)
            if tuple(names[i:i + n]) != pattern:
                continue
            assert events[i]["step"] == events[i + n - 1]["step"]
            compiled.append(dict(events[i], name=label, us=benchmarks[label]))
            counts[label] += 1
            i += n
            break
        else:
            compiled.append(events[i])
            i += 1
    expected = Counter(residual_adarmsnorm=360, adarmsnorm=10, scale_mask_softmax_cast=180, gelu_gate=180)
    if gpu_rope:
        expected["rope_qk"] = 180
    assert counts == expected
    assert [e for e in compiled if e["kind"] == "pim"] == [e for e in events if e["kind"] == "pim"]
    return compiled


if __name__ == "__main__":
    config_path, bench_path, out = map(Path, sys.argv[1:])
    config = json.loads(config_path.read_bytes())
    with bench_path.open(newline="") as handle:
        times = {row["case"]: float(row["compile_device_p50_us"]) for row in csv.DictReader(handle)}
    schedule = json.loads(Path(config["ae_schedule_path"]).read_bytes())
    events = compile_schedule(schedule["events"], times)
    program_path = prepare_program(config, out)
    for steps in (1, 10):
        dest = out / str(steps)
        dest.mkdir()
        ae_path = dest / "ae.json"
        write(ae_path, dict(schedule, denoising_steps=steps, events=[e for e in events if e["step"] < steps]))
        write(dest / "config.json", dict(config, run_kind="ae_solo", program_path=str(program_path),
                                         ae_schedule_path=str(ae_path)))
