"""The five design switches and fixed hardware validation from the rewrite plan."""
from __future__ import annotations

from copy import deepcopy
import math

ARBITRATIONS = {"frfcfs", "pim_head", "pim_lookahead", "gap_batch"}
CASES = {
    "b": dict(layout="feature", rope="gpu", dataflow="streaming", mode="serial"),
    "c": dict(layout="feature", rope="gpu", dataflow="streaming", mode="concurrent", arbitration="pim_head"),
    "d": dict(layout="token", rope="gpu", dataflow="streaming", mode="concurrent", arbitration="pim_head"),
    "e": dict(layout="token", rope="local", dataflow="streaming", mode="concurrent", arbitration="pim_head"),
    "f": dict(layout="token", rope="gpu", dataflow="fused", mode="concurrent"),
    "g": dict(layout="token", rope="local", dataflow="fused", mode="concurrent"),
    "e-frfcfs": dict(layout="token", rope="local", dataflow="streaming", mode="concurrent", arbitration="frfcfs"),
    "e-lookahead": dict(layout="token", rope="local", dataflow="streaming", mode="concurrent", arbitration="pim_lookahead"),
    "e-gap": dict(layout="token", rope="local", dataflow="streaming", mode="concurrent", arbitration="gap_batch"),
}


def variant_name(config):
    suffix = "after" if config["dataflow"] == "fused" else "before"
    return f"{config['layout']}_{config['rope']}_{suffix}"


def validate_calibration(params):
    required = {"alpha", "window", "write_high", "write_low"}
    if set(params) != required:
        raise ValueError(f"calibration must contain exactly {sorted(required)}")
    alpha = params["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0.05 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0.05, 1]")
    for name in ("window", "write_high", "write_low"):
        if isinstance(params[name], bool) or not isinstance(params[name], int):
            raise ValueError(f"{name} must be an integer")
    if not 1 <= params["window"] <= 4096:
        raise ValueError("window must be in [1, 4096]")
    if not 0 <= params["write_low"] < params["write_high"] <= params["window"]:
        raise ValueError("write watermarks must satisfy 0 <= low < high <= window")
    return params


def validate_config(config):
    execution = config.get("execution_model", "native_dram_requests")
    if execution not in {"native_dram_requests", "fused_kernel_windows"}:
        raise ValueError("unknown execution_model")
    if "vlm_gpu_quantum_us" in config:
        quantum = config["vlm_gpu_quantum_us"]
        if (execution != "native_dram_requests" or config.get("mode") != "concurrent"
                or config.get("run_kind", "combined") != "combined"):
            raise ValueError("vlm_gpu_quantum_us requires native concurrent combined execution")
        if type(quantum) not in (int, float) or not 0 < quantum <= 5_000_000:
            raise ValueError("vlm_gpu_quantum_us must be positive and at most 5000000")
    if execution == "fused_kernel_windows":
        if (config.get("mode"), config.get("dataflow"), config.get("run_kind", "combined")) != ("concurrent", "fused", "combined"):
            raise ValueError("fused_kernel_windows requires concurrent fused combined execution")
        if not config.get("calibrated_vlm_path"):
            raise ValueError("fused_kernel_windows requires calibrated_vlm_path")
    elif "calibrated_vlm_path" in config:
        raise ValueError("calibrated_vlm_path is only valid for fused_kernel_windows")
    if execution == "native_dram_requests":
        release = config.get("vlm_release", "closed_loop")
        if release not in {"closed_loop", "uniform"}:
            raise ValueError("unknown vlm_release")
        if release == "closed_loop":
            w_max = config.get("w_max", 128)
            if "calibration" in config or type(w_max) is not int or w_max not in (128, 256):
                raise ValueError("closed_loop uses w_max 128 or 256 and fixed write watermarks")
        elif "w_max" in config or "mlp_path" in config:
            raise ValueError("uniform does not accept closed-loop settings")
    if "refresh_allbank" in config:
        raise ValueError("refresh_allbank is forbidden: rank0 uses REFpb and rank1 uses REFab as a fixed hardware rule")
    for name, options in (("layout", {"feature", "token"}), ("rope", {"gpu", "local"}),
                          ("dataflow", {"streaming", "fused"}), ("mode", {"serial", "concurrent"})):
        if config.get(name) not in options:
            raise ValueError(f"{name} must be one of {sorted(options)}")
    if config["layout"] == "feature" and (config["rope"] != "gpu" or config["dataflow"] != "streaming"):
        raise ValueError("feature layout requires gpu RoPE and streaming dataflow")
    competing = config["mode"] == "concurrent" and config["dataflow"] == "streaming"
    if competing:
        if config.get("arbitration") not in ARBITRATIONS:
            raise ValueError("concurrent streaming requires an arbitration policy")
    elif "arbitration" in config:
        raise ValueError("arbitration must be absent for serial or fused operation")
    if config.get("mac_freq_mult", 4) not in (1, 2, 4) or isinstance(config.get("mac_freq_mult"), bool):
        raise ValueError("mac_freq_mult must be 1, 2, or 4")
    if config.get("run_kind", "combined") not in {"combined", "ae_solo", "vlm_solo"}:
        raise ValueError("invalid run_kind")
    retired = [k for k in config if k.startswith("pim_") or k in {"shift", "constant_offset", "dq_latency", "row_hit"}]
    if retired:
        raise ValueError(f"retired timing/compatibility switches are forbidden: {retired}")
    if "calibration" in config:
        validate_calibration(config["calibration"])
    return config


def case_config(case, *, mac_freq_mult=4, run_kind="combined", **paths):
    if case not in CASES:
        raise ValueError(f"unknown simulated case {case!r}; case a is measured, never simulated")
    cfg = deepcopy(CASES[case])
    cfg.update(case=case, mac_freq_mult=mac_freq_mult, run_kind=run_kind, **paths)
    return validate_config(cfg)
