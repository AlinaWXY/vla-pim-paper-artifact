"""Report native event timestamps without retiming, offsets, or replay."""
from __future__ import annotations

import math
from pathlib import Path

from experiments.common import write_csv, write_json
from experiments.config import CASES, variant_name
from experiments.native import metrics, total_us


def duration(row):
    return float(row.get("duration_us", row["end_us"] - row["start_us"]))


def kernel_active_duration(row):
    return duration(row) - float(row.get("paused_us", 0))


def stage_key(row):
    return (row["stage_id"], row["name"], row["step"], row["layer"])


def summarize(case, combined, ae_solo, vlm_solo, output, *, program=None, calibration=None, vlm_input=None):
    output = Path(output)
    solo_stages = {stage_key(row): row for row in ae_solo["stages"]}
    current = {stage_key(row): row for row in combined["stages"]}
    if solo_stages.keys() != current.keys():
        raise ValueError(f"{case}: combined PIM stage coverage differs from standalone")
    stages = []
    for row in combined["stages"]:
        solo = solo_stages[stage_key(row)]
        stages.append(dict(stage_id=row["stage_id"], name=row["name"], step=row["step"], layer=row["layer"],
                           solo_us=duration(solo), concurrent_us=duration(row),
                           interference_us=duration(row) - duration(solo),
                           AE_GPU_wait_us=float(row.get("gpu_wait_us", 0)),
                           start_us=row["start_us"], end_us=row["end_us"]))
    solo_kernels = {row["ordinal"]: row for row in vlm_solo["kernels"]}
    current_kernels = {row["ordinal"]: row for row in combined["kernels"]}
    if solo_kernels.keys() != current_kernels.keys():
        raise ValueError(f"{case}: combined VLM coverage differs from standalone")
    kernels = []
    for row in combined["kernels"]:
        solo = solo_kernels[row["ordinal"]]
        entry = dict(ordinal=row["ordinal"], start_us=row["start_us"], end_us=row["end_us"],
                     solo_us=duration(solo), duration_us=duration(row),
                     stall_us=kernel_active_duration(row) - duration(solo))
        if "paused_us" in row:
            entry["paused_us"] = row["paused_us"]
        kernels.append(entry)
    m = metrics(combined)
    ae_us = total_us(ae_solo)
    vlm_us = total_us(vlm_solo)
    interference = sum(row["interference_us"] for row in stages)
    wait = float(m["AE_GPU_wait_us"])
    ae_elapsed = float(m["AE_elapsed_us"])
    total = total_us(combined)
    # The raw scheduler owns the timestamps. These are diagnostics, never a
    # replacement value or latency correction.
    checks = {
        "stage_coverage": len(stages), "vlm_coverage": len(kernels),
        "AE_timestamp_accounting_error_us": ae_elapsed - ae_us - interference - wait,
        "completion_order": total + 1e-8 >= ae_elapsed,
        "all_durations_nonnegative": all(duration(r) >= 0 for r in combined["stages"] + combined["kernels"]),
    }
    if not checks["completion_order"] or not checks["all_durations_nonnegative"]:
        raise ValueError(f"{case}: invalid native completion timestamps")
    # One CK per GPU-duration rounding is permitted because all quantities
    # originate from integer CK in C++, not floating-point model retiming.
    if abs(checks["AE_timestamp_accounting_error_us"]) > 0.002:
        raise ValueError(f"{case}: AE timestamp identity failed: {checks['AE_timestamp_accounting_error_us']}")
    if case == "b":
        checks["serial_sum_error_us"] = total - ae_us - vlm_us
        if abs(checks["serial_sum_error_us"]) > 0.002:
            raise ValueError(f"case b native total != AE_solo + VLM_solo: {checks['serial_sum_error_us']} us")
    if program is not None:
        expected = program.get("expected_dq_bytes_per_layer", program.get("expected_dq_bytes"))
        if expected is None:
            raise ValueError(f"{case}: program has no DQ conservation contract")
        if isinstance(expected, dict):
            expected = expected["total"]
        actual = m.get("pim_read_bytes", 0) + m.get("pim_write_bytes", 0)
        # Full AE schedule instantiates every single-layer template 180 times.
        expected *= 180
        checks["PIM_DQ_expected_bytes"] = expected
        checks["PIM_DQ_actual_bytes"] = actual
        if expected != actual:
            raise ValueError(f"{case}: PIM DQ conservation failed: {actual} != {expected}")
    if vlm_input is not None and calibration is not None:
        alpha = calibration["parameters"]["alpha"]
        expected_read = sum(u["read_sectors"] for u in vlm_input["units"]) * 32
        expected_write = sum(math.ceil(alpha * u["write_sectors"]) if u["write_is_proxy"] else u["write_sectors"]
                             for u in vlm_input["units"]) * 32
        checks["GPU_DQ_expected_read_bytes"] = expected_read
        checks["GPU_DQ_expected_write_bytes"] = expected_write
        checks["GPU_DQ_actual_read_bytes"] = m["gpu_read_bytes"]
        checks["GPU_DQ_actual_write_bytes"] = m["gpu_write_bytes"]
        if (expected_read, expected_write) != (m["gpu_read_bytes"], m["gpu_write_bytes"]):
            raise ValueError(f"{case}: GPU DQ conservation failed")
    bus = []
    for row in combined["bus"]:
        row = dict(row)
        wall = row.get("duration_cycles", 0)
        row["dq_utilization"] = row["dq_cycles"] / wall if wall else 0
        row["ca_utilization"] = row["ca_cycles"] / wall if wall else 0
        if not 0 <= row["dq_utilization"] <= 1 + 1e-12 or not 0 <= row["ca_utilization"] <= 1 + 1e-12:
            raise ValueError(f"{case}: impossible bus utilization {row}")
        bus.append(row)
    completed = [row["ordinal"] for row in combined["kernels"] if row["end_us"] <= ae_elapsed]
    remaining_ms = float(m["VLM_remaining_us"]) / 1000
    # The original 21 columns remain first. Compatibility names are explicit
    # aliases of native measurements; there is no second corrected timeline.
    result = dict(case=case, variant=variant_name(CASES[case]), total_ms=total / 1000,
                  AE_solo_ms=ae_us / 1000, PIM_interference_ms=interference / 1000,
                  AE_GPU_wait_ms=wait / 1000, net_overlap_ms=(ae_us + vlm_us - total) / 1000,
                  AE_native_schedule_ms=ae_us / 1000,
                  VLM_memory_stall_ms=sum(row["stall_us"] for row in kernels) / 1000,
                  VLM_completed_model_ms=sum(duration(solo_kernels[n]) for n in completed) / 1000,
                  VLM_remaining_model_ms=remaining_ms, raw_model_total_ms=total / 1000,
                  completed_units=len(completed))
    for field in ("GPU_act1", "GPU_act2", "GPU_prepb", "GPU_preab", "GPU_reads", "GPU_writes",
                  "GPU_row_hits", "GPU_row_misses"):
        result[field] = m[field]
    result.update(AE_parallel_ms=ae_elapsed / 1000, VLM_solo_ms=vlm_us / 1000,
                  VLM_remaining_ms=remaining_ms, PIM_stages=len(stages), VLM_work_units=len(kernels))
    write_csv(output / f"{case}-PIM-and-AE-waits.csv", stages)
    write_csv(output / f"{case}-VLM-placement.csv", kernels)
    write_csv(output / f"{case}-bus.csv", bus)
    write_json(output / f"{case}-checks.json", checks)
    return result


def write_report(output, rows, calibration, repeat_checks):
    output = Path(output)
    write_csv(output / "evaluation-b-g.csv", rows)
    write_json(output / "summary.json", {
        "schema_version": 1, "evaluation": rows, "calibration": calibration,
        "repeat_checks": repeat_checks,
        "interpretation": "Single Ramulator2 instance, closed-loop synthetic sequential VLM addresses; model estimate, not Thor concurrent hardware measurement",
        "unmodeled": ["GPU SM occupancy of PIM transfers", "GPU/PIM control handoff latency",
                      "GPU compute interference beyond nonpreemptive kernel serialization", "actual VLM burst arrival/address traces"],
        "latency_correction": None,
        "compatibility_columns": {
            "AE_native_schedule_ms": "Alias of AE_solo_ms; the native instance is the only AE timing model",
            "VLM_completed_model_ms": "Sum of standalone native VLM unit durations for units ending by AE completion",
            "VLM_remaining_model_ms": "Alias of actual native tail VLM_remaining_ms, including any partially finished unit",
            "raw_model_total_ms": "Alias of total_ms without correction",
            "completed_units": "VLM units whose native end timestamp is at or before AE completion",
            "GPU_command_counts": "Native rank0 command counters; row hits require no array prerequisite while servicing the request",
        },
    })
    lines = ["# Ramulator2 重写版消融实验", "", "本表直接由同一个原生仿真器的事件时间戳计算，未使用旧时间线重放、常数偏差或事后归一化。", "",
             "| case | AE 单独 ms | AE 并发 ms | PIM 干扰 ms | AE GPU 等待 ms | VLM stall ms | 总时延 ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['AE_solo_ms']:.6f} | {r['AE_parallel_ms']:.6f} | {r['PIM_interference_ms']:.6f} | {r['AE_GPU_wait_ms']:.6f} | {r['VLM_memory_stall_ms']:.6f} | {r['total_ms']:.6f} |")
    v = calibration["validation"]
    lines += ["", f"VLM 校准总时延 {v['total_us']/1000:.6f} ms；目标 {v['target_us']/1000:.6f} ms；相对误差 {v['relative_error']:.3%}。逐 kernel 相对误差中位数 {v['kernel_relative_error_median']:.3%}，p90 {v['kernel_relative_error_p90']:.3%}。",
              "", "alpha 是对 L2 写扇区代理拟合的 DRAM 写回比例，不是已测 DRAM 写流量。VLM 使用均匀释放与连续合成地址，结果不能当作真实 Thor 并发硬件性能。固定省略 GPU 搬运 SM 开销与控制交接延迟；AE GPU 算子与 VLM kernel 按非抢占规则分时。",
              "", "CSV 前 21 列兼容旧表。AE_native_schedule_ms 等于 AE_solo_ms；raw_model_total_ms 等于 total_ms，均无修正。VLM_completed_model_ms 累加 AE 完成时已结束单元的原生 VLM solo 时长；completed_units 统计这些单元。VLM_remaining_model_ms 等于 AE 完成后的实际 VLM 尾部时长，可能包含已启动但未完成的单元，因此不要求它与 completed 列构成 solo 时长的补集。GPU 命令与行命中统计均来自 rank0 原生执行。", ""]
    (output / "REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
