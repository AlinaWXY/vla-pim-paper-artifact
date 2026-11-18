"""Run and report the six-case native scaled-VLM ablation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import ROOT, read_json, sha256, write_csv, write_json
from experiments.run import prepare_configs
from experiments.run_closed_loop import run_closed_loop
from tools.calibrate_scaled_vlm import SPEC, SLOW

CASES = ("b", "e", "e-lookahead", "e-frfcfs", "e-gap", "g")
DEFAULT_OUTPUT = ROOT / "results/ablation-scaled-w128"


def run_case(binary, output, case, program_dir, *, resume=False, allow_unit_errors_sha256=None):
    path = output / "calibration-scaled.json"
    ledger = read_json(path)
    allow_unit_errors = allow_unit_errors_sha256 is not None
    if allow_unit_errors and allow_unit_errors_sha256 != sha256(path):
        raise ValueError("the per-unit waiver is pinned to a different calibration file")
    v = ledger["validation"]
    if (ledger.get("calibration_model") != "scaled_ablation" or abs(v["relative_error"]) > .01 or
        (not v["passed"] and not allow_unit_errors)):
        raise ValueError("scaled solo failed a required timing gate")
    if (ledger["w_max"], ledger["write_high"], ledger["write_low"], ledger["alpha"]) != (128, 96, 32, 1):
        raise ValueError("scaled ablation hardware parameters differ")
    destination = output / "cases" / case
    configs = prepare_configs(destination / "configs", spec_path=SPEC, program_dir=program_dir)
    scaled = ledger["provenance"]["inputs"]["vlm"]["path"]
    for config in configs.values():
        config["vlm_path"] = scaled
    run_closed_loop(binary, configs, destination, path, [case], resume=resume,
                    allow_scaled_unit_errors=allow_unit_errors)


def report(output):
    path = output / "calibration-scaled.json"
    ledger = read_json(path)
    v = ledger["validation"]
    results = []
    for case in CASES:
        case_path = output / "cases" / case / "closed-loop-summary.json"
        if not case_path.is_file():
            results.append(dict(case=case, status="未运行"))
            continue
        saved = read_json(case_path)
        [row] = saved["evaluation"]
        if (saved["mlp_sha256"] != sha256(path) or row["case"] != case or
            not row["raw_repeats_byte_identical"] or abs(row["total_identity_residual_us"]) > .002):
            raise ValueError(f"{case} does not match the calibrated run")
        results.append(dict(case=case, status="已完成",
                            total_ms=row["total_ms"], ae_solo_ms=row["AE_solo_ms"],
                            pim_stage_delta_ms=row["PIM_interference_ms"],
                            ae_gpu_wait_ms=row["AE_GPU_wait_ms"], vlm_tail_ms=row["VLM_remaining_ms"],
                            identity_error_us=row["total_identity_residual_us"],
                            vlm_slowdown_ms=row["VLM_memory_stall_ms"],
                            raw_sha256=row["raw_sha256"],
                            allowed_unit_errors=saved.get("allow_scaled_unit_errors", False)))
    complete = {r["case"]: r for r in results if r["status"] == "已完成"}
    for row in complete.values():
        row["speedup_over_b"] = complete["b"]["total_ms"] / row["total_ms"] if "b" in complete else None
    comparison = None
    if all(case in complete for case in CASES[1:]):
        best = min((complete[case] for case in CASES[1:-1]), key=lambda r: r["total_ms"])
        g = complete["g"]
        comparison = dict(best_streaming_case=best["case"],
                          g_minus_best_ms=g["total_ms"]-best["total_ms"],
                          g_relative_delta=g["total_ms"]/best["total_ms"]-1)
    write_json(output / "ablation.json", dict(calibration_sha256=sha256(path), calibration=v,
                                              results=results, comparison=comparison))
    write_csv(output / "ablation.csv", results)
    lines = [
        "# W_max=128 的缩放流量消融", "",
        "所有仿真使用原生逐请求闭环、α=1.0、写水位 96/32。"
        "阶段 A 中的 874 个单元保持 W_k=128，按同一 s_k 缩减各自的读写请求数。"
        "缩减量表示 W=128 时仿真器在实测时长内能完成的流量。"
        "两类请求都按同一比例计算，再分别向上取整到 32 B 扇区。各自保留原地址区域的顺序前缀。"
        "其余单元沿用阶段 A 的 W_k 和原流量。", "",
        f"实际缩放 {v['scaled_units']} 个单元，占 profile 时间 {v['scaled_profile_time_share']:.3%}。"
        f"读流量删减 {v['removed_read_bytes']:,} B，写流量删减 {v['removed_write_bytes']:,} B。", "",
        f"最终 VLM solo 为 **{v['total_us']/1000:.9f} ms**，相对 profile 总时长 105.220578 ms 的误差为 {v['relative_error']:+.4%}。"
        f"逐单元最大绝对误差 {v['max_absolute_unit_error']:.3%}。", "",
    ]
    if v["passed"]:
        lines.append("VLM solo 已在逐单元与总时长两项 1% 容限内对齐 GPU profile。")
    else:
        over = v["unit_violations"]
        slow = sum(row["classification"] == SLOW for row in over)
        lines.append(f"**逐单元验收未通过。** {len(over)} 个单元超出 1%，其中 {slow} 个属于允许缩放的 874 个单元，"
                     f"{len(over)-slow} 个沿用原流量和窗口。")
        if any(r.get("allowed_unit_errors") for r in results):
            lines.append("本轮按授权豁免逐单元启动门槛，复用原校准文件和流量输入。")
        lines.extend(["", "| ordinal | 分类 | W_k | s_k | profile（μs） | solo（μs） | 偏差 |",
                      "|---:|---|---:|---:|---:|---:|---:|"])
        for row in over:
            lines.append(f"| {row['ordinal']} | {row['classification']} | {row['w_k']} | {row['s_k']:.8f} | "
                         f"{row['measured_us']:.6f} | {row['solo_us']:.6f} | {row['relative_error']:+.3%} |")
    lines.extend(["", "每个 ordinal 的缩放、时长、窗口和 s_k 分布见 [calibration-scaled.json](calibration-scaled.json)。", "",
                  "| Case | 总时长（ms） | 相对 b 加速比 | AE solo（ms） | PIM 增量（ms） | AE GPU 等待（ms） | VLM 尾部（ms） | VLM 延迟增量（ms） |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in results:
        if row["status"] != "已完成":
            lines.append(f"| {row['case']} | 未运行 | — | — | — | — | — | — |")
            continue
        speedup = f"{row['speedup_over_b']:.6f}×" if row["speedup_over_b"] is not None else "—"
        lines.append(f"| {row['case']} | {row['total_ms']:.6f} | {speedup} | "
                     f"{row['ae_solo_ms']:.6f} | {row['pim_stage_delta_ms']:.6f} | {row['ae_gpu_wait_ms']:.6f} | "
                     f"{row['vlm_tail_ms']:.6f} | {row['vlm_slowdown_ms']:.6f} |")
    if comparison:
        lines.extend(["", f"四个 e 变体中，**{comparison['best_streaming_case']}** 总时长最短。"
                      f"g 相比它的总时长差为 **{comparison['g_minus_best_ms']:+.6f} ms**"
                      f"（{comparison['g_relative_delta']:+.3%}）。"])
    (output / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    case = commands.add_parser("case")
    case.add_argument("--case", choices=CASES, required=True)
    case.add_argument("--binary", type=Path, default=ROOT / "build/pim_vla_sim")
    case.add_argument("--program-dir", type=Path)
    case.add_argument("--resume", action="store_true")
    case.add_argument("--allow-unit-errors-sha256",
                      help="allow per-unit errors only for the scaled ledger with this SHA-256")
    for command in (case, commands.add_parser("report")):
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "case":
        run_case(args.binary.resolve(), args.output.resolve(), args.case, args.program_dir,
                 resume=args.resume, allow_unit_errors_sha256=args.allow_unit_errors_sha256)
    else:
        report(args.output.resolve())


if __name__ == "__main__":
    main()
