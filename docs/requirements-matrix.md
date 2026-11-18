# 重写方案需求与验收映射

> 本页下述验收表是 Version1 / 源码 `1e0b73b` 的历史记录；完整 AE 数字不能作为 Version2 或当前 FIFO32 分支的验证。[Down 同排单层报告](down-row-single-layer.md) 同样绑定其历史源码，采用的是有限乱序版本。

当前分支落实用户 2026-11-09 的最新澄清：fused 描述符内部 **FIFO32 仅表示缓冲容量，必须按程序顺序发射**。只允许队首及其原生前置命令发射，不允许后续指令绕过队首或提前预开行；队首发射后推进而非等待完成，保留显式依赖、资源占用和 Ramulator 原生时序所允许的流水重叠。该要求替代 Version2 的“32 项窗口内有限前瞻”条款；原始需求附件保留为历史证据。当前验证范围为单层、单个 denoising step 的 AE solo，以 `6bcd57b` 为修改前对照，同时比较模拟 latency 与 native 运行耗时。验收需确认 FIFO 顺序、流水重叠、刷新和 barrier、两版相同输入、字节守恒及 streaming 对照不变；本轮不启动校准或完整端到端实验。

依据 [2026-11-09 任务文档](rewrite-plan.txt)。本次上传已完成：**38 项 Python 测试、132 项 native 检查、5 个微基准（含 lookahead barrier）、5 种 PIM 单层、8 种并发单层配置，以及 5 种完整 18 层 × 10 步 AE solo**。证据见 [Python 测试](../reference/validation/python-tests.txt)、[原生构建与测试](../reference/validation/native-build-and-tests.txt)、[PIM 单层](../reference/validation/pim-single-layer.txt) 与 [并发单层](../reference/validation/concurrency-smoke.txt)。

完整 AE solo 的冻结源码为 1e0b73bc8da695268fbf79f1013653b54120316e，作业数组为1976674；五个任务均完成并通过独立审核，见 [full-k4-final-solos.json](../reference/validation/full-k4-final-solos.json)。[已完成 AE 原始结果归档](../reference/completed-ae-solos.tar.gz) 包含五组 raw/provenance/config、冻结清单及 audit-frozen，SHA-256 为 e636d98422f61e2d1600b357776b27be39ae0e1625d75330c0a87c9003a280bc。

**2026-11-09 用户要求先上传 branch，暂停 VLM 校准及之后的工作。VLM 校准尚未通过验收；b–g 与三个仲裁消融共9个完整 case 未运行。** 校准作业1976675_0..5及收集作业1976676已取消。完整 AE solo 已完成不代表校准或并发矩阵通过；不提供未运行的加速比或矩阵结果。下表路径相对仓库根目录；仅纳入本次已完成的可审查证据。

## 基础架构、时序与资源

| 文档要求 | 实现与产物 | 验证 / 当前状态 |
|---|---|---|
| §1–2 一套原生时序；GPU/PIM 同控制器竞争；取消重放与偏差抵消 | `pim_controller.cpp`、`pim_vla_frontend.cpp`；每通道一个包含双 rank 的原生 `DRAMDevice` | microbench、synthetic frontend、8种并发单层已通过；完整并发矩阵按用户要求暂停，未运行 |
| §3 16 通道、两 rank、x16、4 BG×4 bank、2 KiB 行、BL16=32B、CK938ps、WCK4:1 always-on | `ramulator2/python/ramulator/dram/lpddr5x_pim.py`；`reference/spec/spec.json` | `tests/test_spec.py` 固定拓扑/数值测试、`native_spec_tests.cpp` 已通过 |
| §4.1 ACT1/2、PRE、REF、RD/WR/RDA/WRA、WRGB/RDRES；细 MAC 与私有本地命令 | 同一 spec 生成原生命令、prerequisite、timing 和完成 metadata | 原生命令状态/前置条件、全 bank 激活、paired REFpb 检查已通过 |
| §4.2–4.3 写类 WRGB / 读类 RDRES 自动继承接口约束；无阵列行前置及恢复；全 BG | spec 的 transfer alias 派生；无控制器手写第二份时序 | spec/native 边界测试通过；文档例值冲突按下文处理 |
| §4.2 WRGB 在 CWL+BL 后可用；RDRES 在 CL 后传输且等待生产者 | spec metadata、`pim_resources.h` 的 ready/reader 生命周期 | SRF 未填充、RDRES 无输出、在途读被覆盖的故障检查通过 |
| §4.4 MAC 完成与间隔随 M/k；LOCALRD/WR 本地银行约束；描述符21CK+decode1CK | spec 支持 M=1/2/4/8/16/32、k=1/2/4；同原生状态中的私有 alias | spec scaling 和 native microbench 已通过；完整 k 灵敏度尚未运行 |
| §4.4/§6.1 刷新阻塞访问 | 原生 REFpb/REFab；主实验轮换 paired per-bank refresh | spec/native refresh、五种完整 AE solo 通过；完整并发矩阵尚未验证 |
| §4.5 4×4×3 命令对矩阵 | `reference/spec/command-pair-matrix.json` | Python 和真实 `DRAMDevice` 的全部 48 项边界通过 |
| §5.1 1KiB SRF/bank、2KiB FP32 输出/bank、一 rank RoPE 单元；无 rank 输入/score cache | `pim_resources.h`；精确 FP32 source_regions 与 SRF validity | `test_pim_resources.cpp`、`test_programs.py`，五程序单层和完整 AE solo 通过 |
| §5.1 描述符队列1、内部 FIFO32、256bit wire 格式 | `programs/descriptors.py`；controller descriptor cursor / decoded_until 上限 | descriptor 几何/CRC、FIFO/assertion、f/g 单层和完整 AE solo 通过；存盘展开列表不是额外硬件 FIFO |

## Layout、RoPE 与 PIM 数据路径

| 文档要求 | 实现与产物 | 验证 / 当前状态 |
|---|---|---|
| §5.2/§7.1 保留原 tensor 和切分定义；仅五种结构组合 | `workloads/layout.py`、`programs/gen_pim_program.py`；`tensor_contract` | 几何、标量 MAC 数、物理地址范围、源数据依赖与五组合校验通过；不执行张量数值 |
| QKV 融合、PV token/feature 分片、PIM O、Gate/Up、Down | 生成器的 qkv/pv/o/gate_up/down stage；权重有效尺寸见 `program-design.md` | 五程序单层及完整18×10 AE solo 均通过独立审核 |
| 完整 W_O 每 rank 常驻、输出按 tile 复用 | 每 bank 256KiB W_O；O 每 tile 输出≤2KiB，第二 tile 等待第一 tile 全部 RDRES | 布局容量与 slot 生命周期测试通过；初始化不计入稳态程序 |
| §5.3 feature/token、gpu/local RoPE | feature 8 lane 分片、token pair-major/parity；GPU export/import、本地单 rank unit | 五程序单层、完整 AE solo 和 packet source_regions 测试通过；精确映射见 `program-design.md` |
| §5.4 streaming 逐 fragment WRGB；PV→O 导出、O 两次输入扫描 | `*_before.json`；补给依赖 MAC 完成，输出读保护 | 程序请求/DQ 守恒测试、streaming 单层及完整 AE solo 通过 |
| §5.4 fused 暂存 bank0/4、本地广播、私有 CA、外部 I/O epoch | `*_after.json`、MAC_RUN/ROPE_RUN children、controller epoch | fused 单层、并发单层及完整 AE solo 通过；完整 VLM 竞争实验按用户要求暂停 |
| §5.2 QK streaming 逐列归约后立即 RDRES；fused LOCALWR score 后集中 RD | qkt 请求；无 rank score tile cache | 请求结构、score 地址与来源依赖、单层和完整 AE solo 通过 |
| §7.1 JSON 程序一次生成，附哈希；显式 completion 依赖 | `programs/generated/<variant>.json`；单层模板，stage 全通道 barrier | 生成确定性、SRF/output/DRAM 暂存生产者测试通过；快速 JSON 与原字段对照在 `reference/spec/json-input-verification.json` |
| §7.1 RD/WR/LOCALRD/LOCALWR 地址依赖 | 生成器显式 deps、stage barrier；静态 transient producer 检查 | 错地址和旧 P fragment 故障测试通过；运行时不是任意用户程序的通用 DRAM 地址 hazard 检测器 |
| 18 层不同权重/KV 地址，10 步复用同层 | `layer_stride_bytes_per_bank=1048576`，native row += layer×512 | 18层区间不相交且落在64MiB/bank内；测试及五完整 AE 的阶段覆盖审核通过 |

五种程序单层、全部16通道的外部 DQ 请求量分别为：feature_gpu_before **6,242,304 B**、token_gpu_before **5,480,448 B**、token_local_before **5,283,840 B**、token_gpu_after **5,087,232 B**、token_local_after **4,890,624 B**。计数由请求计算，另与保留语义的独立预期核对；这些是流量契约，不是性能结果。

## 控制器、前端与实验

| 文档要求 | 实现与产物 | 验证 / 当前状态 |
|---|---|---|
| §6.1 GPU/PIM 双来源、GPU 开页、write high/low watermark | `pim_controller.cpp` 的按 bank/方向队列；生成流中比较各队首的 row-hit/arrival | microbench / frontend smoke / 并发单层通过；实测全流量校准已暂停 |
| §6.2 frfcfs / pim_head / pim_lookahead / gap_batch | `pim_controller.cpp`、`pim_lookahead.cpp`；未来状态由 cloned native DRAMDevice 预测 | pim_head 连续段保护、lookahead barrier/horizon/FIFO32/无 live mutation 及8种并发单层通过；完整仲裁消融未运行 |
| §6.2 gap 阈值来自 spec，不增自由旋钮 | controller 使用 spec latency/burst 与预测 DQ 空档 | gap_batch 并发单层通过；完整策略效果及复现按用户要求暂停 |
| §6.3 fused 外部 I/O epoch、已发 GPU burst drain、私有 CA | 同 controller 中的固定 fused 调度，不暴露 arbitration | f/g 单层、并发单层及完整 AE solo 通过；完整实测流量并发验证暂停 |
| §6.4/§8.3 b = AE_solo + VLM_solo | 串行独立 epoch 重置机制，见 `native-model.md` | synthetic b 恒等式通过；完整 b 未运行；状态重置是已披露建模决定 |
| §7.2 2353 kernel+4 memcpy；实测 T/读 miss/写 sector；每通道 M，uniform 闭环 | `input-data/measured` → `experiments/prepare_inputs.py` → `pim_vla_frontend.cpp` | 数量、实测时长不变、每通道读写混合、闭环合成测试通过；完整 VLM 校准已暂停 |
| §7.3 全局 alpha/M/写高低水位；104.950496035ms ±5%；kernel median/p90 | `calib/calibrate_vlm.py`；计划产物 `calibration.json` 与候选明细 | 评估公式/冻结门禁已实现；**用户要求暂停，校准尚未通过验收**；不得宣称达到5% |
| §8.1 AE 18×10，实测 GPU blocks 与 PIM stage 交替 | 五个 `input-data/ae-*.json`；保留 step/layer；算术 helper 计时 | **五种完整 AE solo 通过独立审核**；阶段、DQ及 GPU 实测时长按 CK 取整后的时间戳恒等式见 reference/validation/full-k4-final-solos.json |
| §8.2 非抢占 VLM kernel，AE GPU wait，AE 后 VLM tail | `pim_vla_frontend.cpp` | synthetic 与并发单层因果/完成检查通过；完整 case 未运行 |
| §8.3 指标直接来自时间戳，无 offset | `analysis/report.py`；AE identity / serial sum / kernel coverage | 合成报表与完整 AE solo 时间戳审核通过；完整并发报表尚未产生 |
| §9.1 五旋钮严格校验、k=1/2/4 硬件参数 | `experiments/config.py`；禁止历史开关 | `tests/test_config.py` 通过；完整矩阵共用参数由 runner 校验 |
| §9.2 b–g + e-frfcfs/e-lookahead/e-gap | `experiments/run.py` 九 case；a 保留实测 | 配置准备、单层验证与五完整 AE solo 已完成；**用户要求暂停后续，九 case 完整消融未运行** |
| §9.3 CSV、逐 stage/kernel/bus、provenance | `analysis/report.py`、`experiments/native.py`；源码/输入/二进制/运行库哈希 | 旧表前21列名称/顺序已保留并通过兼容测试，追加 canonical 列；完整消融 CSV 尚未产生 |
| §11.2 迁移绘图用途，同时移除旧 trace 依赖 | `analysis/plot.py` 只消费新 evaluation CSV；PNG/SVG/PDF、输入哈希与 accounting manifest | 已用真实 c/d/e/f 单层输出生成并视觉检查；正/负/零干扰、错误总时间拒绝已验证；正式完整消融图尚未产生 |
| §10.6 两次 byte-identical、DQ 字节守恒 | runner 比较 raw 输出；report 核对 PIM×180、GPU alpha/精确 memcpy 与总线界限 | synthetic 重复、单层和完整 AE solo DQ 通过；**完整九 case 两次复现未执行** |

## 验证阶梯及交付边界

| 任务级别 | 已完成证据 | 尚未完成 |
|---|---|---|
| 1 spec | 38项 Python 测试、132项 native 检查通过；包含48命令对、阵列/接口/刷新检查 | 后续 spec 更改需重跑 |
| 2 常开 checker | 微基准、五 PIM 单层、八并发单层、五完整 AE solo 无断言；删除一条派生约束后被独立 DQ observer 抓获 | 完整 VLM/并发矩阵证据，按用户要求暂停 |
| 3 文档四微基准与新增 barrier 微基准 | pure RD、批量读写交替、pure WRGB、WRGB 中 GPU RD、lookahead_barrier 共5项通过 | 无 |
| 4 五配置 PIM solo | 五真实程序单层及完整18×10 AE solo 均通过；完整结果已独立核验并保存哈希、DQ和阶段/时间戳证据 | 旧结果差异的定量对照仍未完成；仅 I/O 差异条款存在下述冲突 |
| 5 VLM solo | 校准实现、实测输入、合成闭环验证 | 用户于2026-11-09要求暂停；尚无通过5%门槛的正式验收结果 |
| 6 完整 case | runner / 报表 / 守恒及重复检查实现，8种并发单层已验证 | 九个完整 case 未运行；重复一致性、正式CSV/图和完整矩阵 provenance 未产生 |

## 文档冲突、保留语义和实现偏离

1. **§4.3 alias 范围与 §4.5 示例冲突。** 仅派生 transfer-to-transfer 接口规则，排除 PREab 前的阵列恢复。全 BG 同 rank RDRES→WRGB 为17CK、跨 rank15CK；WRGB→read 保留 WTR_L。依据与测试数值见 [时序决定](timing-source-and-decisions.md)。
2. **§10.4 不能仅归因于 buffer 转向。** 新任务还要求一个 rank RoPE 单元、删除 score tile cache、正确 MAC completion；合法全 bank ACT1/2 取代 ACTAB。Q 分片交换必须有地址，使用 WR+LOCALRD。不得把这些真实结构差异归一化消除，详见 [程序设计](program-design.md)。
3. **保留 layout 不代表算术量相等。** token O 在两个 KV parity rank 分别投影再 GPU 相加，标量 MAC 数是 feature O 的两倍。特征/偶奇映射、V tail RMW、完整 W_O 驻留和各层地址分离均列入契约；模型不执行 BF16/FP32 数值精度验证。
4. **§6.4/§8.3 serial 定义。** 为精确比较两个独立 solo，case b 在同一模拟器内重置 AE→VLM 边界的 DRAM 行状态/刷新相位，保留全局时间与统计。真实保留内存状态的 serial handoff 不在本基准定义内。
5. **§11.1 文件组织。** policy 与 descriptor/private issue 集成在 `pim_controller.cpp`，未机械建立 `policies/` 和 `rank_controller.cpp`；lookahead 与资源检查独立为文件。运行语义由原生共同状态维持。
6. **§11.2 迁入范围。** 张量/descriptor/Thor 测量已迁入；原外部时序、trace 后处理、延迟修正与历史开关未迁入。新的 `analysis/report.py` 只读原生时间戳；`analysis/plot.py` 延续旧 `plot_comparison.py` 的横向分解展示，改为只读新版 CSV，保留负干扰且不重建时间线。旧 trace 插入位置图没有合法的新输入，未把它迁成重放器。旧 CSV 的前21列名称/顺序已保留并验证，新增 canonical 列在其后；下表列出不应按旧重放语义解释的兼容字段。
7. **模型固定边界。** 不计 WRGB transport SM 或额外交接时间；AE GPU/VLM 不同时执行；无窗口适配；GPU 地址采用有限物理容量内合成顺序区域；uniform arrivals 与 alpha 写回为模型假设。完整 VLM 校准未通过前不可宣称预测精度达标。
8. **lookahead 是最早时间预测。** 最多32条，clone 原生状态与依赖，忽略未来 GPU/刷新及资源争用；真实 issue 仍核验 live native device。它不是额外时序引擎或对未来完成时间的保证。

| 兼容字段 | 新版准确语义 |
|---|---|
| `variant` | case 对应的五种实际 PIM 程序名 |
| `AE_native_schedule_ms` | `AE_solo_ms` 的别名；不存在另一套外部调度时间 |
| `net_overlap_ms` | `AE_solo_ms + VLM_solo_ms - total_ms`，保留符号 |
| `VLM_completed_model_ms` | AE 结束时已经完成的 VLM 单元，在 native solo 运行中的时长之和 |
| `completed_units` | 上述集合的单元数，不是全部2357个单元 |
| `VLM_remaining_model_ms` | `VLM_remaining_ms` 的别名；实际 native tail，包含可能尚未完成的单元尾部 |
| `raw_model_total_ms` | `total_ms` 的别名，未减去任何 bias |
| `GPU_act1/act2/prepb/preab/reads/writes` | rank0 实际发射命令计数 |
| `GPU_row_hits/row_misses` | 请求服务过程中是否需要 array prerequisite，命中与缺失之和等于完成 GPU RD/WR 请求数 |

`tests/test_config.py::ReportingTests` 覆盖负 PIM 干扰、AE 结束时的 completed 集合、实际 tail、兼容别名与原生命令计数。保留字段名不保留旧常数修正，也不把 completed 与 remaining 强行构造成 solo 总和。

旧仓库 tag、备份目录与三个完整归档 SHA-256 见 [legacy-backup.json](legacy-backup.json)。干净上游为 `7bead6cb7c4b03e095c6d4769ae3181a8e0bf6da`。最终实验 provenance 必须绑定实际运行的源提交、spec、程序、校准和二进制，不能用备份提交替代新版实验来源。
