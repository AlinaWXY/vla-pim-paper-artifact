# Version2：单层、单个 denoising step

> 历史结果：本页作业 `1977315` 使用 rank0/rank1 均为 REFpb 的旧刷新策略。
> 用户随后明确 GPU rank0 用 REFpb、PIM rank1 用 REFab。修正后的结果与阶段分析见
> [按 rank 刷新的单层报告](rank-refresh-single-layer.md)；本页及归档保留用于对照，不代表当前模型。

用户于 2026-11-09 要求先观察单层 latency，暂停校准和端到端仿真。本轮只运行 AE 的 step 0 / layer 0；不加载 VLM、不恢复旧校准作业、不运行 18 层 × 10 步或并发消融矩阵。Version2 文档中的完整实验要求不改变本轮范围。

任务文档 SHA-256：`a8273123c0f306776eb09a8cc98269071b11d5e379e52398c1ae18f8a4c1fbfa`。原件保存在服务器 `results/task-documents/version2-rewrite.docx`；分支保存实现说明及来源哈希。

## 统计边界

单层包含自然 kernel ledger 的 ordinal 30–84，以及对应的 PIM 阶段、GPU RoPE / helper、注意力和 FFN 残差与归一化。mask、RoPE 系数已在层外准备；全局 action 输入投影、下一层、最终 norm / 输出投影不计入。单层总时延 = 原生 PIM 阶段时延 + 实测 GPU 算子时延（向上取整到 CK）。GPU 部分采用已有实测，不在本轮重新测量或校准。

streaming 在 PV 与 O 之间恢复实测转置：feature 4.352 μs，token 4.992 μs；删除原始序列中对应的自然 contiguous 记账，避免重复计算。fused 无此转置。表中的 layout / RoPE / dataflow 是五种 AE 结构；这里没有 VLM 并发，因此不等同于完整 b–g 消融。

Layout 继续遵循用户要求保留 1108 的实际 feature/token 映射，包括 feature PV 的分片语义；没有把文档中“PV 按 token 分片”的概述用于悄然改变 feature 路径。详细映射见 [程序设计](program-design.md)。

## 本轮实现

- ACTAB 由原生 spec 展开为四批、每批四个 bank 的激活，批间 16 CK，共用滚动 FAW 历史；读/计算就绪分别为 20/36/52/68 CK。外部 CA 2 CK，私有 ACTAB 1 CK。
- 每个 bank 独立 RoPE，两输入槽、双输出 FIFO；八对元素向量完成 11 CK、启动间隔 8 CK，Q 执行可与 K 装载重叠。streaming 固定功能操作不占 CA，fused 私有命令占 1 CK。
- QKT 的归约结果存入现有每 bank 2 KiB 输出池，496 列批满后集中 LOCALWR；没有额外 rank score cache。streaming 仍逐列 RDRES。
- 描述符内部 FIFO32 允许已满足依赖的后续操作及安全 ACT/PRE 提前发射。完成依赖保护数据，issue 依赖保护流水顺序；barrier、I/O epoch 和未发请求所需行限制重排。
- Q 导入先完成整个 WR epoch，再 LOCALRD 装载 SRF。生成器为阵列、SRF、输出池、gather/CAST 和 RoPE 资源生成 RAW/WAR/WAW 依赖。
- RDRES / LOCALWR 直接读取精确的输出池区域，不锁住另一份虚构的 gather 数据缓冲。不同输出字的 gather 与传输可以重叠；相同输出字的冲突、GATHER→CAST 元数据和 V 尾部 RMW scratch 生命周期仍受保护。

## 结果与验证

最终模型源提交：`f7c7af6eab0655b526773b9b15ed63e8e6f73add`；Slurm 作业 `1977315` 成功退出。LPDDR5X-8533、16 通道、`mac_freq_mult=4`，单层原生行状态与周期刷新均保留。每种配置独立执行两次，两份 `raw.json` 逐字节一致。

| 配置 | PIM / μs | GPU / μs | 单层总计 / μs |
|---|---:|---:|---:|
| Feature / GPU RoPE / streaming | 322.153 | 138.697 | 460.851 |
| Token / GPU RoPE / streaming | 296.824 | 125.561 | 422.385 |
| Token / bank RoPE / streaming | 295.626 | 97.665 | 393.291 |
| Token / GPU RoPE / fused | 296.195 | 120.569 | 416.764 |
| Token / bank RoPE / fused | 294.825 | 92.673 | 387.498 |

GPU 列是已有实测 GPU 算子向上取整到 CK 后的总和；PIM 列来自原生事件时间戳。这不是完整 AE 或端到端时间，不能直接乘 180 充当完整模型结果。

PIM 各阶段如下，单位 μs；“—”表示该配置不存在该阶段：

| 阶段 | Feature/GPU/stream | Token/GPU/stream | Token/local/stream | Token/GPU/fused | Token/local/fused |
|---|---:|---:|---:|---:|---:|
| qkv | 64.809 | 64.081 | 64.081 | 64.027 | 64.027 |
| rope | — | — | 1.180 | — | 1.367 |
| rope_export | 1.698 | 1.786 | — | 1.840 | — |
| rope_import | 1.324 | 1.430 | — | 1.430 | — |
| q_exchange_export | 0.254 | 0.044 | 0.686 | 0.044 | 0.857 |
| q_exchange_import | 0.000 | 1.288 | 1.275 | 1.288 | 1.044 |
| qkt | 81.253 | 41.643 | 41.237 | 23.310 | 23.276 |
| pv | 22.449 | 19.657 | 19.333 | 21.532 | 20.821 |
| o | 21.599 | 37.958 | 37.748 | 37.030 | 37.739 |
| gate_up | 68.156 | 68.260 | 68.260 | 70.700 | 70.700 |
| down | 60.611 | 60.678 | 61.825 | 74.994 | 74.994 |

在 token/GPU 配置中，fused 的 QKT 从 41.643 μs 降至 23.310 μs，但 Down 从 60.678 μs 增至 74.994 μs；PIM 合计仅减少约 0.629 μs。因此本次总时间收益不能全部归于 QKT，也不表示所有 fused 阶段都更快。

验证通过：44 项 Python 测试、222 项原生时序检查、5 个总线微基准、lookahead 回归、8 个 FIFO32 调度回归、资源生命周期及 154,368 条生成请求重放、故障注入。快速 JSON 解析器与独立 Python 对照逐字段匹配 630,608 条 fused/local 操作，并拒绝 7 类非法输入。逐阶段/逐通道 DQ 字节和占用界限检查通过；全部 rank0/VLM 请求、VLM kernel、AE GPU 等待和 VLM 尾部均为零。

原生运行库来自 `292cfbc` 的干净构建；最终 `f7c7af6` 只修改生成器、其回归测试和说明文档。运行脚本核对两提交的原生代码/spec 完全相同，并逐字节比较二进制与运行库，再重新生成全部程序、重做单层计时。构建和差异证据均在验证日志中。

此前 `1977130` 的结果包含过度保守的 gather 依赖，只作为诊断留存；不纳入本报告或最终归档。历史 Version1 完整 AE 结果保持原样。

- [汇总 CSV](../reference/version2-single-layer/single-layer-latency.csv)、[逐阶段 CSV](../reference/version2-single-layer/stage-latencies.csv)、[GPU 块 CSV](../reference/version2-single-layer/gpu-blocks.csv)。
- [来源记录](../reference/version2-single-layer/provenance.json)、[独立审计](../reference/version2-single-layer/independent-audit.json)、[验证日志](../reference/version2-single-layer/validation/)。
- [十份原始输出及配置归档](../reference/version2-single-layer.tar.gz)，SHA-256 `f617fd45a6f53bbac3ffefa04f783e56e9d54dc20bb514da8900431a502ba83e`。
- 服务器冻结目录：`measurement-host:/artifact-source/pim_vla_sim2_20261109/results/version2-single-layer-1977315/`；完整程序、可执行文件、运行库及哈希清单保存在 `frozen/`。

复现入口：

```bash
python programs/gen_pim_program.py --output programs/generated --compact
python experiments/run_single_layer.py --binary build/pim_vla_sim \
  --output results/version2-single-layer --repeats 2
```

后续恢复并发实验前仍需验证 GPU 仲裁路径；本轮所有 rank0/VLM 请求计数应为零，不能据此证明并发性能或校准误差。

已知后续检查项：`next_gpu` 在外部 ACT1/ACT2 lane 被占用时，可能先选中不能发射的 ACT1 候选而没有回选其他 bank 的就绪请求。本轮 PIM 窗口中的同类问题已修复并回归；GPU 路径须在恢复校准/并发前处理。它不影响这里没有 rank0 请求的单层 AE 结果。
