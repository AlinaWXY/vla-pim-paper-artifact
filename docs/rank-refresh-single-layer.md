# GPU REFpb / PIM REFab：单层结果与 latency breakdown

> 本页是Down使用独立暂存区的冻结基线（作业1980912）。用户随后要求Down权重与输入同排；当前结果见 [Down同排单层报告](down-row-single-layer.md)。本页原始结果与分析保留用于控制变量对照。

按用户 2026-11-09 的明确要求，固定 **rank0（GPU）使用 REFpb、rank1（PIM）使用 REFab**。本次只重跑 AE 的 **step 0 / layer 0**，五种结构各重复两次；未执行 VLM 校准、完整 AE、VLM 并发或端到端仿真。

刷新修正后的结果改变了此前结论：token/GPU 的 fused PIM 比 streaming **慢 24.022 μs（12.60%）**，token/bank RoPE 也慢 **23.936 μs（12.63%）**。旧双 REFpb 结果中的“PIM 几乎不变”不能代表用户指定的硬件。主要差值来自 Down，超过 QKT 的收益。

## 运行与统计边界

- 模型源提交：`ea217bad2fce69c3c5773f690fde64d633041791`；Slurm 作业 `1980912`，`COMPLETED 0:0`。LPDDR5X-8533，16 通道，`mac_freq_mult=4`，CK=938 ps。
- rank0：REFpb 每 520 CK 到期，轮转 bank n/n+8 的八组配对，恢复 128 CK；rank1：REFab 每 4164 CK 到期，恢复 224 CK。实际发射服从原生 PRE/REF/ACT prerequisites 和 timing 约束，不叠加控制器延迟常数。旧 `refresh_allbank` 键被拒绝，包括 false/null。
- 单层包含自然 kernel ledger ordinal 30–84 对应的全部层内 GPU 算子及 PIM 替换阶段；mask/RoPE 系数已准备。不含 action 输入投影、其他层/步、最终 norm/输出投影。
- **总时间 = 原生 PIM 阶段和 + 已有实测 GPU 算子和（向上取整到 CK）**。GPU 值本轮未重新测量。设备状态连续，GPU 算术间隔内周期刷新继续推进。
- 五份程序、spec、GPU 实测数据与旧作业 `1977315` 保持相同字节；生成器/布局/输入的 Git 内容也未变。本次 clean build 更新控制器；程序复用的来源和 SHA 检查见 `validation/input-reuse.json`。归档中的 `program-generation.jsonl` 是这些程序在旧作业生成时的原始记录。

## 单层总计

单位均为 μs。表中 bank RoPE 对应程序名的 `local`。

| 配置 | PIM | GPU | 总计 |
|---|---:|---:|---:|
| Feature / GPU RoPE / streaming | 205.897 | 138.697 | 344.594 |
| Token / GPU RoPE / streaming | 190.664 | 125.561 | 316.225 |
| Token / bank RoPE / streaming | 189.495 | 97.665 | 287.160 |
| Token / GPU RoPE / fused | 214.687 | 120.569 | 335.255 |
| Token / bank RoPE / fused | 213.431 | 92.673 | 306.104 |

fused 去掉 token PV→O 的 4.992 μs GPU 转置，但不足以抵消 PIM 增量。因此 GPU RoPE 组合的总时间增加 19.030 μs（6.02%）；bank RoPE 组合增加 18.944 μs（6.60%）。这是单层 AE solo 对照，不能推导 VLM 并发性能或完整端到端加速比。

## PIM 阶段 breakdown

单位 μs；“—”表示该配置不存在该阶段，0.000 表示存在零时长阶段。

| 阶段 | Feature/GPU/stream | Token/GPU/stream | Token/bank/stream | Token/GPU/fused | Token/bank/fused |
|---|---:|---:|---:|---:|---:|
| QKV | 44.657 | 44.641 | 44.641 | 45.232 | 45.232 |
| Bank RoPE | — | — | 0.744 | — | 0.920 |
| RoPE export | 1.158 | 1.176 | — | 1.176 | — |
| RoPE import | 0.966 | 0.965 | — | 1.358 | — |
| Q exchange export | 0.074 | 0.044 | 0.508 | 0.044 | 0.500 |
| Q exchange import | 0.000 | 1.007 | 0.874 | 0.874 | 0.874 |
| QKT | 50.003 | 25.308 | 25.555 | 16.110 | 16.302 |
| PV | 13.412 | 11.887 | 11.834 | 14.547 | 14.365 |
| O | 14.368 | 24.493 | 24.325 | 24.766 | 25.086 |
| Gate/Up | 44.838 | 44.525 | 44.646 | 48.660 | 48.375 |
| Down | 36.420 | 36.617 | 36.367 | 61.918 | 61.778 |
| **PIM 合计** | **205.897** | **190.664** | **189.495** | **214.687** | **213.431** |

## GPU 阶段 breakdown

以下为已有实测值向上取整到 CK 后的阶段和，单位 μs；包含转换、残差及 norm，不能只计 RoPE/softmax/激活而遗漏这些工作。逐 GPU 块类别、source_ordinal 与边界保存在 [gpu-blocks.csv](../reference/rank-refresh-single-layer/gpu-blocks.csv)，完整 kernel 名称可通过 ordinal 在原始 ledger 中查询。

| GPU 工作 | Feature/GPU/stream | Token/GPU/stream | Token/bank/stream | Token/GPU/fused | Token/bank/fused |
|---|---:|---:|---:|---:|---:|
| QKV 前：copy/cast、norm | 22.763 | 22.763 | 22.763 | 22.763 | 22.763 |
| GPU RoPE | 27.895 | 27.895 | 0 | 27.895 | 0 |
| QKT 后：score helper、scale/mask/softmax/cast | 28.965 | 15.189 | 15.189 | 15.189 | 15.189 |
| PV→O 转置 | 4.352 | 4.992 | 4.992 | 0 | 0 |
| O 后：rank sum/cast、残差、FFN 前 norm | 36.302 | 36.302 | 36.302 | 36.302 | 36.302 |
| Gate/Up 后：cast、GELU、门控乘法 | 11.876 | 11.876 | 11.876 | 11.876 | 11.876 |
| Down 后：cast、残差 | 6.543 | 6.543 | 6.543 | 6.543 | 6.543 |
| **GPU 合计** | **138.697** | **125.561** | **97.665** | **120.569** | **92.673** |

## 为什么 fused 的 PIM 更慢

先看直接观测到的差值。Token/GPU 的 fused−streaming：

| PIM 阶段 | 差值 / μs |
|---|---:|
| QKT | −9.198 |
| Down | +25.302 |
| Gate/Up | +4.135 |
| PV | +2.660 |
| O | +0.273 |
| QKV、RoPE/Q 传输等其余阶段 | +0.851 |
| **合计** | **+24.022** |

**QKT 的收益真实存在，但被 FFN 的代价超过。** QKT fused 先把归约结果放入既有输出池，再批量 LOCALWR/导出，改善逐列处理；然而 fused FFN 的输入仍需从 GPU 输入，外部 WRGB 被同量 staging WR 替换，再增加 LOCALRD 和 BCAST。减少命令占用不等于减少阵列工作。

生成器静态展开给出以下结构；数字均为每 channel、每阶段，streaming/fused 的外部字节相同：

| 结构 | Gate/Up | Down |
|---|---:|---:|
| 外部输入 / 输出 | 32 / 32 KiB | 128 / 4 KiB |
| 两片输入组成的 SRF 批数 | 32 | 128 |
| MAC 总数 / 每批 MAC | 2048 / 64 | 1024 / 8 |
| Streaming WRGB | 1024 | 4096 |
| Fused WR / LOCALRD / BCAST | 各 1024 | 各 4096 |
| 权重行数 / 每权重行覆盖批数 | 32 / 1 | 16 / 8 |

Down 每批仅 8 个 MAC，局部搬运更难摊薄：新增 LOCALRD+BCAST 的操作数与 MAC 数之比为 8，Gate/Up 为 1。并且 staging 使用 bank 0/4 的独立行（Gate/Up 256–263，Down 256–287），与全 bank 权重行（分别 128–159、160–175）冲突。Streaming WRGB 写 SRF，不改变打开的权重行；fused 读取 staging 后，全 bank MAC 必须恢复统一权重行。Down 原本每个权重行可服务八批，跨批复用尤其容易受损。

这解释了变慢的具体机制，但当前日志没有独立的逐命令 ACT/PRE 和互斥 stall 分解，**不能把 Down 的全部 25.302 μs 精确归为某一种等待**。刷新相位、行切换、私有 CA、依赖和资源容量都会影响阶段时间。

Token/GPU 的阶段内平均外部 CA 占用从 9.843% 降到 4.852%，DQ 从 10.532% 降到 8.682%（分母为 16×PIM 阶段周期和）。因此 fused 确实减少外部接口压力；低接口占用同时也可能来自内部等待，不能据此宣称计算单元已饱和。本轮没有 VLM 并发，不能验证省出的 CA 是否改善 VLM。

## 为什么刷新修正改变了之前的判断

对比保留的旧双 REFpb 作业 `1977315`，只改变 rank1 刷新策略后：

| Token/GPU | 旧 PIM / μs | 修正后 PIM / μs | 减少 / μs |
|---|---:|---:|---:|
| Streaming | 296.824 | 190.664 | 106.160 |
| Fused | 296.195 | 214.687 | 81.508 |

旧 REFpb 每 520 CK 到期，需要关闭一对 bank。all-bank MAC 依赖全部 bank，所以“只刷新两个 bank”也会阻塞它，并破坏统一的权重行状态。改为每 4164 CK 到期的 REFab 后，恢复虽由 128 CK 增至 224 CK，但刷新频率约降为八分之一。两条路径都明显变快，streaming 的改善更大；旧刷新设置因此掩盖了 fused 的 staging/行复用代价。上表的改善量是实际控制变量对照，内部等待的精确归因仍需额外计数。

## 与 VLM 的比较边界

本轮没有运行 VLM。已有 GPU 实测来自 [warm-summary.json](../input-data/measured/warm-summary.json) 的 `baseline.phase_ms`，各 100 次测量：

| 已有实测项 | 均值 |
|---|---:|
| ViT（3 路 224×224 输入） | 26.290488 ms |
| VLM 语言部分（18 层） | 78.660008 ms |
| 两者合计 | 104.950496 ms |

语言部分除以 18 得到 **4.370000 ms/层**，只是按层数平均，不是某一层的单独实测。VLM 的 D=2048、FFN=16384，AE 的 D=1024、FFN=4096，输入长度和计算形态也不同；不能把这个均值与本页 AE 单层 0.287–0.345 ms 的差当作同工作负载加速比。

最近的纯 GPU AE 对照是 [natural-ae-kernel-ledger.json](../input-data/measured/natural-ae-kernel-ledger.json) 中 ordinal 30–84 的 `step_mean_us[0]`：55 个 kernel duration 之和 **573.6576 μs**，其中 ordinal 54/55 的 KV concat 合计 **224.7488 μs**。它是 profiler 的 kernel 时长和，含 kernel 重叠，不含 CPU launch gap、独立 DMA 和 PIM 交接成本，因此也不是 wall latency，不据此宣称单层加速比。该数据源的 `mean_us` 跨步均值不能替代 step 0 数字。

## 验证与归档

46 项 Python 测试、224 项 native spec 检查、5 个总线微基准、lookahead、8 个 FIFO 调度回归、5 组 rank 刷新回归、资源检查及故障注入通过。刷新回归覆盖配对轮换、两种周期、全 bank 恢复、恢复后的 ACTAB prerequisite、跨 rank 独立和 serial epoch 重置；原生入口还拒绝 true/false/null/0 四种旧开关配置。

独立审计直接读取十份 raw/config/provenance、实际冻结二进制/库/五程序及对应 Git 源文件，未导入 runner 逻辑。确认两次 raw 字节一致、只含 step0/layer0、PIM 字节逐阶段/逐通道守恒、DQ 占用守恒、PIM+GPU=总计、CSV 与 raw 一致、SHA 绑定一致。rank0 数据读写及 VLM kernel 均为零，但 rank0 的维护性 REFpb 正常执行。

| 配置 | rank0 REFpb | rank1 REFab |
|---|---:|---:|
| feature_gpu_before | 11296 | 1408 |
| token_gpu_before | 10368 | 1280 |
| token_local_before | 9408 | 1168 |
| token_gpu_after | 10992 | 1360 |
| token_local_after | 10032 | 1248 |

上述是全部 16 通道的合计；所有配置 **rank0 REFab=0、rank1 REFpb=0**。

- [汇总](../reference/rank-refresh-single-layer/single-layer-latency.csv)、[PIM 阶段](../reference/rank-refresh-single-layer/stage-latencies.csv)、[GPU 块](../reference/rank-refresh-single-layer/gpu-blocks.csv)。
- [来源](../reference/rank-refresh-single-layer/provenance.json)、[独立审计](../reference/rank-refresh-single-layer/independent-audit.json)、[验证日志及审计脚本](../reference/rank-refresh-single-layer/validation/)。
- [原始结果归档](../reference/rank-refresh-single-layer.tar.gz)，SHA-256 `2e36f97cb9d70b976ddbbe0c3ceb5f338780981520df5b073db6ebbb529cad77`。
- 服务器冻结目录：`measurement-host:/artifact-source/pim_vla_sim2_20261109/results/rank-refresh-single-layer-1980912/`。完整程序、二进制与库保存在 `frozen/`；旧结果不覆盖。

校准和 e2e 继续暂停。恢复并发前仍须处理 [旧报告记录的 GPU 候选选择检查项](version2-single-layer.md)；本次无 rank0 数据请求，不能证明该路径的并发性能。
