# Down 权重与输入同排：单层验证

按用户要求，将 Down 改为每 bank、每行 12 个 fragment，每个 fragment 为 **4 列权重＋1 列输入**。最终单层作业 `1981031` 完成，模型源提交 `5fcd84be57f7411ec967107511aac65693070481`。Token/GPU fused 的 Down 从 **61.918 μs 降到 45.000 μs（−27.32%）**；bank RoPE fused 从 **61.778 μs 降到 45.471 μs（−26.39%）**。

范围仍为 AE **step 0 / layer 0**，五配置各重复两次。固定 **rank0 REFpb、rank1 REFab**，16 通道、LPDDR5X-8533、`mac_freq_mult=4`。校准、完整 AE、VLM 并发和 e2e 未运行。

## 实现与容量

对 fragment `f`、query `q`、输出列 `c`：

```text
row          = 160 + f // 12
weight_col   = 5 * (f % 12) + c       # c = 0..3，全部 bank
input_col    = 5 * (f % 12) + 4       # 输入放在 bank q，q = 0..15
SRF_slot     = 16 * (f % 2) + q
```

256 个 fragment 占 22 行，保留区域 `[0x50000,0x5B000)`。每 bank 权重有效载荷仍为 32 KiB，加 8 KiB 输入及 4 KiB 总空余，占用 **44 KiB**；相对原权重区 32 KiB 增加 37.5%。前 21 行每行空 4 列，末行只放 4 个 fragment，使用列 0–19。它与相邻区域及每层 1 MiB 地址区均不重叠。

Streaming 和 fused 使用相同的 22 行权重映射。Streaming 仍用 WRGB 直接装 SRF，不写阵列输入列；fused 先用普通 WR 散写输入，再逐 bank LOCALRD＋BCAST。

每批仍是两个 fragment，即 **32 条 LOCALRD＋32 条 BCAST、8 条 MAC**，不是总共32条本地操作。每个新行组有一条显式 ACTAB，请求完成依赖前一行全部48条MAC，避免有限乱序调度提前关闭旧行。首批 LOCALRD 对 ACTAB 使用 issue dependency；各 bank 仍由 native timing 决定 20/36/52/68 CK 的就绪时刻，没有人为添加统一68 CK屏障。PREab、ACTAB 展开和刷新恢复继续走同一原生设备。

程序每 channel 显式包含22条ACTAB；行内 REFab 仍可能引入额外动态开行前置命令，因此不能把22当作全部实际ACT次数。Down descriptor 的地址字段同步更新，仍为8个word；不改变原生CA或decode时序。详见 [程序设计](program-design.md)。

## Down 阶段前后对照

基线为已正确区分 rank 刷新的作业 `1980912`，不是更早的双REFpb结果。单位 μs：

| 配置 | 原独立暂存布局 | 同排布局最终值 | 变化 |
|---|---:|---:|---:|
| Feature / GPU RoPE / streaming | 36.420 | 36.839 | +0.419 |
| Token / GPU RoPE / streaming | 36.617 | 37.149 | +0.532 |
| Token / bank RoPE / streaming | 36.367 | 36.838 | +0.471 |
| Token / GPU RoPE / fused | 61.918 | 45.000 | **−16.919** |
| Token / bank RoPE / fused | 61.778 | 45.471 | **−16.306** |

Streaming 权重行数增加，Down 多出约0.42–0.53 μs。Fused 则显著降低暂存行与权重行反复切换的代价。独立对照确认：**所有非Down阶段的起止时间戳完全相同**，GPU算子时间和逐阶段传输字节不变。因此整层时间变化恰好等于Down阶段变化，没有把其他优化计入本次收益。

实现同时调整了 fused 输入预写顺序。旧的“两fragment交替”针对 bank0/4 暂存区设计；直接沿用到 `bank=query` 后，会连续两次写同一bank。最终版本改为逐fragment、query顺序 `0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15`，相邻WR跨BG；地址、数据标签、epoch和字节集合不变。

保留的首轮同排诊断作业 `1980945` 使用旧WR顺序，两个 fused Down 为51.392/51.208 μs。只调整WR顺序后，分别再减少 **6.392/5.737 μs**。对照脚本核对全部16通道：除Down的WR排列及其操作ID外，请求内容一致，descriptor children（包括LOCALRD顺序）完全相同。该差值包含请求顺序引起的实际刷新相位变化，不能解释为某个固定延迟常数。

## 最终单层 breakdown

单位 μs；GPU列仍使用已有实测算子时长向上取整到CK，本轮没有重新测量GPU。

| 配置 | PIM | GPU | 单层总计 |
|---|---:|---:|---:|
| Feature / GPU RoPE / streaming | 206.316 | 138.697 | 345.013 |
| Token / GPU RoPE / streaming | 191.196 | 125.561 | 316.757 |
| Token / bank RoPE / streaming | 189.966 | 97.665 | 287.631 |
| Token / GPU RoPE / fused | 197.768 | 120.569 | 318.337 |
| Token / bank RoPE / fused | 197.124 | 92.673 | 289.798 |

PIM各阶段如下；GPU各块不变，完整明细见 [gpu-blocks.csv](../reference/down-row-single-layer/gpu-blocks.csv)。

| PIM 阶段 | Feature/GPU/stream | Token/GPU/stream | Token/bank/stream | Token/GPU/fused | Token/bank/fused |
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
| Down | 36.839 | 37.149 | 36.838 | 45.000 | 45.471 |

## 为什么 fused 尚未全面超过 streaming

Token/GPU 的 fused−streaming 分解：

| 项目 | 差值 / μs |
|---|---:|
| QKT | −9.198 |
| Down | +7.851 |
| Gate/Up | +4.135 |
| PV | +2.660 |
| O | +0.273 |
| 其余PIM阶段 | +0.851 |
| **PIM合计** | **+6.572** |
| GPU（省去PV→O转置） | −4.992 |
| **整层合计** | **+1.580** |

因此同样22行权重布局下，GPU RoPE fused 的PIM仍慢约3.44%，整层慢约 **0.50%**；bank RoPE fused 的PIM慢约3.77%，整层慢约 **0.75%**。Down的布局问题明显缓解，但Gate/Up、PV等路径没有在本轮修改。

同排放置没有消除fused的输入中转：每channel仍有 **4096 WR（128KiB）→4096 LOCALRD→4096 BCAST**，加1024个MAC和128 RDRES（4KiB）。Streaming直接WRGB到SRF。每批“约64 CK装载＋128 CK MAC”只描述理想局部服务量，不含全阶段的输入WR、输出RDRES、流水首尾、真正换行和周期REFab。现有日志未提供互斥stall细分，不把剩余Down差值全部归给某一种等待。

VLM参考测量与比较边界沿用 [上一轮报告](rank-refresh-single-layer.md#与-vlm-的比较边界)；没有新的VLM测量或并发收益结论。

## 验证与来源

52项Python测试、224项原生时序检查、8个FIFO调度回归、5组rank刷新回归通过。地址与依赖测试覆盖权重/输入不相交、全部fragment/query/bank、行尾空列、末行、18层隔离、SRF槽与producer、两fragment批不跨行、显式ACTAB的完成/发射依赖、跨BG预写顺序，以及所有非Down程序的语义不变。原生解析器与独立Python逐字段匹配 **630,960** 个请求，并拒绝7类非法输入。

本次原生二进制/库逐字节复用 `ea217ba` 的干净构建，native源码、spec、runner和GPU输入均核对不变。新五程序绑定最终source commit。独立审计检查实际冻结文件SHA、十份raw/config/provenance、两次逐字节一致、逐阶段/通道DQ与占用守恒，以及PIM＋GPU＝总时间；全部通过。所有配置rank0 REFab和rank1 REFpb计数为零，正确刷新类型计数均非零。

- [汇总](../reference/down-row-single-layer/single-layer-latency.csv)、[PIM阶段](../reference/down-row-single-layer/stage-latencies.csv)、[GPU块](../reference/down-row-single-layer/gpu-blocks.csv)。
- [独立审计](../reference/down-row-single-layer/independent-audit.json)、[来源](../reference/down-row-single-layer/provenance.json)、[布局及WR顺序对照](../reference/down-row-single-layer/validation/down-layout-comparison.json)、[验证记录](../reference/down-row-single-layer/validation/)。
- [原始结果归档](../reference/down-row-single-layer.tar.gz)，SHA-256 `e525106d94321d01cb8ff12e3cba6f3d90450d54ed7da83ba6543e1b883abf61`。
- 服务器最终冻结目录：`measurement-host:/artifact-source/pim_vla_sim2_20261109/results/down-row-single-layer-1981031/`；首轮同排诊断 `1980945` 和原暂存布局基线 `1980912` 均保留。

后续文档/结果提交不改变上述冻结模型、二进制或输入；校准和e2e继续暂停。
