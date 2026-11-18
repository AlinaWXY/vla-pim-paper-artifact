# 新旧 AE 差异：单层快照、完整 AE 与归因边界

> 本页记录 Version1 历史比较，其中“新版”仍使用双 rank REFpb。当前模型已按用户要求
> 改为 GPU rank0 REFpb、PIM rank1 REFab；最新单层结果见 [刷新修正报告](rank-refresh-single-layer.md)。

本页比较旧发布目录 `publish-anonymous_1108` 的冻结结果、新版单层验证作业 **1976509**，以及已通过验收的完整 AE solo 作业 **1976674**。单层表仍为 step 0、layer 0，包含该层实测 GPU 算子块；完整 AE 表来自实际执行的 **18 层 × 10 步**，没有用单层时间乘 180 代替运行。所有差值仅用于核查与解释，不用于修正新版时长。用户已要求暂停校准及后续实验；校准作业 1976675、收集作业 1976676 已取消，本页不报告尚未完成的校准或并发消融结果。

## 统计口径

- 旧原生基线取 `reference/evaluation-1108/direct-before-corrected/native-provenance/<variant>/`（最终原生作业 1971385），不取更早已撤回的 before 数据。旧 native library SHA256 为 `5cdc451ff5dbc3f69a6fd030c273653c2fa9d5c9f0b3328b83e76bed01ec0ba6`。
- 旧“四段和”相加 `chain`、`pv`、`gate_up_gpu`、`down_gpu_input` 四个独立 native case；旧 `pv` 已包含 O。旧 `serial_layer_control` 是另一次连续内存/PIM 运行，数值单列。二者都不含实测 GPU 算术。
- 新版“PIM 阶段和”直接相加 `raw.json:stages[].duration_us`；“GPU 块”是 `metrics.total_us` 减该阶段和。新版保持设备状态连续，GPU 块期间刷新仍在推进，因此与旧独立 case 相加有不同的初始状态与刷新相位。
- 外部 DQ 统一取全部 16 channels、rank1、一层一步的实际 32 B burst 字节和，包含输入、输出、暂存、KV、score 等既定传输；排除 GPU rank0 流量、内部广播 fanout 与一次性模型初始化。一条 WRGB 仍只计 32 B。
- b/c 共享同一 `feature_gpu_before` AE 程序；本页 b 是该程序的 AE solo 检查，并不代表新完整串行 b 的端到端延迟。

## 已核对的单层时间与流量

单位：时间 µs；DQ KiB，1 KiB = 1024 B。五种配置不仅 DQ 总量一致，按下文四个算子分组的 DQ 也逐组完全相等。

| case / variant | 旧四段 PIM 和 | 旧连续 PIM 层 | 新 PIM 阶段和 | 新 GPU 块 | 新单层 AE | 旧 = 新 DQ KiB |
|---|---:|---:|---:|---:|---:|---:|
| b / `feature_gpu_before` | 155.260574 | 155.626394 | 303.253524 | 134.345050 | 437.598574 | 6096 |
| d / `token_gpu_before` | 151.214980 | 151.156824 | 293.129690 | 120.568644 | 413.698334 | 5352 |
| e / `token_local_before` | 155.090796 | 155.062656 | 301.514472 | 92.673462 | 394.187934 | 5160 |
| f / `token_gpu_after` | 206.440668 | 206.919986 | 406.598612 | 120.568644 | 527.167256 | 4968 |
| g / `token_local_after` | 210.076356 | 210.341810 | 414.983394 | 92.673462 | 507.656856 | 4776 |

新版 PIM 阶段和相对旧四段和的增加分别为：

- b：+147.992950 µs（+95.319%）。
- d：+141.914710 µs（+93.850%）。
- e：+146.423676 µs（+94.412%）。
- f：+200.157944 µs（+96.957%）。
- g：+204.907038 µs（+97.539%）。

这项核对未发现外部传输总量差异；总量相等本身不能排除相互抵消的缺失/重复，也不证明地址、payload、调度、资源数、等待关系或时序约束相同。地址和生产者对应仍须依靠程序覆盖验证。尤其不能把新单层 AE 的 437.599 µs 直接除以旧纯 PIM 的 155.261 µs，作为“时序修复倍率”。

## 差异集中在哪些算子

分组映射：`chain` = QKV、RoPE/其外部收发、Q 交换、QKᵀ；`pv` = PV + O；其余分别为 Gate/Up、Down。括号中是新减旧。仍须保留上述设备连续性差异。

| case | chain 旧 → 新（Δ） | PV+O 旧 → 新（Δ） | Gate/Up 旧 → 新（Δ） | Down 旧 → 新（Δ） |
|---|---:|---:|---:|---:|
| b | 55.635 → 137.990 (+82.355) | 28.874 → 42.307 (+13.432) | 40.112 → 68.017 (+27.906) | 30.640 → 54.940 (+24.300) |
| d | 43.085 → 114.940 (+71.855) | 37.378 → 55.510 (+18.132) | 40.112 → 67.770 (+27.659) | 30.640 → 54.910 (+24.270) |
| e | 46.961 → 122.527 (+75.566) | 37.378 → 55.453 (+18.074) | 40.112 → 68.625 (+28.513) | 30.640 → 54.910 (+24.270) |
| f | 56.257 → 178.902 (+122.645) | 44.398 → 60.915 (+16.516) | 47.139 → 76.325 (+29.186) | 58.647 → 90.457 (+31.810) |
| g | 59.892 → 187.287 (+127.394) | 44.398 → 60.915 (+16.516) | 47.139 → 76.325 (+29.186) | 58.647 → 90.457 (+31.810) |

## 已有证据支持的机制差异

这些是代码与记录可直接证实的差异；下表没有把总时间增量武断分摊给各机制。多个约束会同时限制同一条命令，不能把它们的等待分别求和。

| 机制 | 旧版证据 | 新版行为及归因限制 |
|---|---|---|
| WRGB/RDRES 接口与完成 | 旧 `lpddr5pim.py:resolve()` 在 AttAcc 路径将 nPIMWrLatency/nPIMRdLatency 置零；输入在 issue+2 CK ready，结果通常按 2 CK packet 服务；WRGB 连续段仅 2 CK | 新 spec 写完成为 CWL+BL=14 CK，读完成为 CL+BL=25 CK；DQ 分别从 issue+12/23 CK 开始。广播命中所有 BG，连续 WRGB/RDRES 为 4 CK，并继承读写切换。必须按依赖和重叠评估，不能将“每条多 12/23 CK”乘条数当总增量。 |
| 全 bank 激活 | 旧 `commands/ACTAB.h` 一条 2 CK 命令直接将 16 bank 的行设为 Opened。旧连续层每 channel ACTAB 次数 b/d/e/f/g 为 132/160/160/298/297 | 新版没有 ACTAB；由同一 native device 给每个 bank 展开 ACT1/ACT2，并受 RRD/FAW/行恢复约束。理想全闭合行至少涉及 32 条单 bank 激活命令，而不是旧的 2 CK 广播。这是行控制架构修正，不能归入 WRGB/RDRES。 |
| QKᵀ 输出驻留与直接落盘 | 旧 `score_spill_extension.inc` 的 SCORE_TILES=31、SCORE_PACKETS=32；每 16 个 K 列形成 1024 B score tile 后批量 spill。streaming trace 也按 qk_tile_compute/export 成组 | 任务要求归约结果不在 rank 内缓存。新程序每 K 列归约后即两次 RDRES 或 LOCALWR；fused 的 score bank0/4 与全 bank 权重读取因此反复切换行。外部 score 字节不变，等待与 ACT/PRE 次数可显著变化。 |
| RoPE 执行资源 | 旧 `rank_rope_extension.inc` 在 `bank.next_pair_issue` 上检查/推进下一批，属于 bank 级独立执行资源 | 新任务固定每 rank 一个 RoPE 单元；新程序/资源记账串行共享该单元。旧 e/g 纯 RoPE 段 0.878906/0.781354 µs，新为 10.123834/10.268286 µs；这些段没有外部 DQ，不能用 WRGB/RDRES 延迟解释。此差值同时包含本地访问与程序调度，不能全部称为 PE 数量的纯因果效应。 |
| 本地端口与控制通路 | 旧结果声明本地读完成为 CL+BL=25 CK、本地写为 nLOCALWR+nWR=41 CK，部分 multicast/gather/cast/reduce/RoPE 控制无 CA | 新版本地 32 B 读写完成为 4 CK，array recovery 仍在 Bank spec；所有内部指令占 1 CK private CA，各执行单元各自按 spec 服务。它既有缩短也有新增约束，不能预设全部差异只会增加时间。固定假设见 timing 文档。 |
| 刷新与阶段状态 | 旧连续层记录使用 REFab，REFpb=0；四个 native case 各自初始化 | 当前新单层默认 paired REFpb，实际 GPU 算术间隔也推进同一设备状态；这些相位和粒度变化会影响行命中与尾部。要精确分离，需要相同初始化/刷新策略的另行诊断；不能从总差值推定。 |
| PIM 资源与计算语义 | 旧 completion/port/compute 条件分散在多个 controller extension | 新 spec 给出 M×4/k 和 max(tCCD_L,M×4/k)，SRF/output pool 记账只跟踪容量与生命周期。k=4 下不少 MAC 服务与旧 M CK 相同；不应把全部 MAC 阶段增长称为计算频率变化。 |

普通 JEDEC 数字不是本次主要变更来源：旧冻结 result 的 CL/CWL=23/12、RCD=20/9、RP=20/23、RAS/RC=45/65、RRD/FAW=4/16、RFCab/RFCpb=224/128 等与新版相同。主要问题在这些约束作用于哪些命令、命令如何展开，以及完成/依赖的含义。

### PV→O 不额外收取旧 GPU transpose

旧 `corrected-overlap/REPORT.zh-CN.md` 记录 feature/token 各 4.352/4.992 µs 的 `transpose(...).contiguous()` 实现。新版遵循零交接延迟、搬运 SM 开销不计的固定范围，用请求顺序表达这种 packet 重排：

- `export_results()` 以 `[query][bank][feature]` 导出连续 32 B BF16 包；token 每 query 16 包，feature 每 query 8 包。feature 的一包由相邻两个 bank 各 8 元素构成。
- 对 O 的 fragment f、query q，所需完整包就是 GPU 导出序号 `q*F+f`，F=16 或 8。O 的两个输出 tile 可按 fragment-major 访问次序各读取同一批包一次，无须改变包内 BF16 lane，也没有额外算术。fused 的 `staging(f,q,16)` 表达同一逻辑对应。
- 因此新版不收费这次额外复制实现；已有实测 rank sum/cast 等算术 helper 仍保留。这是公开的计费差异，不是删除一项必须执行的数值计算。当前模拟器验证 packet/地址/生命周期，未执行 tensor 数值，不能据此宣称完成了 BF16 数值精度验证。

### 可以精确核对的 GPU 输入账本差异

这里比较请求的 GPU 算子时长；已完成的完整 AE 运行时间另见下表。旧 GPU 账本由 `AE_native_schedule_ms` 减去旧四个独立 PIM case 的时长和 ×180 重建；该乘法只复核旧版已经采用的拼接算法。新版 GPU 账本直接汇总 `input-data/ae-<variant>.json` 中所有 GPU event 的 `us`。实际模拟会对每个 event 向上取整到 CK，完整 AE 表中的 GPU 时间可能有相应的小量量化增量。

| case | 旧 GPU 输入账本 / ms | 新 GPU 输入账本 / ms | 旧减新 / ms | 已核对来源 |
|---|---:|---:|---:|---|
| b/c | 27.0646306 | 26.2812706 | 0.7833600 | PV→O 复制 4.352 µs ×180 |
| d | 24.7001506 | 23.8015906 | 0.8985600 | PV→O 复制 4.992 µs ×180 |
| e | 19.1716258 | 18.1604770 | 1.0111488 | 上项 0.8985600 ms + 离线 RoPE 系数生成 0.1125888 ms |
| f | 23.8015906 | 23.8015906 | 0 | GPU 输入账本相同 |
| g | 18.2730658 | 18.1604770 | 0.1125888 | 离线 RoPE 系数生成 |

按冻结输入逐 event 向上取整到 938 ps，GPU 块总和为 b/c 26.285405160 ms（8710 个 event）、d/f 23.805652080 ms（8530 个）、e/g 18.163703082 ms（6670 个）。相对原始输入账本仅增加 4.134560、4.061480、3.226082 µs；这些是整数时钟表示产生的量化，不是校准补偿。

RoPE 系数生成量直接来自同一 `natural-ae-kernel-ledger.json`：每步 6 个 `rope_generation` kernel，10 步累计 112.5888 µs；新版本地 RoPE 模式显式删除它们，因为系数离线生成。旧 `AE_solo_ms` 比 `AE_native_schedule_ms` 另外多出的约 1–2.5 µs 来自旧重放的孤立 PIM 完成与 native 阶段包络差异，不应误归为 GPU 算术。该表是独立账本核对，未从新版总时间扣减任何数字。

## 已完成的完整 AE solo：18 层 × 10 步

新版使用冻结源码 `1e0b73bc8da695268fbf79f1013653b54120316e`、`results/full-k4-final`，五种 AE solo 对应作业数组 **1976674**，均已完成并通过[独立验收](../reference/validation/full-k4-final-solos.json)。验收包括每阶段 180 次、单独读写字节守恒、GPU rank0 零访存、总时长等于 PIM 阶段和加逐 event 取整后的 GPU 块，以及冻结 binary/library/input/program 的哈希一致性。

旧值取 `native-timing-fix/with-rows/evaluation-b-g.csv` 的 AE_solo_ms，是旧 native 阶段与 GPU 账本拼接的诊断值；新版为单实例连续运行。两者都不是 Thor 硬件端到端测量，且前文列出的架构与计费差异仍存在。以下是总量对比，不是某一时序修复的纯因果效应。所有时间单位为 ms。

| variant / case | 旧完整 AE | 新 PIM 阶段和 | 新 GPU 块 | 新完整 AE | 新减旧 | 增幅 |
|---|---:|---:|---:|---:|---:|---:|
| `feature_gpu_before` / b | 55.012884640 | 54.540327982 | 26.285405160 | 80.825733142 | +25.812848502 | +46.921% |
| `token_gpu_before` / d | 51.920704240 | 52.842904422 | 23.805652080 | 76.648556502 | +24.727852262 | +47.626% |
| `token_local_before` / e | 47.089319800 | 54.217928940 | 18.163703082 | 72.381632022 | +25.292312222 | +53.711% |
| `token_gpu_after` / f | 60.963432887 | 73.323279904 | 23.805652080 | 97.128931984 | +36.165499096 | +59.323% |
| `token_local_after` / g | 56.088830684 | 74.714909836 | 18.163703082 | 92.878612918 | +36.789782234 | +65.592% |

完整 AE 的外部 DQ 与旧固定工作量仍相同，全部 16 channels 合计如下。MiB = 2^20 B。读写数来自完整运行，没有按单层外推；与单层定义 ×180 的关系仅作为覆盖校验。

| case | 读字节 | 写字节 | 外部 DQ / MiB | 完成 job | wall / s |
|---|---:|---:|---:|---|---:|
| b | 418775040 | 704839680 | 1071.562500 | `1976674_0` | 535 |
| d | 339148800 | 647331840 | 940.781250 | `1976674_1` | 558 |
| e | 327352320 | 623738880 | 907.031250 | `1976674_2` | 571 |
| f | 315555840 | 600145920 | 873.281250 | `1976674_3` | 700 |
| g | 303759360 | 576552960 | 839.531250 | `1976674_4` | 700 |

完整运行 raw 位于 `results/full-k4-final/solos/ae-solo/<variant>/raw.json`，验收报告保存全部程序、输入及 provenance 哈希。对应 raw SHA256：

| case | 完整 AE raw.json SHA256 |
|---|---|
| b | `0fca03e6b9077a124a5a69d7a011684f75a09989a17c16c31dedc2008dfd4dac` |
| d | `f4b4af4b198ec24f1bff0515328be395e262a023a12ef14a0755718e59022dcf` |
| e | `e60523cf0b0ad86528f174c297cfdb8f828b8d57b0e9a6a23c7cd5ac95004f8e` |
| f | `4e7f99d708c116082398beee8d7ab565d45e42e8e6d48f2296173d881990e84e` |
| g | `5d9cc1ec20df3558832715a438c2feca3f767bbfa66e8c981fde0ce54451d5c3` |

五项使用相同 binary SHA256 `60c854f7f4bfbb75ad4308de8cd3acd657bb2ae4a257dde600c740df48d6d7d2`，library SHA256 `ca17a92971c415df22bd40276d3a8f8b75b9f4c464441ea470aad5b53c683447`。旧并发总表曾用模型偏差作偏移修正，因此本页不比较其 total_ms，也不继承任何偏移量。

**当前停止位置：完整 AE solo 已验收；VLM 校准和之后的并发／仲裁消融按用户要求暂停。** 本页中的后续归因方法仅为可复核的分析计划，不表示实验已经执行。
## 下一步定量归因方法

1. 固定输入/程序/spec 哈希、相同 k、单层初态与刷新设置；先逐算子核对 MAC 工作量、LOCALRD/WR、DQ 字节及生命周期，再比较时间。除字节以外，当前新 raw 尚未包含按 PIM opcode 的计数，不能用 checked_commands 与旧 issued_commands 直接对比：旧伪控制事件也没有相同计数口径。
2. 在 native device 旁路采集实际命令族计数与 issue timestamp（ACT1/2、PRE、REF、MAC、LOCAL、buffer IO），记录资源 ready 与 native ready 对当前 head 的限制。记录不参与调度，不复制时序常量。
3. 对每个无法发射的区间只按真正限制该次推进的最大 ready timestamp 归一归类；并列约束单独标记，避免把重叠等待相加。分别报告接口、行激活/恢复、刷新、显式依赖、执行资源等待与完成尾部。
4. 若需要“某项修正贡献了多少 µs”的因果值，再做明确隔离的诊断实验，例如固定 score batching/PE 数/激活协议逐项比较。必须标注旧机制可能不满足新任务，独立保存，不把它们暴露为正式实验开关，也不能用失效旧时序回填主结果。当前只能报告上文的算子差异与机制证据，不能宣称已完成数值因果拆分。

## 新数据定位与校验

来源：measurement-host `/artifact-source/pim_vla_sim2_20261109/results/pim-smoke/{case}/raw.json`；验证日志 `validate-1976509.log`。以下哈希固定本页实际读取的单层 raw，防止以后完整运行或新验证覆盖后混用数字。

| case | raw.json SHA256 |
|---|---|
| b | `8dec110de79d87f4cbc27c65f362bf76406a826c15a5619477c18e18167c8173` |
| d | `d2152fff9b86f2040958ab227ab44952364a4604df8ba6bb6acf6179a44649bb` |
| e | `1a3acadf317af62db74ed733b227a12ca35a095df242bc26efe44df3d5f1014c` |
| f | `a0bed56bf6e3f784350e85907e27d56b6eadad865cda76f8ddfffe8d2a607371` |
| g | `1b8d2bd88256b098a5c1e668cba0b88c3f9ae431128be472cfc8f4514d64e5fc` |

相关新版实现：[spec 与时序依据](timing-source-and-decisions.md)、[PIM 程序](../programs/gen_pim_program.py)、[地址布局](../workloads/layout.py)。
