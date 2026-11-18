# VLM 闭环访存与统一的 e/f/g 对比

`anonymous_1109_V2-closed-loop` 从 `79b4d84` 建立。b–g 和 e 的三个仲裁变体统一使用 `native_dram_requests` 与 `vlm_release=closed_loop`。f/g 继续使用原生 fused I/O epoch 和私有 CA，PIM 程序、布局、DRAM spec、GIO 与 BCAST 约束沿用基线。

## 请求和完成条件

每个 kernel 开始时，各通道按原有条带化、读写交错和地址顺序填充窗口 `W_k`。在飞请求包含排队和已发射但尚未完成的请求。完成事件退役后，同一周期补足空位。fused epoch 期间请求仍可进入窗口，GPU 命令发射沿用原有的 epoch 门控。

kernel 的实测时长向上取整到原生时钟周期，作为 floor。kernel 完成时间取 floor 与 16 个通道最后一次访存完成的较晚者。计算受限单元使用 `W_max`，其访存结束到 floor 之间可与 PIM 重叠。

每通道硬件队列深度 `W_max` 默认为 128，敏感性取 256。写水位固定为高 `3 × W_max / 4`、低 `W_max / 4`。读队列为空时，写请求可在低于高水位的情况下继续发射。闭环的 `gap_batch` 用已知的 PIM 队首就绪时间减去当前周期，与往返切换开销比较。队首时间未知时放行 GPU。写批清空后切回读阶段。

`vlm_release=uniform` 保留原有定时释放、原有水位配置、原有 `gap_batch` 行为和原始输出格式，用于历史复现。

## 校准和验收

1. 每个工作单元独立从 CK0、关闭的 DRAM bank、确定的刷新相位开始。rank0 第一次 REFpb 到期于 `nREFIpb`，从 pair 0 起步。rank1 第一次 REFab 到期于 `nREFI`。探测使用原生 FR-FCFS，按通道条带读写计数选择等价通道，每种计数仿真一个通道并取最慢完成时间。
2. `W_max` 的访存用时至多为实测时长的 80% 时归为计算受限。超过实测时长时单列为仿真器慢于 Thor。其余归为访存受限，在 `[1, W_max]` 二分寻找满足实测时长的最小窗口。每次探测的周期数与观察到的非单调 ordinal 写入 ledger。
3. 在连续 DRAM 状态和刷新相位下顺序运行 2,353 个 kernel 和 4 个 memcpy。对其中误差超过 5% 的访存受限单元执行至多一轮局部二分，用连续运行与隔离探测的访存周期差收紧隔离预算。计算受限与仿真器慢于 Thor 的单元保留 `W_max`。
4. 最终向量用两次完整 solo 验证，两个 `raw.json` 须逐字节一致。总时长以 104.950496 ms 为基准，误差容限 ±5%。`calibration-mlp.json` 包含每个 ordinal 的分类、窗口、隔离和连续时长、误差，以及源码归档、可执行文件、运行库、spec 与 VLM 输入的 SHA-256。

在 Linux 上构建并生成既有程序后运行以下命令。阶段 A 通过并提交结果后才执行阶段 B，然后依次提交 C、D。

```bash
bash build.sh
python3 programs/gen_pim_program.py --output programs/generated --compact
python3 tools/calibrate_kernel_mlp.py --w-max 128 --output results/stage-a
python3 experiments/run.py --spec reference/spec/spec.json \
  --calibration results/stage-a/calibration-mlp.json --output results/stage-b \
  --cases b c d e e-frfcfs e-lookahead e-gap
python3 experiments/run.py --spec reference/spec/spec.json \
  --calibration results/stage-a/calibration-mlp.json --output results/stage-c --cases f g
```

各组在 measurement-host 提交独立作业时可用 `--cases` 和独立输出目录拆分。阶段 D 用 256 重做校准和九组对比。每个 case 保存 AE solo 和并发的两个原始结果及校验和。总时间按 AE solo、PIM 阶段增量、AE GPU 等待和 VLM 尾部核对，残差容限为 0.002 μs。VLM 延迟差按 ordinal 累计，计算侧与访存侧分开。访存侧汇总包含单列的「仿真器慢于 Thor」子集。CA 和 DQ 的占用分母为相应阶段的通道周期数，刷新按所属 rank 归类，共享 CA 与 PIM 私有 CA 分列。

## 阶段 B 的历史上界参考

历史 `fused_kernel_windows` 在上传与导出时执行独占交接。g 的上传、导出及 Q 交换合计约 6.78 ms，完全串行地加到 GPU 时间线上。闭环逐请求模型允许计算受限 kernel 在 I/O epoch 内继续推进，epoch 也可在 kernel 中途开始。新增方向切换与描述符 CA 开销预计为 μs 量级。按这个调度假设，以闭环 solo 替换历史 solo 后的参考上界为

```text
g_upper = 131.832380624 - 106.893161172 + V_closed
f_upper = 137.737998608 - 106.893161172 + V_closed
```

各项单位均为 ms。阶段 B 用 `e_closed` 对照 `g_upper`，用 `d_closed` 对照 `f_upper`。仅当 streaming 实测值达到或超过对应参考值时，给出「基于上界」的判断。阶段 C 用原生 f/g 实测检验两条上界关系，并记录超出量和造成差异的阶段。

历史窗口结果来自 [连续 fused 窗口实验](../reference/fused-kernel-continuous-e2e/summary.json)。历史逐请求均匀释放结果来自 [b–e 与仲裁实验](../reference/e2e-gio-bcast-20261109/data/completed-cases.json)。

## 局限

- L2 写扇区代理到 DRAM 写的比例 α 固定为 1.0，实际 DRAM 写回比例仍待测。
- GPU 地址是合成的顺序地址。真实地址、burst 到达分布和每个 kernel 的硬件 MLP 尚无直接观测。
- 模型覆盖 GPU/PIM 的原生内存时序竞争和 AE GPU/VLM 的非抢占分时。SM 竞争处于模型范围之外。

## 阶段 A 实测

measurement-host 作业 `1983536` 使用 `a042beb` 和 `W_max=128`。第一次连续 solo 用时 143.401551622 ms，34 个访存受限单元完成局部细化。最终两次均为 **134.272003250 ms**，相对 104.950496 ms 的误差为 **+27.938%**，超过验收上限 24.073982 ms。两个原始结果经 `cmp` 确认逐字节一致。

| VLM solo | 时长（ms） | 全程 GPU DQ 占用 |
|---|---:|---:|
| 历史均匀释放，窗口 1024 | 106.893161172 | 67.984% |
| 闭环，硬件窗口 128 | 134.272003250 | 54.122% |

闭环 solo 较历史均匀结果增加 27.378842 ms。两轮使用不同的释放机制、窗口和硬件写水位。在本轮固定的 128 深度和水位下，隔离探测已观测到大量访存完成时间超过 Thor，连续运行的时间差主要集中在这一类。

| 分类 | 单元数 | 最终 solo（ms） | 较采集时长之和增加（ms） |
|---|---:|---:|---:|
| 计算受限 | 1314 | 23.161541 | 0.000627 |
| 访存受限 | 169 | 8.868389 | 0.022995 |
| 仿真器慢于 Thor | 874 | 102.242074 | 29.027803 |

按名称汇总，`nvjet_sm110_tst_256x248_64x3_1x2_h_bz_TNT` 的 36 次调用多用 7.207 ms，`nvjet_sm110_tst_256x200_64x3_2x1_v_bz_TNT` 的 90 次调用多用 6.199 ms。隔离超出量最大的 ordinal 为 1742，实测 765.728 μs，隔离访存 1108.289 μs，连续 solo 1108.612 μs。

计算受限单元占最终 solo 时间的 **17.250%**。逐 kernel 误差中位数 **0.009%**，P90 **44.330%**，最大值 185.920%。分布只统计 kernel，四个 memcpy 的记录保留在校准文件中。

| 逐 kernel 相对误差 | 数量 |
|---|---:|
| ≤5% | 1589 |
| (5%, 25%] | 282 |
| (25%, 50%] | 358 |
| (50%, 100%] | 104 |
| >100% | 20 |

| W_k 区间 | 工作单元数 |
|---|---:|
| 1–95 | 2 |
| 96–111 | 20 |
| 112–119 | 36 |
| 120–127 | 73 |
| 128 | 2226 |

真实工作单元的采样中有 56 个 ordinal 出现更大窗口反而用时更长。合成单调性测试通过。二分法只在访存用时随窗口单调时保证全局最小，实测逆序记录保留在校准文件。

全部 **874** 个仿真器慢于 Thor 的单元及其隔离和连续超出量已按隔离超出量降序列出。校准文件同时保存逐 ordinal 时长、逐窗口直方图 `w_k_histogram` 和按 kernel 名称分组的直方图 `w_k_by_name`。 [超时清单](../reference/closed-loop-vlm/stage-a/slower-than-thor.csv) · [校准文件](../reference/closed-loop-vlm/stage-a/calibration-mlp.json)

重复原始结果的 SHA-256 为 `b16c27376059cb8cb6ef2bc86d39db5b035513d6ed36939d29c458de0aae83e0`。 [原始结果](../reference/closed-loop-vlm/stage-a/raw.json) · [验证记录](../reference/closed-loop-vlm/stage-a/validation.json)

**阶段 B、C、D 未运行。** 128 深度的校准未通过，按验收顺序在阶段 A 停止。
