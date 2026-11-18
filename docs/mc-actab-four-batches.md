# 顶层 MC 的四批 ACTAB 对照实验

从 `anonymous_1113` 的 `e76d7e7` 分出实验工作树。两组都使用 `--mc-control` 和 MAC 2×，各跑 b–g 的 step 0 / layer 0 与 step 9 / layer 17。两组使用同一个原生仿真二进制、同一份 AE 请求程序和 GPU 时间表。每个切片分别关闭和打开 trace 运行，比较原始 JSON 是否逐字节一致。

| 行为 | 发布版 ACTAB | 四批 ACTAB |
|---|---:|---:|
| 外部 CA | 一条命令，2 CK | 一条命令，2 CK |
| 阵列激活 | 同时打开 16 bank，FAW 历史加一次 | 偏移 0、16、32、48 CK，每批每个 BG 打开一个 bank，FAW 历史每批加四次 |
| 首条整 rank 计算 RD | 20 CK | 68 CK |
| 无列访问或刷新时 `ACTAB → PREab → ACTAB` | 68 CK | 116 CK |

四批模式沿用原生 `ACT2_PULSE` 展开。每批占满滚动四激活窗口，外部 `ACT2` 与内部脉冲共用该窗口。`PREab` 等最后一批的 `tRAS`，同一 bank 的预充电和列访问按各自的激活时刻计时。`SRF_SELECT` 仍只负责选择 Host SRF 写入路径。

四批内部脉冲是 PIM 扩展的架构假设。同批四个 bank 的供电和时序需要器件实现验证。标准单 bank ACT 还受 `tRRD` 约束。

## 结果

原生时序检查通过 232 项。24 份 AE 切片的 trace 开关对照逐字节一致。两组的 AE 请求程序、GPU 时间表、原生二进制和 PIM DQ 字节数逐切片一致。独立读取 CA trace 后，整 rank 首条 RD 的最短间隔从 20 CK 变为 68 CK，四批模式中后续阵列激活符合滚动 FAW 窗口。

单切片 AE solo 增加 8.182–15.847 µs（2.27%–3.74%）。下表展示 step 9 / layer 17 的 CA 延发回放。完整的两个切片与关闭 CA 冲突时的结果在 `results/actab-four-batches/comparison.csv`，逐阶段原生耗时在 `stage-comparison.csv`。

| Case | AE solo 增量 (µs) | 发布版 E2E (ms) | 四批 E2E (ms) | E2E 增量 (ms) |
|---|---:|---:|---:|---:|
| b | 9.818 | 180.251 | 182.018 | +1.767 |
| c | 9.818 | 144.945 | 145.365 | +0.420 |
| d | 9.759 | 141.478 | 141.413 | −0.066 |
| e | 10.203 | 133.592 | 133.485 | −0.107 |
| f | 15.821 | 134.741 | 138.927 | +4.186 |
| g | 15.847 | 125.555 | 129.992 | +4.437 |

g 在两个切片的 180 次回放中仍最快。step 9 / layer 17 的 e−g 差值从 8.037 ms 缩到 3.492 ms。g 单切片增加的时间主要出现在 Down（6.605 µs）、QKV（3.157 µs）和 Gate/Up（2.656 µs）。该切片在原模型中，g 的 Down 平均每通道发出 136 次数据行 ACTAB，e 是 31 次。f/g 的 Down 在从 staging 行广播装入下一批 SRF 后重新开权重行。d/e 用 Host SRF 写入与 MAC 交错，权重行可以保持打开。

### 并发时序交叉检查

下表拆开 step 9 / layer 17、180 次的 CA 延发回放，增量均为 ms。f/g 有一部分 native 数据搬运窗口已经计入 GPU profile，回放扣除这部分重复时间。表中采用回放实际计入的 AE 时间。

| Case | Δ AE（GPU 等待前） | Δ GPU 交接等待 | Δ VLM 尾部 | Δ E2E |
|---|---:|---:|---:|---:|
| d | +1.757 | −1.822 | 0 | −0.066 |
| e | +1.836 | −1.944 | 0 | −0.107 |
| f | +2.781 | +1.405 | 0 | +4.186 |
| g | +2.803 | +1.702 | −0.068 | +4.437 |

在该回放中，d/e 的 PIM 期间分别多服务了 766 万和 1,232 万笔 VLM 请求，f/g 分别少服务了 745 万和 1,171 万笔。逐阶段核对时，g 的 QKV 和 Down 都出现了可用 DQ 窗口变长、实际服务请求数减少的情况。VLM kernel 进度、独占区间和 GPU 交接时点共同决定请求落在哪一段。关闭 CA 冲突时仍看到相近的 E2E 差异。

另用同一原生二进制运行 d–g，从相同初始状态并发执行两个 step 9 / layer 17 切片和前 64 个 VLM kernel。VLM 使用原有均匀释放模式（α=1.0，窗口 1024），原生 MC 共同调度两个 rank 的命令。四批模式下，d/e/f/g 的 PIM 时间分别增加 21.630、22.650、30.125、30.710 µs，E2E 分别变化 +4.649、+14.081、−7.015、−5.908 µs。匹配该短输入的 CA/DQ 回放也未重现 180 次回放中的 f/g E2E 惩罚。原生短跑和长回放都支持四批模式增加 f/g 的 PIM 时间。长回放中的 GPU 等待与 E2E 差值限定在原测试时间表和回放口径。短跑结果、回放短跑和 Slurm 命令保存在 `results/actab-four-batches/`，服务器原始输出在实验目录的 `diagnosis/`，作业 `2039896`。

Slurm 作业 `2039888` 完成原生捕获。新 trace 暴露了回放器的浮点批次边界问题。修正并加入回归后，同一个回放二进制在作业 `2039890` 中重新运行两组全部切片。同步开 16 bank 的对照在发布切片上复现了 b–g 的 E2E 和总请求数。服务器的原始 trace 和回放记录位于 `measurement-host:/artifact-source/pim-mc-actab-4x4-20261114`。

## 复现

### QK score 写回的 bank mask

QK 的 `WB_OBUF` 使用 `0x0011`，只访问 bank 0、4。MC 在选择候选周期和发令时都检查这两个 bank 的开行与列时序，并只更新被选中 bank 的阵列恢复时序。CA 和共享层时序仍按一条 WR 记账。整 rank MAC、全掩码写回和 FIFO 顺序沿用原有规则。

Slurm 作业 `2039975` 通过 245 项原生时序检查，并重跑 step 9 / layer 17 的 b、d、e、f、g。四批模式下，trace 中 `WB_OBUF` 距目标 score 行 `ACTAB` 的最短间隔从 57 CK 变为 9 CK。数据包数量与 PIM DQ 字节数一致。同步模式的最短间隔仍是 9 CK。阵列恢复的变化可以改变后续刷新相位，因此部分其他阶段的实际时长也发生变化。结果见 `results/actab-four-batches/score-writeback-comparison.csv`。上文的 E2E 回放使用这一改动之前的 trace。

### ACTAB 对照

在工作树根目录构建 `pim_vla_sim`，用同一个二进制生成两组 trace。

```bash
python3 tools/capture_gap_traces.py --binary build/pim_vla_sim \
  --output results/actab-four-batches/instant --mac-freq-mult 2 --mc-control --cases b c d e f g
python3 tools/capture_gap_traces.py --binary build/pim_vla_sim \
  --output results/actab-four-batches/four-batch --mac-freq-mult 2 --mc-control --four-batch-actab --cases b c d e f g
c++ -std=c++20 -O3 tools/dq_request_replay.cpp -o build/dq_request_replay
for model in instant four-batch; do
  python3 tools/run_dq_request_replay.py --engine build/dq_request_replay \
    --traces "results/actab-four-batches/$model" --prior-summary reference/mc-dq-replay/prior-summary.csv \
    --output "results/actab-four-batches/$model-replay" --compare-ca --preserve-pace
done
python3 tools/compare_actab.py --traces-root results/actab-four-batches \
  --output results/actab-four-batches
```

E2E 回放对每个切片重复 180 次，固定 AE 的原生 CA/DQ trace，根据 CA 冲突延发 VLM 列命令。它使用发布结果的 VLM profile、GPU 交接和共享 I/O 口径。真实芯片的功耗和供电裕量需要独立验证。
