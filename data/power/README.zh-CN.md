# 方法 g：完整 DRAM 功耗重跑

作业 **2062938** 已完成，退出码 `0:0`。2026-11-17 19:58:58 开始，
20:58:33 结束（UTC+8），实际运行 **59 分 35 秒**。
资源：cpu-scavenger，1 个任务、1 CPU、8 GiB、无 GPU，时限 4 小时。
预计运行 1–2 小时；按用户确认方案关闭自动重新排队。

## 完整运行结果

- 模拟的工作负载时长：**77.677674 ms**；发出 **667,676,820** 条原生命令。
- 完整 `raw.json` 与原始备份逐字节一致。bank/阶段时间守恒、状态分布、
  命令数、DQ 字节数、刷新范围和激活展开均核对通过。
- Micron IDD 模型参考平均功耗：**12.677460 W**；参考能量：**0.984755619 J**。
  其中 GPU 侧 DRAM rank 为 6.733116 W，PIM 侧 DRAM rank 为 5.944344 W。
  这两项均不包含计算单元能耗。
- 实际激活计数：GPU rank 为 9,800,689，PIM rank 为 16,396,304。
- 记录了 1,620 个 PIM 阶段及阶段外区间；分项见 `results/combined-g/power-report.json`。

上述功耗对应未校准参考参数 `alpha_read = alpha_write = 1`。一般形式为：

`P_total = 7.538090 + 4.885828 * alpha_read + 0.253542 * alpha_write` W。

这是带器件和 PIM 能耗假设的模型结果，不能称为已验证的真实硬件功耗。

## 加入 PIM 计算单元和 Buffer 后的当前估算

按用户提供的图片，采用 ASAP7、0.7 V、约 533 MHz 的每 bank 活动功耗粗估。
当前每通道仅 rank1 配置 PIM 单元，因此单元数量为：

`8 package * 2 channel/package * 1 PIM rank/channel * 16 bank/rank = 256 PIM banks`。

每个 package 有 32 个 PIM bank，另有 32 个普通 GPU/DRAM bank。
原 DRAM 模型仍覆盖全部 512 个 bank；新增逻辑只按 256 个 PIM bank 累加。
每 bank 配 1 套 16 路 MAC 模块、1 个 RoPE 模块、1 KiB 输入 Buffer 和
2 KiB 输出 Buffer。因此共有 256 套 MAC 模块（4,096 条 MAC lane），
其余三个模块也各为 256 个。图片已按每 bank 给出整套模块功耗，
MAC 功耗不再额外乘 16。

| 单元 | 数量 | 图中每 bank 活动功耗 | 全部 PIM bank 分项功耗 |
|---|---:|---:|---:|
| MAC 模块 | 256 套 | 3.28 mW | 0.83968 W |
| RoPE | 256 个 | 0.95 mW | 0.24320 W |
| 输入 Buffer，1 KiB | 256 个 | 6.06 mW | 1.55136 W |
| 输出 Buffer，2 KiB | 256 个 | 10.85 mW | 2.77760 W |

图中合计为 21.13 mW/bank，显示分项之和为 21.14 mW/bank。
本次遵循图片给出的合计，新增 `21.13 mW * 256 = 5.40928 W`；
用显示分项求和则为 5.41184 W，相差 0.00256 W，二者均约 5.41 W。
此前 Micron 模型不含这四类 PIM 逻辑，因此这里加入整套逻辑功耗。
图片中“原有小 Buffer 基线”的 0.115 W/die 不在原模型中，不作扣除。

保持 `fresh_cache` 的 GPU 平均功耗代理后：

`P_g = 50.358117 + 12.677460 + 5.409280 = 68.444857 W`。

`E_g = P_g * 77.677674 ms = 5.316637 J`。

`R = E_board_fresh_cache / E_g = 13.418796 / 5.316637 = 2.523925`。

按用户指定的简化比较口径，计入这些逻辑后的能耗比约 **2.52**，
能耗减少约 **60.38%**。下面仅包含 GPU+DRAM 的 2.74 是未加上述逻辑的参考结果。

当前采用**全部 256 套单元在完整 g 区间维持图中活动功耗**的常数情形，
没有从 trace 推导活动率，不是工作负载平均逻辑功耗。图中“数据静止、
时钟运行”约 0.27 W/16 bank，对应全部 PIM 单元约 **4.32 W**；
这个空转参考不再叠加到 5.40928 W 活动总功耗中，也不能把未门控 Buffer
功耗简单按 MAC 占空比全部缩小。没有施加面积 45.94× 倍率，也没有额外
缩放频率/电压。g 的 `nMAC1=2 CK`、`CK=938 ps` 对应约 533.05 MHz，
与所给 MAC 参考频率相符。

这些值来自逻辑工艺的综合活动假设，尚不是 DRAM 工艺中的实测。
原先整板/部分部件功耗范围不一致、稳态周期/单 pair 时间不一致、
PIM 阵列访问能量未校准等限制仍然存在。
配置见 `configs/pim-logic-power-533mhz.json`；当前结果见
`results/combined-g/fresh-cache-energy-ratio-with-pim-logic.json`；
原始图片保存在 `evidence/pim-unit-power-user-reference.png`。

## GPU+DRAM 参考：fresh_cache 基线与 g，未计 PIM 逻辑

当前比较采用单 GPU 中 AE 等待最新 VLM cache 的 `fresh_cache` 模式，
分子使用该模式的板上 `VIN` 功耗，分母将该模式的 `VDD_GPU` 平均功耗
作为 g 的 GPU 常数代理，再加入 g 的 DRAM 仿真功耗：

`R = (P_board_base * T_base) / ((P_GPU_base + P_DRAM_g) * T_g)`。

三轮各 30 秒实测共完成 663 个 AE chunk。`T_base = 90 / 663` 秒，
即平均完成周期 135.746606 ms；这不是输入到动作的延迟。
`T_g = 77.677674 ms` 来自完整 g 的单 pair 原生仿真。
两个时间分别对应各自方案，不共用基线时间。

| 比较项 | 使用的平均功耗 | 使用的时间 | 对应能量 |
|---|---:|---:|---:|
| 单 GPU，AE 等最新 cache，板上 VIN | 98.851798 W | 135.746606 ms | 13.418796 J |
| g，GPU 代理加 DRAM 模型 | 63.035577 W | 77.677674 ms | 4.896457 J |

分母功耗为 `50.358117 + 12.677460 = 63.035577 W`。
按上述公式得到 **R = 2.740511**，对应简化口径下能耗减少 **63.51%**。
等价分解为功耗比 `1.568191` 乘时间比 `1.747563`。

**这是用户指定口径的简化能耗估算，尚不是统一口径的整板能效测量。**
分子是板上 VIN，分母缺少 g 的 CPU、其他 SoC 部件、载板和外设等功耗；
GPU 代理来自 VLM+AE 的全 GPU 运行，DRAM 的 PIM 内部访问能耗也仍未校准。
此外，基线使用稳态 AE 完成周期，g 使用单 pair 完成时间，尚无多帧稳态
仿真来核对时间口径。因此不能把 2.740511 标注为已经验证的整板能效提升。

计算参数见 `configs/fresh-cache-energy-ratio.json`，结果、原始区间、
输入哈希和核对记录见 `results/combined-g/fresh-cache-energy-ratio.json`。

## 串行模式 GPU 代理的独立参考结果

以下保留先前用 `serial` GPU 均值计算的独立结果；当前 `fresh_cache`
能耗比使用上一节的 50.358117 W，不使用这里的 49.858615 W。

按用户接受的常数平均功耗口径，使用已归档 Thor 实测记录中 `serial`
模式的三个 30 秒区间，按 `sum(energy_j) / sum(seconds)` 得到
GPU 电源轨 `VDD_GPU` 平均功耗 **49.858615 W**，并将该常数应用于
完整 g 的 77.677674 ms 区间。GPU 与 DRAM 使用同一统计时间窗口，
不另乘 GPU 占空比。

| 分项 | 参考平均功耗 | 对应能量 |
|---|---:|---:|
| DRAM，含 GPU rank 和 PIM rank | 12.677460 W | 0.984756 J |
| GPU 电源轨常数代理 | 49.858615 W | 3.872901 J |
| **GPU + DRAM 合计** | **62.536075 W** | **4.857657 J** |

**49.858615 W 来自 VLM+AE 在 GPU 上串行运行的实测，暂作 g 的 GPU/VLM
平均功耗代理，不能标注为 VLM 单独运行的实测功耗。** 当前估算也未刻画
GPU 在 PIM 并行期间的功耗变化。PIM 内部能耗的未校准假设保持原样；
仍不额外计入 PIM 计算单元和寄存器，CPU 及其他整机部件不在本次合计范围。

只加入 `VDD_GPU`，不加入整机 `VIN`；GPU 侧 DRAM 已包含在上表 DRAM
项中。[NVIDIA 官方电源轨定义](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html#system-power-measurement)
将 `VDD_GPU` 定义为 GPU 功耗，将 `VIN` 定义为包含模块和载板的整机功耗。

新增结果为 `results/combined-g/gpu-plus-dram-power.json`；常数、聚合方式、
原始测量位置及校验哈希保存在 `configs/gpu-average-power.json`。
三个测量区间的功率/能量关系、总量关系及原仿真结果哈希均核对通过。
原 `power-report.json` 和 `power-summary.json` 继续保留 DRAM 专用口径。

## 保持原有仿真行为

源代码来自 `backup.json` 指向的原始备份。新增代码只观察已经发出的命令，
不改变调度、数据流或时序参数。完整输入配置相对原配置只增加
`dram_power_stats: true`。原始源代码和结果保留在原位置。

重跑前完成 4 个原生命令统计测试、4 个功耗计算测试。单层 g 和
g/VLM 并行运行两个样例的 `raw.json` 均与原始可执行文件逐字节一致，
证据见 `preflight.json`。完整运行结束后还会执行同样的原始结果核对，
核对失败时不生成正式功耗报告。

## 输出和计算口径

所有完整运行结果写入 `results/combined-g/`：

- `raw.json`：原生仿真结果。
- `dram-power-stats.json`：16 个通道、512 个 bank 的实际命令计数、
  状态驻留时间，以及各 rank、各阶段的 active/refresh bank 数量分布。
- `power-report.json`：上述统计通过核对后的能量分项与平均功耗。
- `power-summary.json`：总量摘要。
- `native.log`、`resource.log`：仿真日志和运行资源记录。

这里保存完整运行的命令统计及状态时间，不输出逐条命令的大型文本 trace。
ACTAB 按原生模型的实际分批激活时间展开；REFpb 按两个目标 bank 统计；
刷新及预充电的结束事件按实际时间计入，仿真结束时截断。

采用 Micron LPDDR5X-8533 AAT 的 IDD 参数。后台、BG 读写和刷新计算
参考 DRAMPower 的 LPDDR5 公式；这是本地公式实现，尚未与上游库交叉验证。
计算使用实际 bank 状态分布，不使用整段始终处于 active 的近似。
`rho=0.5` 是模型假设。ACT/PRE 只报告每次实际激活对应的**合并增量能量**，
现有 IDD 参数不足以独立校准 ACT 和 PRE 两项。

PIM 内部读写能量没有厂商校准参数，报告明确给出：

`P_total = fixed_W + alpha_read * PIM_read_reference_W + alpha_write * PIM_write_reference_W`

默认参考情形取两个 alpha 为 1，不能将其称为已验证的真实 PIM 硬件功耗。
Micron 参数来自 16 Gb die，原仿真组织为 8 Gb/die；规格中的最大值和
部分典型值混用也会限制物理解释。原 DRAM 单项中不计计算单元、寄存器、
主机控制器/PHY 和供电损耗；上文的当前合计已另行加入用户给出的
MAC、RoPE 和输入/输出 Buffer 逻辑粗估。

`configs/micron-8533-aat-proxy.json` 从旧基线复制，保留的
`background_policy`、`row_policy`、`refresh_policy` 描述旧脚本策略，
**新计算器不读取这三个字段**；本次实际算法以上述说明及
`source/tools/g_power_report.py` 为准。冻结输入的哈希保存在 `manifest.json`。

## 运行记录

- `approval.json`：用户确认的资源方案。
- `jobs.json`：作业编号及状态快照。
- `rerun-g.sbatch`：已提交脚本；运行成功后自动核对并生成报告。
- `logs/g-power_2062938.log`：Slurm 日志。

可用 `squeue -j 2062938` 查看运行状态；结束后用
`sacct -j 2062938 --format=JobID,State,ExitCode,Elapsed,MaxRSS` 查看结局。
