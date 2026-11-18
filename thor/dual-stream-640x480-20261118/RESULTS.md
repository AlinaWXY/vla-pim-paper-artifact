# GR00T N1.6：双 CUDA stream 与最新完成的 cross-KV 缓存

已在 NVIDIA Thor、MAXN 上完成 eager 和 compile 两种后端测量。三路原始图像为 640×480，checkpoint 默认预处理后的 ViT 入口均实测为 `[1,3,252,336]`，即宽×高 336×252。总序列 371 token，其中视觉 324、文本及格式 47。batch=1、BF16、H=15，包含一个 state token 后 DiT 有 16 个 query；32 层、4 步去噪。

## 结果

延迟为未开启 profiler 的 CUDA Event 中位数，单位 ms。Hz 为完整 action chunk/s 或完整 VLM+KV 刷新次数/s，按测量窗口内完成数计算，不能用延迟中位数的倒数替代。

| 后端 | 缓存 AE 单跑 | 双流 AE | 双流 VLM forward | 双流 K/V 投影 | 双流完整 VLM+KV 刷新 | 双流 AE Hz | 双流 VLM+KV Hz |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager | 45.709 | 109.463 | 105.993 | 1.482 | 107.477 | 9.039 | 9.491 |
| compile | 39.518 | 98.268 | 108.729 | 1.756 | 110.527 | 10.415 | 9.243 |

AE 单跑吞吐分别为 eager 22.098 Hz、compile 25.243 Hz。双流使 AE 的中位耗时分别变为单跑的约 2.39 倍和 2.49 倍。当前相同优先级、持续刷新 VLM 的双流配置，没有让 AE 保持单流速度。

VLM+KV 在没有 AE 并发时的基线：

| 后端 | VLM forward p50 | K/V 投影 p50 | 完整刷新 p50 |
|---|---:|---:|---:|
| eager | 71.939 | 1.039 | 72.977 |
| compile | 57.604 | 1.026 | 58.627 |

两种后端在双流中都承担不同强度的并发 AE 工作，因此不能用双流 VLM 的 eager/compile 两个数单独判断 VLM 编译收益。以上结果包含实际的 Python/CUDA 提交、同步和 GPU 竞争，本实验没有把它们进一步分离为独立瓶颈。

## 缓存语义与验证

VLM stream 执行 `ViT + 视觉投影 + 语言模型 → VL LayerNorm → 各层 cross-attention K/V 投影`。16 个 cross-attention 层各有 K、V，共 32 个 BF16 张量，形状均为 `[1,32,371,48]`。每版实际缓存为 36,470,784 B，即 34.78125 MiB；保留完整 371 长度，text/image 使用原来的 mask。

AE stream 只在一个完整推理开始时选择最新完成的一版缓存；该 action chunk 的 4 步去噪始终使用同一版。VLM 在后台继续计算，AE 不等待下一版。完成事件确保不会读到计算中的 K/V；旧缓存保持强引用并标记 consumer stream，直到使用它的 AE 完成。每条 stream 最多一个推理在执行，没有预排队下一次 AE。

每次 AE 选择时也检查已提交 VLM 的完成事件，避免生产线程的发布延迟掩盖已经完成的缓存。随后根据 GPU 事件时间线重新核对：正式双流样本共 160 个（每后端 80 个），全部选择了 GPU 启动时最新完成的缓存；过期选择、缓存版本倒退、读取未完成缓存均为 0。

| 后端 | AE 启动时缓存年龄 p50 | p90 | AE 完成时缓存年龄 p50 | 复用上一版的比例 |
|---|---:|---:|---:|---:|
| eager | 14.181 ms | 101.557 ms | 125.137 ms | 0/80，0% |
| compile | 70.895 ms | 98.095 ms | 170.664 ms | 9/80，11.25% |

这里的“缓存年龄”从 K/V 完成时计算。VLM 自身还需要处理时间：距该次 VLM 刷新开始，eager/compile 在 AE 启动时的年龄中位数分别为 121.896/177.218 ms，在 AE 完成时分别为 233.057/273.020 ms。四幅输入图像已预先驻留 GPU，这些时间不是实际摄像头曝光到动作输出的端到端延迟。

K/V 投影开销计在 VLM 侧，AE 读取缓存；本次 AE 单流和双流都使用这一缓存版实现。之前的完整 `get_action` 在每个去噪步骤重新投影 K/V，两组数据应按各自的工作范围阅读。

## 测量方法

- 两个 Python worker、两条同优先级 CUDA stream；持续提交 VLM 和 AE。
- 每个后端按 `AE 单跑 → 双流 → 双流 → AE 单跑` 顺序，降低固定顺序造成的影响。每块排除 6 次预热，并使用一个完成边界后测量 40 个 completion period；每种条件合计 80 个正式样本。
- AE 每次生成新的随机初始动作噪声，其生成开销包含在 AE Event 时间内。state 固定为零，embodiment ID=2。
- VLM 轮换四组固定输入：原图、水平翻转、水平平移 8 像素、垂直翻转。变换在预处理后、正式计时前完成，已核对它们确实改变 K/V 内容。
- 两种后端都排除首次编译与预热，正式测量期间没有重新编译。图像预处理、权重读取、结果保存不在计时内。
- compile 选项沿用前次：`max_autotune=True`、`triton.cudagraphs=False`、`triton.unique_kernel_names=True`、`emulate_precision_casts=True`。VLM、K/V 投影和缓存 AE 分别编译；K/V 投影及缓存 AE 使用 `fullgraph=True`。

独立的 CUDA-only CUPTI trace 用于核对实际重叠，不参与上述延迟和吞吐统计。检测到 eager 的两条 stream 为 CUPTI 13/17，compile 为 21/25；实际 kernel 执行存在交叠。重叠时长占 AE GPU 忙碌时间的比例分别为 38.24% 和 34.08%。该比例来自有 profiler 扰动的诊断窗口，不是 SM 利用率或未插桩性能增益。

## 数值校验与复现文件

对同一初始噪声，缓存版 eager AE 与官方 eager `get_action` 输出逐元素相同，最大绝对差和相对 RMS 均为 0。这验证本次固定输入下的缓存改造。compile 整条 VLM+KV+AE 路径相对 eager 的最终动作相对 RMS 差为 0.3429%，最大绝对差 0.0234375；所有并发测量的输出均为有限值。此处没有进行任务成功率评测。

- `dual-stream-latencies.csv`：单流/双流延迟、吞吐、缓存年龄、复用率。
- `ae-cache-timeline.csv`：每次 AE 的实际缓存版本、最新可用版本、开始/结束时刻及缓存年龄，标记正式样本和预热。
- `vlm-refresh-timeline.csv`：每版缓存的 VLM forward、投影、完成时刻和版本号。
- `dual-stream-validation.json`：缓存形状、数值校验、无重新编译和真实 kernel 重叠证据。
- `results/dual-stream-summary.json`：完整原始样本、输入和策略配置。
- `results/dual-eager.trace.json`、`results/dual-compile.trace.json`：独立 CUPTI trace；`dual-trace-kernels.csv` 为对应 kernel 时间线。
- `source/`、`run.sh`、`environment.txt`：冻结的实际执行代码、命令、环境及权重 SHA256。`thor-measurements.tar.gz` 为远端原始归档。
- `analysis/analyze_gr00t_dual_stream.py`：离线分析代码，仅依赖 Python 标准库。在本目录运行 `python analysis/analyze_gr00t_dual_stream.py .` 可重建 CSV 和验证文件。

checkpoint revision 为 `d0814e7ecb19202e7c8468b46098b0b7ef3a6d61`，官方源码 revision 为 `9b37aa1ce69c73c6d165233fa88128283bba4508`。GPU 测量在共享互斥锁下运行。
