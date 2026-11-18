# π0.5 PIM 计算阶段的 GPU DQ 时间比例

全部关闭采用 case c（feature layout、GPU RoPE、streaming）。全部开启采用
case g（token layout、PIM RoPE、fused input preload）。case e 仍用
streaming，所以全开对照取历史 case g。新同步时序的全量任务只包含 b、c、d、e。

| PIM 计算期 DQ 指标 | 全关 c | 全开 g | 差值 |
|---|---:|---:|---:|
| WCK 常开，GPU 实际占用（16 通道） | 63.62% | 75.57% | +11.95 个百分点 |
| WCK 常开，未被 PIM burst 占用（16 通道） | 94.60% | 96.49% | +1.89 个百分点 |
| WCK 常开，历史 trace 能安排的 GPU 读 burst（通道 0） | 73.35% | 93.80% | +20.46 个百分点 |
| 每次切 rank 同步，历史 trace 能安排的 GPU 读 burst（通道 0） | 55.15% | 93.41% | +38.25 个百分点 |
| 每次切 rank 同步，再保守屏蔽整个融合 I/O epoch（通道 0） | 55.15% | 92.70% | +37.54 个百分点 |

分母为 QKV、QKᵀ、PV、O、Gate/Up 和 Down 六类 PIM 线性阶段在
18 层 × 10 个去噪步上的总通道周期，含各阶段内部的输入输出。独立 RoPE、
Q 交换、AE 的 GPU 算术和 VLM 尾部不计入分母。16 通道指标按通道周期加权。

读容量估算使用同一批历史 PIM burst 的起止时间。对每个相邻 PIM burst
形成的窗口，在两端扣除与一串 GPU 读的切换保护时间，再按 2 CK 对齐，
统计能放进阶段边界内的完整 GPU 读 burst。每次切 rank 同步时，接读的
最小 DQ 间隔为 19 CK，接写为 11 CK。此口径量化 DQ 上能安排的读容量，
最后一行还把历史融合 I/O epoch 全部视为不可放置 GPU 数据，分母相同。
这是保守口径，因为原生模型允许 GPU 在 I/O epoch 内收到此前已发请求的数据。
实际利用率还取决于 CA、刷新、调度和 GPU 请求是否就绪。

同一固定 trace 上，开启逐次同步使 c 的读容量降低 18.19 个百分点，
使 g 的读容量降低 0.40 个百分点。两者之间的差异反映 g 将 PIM 外部
传输集中到 I/O 窗口后，长 DQ 空隙可摊销进出 rank 的成本。

历史输入为 measurement-host 的 `experiments/2026-11-16/results/combined-{c,g}/repeat-1`，
逐次同步只用于静态 DQ 计算。两组原生 `raw.json` 的 SHA-256 及全精度结果见
同目录 `wck-c-g-dq-comparison.json`。
