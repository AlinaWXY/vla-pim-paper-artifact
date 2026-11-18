# Rank 1 分批列命令与完整 AE/VLM 对照

顶层 MC 发一条占 2 CK 的数据行 ACTAB。同步组在同一拍激活 16 bank。四批组在 0、16、32、48 CK 依次激活四个 bank，每批从每个 BG 取同一 BA，内部脉冲进入共享 tFAW 历史。

两个实验组共用分批列命令协议。rank 1 在 PIM mode 以四个 mask `0x1111`、`0x2222`、`0x4444`、`0x8888` 发 MAC、QK、RoPE 阵列读和整 rank 写回。每个子命令占 1 CK 外部 CA，检查目标 bank 的恢复时序以及共享时序。子命令使用原列操作码，mask 表示本实验的 PIM 扩展选择语义。QK 的 score 写回继续使用 `0x0011`。

每个逻辑操作保持 FIFO 队首，直到所有子命令已发出。QK 归约等最后一个 bank 的部分和完成。目标 bank 就绪且输入或输出数据到位后，MC 可以在后续激活脉冲之前先发对应的列命令。原生短跑验证第一批 MAC 最早为 ACTAB 后 20 CK，score 写回最早为 9 CK。DRAM mode 的 Host RD/WR 继续普通 bank 寻址和 DQ 传输。SRF 输入和 Q exchange 继续使用原协议。

## 完整负载

每种配置执行 18 层 × 10 步 AE 的完整调度和 2,357 个 VLM 单元。b 为串行基准，c–g 使用顶层 MC 原生并发。VLM 沿用均匀释放参数，alpha 为 1、窗口为 1024，写水位为 768/256。MAC 频率倍率为 2。

f/g 显式执行 PIM 的真实 I/O，并采用同一套 GPU 计账规则处理 profile 已覆盖的共享张量搬运。预算取此前同步激活对照中两个切片的共享 I/O 区间平均值，冻结后用于两组的全部 180 个对应层。QKV、Gate/Up 和 Down 在同层 GPU 总预算内摊销，保留 GPU kernel 边界。其余共享 I/O 在相邻 GPU 块摊销。Q exchange 计入原生路径。这是用固定预算处理重叠访存的估计，预算来自已有 trace，GPU profile 未重新测量。

作业使用相同二进制、PIM 程序、AE 调度和 VLM profile。通过 `tools/run_full_mc_actab.py` 生成输入及 SHA-256 清单。Slurm `2040022` 执行 b–f，`2040270` 执行 g，`2040271` 审计并归档。

## VLM GPU 交接

原生并发运行的配置可加入 `"vlm_gpu_quantum_us": 2`，也可在 `experiments/run.py` 使用 `--vlm-gpu-quantum-us 2`。这个配置模拟 VLM kernel 每执行 2 µs 到达一个可以中断的边界。AE 需要 GPU 时，VLM 在下一边界停止向 MC 提交请求，排空已接收的请求后让出 GPU。AE 的 GPU 算子结束后，VLM 从原来的请求序号继续执行。AE 的 PIM 阶段仍与 VLM 并发。边界间隔以外的排空时间也计入 AE 等待。

原始 VLM kernel 顺序、地址、流量和 profile 时间保持一致。`paused_us` 记录每个 kernel 的排空与暂停区间，`VLM_GPU_yields` 和 `AE_GPU_wait_max_us` 记录交接次数与最长等待。省略该配置时沿用原有的整 kernel 交接。此参数模拟可中断的执行边界，实测 GPU 上的划分和恢复开销需要通过新的 GPU profile 建模。
