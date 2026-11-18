# GR00T N1.6 H=15，640×480 单读仿真归档

三路 640×480 原图，实际 ViT 输入 336×252，共 371 token；4 个去噪步。时延单位 ms。

| Case | AE 单跑 | AE 完成 | 端到端 |
|---|---:|---:|---:|
| n16-b | 45.950081 | 45.950081 | 104.579663 |
| n16-c | 45.950081 | 47.083475 | 78.358497 |
| n16-d | 44.045465 | 45.167811 | 77.823912 |
| n16-f | 59.458015 | 60.442819 | 74.409592 |

VLM solo 校准 58.629582 ms，相对 Thor 均值 58.547801 ms，误差 +0.140%。
每个 case 两次 raw.json 逐字节一致；PIM/GPU DQ 守恒、冷 L2 单读配对、GPU→PIM 及 cross K/V 写入保留检查均通过。

## 查看结果

[完整报告](matrix/REPORT.zh-CN.md) · [CSV](matrix/evaluation-b-g.csv) · [复跑及输入校验](matrix/640x480-checks.json) · [本地证据校验](local-evidence-checks.json)

## 归档内容

- simulator-source.tar.gz：仿真源码、默认输入、测试、文档和构建依赖源码。
- simulator-runs.tar.gz：本次完整输入程序、AE 时间表、VLM trace、7 组校准、3 种 AE solo、8 份完整原始结果和来源记录，以及 π0.5 回归汇总。
- thor-captures.tar.gz：640×480 的原始计时、逐 kernel trace、NCU 读写计数、AE 非线性测量、冷 L2 消费算子、配置与权重索引。
- simulator-runtime.tar.gz：本次实际使用的原生二进制和 libramulator.so。
- ARCHIVE_MANIFEST.json：每个归档成员与原文件逐字节核验后的 SHA256、输入条件和运行版本。
- SHA256SUMS：整个归档及可直接查看文件的完整性校验清单。

GPU 权重由锁定 revision 和 checkpoint-sha256.txt 定位；仿真回放直接使用已采集的时间和访存数据。

## 解包与复跑

四个 tar.gz 内部路径均相对于原 GR00T N1.6 工程根目录，解压到同一个新目录即可还原本次源码、数据和运行库。
原路径为：GR00T N1.6。迁移目录时，将运行配置和 calibration JSON 中的这一绝对路径前缀替换为解压目录；保留原始 raw.json 和输入数据内容，校验记录按原文件保留。
使用实验 README 中的 --program-dir、--input-dir 和 --calibration 参数重新生成新输出目录。若二进制找不到运行库，将解压目录的 ramulator2 加入 LD_LIBRARY_PATH。

## 测量范围

当前采集使用 packed SDPA。GR00T VLM compile/eager 输出约有 11% RMS 差异；这些结果用于当前实现的性能仿真。
12 处 SigLIP LayerNorm 的计数使用显式线程分块代理，编译代码及说明保存在 Thor 采集中；CUPTI 时间保留，其余启动身份和网格严格核对。
