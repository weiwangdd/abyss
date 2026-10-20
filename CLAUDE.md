# Abyss

家用硬件上的分层推理系统：把模型权重和 KV cache 按热度分布在显存（VRAM）、内存（DRAM）、NVMe 三层，在单机上跑超出显存容量的大模型。

设计文档：https://claude.ai/code/artifact/9303eb10-5298-4f82-b041-4663290579f1

## 硬件

主机名 `blackwell`（`ssh blackwell`），Ubuntu 24.04.3，内核 7.0.0-38，驱动 580.178.04，CUDA 13.1，Docker 29.7 + nvidia-container-toolkit 1.20。

- GPU：2 × RTX PRO 6000 Blackwell **Max-Q**（300 W，上限 325 W），每张 97,887 MiB；PCIe 5.0 x8 / x8，P2P 可用（PHB，经 CPU），没有 NVLink
- CPU：AMD Ryzen 9 9950X（16 核 32 线程，单 NUMA），主板 ASUS ProArt X870E-CREATOR WIFI（BIOS 1715）
- 内存：4 × 48 GiB DDR5（共 192 GiB，系统可用约 186 GiB），**双通道，2DPC 运行在 3600 MT/s**（条子标称 5600），理论带宽约 57.6 GB/s
- 存储：Samsung 9100 PRO 4TB（PCIe 5.0，系统盘 `/`，剩余 2.1 TB）、Samsung 990 PRO 4TB（PCIe 4.0，`/data`，剩余 3.4 TB）
- 系统：IOMMU 未启用；`ulimit -l` 约 23 GiB（锁页内存上限，方案 A 需要调大）
- 平台上限：AM5 最多 4 × 64 GB = 256 GB 内存

实测带宽（2026-10-20，工具在 blackwell 的 `~/bench`）：

- 内存：纯读约 44 GB/s；STREAM Copy 42、Triad 32 GB/s。8 线程就已饱和
- 主机 → 单卡 28.9 GB/s，单卡 → 主机 28.7 GB/s；**两卡同时从主机读合计只有 43.4 GB/s**，被内存带宽卡住
- 卡间 P2P 单向约 27.7 GB/s，双向合计 104 GB/s

## 目标模型：DeepSeek V4.1 Flash

- 总参数 763B：552B 主干 + 196.6B Engram 表 + 约 14B DSpark 草稿模型
- 40 层，384 个专家，每 token 激活 top-6
- 原生 MXFP4 主干约 268.9 GiB；Engram 表为 FP8，约 189 GiB

## 已有结论

- 原生总量约 474 GiB，超过显存加内存的 383 GiB，所以 Engram 表放 NVMe。
- **方案 A（推荐）**：专家量化到 3.0 bpw EXL3，用 vLLM TP2，预计 55–75 tok/s。公开配方需要约 250 GiB 空闲内存，要么改成从 NVMe 读 Engram，要么加内存。
- **方案 B**：原生 MXFP4，用 mx-llama.cpp；第 0–21 层的专家放 GPU，第 22–39 层放 CPU，预计 12–45 tok/s；按本机双通道 3600 MT/s 估算只有约 12–15 tok/s（不开投机解码），待实测。
- Strata（https://github.com/Niko1221/Strata）只作设计参考，不直接依赖。
