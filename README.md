# Abyss

家用硬件上的分层推理系统：把模型权重和 KV cache 分布在显存（VRAM）、内存（DRAM）和 NVMe 三层，在单机上跑超出显存容量的大模型。

## 目标

- 目标模型：DeepSeek V4.1 Flash
- 在家用/工作站级硬件上实现可用的本地推理，按访问热度在 VRAM / DRAM / NVMe 之间调度权重与 KV cache

## 硬件

- GPU：2 × NVIDIA RTX PRO 6000 Blackwell Max-Q（每张 95.6 GiB，共约 191 GiB 显存），PCIe 5.0 x8 / x8，无 NVLink
- CPU：AMD Ryzen 9 9950X（16 核）
- 内存：4 × 48 GiB DDR5（共 192 GiB，系统可用约 186 GiB），双通道，实测纯读带宽约 44 GB/s
- 存储：Samsung 9100 PRO 4TB（PCIe 5.0）+ 990 PRO 4TB（PCIe 4.0），作冷数据层

详细配置和实测带宽见 [CLAUDE.md](CLAUDE.md)。

## 设计文档

- [Abyss 设计文档](https://claude.ai/code/artifact/9303eb10-5298-4f82-b041-4663290579f1)
