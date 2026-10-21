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
- **方案 B**：原生 MXFP4，用 mx-llama.cpp；第 0–21 层的专家放 GPU，第 22–39 层放 CPU。2026-10-20 本机实测：解码 16.5 tok/s，预填充 307 tok/s（2048 token，ubatch 2048）。

## 方案 B 运行要点（blackwell）

- 模型在 `~/models/DeepSeek-V4.1-Flash-GGUF`（mxxm-t 的 MXFP4 GGUF，383 GiB），mx-llama.cpp 在 `~/src/mx-llama.cpp`（分支 `abyss-expert-stats`，基于 `c4d12e2`）
- 必须设 `GGML_CUDA_Q8_1_CACHE=0`：分支的 q8_1 激活缓存按 FIFO 淘汰，和 CUDA VMM 内存池的 LIFO 释放冲突，会触发 `ggml_cuda_pool_vmm::free` 断言
- 必须设 `GGML_CUDA_MMQ_PREC=q8`：Blackwell 上 MXFP4 的 MMQ 默认用 W4A4（激活量化到 4 bit），只影响超过 8 个 token 的批次，也就是预填充。官方参考实现是 W4A8（`inference/model.py` 的 `linear`）。2026-10-21 对拍：W4A4 预填充对解码路径 KLD 0.044、top-1 91%；改 q8 后 KLD 0.012、top-1 95%，和数值底线相当；预填充只慢约 5%
- 对拍的数值底线约 KLD 0.01（中位数约 0.005）：任何改变舍入顺序的改动（换内核、关融合、换 ubatch）都会到这个量级，同一配置重跑是 0。判断是否有问题要看是否明显超过这个底线
- 层切分用 `-ts 12/29`；`12/28` 会把 13 层分给 GPU0，放不下计算缓冲区
- 参数：`-ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 -ot "blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU"`
- 这个分支对提示词也跑完 40 层（没有实现 CED 的只走编码器），所以预填充同样要经过 CPU 专家；Engram 随机读约占预填充时间的 37%
- DSpark 投机解码在 V4.1 上还不支持
- 工具调用：V4.1 的 DSML 标签名带前导空格（`<｜DSML｜ calls>`、` invoke`、` parameter`，见官方 `encoding/encoding.py`），GGUF 内嵌的是 V4 模板。要用 `mx-llama.cpp/0002` 补丁加 `--chat-template-file mx-llama.cpp/deepseek-v4.1.jinja`
- server 用 `~/abyss-stats/restart.sh` 重启（仓库副本是 `tools/restart-server.sh`；编译产物在 `build-v41`，会先备份统计文件）；不要用 `pkill -f llama-server`，它会匹配到 ssh 命令自身
- 专家热度统计：补丁在 `mx-llama.cpp/`，分析脚本是 `tools/expert_hotness.py`
- 对拍和测速：`tools/eval/`（`kl.sh` 存基准 logits 并比 KLD，`bench.sh` 测速），复制到 blackwell 的 `~/abyss-tools/eval` 运行，结果在 `~/abyss-eval`；运行前要先停 server
- Strata（https://github.com/Niko1221/Strata）只作设计参考，不直接依赖。
