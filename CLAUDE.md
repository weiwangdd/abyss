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
- **方案 B**：原生 MXFP4，用 mx-llama.cpp；第 0–21 层的专家放 GPU，第 22–39 层放 CPU。2026-10-20 本机实测：解码 16.5 tok/s，预填充 307 tok/s（2048 token，ubatch 2048）。2026-10-21 开 `--moe-cache-mib 14000` 后，4.8 万上下文解码从 14.1 提到 25.2 tok/s，短上下文约 29–30 tok/s，预填充不变（约 354 tok/s）。再开 CED 感知预填充（`ABYSS_CED_PREFILL=1`，`mx-llama.cpp/0003`）后，4.8 万 token 预填充从 355 提到约 890 tok/s，对拍 KLD 0.00004。再开 Engram 预取（`ABYSS_ENGRAM_PREFETCH=8`，`mx-llama.cpp/0004`）后，Engram 页缓存冷时的预填充从 91.1 s 降到 54.3 s，热时约 42 s（约 1,140 tok/s）。再让解码器层的注意力也只算子集（`mx-llama.cpp/0005`）后，热时 27.9 s（1,715 tok/s），对拍 KLD 0.00004–0.00007。再开 V4.1 稀疏注意力索引器（`ABYSS_DSV41_INDEXER=1`，`mx-llama.cpp/0006`）后，热时 22.7 s（2,111 tok/s），12.8 万 token 时 51.8 s（2,464 tok/s）。

## 方案 B 运行要点（blackwell）

- 模型在 `~/models/DeepSeek-V4.1-Flash-GGUF`（mxxm-t 的 MXFP4 GGUF，383 GiB），mx-llama.cpp 在 `~/src/mx-llama.cpp`（分支 `abyss-expert-stats`，基于 `c4d12e2`；当前 server 用 `build-idx`（含 0003–0006）；`build-b2` 不含 0006，`build-eng` 不含 0005，`build-ced` 只含 0003，`build-v41` 都不含）
- 必须设 `GGML_CUDA_Q8_1_CACHE=0`：分支的 q8_1 激活缓存按 FIFO 淘汰，和 CUDA VMM 内存池的 LIFO 释放冲突，会触发 `ggml_cuda_pool_vmm::free` 断言
- 必须设 `GGML_CUDA_MMQ_PREC=q8`：Blackwell 上 MXFP4 的 MMQ 默认用 W4A4（激活量化到 4 bit），只影响超过 8 个 token 的批次，也就是预填充。官方参考实现是 W4A8（`inference/model.py` 的 `linear`）。2026-10-21 对拍：W4A4 预填充对解码路径 KLD 0.044、top-1 91%；改 q8 后 KLD 0.012、top-1 95%，和数值底线相当；预填充只慢约 5%
- 对拍的数值底线分两档（同一配置重跑是 0）：改动只落在解码器层（20–39）时 KLD 约 0.0005，超过 0.001 就要查；改动经过编码器层（0–19，包括换内核、关融合、换 ubatch）时 0.005–0.01 属正常。原因是编码器的输出决定了所有位置共享的全局 KV，数值差别会扩散到之后所有位置；解码器层的差别只影响当前 token。2026-10-21 实测：只把第 5 层专家改到 CPU 算，KLD 0.0064；把第 22–39 层专家改到 GPU 算，KLD 0.0005
- 层切分用 `-ts 12/29`；`12/28` 会把 13 层分给 GPU0，放不下计算缓冲区
- 参数：`-ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 -ot "blk\.(2[2-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CPU"`
- 原版对提示词也跑完 40 层。`0003` 补丁让解码器层只对后面的 token 还能通过滑动窗口看到的 token 算 FFN，`0005` 让注意力也只对这些 token 算查询，KV 照常全部写入。每次 `llama_decode` 调用按自己的末尾规划，所以 server 用 `-b 65536`，让整段提示词进一次调用。2026-10-23 测 Engram 占比（开 CED，4.8 万 token，`tools/eval/engram-share.sh`）：页缓存冷时 Engram 约 38.6 s / 91.0 s，热时约 0.4 s / 41.7 s（热时预填充约 1,150 tok/s）；server 启动后第一个大请求另有约 11 s 一次性开销（发一个短请求热身消不掉）。`0004` 的预取把冷读开销从约 39 s 降到约 2 s（`tools/eval/engram-cold.sh`）
- 专家缓存：`--moe-cache-mib 14000`，分支自带，在 GPU1 上为第 22–39 层做 LRU 专家缓存，约 780 个槽位，未命中时上传。KLD 0.0005，GPU1 用到 93.3 / 95.6 GiB。`~/abyss-stats/restart.sh` 已带上，`restart-nocache.sh` 是不带缓存的版本
- DSpark 投机解码在 V4.1 上还不支持
- 稀疏注意力索引器：fork 原本对压缩 KV 做稠密注意力，和官方的 top-512 稀疏选择不一样（压缩比 2 的层超过 1,024 token、压缩比 1 的层超过 512 token 后开始不同）。`0006` 按官方 `inference/model.py` 补上：512 token 以内和稠密版 KLD 为 0；8k / 16k / 32k 时稀疏版 PPL 比稠密版低 0.9% / 3.1% / 0.23%，稀疏对稠密 KLD 0.06–0.09；和 CED 叠加 KLD 0.00004（`tools/eval/d-indexer.sh`）。GPU1 计算缓冲区会涨到约 6.1 GiB，12.8 万 token 时峰值 96,685 / 97,887 MiB，余量约 1.2 GiB
- 工具调用：V4.1 的 DSML 标签名带前导空格（`<｜DSML｜ calls>`、` invoke`、` parameter`，见官方 `encoding/encoding.py`），GGUF 内嵌的是 V4 模板。要用 `mx-llama.cpp/0002` 补丁加 `--chat-template-file mx-llama.cpp/deepseek-v4.1.jinja`
- server 用 `~/abyss-stats/restart.sh` 重启（仓库副本是 `tools/restart-server.sh`，会先备份统计文件）；退路是 `restart-noindexer.sh`（不开索引器）、`restart-noced-attn.sh`（不含 0005）、`restart-noprefetch.sh`（不开 Engram 预取）、`restart-noced.sh`（不开 CED）和 `restart-nocache.sh`（不开 CED 和专家缓存）；不要用 `pkill -f llama-server`，它会匹配到 ssh 命令自身
- 专家热度统计：补丁在 `mx-llama.cpp/`，分析脚本是 `tools/expert_hotness.py`
- 对拍和测速：`tools/eval/`（`kl.sh` 存基准 logits 并比 KLD，`bench.sh` 测速，`prefill-warm.sh` 测 server 热缓存预填充，`profile-prefill.sh` 用 nsys 剖析预填充），复制到 blackwell 的 `~/abyss-tools/eval` 运行，结果在 `~/abyss-eval`；运行前要先停 server
- Strata（https://github.com/Niko1221/Strata）只作设计参考，不直接依赖。
