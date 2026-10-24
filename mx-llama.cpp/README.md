# mx-llama.cpp 补丁

给 [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) 的补丁，基于提交 `c4d12e2`（2026-10-15）：

| 文件 | 作用 |
|---|---|
| `abyss-expert-stats.{h,cpp}` + `0001-hook-expert-stats.patch` | `llama-server` 的专家命中统计 |
| `0002-deepseek-v41-dsml-tags.patch` | 解析 V4.1 的工具调用标签（`<｜DSML｜ calls>` 等，标签名带前导空格） |
| `deepseek-v4.1.jinja` | V4.1 对话模板：GGUF 内嵌的是 V4 的模板，工具调用标签是 V4 写法 |
| `0003-ced-prefill.patch` | CED 感知预填充：解码器层只对后面的 token 还能通过滑动窗口看到的那些 token 算 FFN（`ABYSS_CED_PREFILL=1`） |
| `0004-engram-prefetch.patch` | Engram 预取：`llama_decode` 拿到整批后，后台线程按哈希算出的行号对表做 `MADV_WILLNEED`（`ABYSS_ENGRAM_PREFETCH=<线程数>`） |
| `0005-ced-attention.patch` | CED 感知预填充第二步：解码器层的注意力也只对同一子集的 token 计算查询，KV 照常全部写入 |
| `0006-dsv41-indexer.patch` | V4.1 稀疏注意力索引器：每个查询只看索引源选出的 512 个压缩位置（`ABYSS_DSV41_INDEXER=1`） |
| `abyss-dump/` + `0007-abyss-dump.patch` | `llama-abyss-dump`：跑一段提示词，把名字匹配正则的计算图张量存成文件，给 `tools/eval/indexer_oracle.py` 对拍用 |
| `0008-dsv41-decode-gather.patch` | 单 token 解码时，压缩注意力层先取出索引器选中的 top-k 行，再和滑动窗口拼接，不再每层拼接整个压缩 KV（`ABYSS_DSV41_DECODE_GATHER=1`，要和 `ABYSS_DSV41_INDEXER=1` 一起开） |
| `0009-moe-cpu-miss.patch` | 专家缓存的未命中，一部分按热度挑出来上传进缓存，其余交给 CPU 算，两边同时进行（`ABYSS_MOE_CPU_MISS=1`） |

## 安装

```bash
cd mx-llama.cpp
cp <abyss>/mx-llama.cpp/abyss-expert-stats.{h,cpp} tools/server/
git apply <abyss>/mx-llama.cpp/0001-hook-expert-stats.patch <abyss>/mx-llama.cpp/0002-deepseek-v41-dsml-tags.patch \
  <abyss>/mx-llama.cpp/0003-ced-prefill.patch <abyss>/mx-llama.cpp/0004-engram-prefetch.patch \
  <abyss>/mx-llama.cpp/0005-ced-attention.patch <abyss>/mx-llama.cpp/0006-dsv41-indexer.patch
cp -r <abyss>/mx-llama.cpp/abyss-dump examples/
git apply <abyss>/mx-llama.cpp/0007-abyss-dump.patch <abyss>/mx-llama.cpp/0008-dsv41-decode-gather.patch \
  <abyss>/mx-llama.cpp/0009-moe-cpu-miss.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server
cmake --build build -j --target llama-abyss-dump   # 对拍工具，可选
```

## 使用

```bash
ABYSS_EXPERT_STATS=~/expert-stats.json llama-server ... -np 1 \
  --jinja --chat-template-file <abyss>/mx-llama.cpp/deepseek-v4.1.jinja
python3 tools/expert_hotness.py ~/expert-stats.json
```

- 工具调用必须同时用 `0002` 补丁和 `deepseek-v4.1.jinja`：解析器只在模板里出现 `' calls>` 时才按 V4.1 标签解析。
- 不设 `ABYSS_EXPERT_STATS` 时不挂回调，行为和原版一致。
- 统计文件已存在时，启动后接着累加，不会覆盖。
- 统计按 ubatch 的 token 数区分阶段：1 个 token 算解码，多个算预填充。要用 `-np 1`，否则多路并发的解码会被算成预填充。
- `ABYSS_EXPERT_STATS_FLUSH_SEC` 控制写文件间隔，默认 30 秒；只在有请求时写，进程正常退出时也会写一次。
- 回调会让调度器在每层的路由结果处同步一次，速度会略降。测速时不要开。
- `ABYSS_CED_PREFILL=1` 打开 CED 感知预填充，只对 DeepSeek V4.1 生效。每次 `llama_decode` 调用按自己的末尾规划，所以 server 要设大 `-b`（比如 65536），让整段提示词进一次调用才省得多。`ABYSS_CED_MARGIN`（默认 256）是末尾额外保持有效的位置数。`0003` 跳过 FFN，`0005` 再跳过注意力的查询，KV 的写入始终不变。
- `ABYSS_DSV41_INDEXER=1` 打开 V4.1 的稀疏注意力索引器：KV 源层写索引键（第 2、8、14 层用已有的索引键 cache，第 20 层用新增的 `kv_lidp`），索引源层打分选 top-512，第 20 层为后面的压缩比 1 索引源挑 2,048 个候选块，其余层沿用最近索引源的 top-k。DSV4 状态格式版本因此升到 2
- `llama-abyss-dump` 由 `ABYSS_DUMP_RE`（张量名正则，名字带 `-<层号>` 后缀）和 `ABYSS_DUMP_DIR` 控制，整段提示词作为一批送入，每个 ubatch 存一个 `<名字>.<序号>.bin`。设 `ABYSS_DUMP_GEN=N` 时不存提示词，而是之后贪心解码 N 个 token，只存这些单 token 的计算。`0007` 给索引键的输入起名 `idx_latent`，`0008` 给压缩注意力的查询起名 `hca_q`，都不影响计算
- `ABYSS_DSV41_DECODE_GATHER=1` 只在单 token、开了索引器、压缩行多于 top-k 时生效。取出的行补齐到 256 的倍数（补的行掩码为 -inf），走普通的 flash attention；和原来的掩码收集是同一组 KV，差别只在 F16 舍入
- `ABYSS_MOE_CPU_MISS=1` 只对专家缓存覆盖的层、32 个 token 以内的批次（也就是解码）生效，目前只支持 DeepSeek V4 的 SwiGLU 专家。每层先由一个 CPU 算子规划：命中的照常读缓存；未命中的按衰减热度挑出 `ABYSS_MOE_ADMIT`（默认 0.5）比例异步上传进缓存，其余由 CPU 用主机内存里的原始专家计算，GPU 那边读全零的备用槽位，最后两边相加。热度每层每步乘 `ABYSS_MOE_HEAT_DECAY`（默认 0.99）。CPU 的 `mul_mat_id` 遇到 id < 0 会跳过并把输出清零。server 退出时在 stderr 打印 `abyss: moe cache` 统计
- `ABYSS_ENGRAM_PREFETCH=8` 打开 Engram 预取，只对 DeepSeek V4.1、单序列、64 个 token 以上的批次生效。只做预读，不改计算结果；新批次到来时会叫停上一批还没做完的预取。
