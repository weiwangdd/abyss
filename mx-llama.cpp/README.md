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
| `0010-fix-ced-inputs-m2-small-batches.patch` | 修两处崩溃：`0006` 的 CED 行映射在计算图用不到时没有分配，提示词约 2,500–16,000 token 时断言失败；`0009` 在 9–32 个 token 的批次里，同一 token 的多个 CPU 未命中指向同一个全零槽位，CUDA MMQ 越界读 |
| `0011-fix-candidate-input.patch` | 修 `0006` 的同类崩溃：CED 让第 24–36 层在某个 ubatch 里没有 token 时，第 20 层的候选块被整段丢掉，它的输入没有分配；提示词超过约 1.7 万 token、长度除以 2,048 余约 114–622 时触发 |
| `0012-prompt-cache-usable-tokens.patch` | server 的 prompt 缓存按「能直接用上的 token 数」挑选，内存缓存里的长对话能换回槽位；只记一个位置的检查点（DeepSeek V4）在分叉点之前就能用，不再要求早 128 个 token |
| `0013-sched-reserve-size-reset.patch` | 修 `ggml_backend_sched_reserve_size`：它在切分计算图之前重置调度器，清掉了构建计算图时手动指定的后端，按错误的切分估算计算缓冲区 |
| `0014-moe-big-split.patch` | 专家缓存覆盖的层在大批次（超过 32 个 token）时，按规划把专家分成两路同时计算：分到 token 少的交给 CPU（紧凑批次），其余照常上传到 GPU0（`ABYSS_MOE_BIG_SPLIT=1`） |

## 安装

```bash
cd mx-llama.cpp
cp <abyss>/mx-llama.cpp/abyss-expert-stats.{h,cpp} tools/server/
git apply <abyss>/mx-llama.cpp/0001-hook-expert-stats.patch <abyss>/mx-llama.cpp/0002-deepseek-v41-dsml-tags.patch \
  <abyss>/mx-llama.cpp/0003-ced-prefill.patch <abyss>/mx-llama.cpp/0004-engram-prefetch.patch \
  <abyss>/mx-llama.cpp/0005-ced-attention.patch <abyss>/mx-llama.cpp/0006-dsv41-indexer.patch
cp -r <abyss>/mx-llama.cpp/abyss-dump examples/
git apply <abyss>/mx-llama.cpp/0007-abyss-dump.patch <abyss>/mx-llama.cpp/0008-dsv41-decode-gather.patch \
  <abyss>/mx-llama.cpp/0009-moe-cpu-miss.patch <abyss>/mx-llama.cpp/0010-fix-ced-inputs-m2-small-batches.patch \
  <abyss>/mx-llama.cpp/0011-fix-candidate-input.patch <abyss>/mx-llama.cpp/0012-prompt-cache-usable-tokens.patch \
  <abyss>/mx-llama.cpp/0013-sched-reserve-size-reset.patch <abyss>/mx-llama.cpp/0014-moe-big-split.patch
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
- `ABYSS_MOE_CPU_MISS=1` 只对专家缓存覆盖的层、32 个 token 以内的批次（也就是解码）生效，目前只支持 DeepSeek V4 的 SwiGLU 专家。每层先由一个 CPU 算子规划：命中的照常读缓存；未命中的按衰减热度挑出 `ABYSS_MOE_ADMIT`（默认 0.5）比例异步上传进缓存，其余由 CPU 用主机内存里的原始专家计算，GPU 那边读全零的备用槽位（每个 token 的第 k 个专家用第 k 个，同一 token 里不能重复，否则 CUDA MMQ 会越界），最后两边相加。CPU 这一路固定在 CPU 上，32 个 token 的批次也不会被卸载到 GPU。热度每层每步乘 `ABYSS_MOE_HEAT_DECAY`（默认 0.99）。CPU 的 `mul_mat_id` 遇到 id < 0 会跳过并把输出清零。server 退出时在 stderr 打印 `abyss: moe cache` 统计
- `0012` 改的是 server 的 prompt 缓存：新请求到来时，比较当前槽位和内存缓存里各条目能直接用上的 token 数（检查点受限的缓存取不晚于分叉点的最近检查点），缓存里的条目多出 256 个以上就换进来；只用到条目一小部分时复制一份，原条目留在缓存里。智能体的辅助请求（共享系统提示、只问一句）因此只算几十个 token，也不会把主对话冲掉。`-lv 4` 能看到检查点和缓存的换入换出
- `ABYSS_MOE_BIG_SPLIT=1` 只对专家缓存覆盖的层、超过 32 个 token 的批次生效。每层一个 CPU 算子统计每个专家的 token 数，按 token 数从少到多把专家交给 CPU，直到 CPU 的预计时间追上其余专家的上传时间（`ABYSS_MOE_BIG_UPLOAD_US`、`ABYSS_MOE_BIG_CPU_EXPERT_US`、`ABYSS_MOE_BIG_CPU_TOKEN_US`，默认 650 / 450 / 50 微秒），而且这些「token × 专家」不超过 `ABYSS_MOE_BIG_CPU_ROWS`（默认 1,024）行。GPU 那一路的这些条目为 -1：CUDA MMQ 跳过它们并把输出清零，上传回调不传它们，而且直接用规划算子在主机上的结果，不必从 GPU 读回 id。CPU 那一路在 GPU1 上取出这些 token，在 CPU 上按一行一个专家计算，再用 `set_rows` 写回、和 GPU0 的结果相加；合并固定在 0 号后端（GPU0）。退出时在 stderr 打印 `abyss: moe cache large batches` 统计
- `ABYSS_ENGRAM_PREFETCH=8` 打开 Engram 预取，只对 DeepSeek V4.1、单序列、64 个 token 以上的批次生效。只做预读，不改计算结果；新批次到来时会叫停上一批还没做完的预取。
