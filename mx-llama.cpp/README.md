# mx-llama.cpp 补丁

给 [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) 的补丁，基于提交 `c4d12e2`（2026-10-15）：

| 文件 | 作用 |
|---|---|
| `abyss-expert-stats.{h,cpp}` + `0001-hook-expert-stats.patch` | `llama-server` 的专家命中统计（经 `0026` 的 `cb_moe_ids` 读选择结果，要和它一起用） |
| `0002-deepseek-v41-dsml-tags.patch` | 解析 V4.1 的工具调用标签（`<｜DSML｜ calls>` 等，标签名带前导空格） |
| `deepseek-v4.1.jinja` | V4.1 对话模板，按官方 `encoding/encoding.py` 逐字对齐：GGUF 内嵌的是 V4 的模板，工具调用标签是 V4 写法，也没有 `<｜System｜>` 和思考强度行 |
| `0003-ced-prefill.patch` | CED 感知预填充：解码器层只对后面的 token 还能通过滑动窗口看到的那些 token 算 FFN（`ABYSS_CED_PREFILL=1`） |
| `0004-engram-prefetch.patch` | Engram 预取：`llama_decode` 拿到整批后，后台线程按哈希算出的行号对表做 `MADV_WILLNEED`（`ABYSS_ENGRAM_PREFETCH=<线程数>`） |
| `0005-ced-attention.patch` | CED 感知预填充第二步：解码器层的注意力也只对同一子集的 token 计算查询，KV 照常全部写入 |
| `0006-dsv41-indexer.patch` | V4.1 稀疏注意力索引器：每个查询只看索引源选出的 512 个压缩位置（`ABYSS_DSV41_INDEXER=1`） |
| `abyss-dump/` + `0007-abyss-dump.patch` | `llama-abyss-dump`：跑一段提示词，把名字匹配正则的计算图张量存成文件，给 `tools/eval/indexer_oracle.py` 对拍用；`ABYSS_DUMP_GEN=N` 时只导出之后逐个解码的 N 个 token，默认贪心，`ABYSS_DUMP_TEMP` 按温度采样（`tools/eval/expert-trace.sh` 用它导出路由）；`ABYSS_DUMP_FORCE` 逐个强制喂一段文字，记下每个 token 的负对数似然（`tools/eval/defer-ppl.sh` 用它测解码路径的 PPL） |
| `0008-dsv41-decode-gather.patch` | 单 token 解码时，压缩注意力层先取出索引器选中的 top-k 行，再和滑动窗口拼接，不再每层拼接整个压缩 KV（`ABYSS_DSV41_DECODE_GATHER=1`，要和 `ABYSS_DSV41_INDEXER=1` 一起开） |
| `0009-moe-cpu-miss.patch` | 专家缓存的未命中，一部分按热度挑出来上传进缓存，其余交给 CPU 算，两边同时进行（`ABYSS_MOE_CPU_MISS=1`） |
| `0010-fix-ced-inputs-m2-small-batches.patch` | 修两处崩溃：`0006` 的 CED 行映射在计算图用不到时没有分配，提示词约 2,500–16,000 token 时断言失败；`0009` 在 9–32 个 token 的批次里，同一 token 的多个 CPU 未命中指向同一个全零槽位，CUDA MMQ 越界读 |
| `0011-fix-candidate-input.patch` | 修 `0006` 的同类崩溃：CED 让第 24–36 层在某个 ubatch 里没有 token 时，第 20 层的候选块被整段丢掉，它的输入没有分配；提示词超过约 1.7 万 token、长度除以 2,048 余约 114–622 时触发 |
| `0012-prompt-cache-usable-tokens.patch` | server 的 prompt 缓存按「能直接用上的 token 数」挑选，内存缓存里的长对话能换回槽位；只记一个位置的检查点（DeepSeek V4）在分叉点之前就能用，不再要求早 128 个 token |
| `0013-sched-reserve-size-reset.patch` | 修 `ggml_backend_sched_reserve_size`：它在切分计算图之前重置调度器，清掉了构建计算图时手动指定的后端，按错误的切分估算计算缓冲区 |
| `0014-moe-big-split.patch` | 专家缓存覆盖的层在大批次（超过 32 个 token）时，按规划把专家分成两路同时计算：分到 token 少的交给 CPU（紧凑批次），其余照常上传到 GPU0（`ABYSS_MOE_BIG_SPLIT=1`） |
| `0015-prefill-tail-one-pass.patch` | 提示词末段尽量一遍算完：server 不再在最后 T 个 token 里按用户消息开头和「末尾前 4 + n_ubatch」拆调用（`ABYSS_CKPT_TAIL_MERGE=T`，生产用 2700；末尾前 4 个 token 的检查点和第一个检查点照旧）；一次调用的余数放进第一个 ubatch，最后一个 ubatch 是满的（`ABYSS_CED_ALIGN=1`）；CED 末尾的余量默认等于 `n_rs_seq`（没开 DSpark 时是 0），`ABYSS_CED_MARGIN` 仍可覆盖 |
| `0016-decode-stats.patch` | 解码统计：`ABYSS_DECODE_STATS=N` 时每 N 次单 token 解码打印一行，含每 token 耗时、专家缓存命中率、每 token 的命中、上传和 CPU 计算的专家数，以及每 token 的磁盘缺页次数（Engram 表是 mmap 的，冷读就是缺页）；只统计解码调用本身，不含两次请求之间的预填充 |
| `0017-slot-save-thin-checkpoints.patch` | 槽位被只共享开头一段的请求占用前先存进内存缓存：DSV4 只能退回检查点，新请求按检查点能保留的比槽位上次提示词（最新检查点）少 `ABYSS_SLOT_SAVE_LOST` 个 token 以上时先存（默认 2048，0 = 关）；原来只按共同前缀算，保留不到一半才存。检查点列表满时按 `ABYSS_CKPT_THIN_STEP` 的间距删减（不设时同 `--checkpoint-min-step`），和长提示词拆分建检查点的步长分开 |
| `0018-plain-request-text.patch` | 对话提示词按官方 DeepSeek Recipe 的方式编码（`ABYSS_PLAIN_INPUT=1`）：只有模板自己写的控制 token 按 ID 编码，来自请求的文字（消息、思考、工具调用和结果、工具定义）只用基础词表编码，里面出现的 `<｜User｜>`、`</think>`、`｜DSML｜` 等字样都是普通文字。原来整段提示词按「识别特殊 token」编码，`<think>`、`</think>`、`｜DSML｜` 在 GGUF 里是 user-defined 类型，关掉识别也照样匹配。Jinja 引擎本来就标记每段字符串是否来自请求，补丁补上 `tojson`、数组 `join` 的标记传递，BOS/EOS 改回模板文字，把渲染结果里来自请求的字节范围交给 server；`llama_tokenize_plain` 只用基础词表编码。另有 `llama-abyss-chat-tokens`：只读词表，打印 server 对一组请求生成的提示词、模板的控制 token 和两种编码结果 |
| `0019-graph-memory.patch` | 计算图显存的两处浪费，加一个排查工具。`0014` 大批次分流里每层的全零张量原来是对一个新张量做 fill，那个张量是计算图的叶子，分配器在图开始前就给所有叶子分配好，第 22–39 层 18 个 240 MiB 从头占到尾（GPU0 约 2.6 GB）；现在从一行重复展开，用到时才分配。CED 下没有查询行的注意力层原来照样算全部 token 的查询并放进图里，没人读它，分配器就一直不释放（17 层 × 256 MiB）；现在这些层不算查询，预填充快 3–5%。`GGML_ALLOC_TRACE=1` 在 stderr 打印每次计算图分配的分配、释放和各缓冲区大小，以及 CUDA 内存池的每次增长和当时的算子，`tools/eval/alloc_peak.py` 据此列出缓冲区峰值时有哪些张量 |
| `0020-indexer-chunks.patch` | V4.1 预填充的索引器打分、选 top-k 和压缩注意力按查询分块，每块 N 个（`ABYSS_DSV41_IDX_CHUNK=N`，生产用 256）：分数和 top-k 掩码只存在一块，掩码每块的视图每张图只建一次、各层共用（每层各建一个的话，调度器会给每个视图往 GPU 拷一份）。19.2 万上下文时 GPU1 计算缓冲区 6.9 → 3.2 GB，大段预填充慢 4–8%；单 token 解码不分块 |
| `0021-bench-moe-cache.patch` | `llama-bench` 从环境变量 `LLAMA_ARG_MOE_CACHE_MIB` 读专家缓存大小（server 的 `--moe-cache-mib` 也读它），不开缓存时解码掉回约 16 tok/s，测出来不代表实际配置；`tools/eval/std-bench.sh` 用它 |
| `0022-big-batch-stats.patch` | 大批次统计（`ABYSS_MOE_BIG_STATS=1`）：专家缓存覆盖的层每次规划大批次（超过 32 个 token）时，在 stderr 打印一行 `abyss: big batch`：层号、token 数、用到的专家数、其中在缓存里的专家数和挑选数、交给 CPU 的专家数和其中在缓存里的。只读，不改计算；`tools/eval/big_coverage.py` 按请求汇总 |
| `0023-big-batch-cache.patch` | 大批次用缓存（`ABYSS_MOE_BIG_CACHE=1`）：专家缓存覆盖的层规划大批次时，缓存里有的专家记下槽位，在缓存所在的卡上直接算（gate、up、激活、down，加权求和），其余的照 `0014` 分给 CPU 和上传；三路同时进行，最后相加。原来缓存只能复制给它所在的卡，大批次在第一张卡上算，用不上缓存。规划结果末尾多一列槽位号，统计行多报从缓存算的专家数 |
| `0024-moe-prefetch.patch` | 专家预取（`ABYSS_MOE_PREFETCH=K`，只用于测试，没有上线）：单 token 解码时，专家缓存覆盖的层用下一层的路由门作用在本层的路由器输入上选 top K（和路由同样建图，CUDA 照样融合），缓存里没有的挤掉最久没用、这一步也没用到的槽位，用缓存所在卡上第二个 CUDA backend 的流提前上传，每层最多 `ABYSS_MOE_PREFETCH_MAX` 个；下一层的规划让主流等它的事件。主机到显卡的拷贝按提交顺序排队、不分流，所以调度器加了启动回调（`ggml_backend_sched_set_launch_callback`），预取等这一层的计算提交以后才发，不挡住它的输入拷贝。解码统计行多报每 token 预取数和被读到的比例。实测命中率 78% → 82–85%，解码速度不变（见 CLAUDE.md「专家预取」）；不设开关时和原来一样 |
| `0025-moe-defer.patch` | 专家延后（`ABYSS_MOE_DEFER=1`，生产已开）：单 token 解码时，专家缓存覆盖的层交给 CPU 算的专家不并进本层的 FFN 输出，它们的加权和在下一层注意力的输出处、按本层 FFN 的 post 系数加进残差（下一层的注意力没看到这部分，残差本身是精确的）；图的顺序让 GPU 先算本层缓存命中的专家和下一层的注意力，CPU 同时算未命中，规划算子顺带把 FFN 输入拷到 CPU。调度器加了开关 `ggml_backend_sched_set_async_host_inputs`：GPU 段从 CPU 段来的小输入（≤ 1 MiB）走锁页暂存区在 GPU 的流上异步拷，紧跟 GPU 段、没有输入的 CPU 段不再同步，有逐节点回调（专家统计）时回调不读的段也不同步。会话解码 +11%，解码路径 PPL +0.06% ± 0.14%（见 CLAUDE.md「专家延后」）；不设开关时和原来一样 |
| `0026-moe-ids-stats.patch` | 专家统计不再切图：`llama_context_params`（和 `common_params`）加了 `cb_moe_ids`。设了它，`build_moe_ffn` 在融合的路由节点之后给每层的选择结果加一个 `cont` 输出，每个 ubatch 算完和 logits 一起异步拷进锁页内存，在下一次 `synchronize()`（读 logits 时）交给回调。`abyss-expert-stats` 改用它，不再装逐节点回调：原来的回调在每层的 top-k 处切开计算图，路由的融合内核用不上，每层还多一次同步和一次 CUDA graph 启动；而且按名字前缀匹配，把专家缓存的规划对选择结果做的 reshape、cont 视图也算了进去，第 22–39 层的计数是错的。生产设置下会话解码 +7.8%（见 CLAUDE.md「专家统计不再切图」） |
| `0027-hc-ksplit.patch` | 超连接混合系数的 K 切分（`ABYSS_HC_KSPLIT=1`，生产已开）：CUDA 后端把「`rms_norm` 后接以它为输入的 `mul_mat`」（单列、F32、≤ 64 行、≥ 4,096 列，也就是 DSV4 每个子层的超连接混合系数，24 行 × 20,480 列）换成两个内核：每块算（行, 2,048 列一段）的点积和这段的平方和，第二个内核按固定顺序相加再乘 `rsqrt(mean(x²)+eps)`，结果和调度无关。每次约 12.3 → 3.3 µs，每 token 80 次，会话解码 +4.5%；`test-backend-ops` 加了 `RMS_NORM_MUL_MAT` 用例。不设开关时和原来一样 |
| `0028-engram-decode-prefetch.patch` | 解码的 Engram 预读（`ABYSS_ENGRAM_DECODE_PREFETCH=1`，生产已开）：少于 64 个 token 的批次（解码），Engram 输入一算出行号就对这些行发 `MADV_WILLNEED`，在计算图启动之前，计算图的 CPU `get_rows` 不再逐行等缺页（每个冷行约 86 µs）；64 个以上的照旧由 `0004` 预取。剖析里 Engram 前的 GPU 空等每 token 1.24 → 0.49 ms，输出不变；剩下的主要是 NVMe 的 APST（见 CLAUDE.md「解码的 Engram 预读」） |
| `0029-dist-vec.patch` | CPU 采样向量化（`ABYSS_DIST_VEC=1`，生产已开）：`dist` 采样把 logit 减最大值后拷进连续缓冲区，用能自动向量化的多项式 exp（`target_clones` 出 AVX-512、AVX2 和通用三份）算分子，每 1,024 个一块求和，按块找到目标再在块内找，归一化改乘倒数；随机数和选法不变。129,280 个 token 的采样 0.411 → 0.117 ms，固定种子 2,000 次选中的 token 全同 |
| `0030-moe-cache-gate.patch` | 专家缓存的准入门和淘汰（生产已开）：小批次的规划可以淘汰衰减热度最低的槽位，不再按最久没用（`ABYSS_MOE_EVICT=cold`，这次规划读到或刚放进的不挤）；未命中比要挤掉的那个热才上传进缓存，否则交给 CPU（`ABYSS_MOE_ADMIT_GATE=1`）；单 token 时 CPU 算的专家槽位号填 −1，`mul_mat_vec_q` 对负 id 整块写 0、不读零槽位（`ABYSS_MOE_SKIP_ZERO=1`）。生产设置下每 token 上传 6.9 → 5.2，会话解码 +3.2%，KL 6e-4（交给 CPU、被延后的专家变了）；零槽位跳过本身输出不变（见 CLAUDE.md「缓存的准入门和淘汰」） |
| `0031-moe-defer-upload.patch` | 上传的专家也延后（`ABYSS_MOE_DEFER_UPLOAD=1`，只用于测试，没有上线）：单 token 的 CPU 专家延后到下一层时（`ABYSS_MOE_DEFER`），规划准入的第一个未命中不在规划里上传，命中那段计算里它的槽位号填 −1，CPU 也不算它；启动回调等命中那段计算提交后（它的小拷贝已经排进队列），把它放到缓存所在卡上第二个 CUDA backend 的流上传，再让主流等它的事件；延后求和那一段从槽位算它（单专家宽，按它在选中专家里的位置取权重），和 CPU 的专家一起在下一层注意力输出前加回残差；同一层其余准入的照旧同步上传。要和 `ABYSS_MOE_CPU_MISS`、`ABYSS_MOE_SKIP_ZERO` 一起用，开了 `ABYSS_MOE_PREFETCH` 时不起作用。会话解码 +1%（约 0.3–0.5 ms/token），PPL +0.12% ± 0.10%（见 CLAUDE.md「上传的专家也延后」）；不设开关时和原来一样 |
| `0032-hc-gather-fuse.patch` | 单 token 解码的小内核融合（生产已开），结果和不融合逐位相同。`ABYSS_HC_FUSE=1`：超连接的 K 切分归约（0027）顺带算出紧跟在混合系数后面的 gate 和 comb（Sinkhorn）；pre gate、残差各份的合并（`dsv4_hc_pre`）和其后的 rms_norm 乘权重并成一个内核，块大小和累加顺序与原来的 norm 内核相同。`ABYSS_GATHER_FUSE=1`：带 top-k 压缩行的单 token 注意力（0008）把取行、转 F16、拼在窗口的 K 后面，以及掩码的取行、repeat、转换、拼接，从 7 个内核并成 1 个；融合内核要边读输入边写输出，所以输出和它读的任何张量地址重叠就不融合（段里最后一个 CSA 层的掩码输入，分配器会复用给拼接后的掩码）。每 token GPU 内核 13.87 → 13.30 ms，生产解码 20.15 → 19.68 ms/token。test-backend-ops 加了 `DSV4_GATHER_KV`（见 CLAUDE.md「小内核融合」） |
| `0033-server-stats.patch` | server 每一步的计时（`ABYSS_SERVER_STATS=N`，默认关）：每 N 个单 token 生成步、以及每个请求结束时打印一行，把一步的墙钟时间拆成主循环和队列、组 batch、解码的准备、`llama_process`、其后 `llama_synchronize` 里等 GPU 的时间（最后一段留下的输出层和 logits 拷回）、yield 收尾、采样、accept、token 的文字和 `process_token`，以及其余部分。server 自己每 token 的处理合计约 0.04 ms（见 CLAUDE.md「token 之间的开销」） |
| `0034-head-split.patch` | 输出层分两卡（`ABYSS_HEAD_SPLIT=F`，生产 0.5）：加载时把输出层前 F 的行复制到另一张 GPU，另建一个绑在输出层缓冲区上的视图表示其余的行；单个输出时两张卡同时算，结果在副本那张卡上拼起来（上下文的建图回调把 concat 放过去），logits 逐位相同。调度器加了 `ggml_backend_sched_set_overlap_gpus`，开了分两卡时打开：没有输入的 GPU 分段接在另一张 GPU 的分段后面时，主机不等那一段；从另一张 GPU 来的输入，先让那张卡的流在 GPU 上等本分段那张卡的事件再拷，主机不等。每 token 约快 0.4 ms，GPU0 多占 631 MiB（见 CLAUDE.md「输出层分两卡」）；不设开关时和原来一样 |
| `0035-sample-fast.patch` | 采样快路径（`ABYSS_SAMPLE_FAST=1`）：采样链里只剩空采样器（各 init 函数把不改候选的设置做成空采样器）加 dist（要 `ABYSS_DIST_VEC=1`）或贪心（前面可以有温度 0）时，`llama_sampler_chain_sample_logits` 直接在一个输出的 logits 上抽，不建 129,280 项的候选数组、不写回概率，选中的 token 和原路径相同（最大值用 16 路向量化的比较，和 `std::max` 逐个比得到同一个值）；`common_sampler_sample_fast` 在不要概率、推理预算没在强制输出时用它，语法检查和重采样照旧，server 解码单 token 时调用它。要概率、有 logit bias（含 `ignore_eos`）时退回原路径，快路径之后候选数组是空的。温度 1.0 每 token 约省 0.16 ms（见 CLAUDE.md「采样快路径」）；不设开关时和原来一样 |

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
  <abyss>/mx-llama.cpp/0013-sched-reserve-size-reset.patch <abyss>/mx-llama.cpp/0014-moe-big-split.patch \
  <abyss>/mx-llama.cpp/0015-prefill-tail-one-pass.patch <abyss>/mx-llama.cpp/0016-decode-stats.patch \
  <abyss>/mx-llama.cpp/0017-slot-save-thin-checkpoints.patch <abyss>/mx-llama.cpp/0018-plain-request-text.patch \
  <abyss>/mx-llama.cpp/0019-graph-memory.patch <abyss>/mx-llama.cpp/0020-indexer-chunks.patch \
  <abyss>/mx-llama.cpp/0021-bench-moe-cache.patch <abyss>/mx-llama.cpp/0022-big-batch-stats.patch \
  <abyss>/mx-llama.cpp/0023-big-batch-cache.patch <abyss>/mx-llama.cpp/0024-moe-prefetch.patch \
  <abyss>/mx-llama.cpp/0025-moe-defer.patch <abyss>/mx-llama.cpp/0026-moe-ids-stats.patch \
  <abyss>/mx-llama.cpp/0027-hc-ksplit.patch <abyss>/mx-llama.cpp/0028-engram-decode-prefetch.patch \
  <abyss>/mx-llama.cpp/0029-dist-vec.patch <abyss>/mx-llama.cpp/0030-moe-cache-gate.patch \
  <abyss>/mx-llama.cpp/0031-moe-defer-upload.patch <abyss>/mx-llama.cpp/0032-hc-gather-fuse.patch \
  <abyss>/mx-llama.cpp/0033-server-stats.patch <abyss>/mx-llama.cpp/0034-head-split.patch \
  <abyss>/mx-llama.cpp/0035-sample-fast.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server
cmake --build build -j --target llama-abyss-dump llama-abyss-chat-tokens   # 对拍工具，可选
```

## 使用

```bash
ABYSS_EXPERT_STATS=~/expert-stats.json llama-server ... -np 1 \
  --jinja --chat-template-file <abyss>/mx-llama.cpp/deepseek-v4.1.jinja --no-reasoning-preserve \
  --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0
python3 tools/expert_hotness.py ~/expert-stats.json
```

- 工具调用必须同时用 `0002` 补丁和 `deepseek-v4.1.jinja`：解析器只在模板里出现 `' calls>` 时才按 V4.1 标签解析。
- 模板和官方编码的对拍用 `tools/eval/template_check.py`（10 组对话：system、developer、工具定义、工具调用和结果、多轮思考、思考强度），全部逐字一致。
- `ABYSS_PLAIN_INPUT=1`（`0018`）：请求文字里的控制 token 字样不再变成控制 token。逐 token 对拍用 `tools/eval/special_check.py`：上面 10 组加 7 组把这类字样放在用户、system、工具结果、助手历史、字符串和对象参数、工具定义里的对话，参照是官方编码器（控制 token 常量加标记）加去掉 added token 的 HF 分词器，全部一致；不含这类字样时和原来的编码逐 token 相同。只对 DeepSeek V4 的模板生效，续写助手消息（prefill）的请求仍按原来的方式编码。
- `--no-reasoning-preserve`：llama.cpp 默认保留历史轮次的思考（把 `drop_thinking` 设成 false），官方在没有工具时只保留最后一条用户消息之后的思考；有工具时两者都保留。
- 思考强度：请求的 `reasoning_effort` 取 low / high / max（官方，对应 50 / 75 / 100，默认 high）或 1–100 的整数；minimal / medium / xhigh 是自定义的 25 / 62 / 88。
- 采样用官方推荐的 temperature 1.0、top_p 0.95；llama.cpp 默认是 0.8、top_k 40、min_p 0.05。请求里带的采样参数会覆盖默认值。
- 不设 `ABYSS_EXPERT_STATS` 时不挂回调，行为和原版一致。
- 统计文件已存在时，启动后接着累加，不会覆盖。
- 统计按 ubatch 的 token 数区分阶段：1 个 token 算解码，多个算预填充。要用 `-np 1`，否则多路并发的解码会被算成预填充。
- `ABYSS_EXPERT_STATS_FLUSH_SEC` 控制写文件间隔，默认 30 秒；只在有请求时写，进程正常退出时也会写一次。
- 统计在每个 ubatch 之后读选择结果（`0026`），不切计算图，每 token 只多 40 次 24 字节的拷回（约 0.1 ms）；`0026` 之前的逐节点回调每层同步一次，测速时要关。
- `ABYSS_CED_PREFILL=1` 打开 CED 感知预填充，只对 DeepSeek V4.1 生效。每次 `llama_decode` 调用按自己的末尾规划，所以 server 要设大 `-b`（比如 65536），让整段提示词进一次调用才省得多。`ABYSS_CED_MARGIN`（默认 256）是末尾额外保持有效的位置数。`0003` 跳过 FFN，`0005` 再跳过注意力的查询，KV 的写入始终不变。
- `ABYSS_DSV41_INDEXER=1` 打开 V4.1 的稀疏注意力索引器：KV 源层写索引键（第 2、8、14 层用已有的索引键 cache，第 20 层用新增的 `kv_lidp`），索引源层打分选 top-512，第 20 层为后面的压缩比 1 索引源挑 2,048 个候选块，其余层沿用最近索引源的 top-k。DSV4 状态格式版本因此升到 2
- `llama-abyss-dump` 由 `ABYSS_DUMP_RE`（张量名正则，名字带 `-<层号>` 后缀）和 `ABYSS_DUMP_DIR` 控制，整段提示词作为一批送入，每个 ubatch 存一个 `<名字>.<序号>.bin`。设 `ABYSS_DUMP_GEN=N` 时不存提示词，而是之后贪心解码 N 个 token，只存这些单 token 的计算。`0007` 给索引键的输入起名 `idx_latent`，`0008` 给压缩注意力的查询起名 `hca_q`，都不影响计算
- `ABYSS_DSV41_DECODE_GATHER=1` 只在单 token、开了索引器、压缩行多于 top-k 时生效。取出的行补齐到 256 的倍数（补的行掩码为 -inf），走普通的 flash attention；和原来的掩码收集是同一组 KV，差别只在 F16 舍入
- `ABYSS_MOE_CPU_MISS=1` 只对专家缓存覆盖的层、32 个 token 以内的批次（也就是解码）生效，目前只支持 DeepSeek V4 的 SwiGLU 专家。每层先由一个 CPU 算子规划：命中的照常读缓存；未命中的按衰减热度挑出 `ABYSS_MOE_ADMIT`（默认 0.5）比例异步上传进缓存，其余由 CPU 用主机内存里的原始专家计算，GPU 那边读全零的备用槽位（每个 token 的第 k 个专家用第 k 个，同一 token 里不能重复，否则 CUDA MMQ 会越界），最后两边相加。CPU 这一路固定在 CPU 上，32 个 token 的批次也不会被卸载到 GPU。热度每层每步乘 `ABYSS_MOE_HEAT_DECAY`（默认 0.99）。CPU 的 `mul_mat_id` 遇到 id < 0 会跳过并把输出清零。server 退出时在 stderr 打印 `abyss: moe cache` 统计
- `0012` 改的是 server 的 prompt 缓存：新请求到来时，比较当前槽位和内存缓存里各条目能直接用上的 token 数（检查点受限的缓存取不晚于分叉点的最近检查点），缓存里的条目多出 256 个以上就换进来；只用到条目一小部分时复制一份，原条目留在缓存里。智能体的辅助请求（共享系统提示、只问一句）因此只算几十个 token，也不会把主对话冲掉。`-lv 4` 能看到检查点和缓存的换入换出
- `ABYSS_MOE_BIG_SPLIT=1` 只对专家缓存覆盖的层、超过 32 个 token 的批次生效。每层一个 CPU 算子统计每个专家的 token 数，按 token 数从少到多把专家交给 CPU，直到 CPU 的预计时间追上其余专家的上传时间（`ABYSS_MOE_BIG_UPLOAD_US`、`ABYSS_MOE_BIG_CPU_EXPERT_US`、`ABYSS_MOE_BIG_CPU_TOKEN_US`，默认 650 / 450 / 50 微秒），而且这些「token × 专家」不超过 `ABYSS_MOE_BIG_CPU_ROWS`（默认 1,024）行。GPU 那一路的这些条目为 -1：CUDA MMQ 跳过它们并把输出清零，上传回调不传它们，而且直接用规划算子在主机上的结果，不必从 GPU 读回 id。CPU 那一路在 GPU1 上取出这些 token，在 CPU 上按一行一个专家计算，再用 `set_rows` 写回、和 GPU0 的结果相加；合并固定在 0 号后端（GPU0）。退出时在 stderr 打印 `abyss: moe cache large batches` 统计
- `ABYSS_ENGRAM_PREFETCH=8` 打开 Engram 预取，只对 DeepSeek V4.1、单序列、64 个 token 以上的批次生效。只做预读，不改计算结果；新批次到来时会叫停上一批还没做完的预取。
- `ABYSS_DSV41_IDX_CHUNK=N`（`0020`）只对开了索引器、单序列、查询多于 N 个的批次生效；不设或 0 时不分块。分块改变编码器层的累加顺序，对不分块 KLD 0.0079、top-1 96.2%（语料 14 × 1,024 token，块大小 128），在编码器层的数值底线以内；同一配置下换上下文大小或专家缓存大小，输出逐 token 相同。
- 排查显存用 `GGML_ALLOC_TRACE=1`（`0019`）加 `tools/eval/alloc-trace.sh`：每次分配计算图（加载时的预留、运行中因为图变大而重新分配）都打印一段，`alloc_peak.py` 列出每段的缓冲区大小，以及节点最多（或指定）那段每个缓冲区峰值时的张量。分配器把所有叶子在图开始前就分配好，没人读的节点不释放，跨设备的输入拷贝每个源张量各一份，这三类最容易悄悄占显存。
