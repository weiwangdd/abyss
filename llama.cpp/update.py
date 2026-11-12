# -*- coding: utf-8 -*-
"""Rebuilds this directory from the port branch: copies the patches up to a given number under their short names and
writes README.md with the table, the patches of mx-llama.cpp still to port, and the settings used so far.

    # in the llama.cpp tree with the port (blackwell: ~/src/llama.cpp-abyss, branch abyss-port)
    git format-patch --no-numbered -o /tmp/port 2cbe0ec..abyss-port
    # here, with the exported files
    python3 llama.cpp/update.py /tmp/port . <last patch number>

A new patch needs its line in P (short name, the patch of mx-llama.cpp it corresponds to, what it does), its step in
STEPS, its switches in settings, and, when it ports a patch listed in TODO, its number there.
"""
import glob
import os
import re
import sys

src, repo, last = sys.argv[1], sys.argv[2], int(sys.argv[3])

# number -> (short name, corresponding fork patch, what it does)
P = {
    1: ("sched-reserve-size-reset", "0013", "修 `ggml_backend_sched_reserve_size`：它在切分计算图之前重置调度器，清掉了构建计算图时手动指定的后端"),
    2: ("bench-moe-cache", "0021", "`llama-bench` 从环境变量 `LLAMA_ARG_MOE_CACHE_MIB` 读专家缓存大小"),
    3: ("dist-vec", "0029", "CPU 采样向量化（`ABYSS_DIST_VEC=1`）"),
    4: ("sample-fast", "0035", "采样快路径：采样链只剩 dist 或贪心时直接在 logits 上抽（`ABYSS_SAMPLE_FAST=1`）"),
    5: ("engram-keepalive", "0039", "解码时每隔几毫秒读一次 Engram 表所在的盘，不让它进省电状态（`ABYSS_ENGRAM_KEEPALIVE_MS=N`）"),
    6: ("prompt-cache-usable-tokens", "0012", "server 的 prompt 缓存按「能直接用上的 token 数」在槽位和内存缓存之间挑"),
    7: ("slot-save-thin-checkpoints", "0017", "槽位被只共享开头一段的请求占用前先存进内存缓存（`ABYSS_SLOT_SAVE_LOST`）；检查点列表满时的删减间距（`ABYSS_CKPT_THIN_STEP`）"),
    8: ("continue-slot-end-gen-checkpoints", "0041", "生成过程中每 N 个 token 存一个检查点，回答被重新分词后还能从中间接着算（`ABYSS_GEN_CKPT=N`）；每个请求打印从哪里接着算。上游已经让 V4.1 从槽位末尾接着算，`ABYSS_CKPT_CONTINUE` 不再起作用"),
    9: ("prefill-tail-one-pass-server", "0015 的 server 部分", "提示词最后 T 个 token 里不再拆调用（`ABYSS_CKPT_TAIL_MERGE=T`）"),
    10: ("ckpt-count-start", "0045", "按检查点步长拆调用时把这次调用开头的检查点算进去，不再多出几百 token 的小调用（`ABYSS_CKPT_COUNT_START=1`）；每个请求打印调用和检查点"),
    11: ("hc-ksplit", "0027", "超连接混合系数的 K 切分（`ABYSS_HC_KSPLIT=1`）"),
    12: ("shared-ptr-checkpoints", "", "上面几个补丁在上游上的修正：检查点列表在上游是 `shared_ptr`，也没有 fork 的无损存储标志"),
    13: ("ced-prefill", "0003、0005、0015、0019 重写", "CED 感知预填充：解码器层只对后面还看得到的 token 算查询和 FFN（`ABYSS_CED_PREFILL=1`），一次调用的余数放进第一个 ubatch（`ABYSS_CED_ALIGN=1`），`ABYSS_CED_DEBUG=1` 打印每个 ubatch 各层留几个 token。每层保留的总是每个 ubatch 末尾的 token，图里直接取视图；只支持单序列"),
    14: ("ced-test-fixture", "", "`tests/test-llama-archs` 加一个适合测 CED 的 V4.1 小模型配置（`ABYSS_TEST_DSV41_CED=1`，`-o <目录>` 写出），`tools/eval/ced-tiny.sh` 用它"),
    15: ("fused-op-sibling-gpu", "", "层分到两张卡时，下一张卡第一层的 HC pre 被调度器放在上一张卡，上游因此把 HC pre 的内核融合对整个模型关掉；同类设备、支持这个算子就接受（mx-llama.cpp 里原有的判定）"),
    16: ("moe-defer", "0025 重写", "专家延后：单 token 解码时 CPU 算的专家晚一层加回残差（`ABYSS_MOE_DEFER=1`）；靠计算图的顺序和上游调度器已有的提前取回，不改调度器。`=2` 在本层就加回，用来检查接线"),
    17: ("decode-stats", "0016", "解码统计行（`ABYSS_DECODE_STATS=N`），计数接上游专家缓存的统计"),
    18: ("ced-debug-log-level", "", "`ABYSS_CED_DEBUG` 的行用 server 默认会显示的日志级别"),
    19: ("moe-admit-gate", "0030 的准入门", "专家缓存的准入门：要挤掉的专家不比未命中的冷，就把未命中交给 CPU（`ABYSS_MOE_ADMIT_GATE=1`，值是比较时的系数），接在上游的 ClassLRU 上"),
    20: ("fused-op-sibling-switch", "", "0015 的判定加开关（`ABYSS_FUSED_SIBLING=1`）"),
    21: ("query-chunks", "0020 重写", "一层的查询分块打分、建掩码、做注意力（`ABYSS_DSV41_CHUNK=N`），打分和掩码只在一块的时间里存在；19.2 万上下文时 GPU1 计算缓冲区 6.6 → 2.8 GB（N=512）"),
    22: ("alloc-trace", "0019 的跟踪部分", "`GGML_ALLOC_TRACE=1` 打印计算图分配器的每次分配和释放，配 `tools/eval/alloc_peak.py`"),
    23: ("candidate-mask-constants", "", "候选块掩码的常量从一个元素 repeat 出来，不再是从图开始占到结束的叶子（计算缓冲区 −832 MiB）"),
    24: ("test-fixture-router-ramp", "", "0014 的小模型配置把路由偏置的斜坡改小，token 才会选不同的专家"),
    25: ("weight-nowait", "0046", "调度器把主机权重交给复制回调之前不先等这张卡（`ABYSS_WEIGHT_NOWAIT=1`）"),
    26: ("moe-big-split", "0014 重写", "预填充的大批次里 token 少的专家交给 CPU 的紧凑批次，其余照常上传（`ABYSS_MOE_BIG_SPLIT=1`）；CPU 的那些位置在设备上改算一个反正要上传的专家，结果用掩码丢掉"),
    27: ("prefill-overlap", "0044 重写", "长提示词两卡流水线（`ABYSS_OVERLAP=1`，配 `CUDA_SCALE_LAUNCH_QUEUES=4x`）：计算缓冲区重排时只在缓冲区要变大才停设备，0 字节的输入不拷也不等"),
    28: ("sync-trace", "0044 的跟踪部分", "`ABYSS_SYNC_TRACE=<ms>` 打印调度器里每次不短于这么久的等待"),
    29: ("plain-request-text", "0018", "对话提示词里来自请求的文字只按基础词表编码，只有模板自己写的控制 token 按 ID 编码（`ABYSS_PLAIN_INPUT=1`）；哪些文字来自请求用上游提示词 parts 的标记，补上 `tojson`、数组 `join` 的标记传递。`llama-abyss-chat-tokens` 只读词表，打印 chat 路径的提示词和 token"),
    30: ("v41-template-effort-levels", "", "V4.1 的模板自己认 minimal、medium、xhigh 时，思考强度原样交给模板；上游为官方模板把它们换成 low、默认、max"),
    31: ("head-split", "0034", "输出层的前 F 行另放一份到另一张卡，单个输出时两张卡各算一部分 logits（`ABYSS_HEAD_SPLIT=F`）。调度器加 `overlap_gpus`：GPU 的段接在另一张 GPU 的段后面时主机不等，跨卡输入的拷贝在 GPU 上等目标卡的事件"),
    32: ("head-split-only", "0040", "输出层不整张加载，两部分从文件直接读到两张卡（`ABYSS_HEAD_SPLIT_ONLY=1`，配 `ABYSS_HEAD_SPLIT=F`），每批最多每个序列一个输出；GPU1 少 631 MiB"),
    33: ("mmvq-tail", "0036 的矩阵乘部分", "单列的量化矩阵乘在整块的 K 循环比单个 warp 多跨 25% 以上时也用每块多行的 `small_k`（`ABYSS_MMVQ_TAIL=1`）；`small_k` 下 MXFP4、Q8_0 每块行数加倍，MXFP4 从对齐的 32 位字取。`test-backend-ops` 加本模型解码的矩阵乘形状作性能用例"),
    34: ("dsv41-window-rows", "0032 里注意力拼装的那部分，做法不同", "单个 token 的注意力只读滑窗里的行（`ABYSS_DSV41_WINDOW_ROWS=1`）：上游的滑窗缓存有 n_swa + n_ubatch 行（128 + 2,048），解码时整块交给注意力、靠掩码挡掉，每层对 2,816 行算；改成给一份滑窗的行号表和对应的掩码，图里按表取行，每层 768 行"),
    35: ("hc-gate-fuse", "mx-llama.cpp 自带的融合", "超连接的门 sigmoid(x·s + b)·scale + bias 由四个逐元素内核合成一个（`ABYSS_HC_GATE_FUSE=1`），数值逐位相同；每 token 少约 480 个内核"),
    36: ("moe-big-cache", "0023 重写", "预填充的大批次里，缓存里已有的专家用槽位在缓存所在的卡上直接算（`ABYSS_MOE_BIG_CACHE=1`，配 `ABYSS_MOE_BIG_SPLIT=1`），只有缓存里没有的才上传或交给 CPU；不在缓存里的（token, 专家）对在那张卡上读全零槽位"),
}

STEPS = [
    (12, "第 1 步：能直接搬的"),
    (14, "第 2 步：CED 预填充"),
    (20, "第 3 步：专家延后、准入门、解码统计"),
    (23, "第 3 步：计算缓冲区"),
    (26, "第 3 步：预填充的大批次"),
    (28, "第 4 步：两卡流水线"),
    (30, "对话提示词的编码（第 1 步里要重做的）"),
    (35, "解码：输出层分两卡、矩阵乘尾部、注意力和超连接的小内核"),
    (36, "预填充的大批次用缓存"),
]

out = os.path.join(repo, 'llama.cpp')
os.makedirs(out, exist_ok=True)
names = []
for n in range(1, last + 1):
    files = glob.glob(os.path.join(src, '%04d-*.patch' % n))
    assert len(files) == 1, (n, files)
    text = open(files[0], encoding='utf-8').read()
    # the fork's first commits carry git's placeholder author, the others the one used since
    text = re.sub(r'^From: Game Developer <developer@example\.com>$', 'From: abyss <abyss@localhost>', text, count=1, flags=re.M)
    name = '%04d-%s.patch' % (n, P[n][0])
    open(os.path.join(out, name), 'w', encoding='utf-8').write(text)
    names.append(name)

rows = []
prev = 0
for end, title in STEPS:
    if prev >= last:
        break
    rows.append('| **%s** | | |' % title)
    for n in range(prev + 1, min(end, last) + 1):
        rows.append('| `%04d-%s.patch` | %s | %s |' % (n, P[n][0], P[n][1], P[n][2]))
    prev = end

settings = {
    12: "ABYSS_CKPT_TAIL_MERGE=2700 ABYSS_CKPT_COUNT_START=1 ABYSS_GEN_CKPT=256 ABYSS_CKPT_THIN_STEP=8192 \\\n  ABYSS_SAMPLE_FAST=1 ABYSS_DIST_VEC=1 ABYSS_HC_KSPLIT=1 ABYSS_ENGRAM_KEEPALIVE_MS=4",
    14: " \\\n  ABYSS_CED_PREFILL=1 ABYSS_CED_ALIGN=1",
    20: " \\\n  ABYSS_FUSED_SIBLING=1 ABYSS_MOE_DEFER=1 ABYSS_MOE_ADMIT_GATE=1 ABYSS_DECODE_STATS=100",
    23: " ABYSS_DSV41_CHUNK=512",
    26: " \\\n  ABYSS_MOE_BIG_SPLIT=1 ABYSS_WEIGHT_NOWAIT=1",
    28: " ABYSS_OVERLAP=1 CUDA_SCALE_LAUNCH_QUEUES=4x",
    30: " \\\n  ABYSS_PLAIN_INPUT=1",
    32: " ABYSS_HEAD_SPLIT=0.5 ABYSS_HEAD_SPLIT_ONLY=1",
    33: " ABYSS_MMVQ_TAIL=1",
    35: " \\\n  ABYSS_DSV41_WINDOW_ROWS=1 ABYSS_HC_GATE_FUSE=1",
    36: " ABYSS_MOE_BIG_CACHE=1",
}
env = ''.join(v for k, v in sorted(settings.items()) if k <= last)
cache = 20900 if last >= 32 else 20300 if last >= 23 else 16000

apply = ' \\\n  '.join('<abyss>/llama.cpp/' + n for n in names)

# the patches of mx-llama.cpp that have no counterpart here yet: (numbers, what, the patch here that ports it)
TODO = [
    ("0001、0026", "专家统计", None),
    ("0003、0005", "CED 预填充", 13),
    ("0007", "`llama-abyss-dump`", None),
    ("0014", "大批次分流", 26),
    ("0016", "解码统计", 17),
    ("0018", "请求文字里的控制 token", 29),
    ("0019、0020", "计算图显存和查询分块", 21),
    ("0022", "大批次统计", None),
    ("0023", "大批次用缓存", 36),
    ("0025", "专家延后", 16),
    ("0030", "准入门", 19),
    ("0032", "小内核融合（门控和注意力的行已做，其余没搬）", 35),
    ("0033", "server 计时", None),
    ("0034、0040", "输出层分两卡", 31),
    ("0036", "矩阵乘尾部", 33),
    ("0044", "两卡流水线", 27),
    ("0046", "主机权重不先等", 25),
]
tools = ''
if last >= 23:
    tools = "cmake --build build -j --target test-llama-archs llama-fit-params   # tools/eval/ced-tiny.sh、fit-print.sh 用，可选\n"
elif last >= 14:
    tools = "cmake --build build -j --target test-llama-archs   # tools/eval/ced-tiny.sh 用，可选\n"
todo = '、'.join('%s（%s）' % (nums, what) for nums, what, at in TODO if at is None or at > last)

readme = """# 上游 llama.cpp 补丁（移植中）

把 [mx-llama.cpp/](../mx-llama.cpp/README.md) 里的改动移到上游 [llama.cpp](https://github.com/ggml-org/llama.cpp) 上，基于提交 `2cbe0ec`（2026-11-11）。mx-llama.cpp 从 2026-10-15 起基本没有再更新，上游这段时间自己加了 DeepSeek V4.1、稀疏注意力索引器、专家缓存的 CPU 分担等，所以这里只搬上游没有的部分；每一步在真模型上和生产（mx-llama.cpp 上的那一套）对比过再往下走，做法和数据见 [CLAUDE.md](../CLAUDE.md) 的「移植到上游 llama.cpp」。生产仍在 mx-llama.cpp 上。

「对应」一栏是 mx-llama.cpp 那边的补丁号；写「重写」的是做法按上游的结构重新写的，空着的是上游这边才需要的。

| 文件 | 对应 | 作用 |
|---|---|---|
%s

mx-llama.cpp 那边的补丁里，上游已经有、不用搬的：0002、0004、0006、0008–0011、0028、0038、0042、0043；用不上的：0024、0031、0037。还没搬的：%s。

## 安装

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 2cbe0ec
git am %s
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server
%s```

## 使用

模型要先用 `tools/convert/to_upstream.py` 改成上游读 V4.1 的布局（权重逐位相同，只改名和重组）。对话模板开头要有一行含字面 `｜DSML｜ calls>` 的注释，上游才按 V4.1 的标签名解析（`tools/eval/upstream-bench.sh` 里有做法）。到这一步为止测试用的设置：

```bash
GGML_CUDA_MMQ_PREC=q8 GOMP_SPINCOUNT=100000000 \\
  %s \\
  llama-server -m <上游布局的 GGUF> -ngl 99 -sm layer -ts 12/29 -fa 1 -lm dio -t 16 \\
  -ot "blk\\.(2[2-9]|3[0-9])\\.ffn_(gate|up|down)_exps\\.=CPU,blk\\.12\\.ffn_(gate|up)_exps\\.=CUDA0" \\
  -c 196608 -b 65536 -ub 2048 -np 1 --moe-cache-mib %d --moe-cache-cpu-share 0.7 --checkpoint-min-step 32768 \\
  --jinja --chat-template-file <模板> --no-reasoning-preserve --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0
```

每个开关不设时都和上游原来一样。这个版本的上游 server 默认不打印库里的 INFO 日志（缓冲区大小、专家缓存统计），要看加 `-lv 4`。
""" % ('\n'.join(rows), todo, apply, tools, env.strip(), cache)
open(os.path.join(out, 'README.md'), 'w', encoding='utf-8').write(readme)
print(len(names), 'patches,', len(rows), 'rows')
