# 标准 benchmark

用三个公开的智能体推理 benchmark 测生产（Abyss，mx-llama.cpp 加补丁）和上游 llama.cpp（`e95426f16`），两边是同一份权重、同样的启动参数（见 CLAUDE.md「上游实测对标」）。这些 benchmark 原本面向多卡集群和高并发，这里按本机的条件跑：一个请求槽位、并发 1，上下文 196,608 token，从官方数据里截一个子集。所以跑出的数只能在这两个服务器之间比：都不是有效的提交，也不能和官方结果比较。其中 MLPerf Agentic Inference 只用了它的数据和客户端，子集、并发和没跑 SWE-bench 那段都不符合 MLPerf 的规则，所以这里的数不是 MLPerf 结果，没有经过 MLCommons 验证（Result not verified by MLCommons Association）；下文的「MLPerf」都指这个子集。MLPerf 是 MLCommons Association 的商标。

## 选了哪三个

| benchmark | 来源 | 负载 | 这里怎么跑 |
|---|---|---|---|
| [MLPerf Agentic Inference](https://github.com/mlcommons/endpoints/tree/main/examples/10_Agentic_Inference)（数据中心） | MLCommons，2026-07 | 613 条真实文本的多轮轨迹：113 条编程（只用一个 bash 工具修仓库里的 bug、加功能）、500 段 Workato 客服对话（约 15 个工具） | 官方 DeepSeek-V4.1-Flash 配置的模型参数照用（温度 1.0、top_p 0.95、思考强度 max、每轮最多 8,192 token），只截 3 段对话（`mlperf/subset.py`）：最短的编程轨迹 `superjson-error-stack-serialization`（38 轮，最长一次请求 89,506 token）和两段 8 轮的客服对话（`sim_000203`、`sim_000424`，最长 36,317 / 38,623 token），并发 1，不跑 SWE-bench 准确率那一段 |
| [InferenceX v3（AgentX）](https://inferencex.semianalysis.com/blog/agentx-inferencexv3-does-cuda-moat) | SemiAnalysis，2026-08 | 393 条真实 Claude Code 会话（`semianalysisai/cc-traces-weka-062126`），只有每次请求的 token 数和 64 token 一块的哈希，客户端按块拼出文本，同一块哈希拼出同样的 token，前缀共享和真实会话一样 | 官方客户端 `agentx-harness` 1.0.13 的 `agentx` 场景（流式、`ignore_eos`、每条轨迹从中段起跑、空闲间隔上限 10 s 都是它锁定的），并发 1，峰值超过 190,000 token 的会话整条去掉（剩约 165 条），测量窗口 900 s（场景允许的最短） |
| [TraceLab](https://github.com/uw-syfi/TraceLab) | 华盛顿大学 uw-syfi，v0.0.1，数据 CC BY 4.0 | 43 个用户的真实 Claude Code 和 Codex 会话：4,265 个会话、357,161 轮，只有每轮的 token 数、引擎报告的缓存前缀长度和时间，没有内容和块哈希 | AgentX 的客户端带 TraceLab 的读取器，按 `agentx` 场景回放，规则和 AgentX 相同；这个场景只收 Weka 轨迹，`--unsafe-override` 放行数据集，结果因此标成 `submission_valid=false`。客户端的 `--max-context-length` 也只认 Weka 轨迹，所以先用 `tracelab/filter.py` 按读取器自己的建树规则（子代理会话按时间包含关系挂到父会话下）去掉峰值超过 190,000 的整棵树，剩 2,893 / 3,460 棵、91,017 轮；并发 1，窗口 900 s |

三者的区别：MLPerf 每轮都按数据里记录的历史重新拼请求，上一轮的回答每轮都要重算；模型真的在思考和调工具，输出长度是自然的。AgentX 的输出长度由轨迹给定，回答也是拼出来的文本，但长上下文和前缀复用的形态来自真实的 Claude Code 使用。TraceLab 也只有 token 数，读取器按 64 token 一块、按记录的缓存前缀长度合成共享部分，新请求只和上一个共享到它最后一个整块，所以每轮都会改写上一条用户消息的最后几十个 token（见 11-09 的结果）。

## 没选的

- **MLPerf Edge Agentic**（v6.1）：参照实现就是 `llama-server -np 1`，但上下文只到 2.35 万 token，测不到长上下文。
- **Inferact 的 Codex SWE-bench Pro 轨迹**（610 条，MIT）：没有配套的回放工具，要改 vLLM 的多轮脚本，回答部分是填充文字。

## 怎么跑

在 blackwell 上：

```bash
~/abyss-tools/benchmark/run.sh                    # 停生产，三个 benchmark 各在生产和上游上跑一遍，打印对比表，重启生产
SKIP="mlperf-up agentx-up" ~/abyss-tools/benchmark/run.sh   # 跳过指定的「benchmark-服务器」组合，这里上游只跑 TraceLab
SERVERS=tiny BENCHES=mlperf ~/abyss-tools/benchmark/run.sh   # CPU 上的小模型，不停生产，只为试流程
SERVERS=fork ~/abyss-tools/benchmark/run.sh       # 只跑生产（上游没变时沿用上次的结果）
SERVERS=port ~/abyss-tools/benchmark/run.sh       # 只跑移植到上游 llama.cpp 的那个分支（llama.cpp/README.md 的设置）
~/abyss-tools/benchmark/compare-runs.sh ~/abyss-eval/bench-1109-0136 ~/abyss-eval/bench-1110-0002   # 两次运行的生产并排比
~/abyss-tools/benchmark/compare-runs.sh ~/abyss-eval/bench-1110-0002 ~/abyss-eval/bench-1112-2206 fork port   # 旧运行的生产对新运行的移植分支
```

- `servers.sh`：起停四种服务器（`fork` 生产、`up` 上游、`port` 移植到上游的分支、`tiny` CPU 小模型），记加载时间和两卡显存峰值。
- `mlperf/run.sh`、`agentx/run.sh`、`tracelab/run.sh`：对一个 URL 跑一个 benchmark；`MLPERF_IDS` 换对话，`AGENTX_DURATION`、`AGENTX_SEED`（TraceLab 是 `TRACELAB_DURATION`、`TRACELAB_SEED`，`TRACELAB_DATA` 换数据文件）换窗口和种子。
- `tracelab/filter.py IN OUT --max N`：用读取器自己的步骤（按会话分组、子代理挂到父会话、每棵树的峰值）写出峰值不超过 N 的树，要用 agx 环境的 python。
- `mlperf/subset.py stats` 按客户端的拼法和官方对话模板数每段对话最长一次请求的 token 数：113 条编程轨迹里 25 条超过 196,608，85 条在 190,000 以内；客服对话 3.1–4.4 万。
- `compare.py`：从两边的原始结果出对比表（TraceLab 用 AgentX 的汇总）。
- `agentx/pairs.py`：两次 AgentX（或 TraceLab）运行只比两边都测到的请求。窗口是固定的，快的一边会多测到后面的请求，整体的百分位数不能直接比。
- `compare-runs.sh OLD NEW [SERVER [NEW_SERVER]]`：两次 `run.sh` 里同一个服务器（默认 `fork`；给了 `NEW_SERVER` 就是旧运行的 `SERVER` 对新运行的 `NEW_SERVER`）的结果并排比：`compare.py` 的表，两列是两次运行；AgentX、TraceLab 再用 `pairs.py` 比两次都测到的请求。

依赖（都在 blackwell 的 `~/abyss-tools/bench`）：MLPerf 客户端 `endpoints`（`uv sync`，520eb4d）和数据集（SHA-256 `1beb24c8…`），AgentX 客户端 `agx`（`pip install agentx-harness==1.0.13`），TraceLab 语料 `tracelab/syfi_coding_trace.jsonl.gz`（release v0.0.1，SHA-256 `9d265eae…`）和 `filter.py` 从它筛出的 `syfi_coding_trace-190k.jsonl.gz`，带官方对话模板的分词器 `dsv41-tokenizer`（`deepseek-ai/DeepSeek-V4.1-Flash` 的 tokenizer 文件）。客户端和服务器在同一台机器上，MLPerf 客户端原本会把自己绑到最快的 5 个核上，配置里关掉了。

## 结果

### 2026-11-12：移植到上游 llama.cpp 的分支

22:06 到 22:57（`~/abyss-eval/bench-1112-2206`，生产停了 51 分钟）。只跑移植分支（`SERVERS=port`：`~/src/llama.cpp-abyss` 的 `abyss-port`，补丁和设置见 [llama.cpp/README.md](../llama.cpp/README.md)，专家缓存 20900 MiB），和 11-10 的生产比（`compare-runs.sh … fork port`，对比表存成 `compare-vs-1110-fork.md`）；生产从 11-10 起没有变。三个都没有失败的请求，加载 48 s，显存峰值 96,069 / 95,271 MiB（生产 96,698 / 95,935）。

| | 移植分支 | 生产（11-10） |
|---|---:|---:|
| MLPerf 子集 TTFT 中位数 / 平均 / p90 (s) | 1.38 / 2.23 / 3.96 | 1.37 / 2.21 / 4.08 |
| MLPerf 子集每个输出 token，中位数 (ms) | 19.9 | 20.4 |
| MLPerf 子集每轮延迟平均 (s) | 14.8 | 14.2 |
| MLPerf 子集端到端交互速度 (tok/s) | 43.15 | 42.94 |
| MLPerf 子集内联准确率 | 0.58 | 0.58 |
| AgentX 30 个相同请求的 TTFT 中位数 / 平均 (s) | 3.19 / 19.5 | 3.22 / 19.9 |
| AgentX 输出 token 间隔平均 (ms) | 18.7 | 19.4 |
| AgentX 请求时间合计 (s) | 1,247 | 1,297 |
| TraceLab 33 个相同请求的 TTFT 中位数 / 平均 (s) | 9.50 / 10.68 | 9.26 / 10.25 |
| TraceLab 输出 token 间隔平均 (ms) | 18.8 | 20.0 |
| TraceLab 请求时间合计 (s) | 721 | 729 |

输出 token 的间隔三个都短 3–6%（解码快），首 token 时间 MLPerf 和 AgentX 持平。TraceLab 的首 token 慢 0.24 s（中位数）：它每轮要重算 1.4–2.3 万 token（见 11-09），这种中长的重算上移植分支比生产慢约 3%，和单独测的长提示词一致（9.6 万 27.6 对 26.8 s）；预热时冷算 14.4 万 token 的前缀 45.1 对 44.3 s，AgentX 的 6.6 万 21.2 对 20.1 s。MLPerf 每轮延迟高 4% 是这次模型每轮多写了 5% 的 token（温度 1.0，两次输出不同）。

### 2026-11-10：生产上了 0046

00:02 到 00:54（`~/abyss-eval/bench-1110-0002`，生产停了 52 分钟）。只跑生产，和 11-09 的生产比（`compare-runs.sh`，对比表存成 `compare-vs-1109.md`）：这中间生产只上了大批次上传不再让主机等（`0046`，智能体每轮预填充约快 16%，解码不变）。加载 49–50 s，显存峰值 96,698 / 95,935 MiB。

| | 这次 | 11-09 |
|---|---:|---:|
| MLPerf 子集 TTFT 中位数 / 平均 / p90 (s) | 1.37 / 2.21 / 4.08 | 1.61 / 2.50 / 4.57 |
| MLPerf 子集每个输出 token，中位数 (ms) | 20.4 | 20.4 |
| MLPerf 子集每轮延迟平均 (s) | 14.2 | 16.2 |
| MLPerf 子集端到端交互速度 (tok/s) | 42.94 | 42.54 |
| AgentX 30 个相同请求的 TTFT 中位数 / 平均 (s) | 3.22 / 19.9 | 3.67 / 20.8 |
| AgentX 输出 token 间隔平均 (ms) | 19.4 | 19.2 |
| AgentX 请求时间合计 (s) | 1,297 | 1,308 |
| TraceLab 33 个相同请求的 TTFT 中位数 / 平均 (s) | 9.26 / 10.25 | 9.67 / 10.83 |
| TraceLab 输出 token 间隔平均 (ms) | 20.0 | 20.0 |
| TraceLab 请求时间合计 (s) | 729 | 748 |

MLPerf 子集的输入逐轮相同，输出因采样不同：这次每轮平均 608 个 token，11-09 是 690，每轮延迟短了有一部分是因为这个；内联准确率 0.578 对 0.591（47 轮），在波动以内。AgentX、TraceLab 用同一个种子，两次测到的请求和输出长度完全相同，可以逐个比。首 token 中位数短了 0.4–0.5 s，正是 `0046` 省下的每轮预填充；解码不变。TraceLab 只快 4%：每轮重算 1.4–2.3 万 token 的原因（`0015` 的末段合并，见下面 11-09 的结果）还在。冷算前缀：AgentX 的 65,756 token 20.9 → 20.1 s，TraceLab 的 144,454 token 45.5 → 44.3 s。

### 2026-11-09：加 TraceLab，生产上了 0044、0045

01:36 到 02:53（`~/abyss-eval/bench-1109-0136`，生产停了 77 分钟）。生产三个都跑，上游只跑新加的 TraceLab：上游的编译（`e95426f16`）和权重都没变，MLPerf、AgentX 的上游用 11-07 的结果。这中间生产上线了长提示词预填充的两卡流水线（`0044`）和检查点拆分的修正（`0045`）。每一档都是新起的服务器，加载 48–49 s，显存峰值生产 96,686 / 95,923 MiB、上游 95,436 / 95,573 MiB。

MLPerf 子集（3 段对话 54 轮，输入和 11-07 逐轮相同，输出因采样不同；不是 MLPerf 结果，未经 MLCommons 验证）：

| | 现在 | 11-07 生产 | 11-07 上游 |
|---|---:|---:|---:|
| TTFT 中位数 / 平均 / p90 (s) | 1.61 / 2.50 / 4.57 | 1.60 / 2.74 / 4.68 | 3.99 / 9.32 / 15.56 |
| 每个输出 token，中位数 (ms) | 20.4 | 20.5 | 25.6 |
| 每轮延迟平均 (s) | 16.2 | 17.5 | 32.7 |
| 端到端交互速度 (tok/s) | 42.54 | 41.96 | 28.42 |
| 每轮输出 token，平均 | 690 | 736 | 928 |
| 跑完 54 轮 (s) | 974 | 1,045 | 1,861 |
| 内联准确率（47 轮） | 0.591 | 0.572 | 0.527 |

AgentX（同一个种子、同一个起点，`agentx/pairs.py` 只比两边都测到的请求）：和 11-07 的生产比，29 个相同请求的 TTFT 平均 23.3 → 21.2 s（中位数都是 3.61 s），请求时间合计 1,325 → 1,251 s，预热时冷算 65,756 token 的前缀 30.0 → 20.9 s（`0044`）；和上游比，22 个相同请求的 TTFT 平均 4.16 对 16.73 s，输出 token 间隔 19.2 对 25.0 ms，请求时间合计 458 对 858 s。窗口里生产测到 30 个请求，和 11-07 一样在末尾换到一条新会话、冷算整段前缀，所以整体的 TTFT p90 是 67.4 s。

TraceLab（同一个种子，900 s 窗口里生产测到 33 个请求、上游 16 个）：

| 16 个相同请求 | 生产 | 上游 | 生产 / 上游 |
|---|---:|---:|---:|
| TTFT 平均 / 中位数 (s) | 14.07 / 10.98 | 33.72 / 7.92 | 0.42 / 1.39 |
| 输出 token 间隔平均 (ms) | 20.0 | 25.7 | 0.78 |
| 请求延迟平均 / 中位数 (s) | 27.5 / 20.1 | 51.2 / 21.5 | 0.54 / 0.93 |
| 请求时间合计 (s) | 440 | 820 | 0.54 |
| 提示词平均 token | 161,473 | 161,474 | |
| 从缓存用上的比例，平均 / 中位数 | 76% / 82% | 93% / 99% | |

预热时冷算 144,455 token 的前缀：生产 45.5 s，上游 358.8 s。整个窗口：系统输出 21.2 对 12.1 tok/s，每用户 50.0 对 39.0 tok/s。

生产的 TTFT 中位数比上游慢，原因在这个回放的前缀共享方式和 `0015`。读取器合成的新请求只和上一个共享到最后一个整块，都断在上一条用户消息的最后一块里，离上一个提示词末尾 6–57 个 token。`0015` 为了每个请求少一次调用（CED 下每次调用都要把第 22–39 层的专家从内存读一遍，0.5–1.5 s），不在最后 2,700 个 token 里拆调用，最后一条用户消息开头没有检查点，只能退回很远的那个，每轮重算 1.4–2.3 万 token（7–9 s）；上游每个请求都在最后一条用户消息开头和末尾前 2,052 处拆调用，下一轮只重算几百个 token。Hermes、Codex 都是在末尾追加，下一轮从槽位末尾接着算（`0042`），碰不到这种情况；AgentX 的新请求断在上一个提示词的最后一个 token，末尾前 4 个 token 的检查点接得住，复用率 87.3%，理论值 89.9%。

三个 benchmark 的结论一致：解码生产快约 22–25%（每 token 20 对 25 ms），同样的工作量用时 0.5–0.55 倍；和 11-07 比，长提示词的冷预填充快约 30%，其余小幅变好。

### 2026-11-07：MLPerf 子集、AgentX，生产和上游

2026-11-07 23:33 到 11-08 01:02（`~/abyss-eval/bench-1107-2333`，生产停了 89 分钟）。每一档都是新起的服务器，两边的加载时间都是 48 s，显存峰值生产 96,653 / 95,883 MiB、上游 95,437 / 95,587 MiB。

#### MLPerf Agentic Inference 的子集

3 段对话共 54 轮，两边都全部成功。不是 MLPerf 结果，未经 MLCommons 验证。输入是数据里记录的历史，两边逐轮相同（平均 45,842 token，最长 89,520）。温度 1.0 采样，两边生成的内容不同，输出长度也不同；交互速度按输出 token 总数除以每轮时间之和算，已经把这一点算进去。

| | 生产 | 上游 | 生产 / 上游 |
|---|---:|---:|---:|
| TTFT 中位数 / 平均 / p90 (s) | 1.60 / 2.74 / 4.68 | 3.99 / 9.32 / 15.56 | 0.40 / 0.29 / 0.30 |
| 每个输出 token，中位数 (ms) | 20.5 | 25.6 | 0.80 |
| 每轮延迟平均 (s) | 17.5 | 32.7 | 0.54 |
| 系统输出 tok/s | 38.0 | 26.9 | 1.41 |
| 端到端交互速度 (tok/s) | 41.96 | 28.42 | 1.48 |
| 每轮输出 token，平均 | 736 | 928 | |
| 跑完 54 轮 (s) | 1,045 | 1,861 | 0.56 |
| 内联准确率（47 轮） | 0.572 | 0.527 | |

内联准确率只用来确认输出没有出错：这只是 3 段对话，两边的差别在抽样波动以内，也不能和 MLPerf 规定的准确率门槛比较。

#### InferenceX v3（AgentX）

两边都是同一个种子、同一条轨迹的同一个起点（`006c98de…` 的第 4 个请求），请求序列和输出长度完全相同；客户端判定两次运行都是有效的 AgentX 运行（`submission_valid`），只是用的是峰值在 19 万以内的那部分会话。900 s 窗口里生产测到 29 个请求、上游 22 个；生产跑得快，后面换到了一条新会话，要冷算整段前缀，所以它的 TTFT p90 反而更高（77.5 s 对 27.0 s），整体百分位数不能直接比。下表只比两边都测到的 22 个请求（`agentx/pairs.py`）：

| 22 个相同请求 | 生产 | 上游 | 生产 / 上游 |
|---|---:|---:|---:|
| TTFT 平均 / 中位数 (s) | 4.61 / 3.13 | 16.73 / 8.41 | 0.28 / 0.37 |
| 输出 token 间隔平均 (ms) | 19.6 | 25.0 | 0.78 |
| 请求延迟平均 / 中位数 (s) | 22.0 / 17.4 | 39.0 / 26.7 | 0.56 / 0.65 |
| 请求时间合计 (s) | 484 | 858 | 0.56 |
| 提示词平均 token（缓存命中） | 97,359（88.5%） | 97,357（88.5%） | |
| 每个请求输出 token，平均 | 903 | 903 | |

预热时冷算 65,756 token 的前缀：生产 30.0 s，上游 166.7 s。整个窗口的数（`compare.py`）：每用户 51.3 对 40.1 tok/s，系统输出 37.0 对 21.6 tok/s。

两个 benchmark 的结论一致：解码生产快约 25%（每 token 20 对 25 ms），首 token 时间快 2.5–3.6 倍，同样的工作量用时约 0.55 倍。差距的来源和自己的对比测试一样（CLAUDE.md「上游实测对标」）：上游没有 CED，每轮新增的几千 token 也要把第 22–39 层的专家经 PCIe 传一遍；缓存小、没有专家延后等补丁让解码慢。
