# 标准 benchmark

用两个公开的智能体推理 benchmark 测生产（Abyss，mx-llama.cpp 加补丁）和上游 llama.cpp（`e95426f16`），两边是同一份权重、同样的启动参数（见 CLAUDE.md「上游实测对标」）。这些 benchmark 原本面向多卡集群和高并发，这里按本机的条件跑：一个请求槽位、并发 1，上下文 196,608 token，从官方数据里截一个子集。所以跑出的数只能在这两台服务器之间比，不是有效的提交，也不能和官方榜单比。

## 选了哪两个

| benchmark | 来源 | 负载 | 这里怎么跑 |
|---|---|---|---|
| [MLPerf Agentic Inference](https://github.com/mlcommons/endpoints/tree/main/examples/10_Agentic_Inference)（数据中心） | MLCommons，2026-07 | 613 条真实文本的多轮轨迹：113 条编程（只用一个 bash 工具修仓库里的 bug、加功能）、500 段 Workato 客服对话（约 15 个工具） | 官方 DeepSeek-V4.1-Flash 配置的模型参数照用（温度 1.0、top_p 0.95、思考强度 max、每轮最多 8,192 token），只截 3 段对话（`mlperf/subset.py`）：最短的编程轨迹 `superjson-error-stack-serialization`（38 轮，最长一次请求 89,506 token）和两段 8 轮的客服对话（`sim_000203`、`sim_000424`，最长 36,317 / 38,623 token），并发 1，不跑 SWE-bench 准确率那一段 |
| [InferenceX v3（AgentX）](https://inferencex.semianalysis.com/blog/agentx-inferencexv3-does-cuda-moat) | SemiAnalysis，2026-08 | 393 条真实 Claude Code 会话（`semianalysisai/cc-traces-weka-062126`），只有每次请求的 token 数和 64 token 一块的哈希，客户端按块拼出文本，同一块哈希拼出同样的 token，前缀共享和真实会话一样 | 官方客户端 `agentx-harness` 1.0.13 的 `agentx` 场景（流式、`ignore_eos`、每条轨迹从中段起跑、空闲间隔上限 10 s 都是它锁定的），并发 1，峰值超过 190,000 token 的会话整条去掉（剩约 165 条），测量窗口 900 s（场景允许的最短） |

两者的区别：MLPerf 每轮都按数据里记录的历史重新拼请求，上一轮的回答每轮都要重算；模型真的在思考和调工具，输出长度是自然的。AgentX 的输出长度由轨迹给定，回答也是拼出来的文本，但长上下文和前缀复用的形态来自真实的 Claude Code 使用。

## 没选的

- **MLPerf Edge Agentic**（v6.1）：参照实现就是 `llama-server -np 1`，但上下文只到 2.35 万 token，测不到长上下文。
- **TraceLab**（华盛顿大学，CC BY 4.0）：8,058 段 Claude Code 和 Codex 会话，也只有 token 数，用 AIPerf 回放，和 AgentX 是同一类测法（AgentX 本身是 AIPerf 的分支，也带 TraceLab 的读取器），再加一项要多停约一小时。
- **Inferact 的 Codex SWE-bench Pro 轨迹**（610 条，MIT）：没有配套的回放工具，要改 vLLM 的多轮脚本，回答部分是填充文字。

## 怎么跑

在 blackwell 上：

```bash
~/abyss-tools/benchmark/run.sh                    # 停生产，MLPerf、AgentX 各在生产和上游上跑一遍，打印对比表，重启生产
SERVERS=tiny BENCHES=mlperf ~/abyss-tools/benchmark/run.sh   # CPU 上的小模型，不停生产，只为试流程
```

- `servers.sh`：起停三种服务器（`fork` 生产、`up` 上游、`tiny` CPU 小模型），记加载时间和两卡显存峰值。
- `mlperf/run.sh`、`agentx/run.sh`：对一个 URL 跑一个 benchmark；`MLPERF_IDS` 换对话，`AGENTX_DURATION`、`AGENTX_SEED` 换窗口和种子。
- `mlperf/subset.py stats` 按客户端的拼法和官方对话模板数每段对话最长一次请求的 token 数：113 条编程轨迹里 25 条超过 196,608，85 条在 190,000 以内；客服对话 3.1–4.4 万。
- `compare.py`：从两边的原始结果出对比表。
- `agentx/pairs.py`：两次 AgentX 运行只比两边都测到的请求。窗口是固定的，快的一边会多测到后面的请求，整体的百分位数不能直接比。

依赖（都在 blackwell 的 `~/abyss-tools/bench`）：MLPerf 客户端 `endpoints`（`uv sync`，520eb4d）和数据集（SHA-256 `1beb24c8…`），AgentX 客户端 `agx`（`pip install agentx-harness==1.0.13`），带官方对话模板的分词器 `dsv41-tokenizer`（`deepseek-ai/DeepSeek-V4.1-Flash` 的 tokenizer 文件）。客户端和服务器在同一台机器上，MLPerf 客户端原本会把自己绑到最快的 5 个核上，配置里关掉了。

## 结果

2026-11-07 23:33 到 11-08 01:02（`~/abyss-eval/bench-1107-2333`，生产停了 89 分钟）。每一档都是新起的服务器，两边的加载时间都是 48 s，显存峰值生产 96,653 / 95,883 MiB、上游 95,437 / 95,587 MiB。

### MLPerf Agentic Inference

3 段对话共 54 轮，两边都全部成功。输入是数据里记录的历史，两边逐轮相同（平均 45,842 token，最长 89,520）。温度 1.0 采样，两边生成的内容不同，输出长度也不同；交互速度按输出 token 总数除以每轮时间之和算，已经把这一点算进去。

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

官方对 DeepSeek-V4.1 的准确率门槛是 0.5236，但这只是 3 段对话，两边的差别在抽样波动以内。

### InferenceX v3（AgentX）

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
