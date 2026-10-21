# mx-llama.cpp 补丁

给 [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) 的补丁，基于提交 `c4d12e2`（2026-10-15）：

| 文件 | 作用 |
|---|---|
| `abyss-expert-stats.{h,cpp}` + `0001-hook-expert-stats.patch` | `llama-server` 的专家命中统计 |
| `0002-deepseek-v41-dsml-tags.patch` | 解析 V4.1 的工具调用标签（`<｜DSML｜ calls>` 等，标签名带前导空格） |
| `deepseek-v4.1.jinja` | V4.1 对话模板：GGUF 内嵌的是 V4 的模板，工具调用标签是 V4 写法 |

## 安装

```bash
cd mx-llama.cpp
cp <abyss>/mx-llama.cpp/abyss-expert-stats.{h,cpp} tools/server/
git apply <abyss>/mx-llama.cpp/0001-hook-expert-stats.patch <abyss>/mx-llama.cpp/0002-deepseek-v41-dsml-tags.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server
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
