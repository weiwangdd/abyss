# 第三方来源说明

Abyss 自己的代码和文档按 MIT 许可证发布（见 [LICENSE](LICENSE)）。下面这些文件来自或改自别的项目，按各自的许可证使用。

| 文件 | 来源 | 许可证 |
|---|---|---|
| `mx-llama.cpp/*.patch`、`mx-llama.cpp/experiments/tp-layers.diff` | 对 [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp)（提交 `c4d12e2`，[llama.cpp](https://github.com/ggml-org/llama.cpp) 的分支）的修改，补丁里带着原项目的代码上下文 | MIT，Copyright (c) 2023-2026 The ggml authors（全文见下） |
| `llama-cpp` 分支 | 上游 [llama.cpp](https://github.com/ggml-org/llama.cpp) 在提交 `2cbe0ec` 的全部文件（根提交，不带历史），加上本仓库对它的修改（之后的提交）；分支里带着原项目的 `LICENSE` | MIT，The ggml authors |
| `tools/eval/corpus.txt` 第 1–473 行 | mx-llama.cpp 的 `FEATURES.md`（`c4d12e2`）原文，用作测试语料；空一行之后是本仓库自己的设计文档和代码 | MIT，The ggml authors |
| `tools/convert/to_upstream.py` | 按上游 llama.cpp（`e95426f16`）读 DeepSeek-V4.1 的布局和它的转换规则重新实现 | MIT，The ggml authors |
| `mx-llama.cpp/deepseek-v4.1.jinja` | 照 [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 的 `encoding/encoding.py` 写成的对话模板 | MIT，Copyright (c) 2023 DeepSeek（全文见下） |
| `benchmark/mlperf/dsv41_abyss.yaml` | 改自 [mlcommons/endpoints](https://github.com/mlcommons/endpoints)（`520eb4d`）的 `examples/10_Agentic_Inference/dsv4_agentic_benchmark.yaml`，改了哪些写在文件开头 | Apache License 2.0（全文见 [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)） |
| `mx-llama.cpp/0029-dist-vec.patch` 和 `llama-cpp` 分支的 `src/llama-sampler.cpp` 里的 exp 多项式 | Cephes Math Library（Stephen L. Moshier）的 `expf` 的系数和区间约简 | Cephes 的作者允许自由使用（[netlib.org/cephes](https://www.netlib.org/cephes/)） |

仓库里不含模型权重和 benchmark 的数据集。脚本会下载或读取的外部内容按它们自己的条款使用，包括 DeepSeek-V4.1-Flash 的权重（MIT）、[mxxm-t 的 GGUF](https://huggingface.co/mxxm-t/DeepSeek-V4.1-Flash-GGUF)、MLPerf Agentic Inference 的数据集、SemiAnalysis 的 AgentX 客户端（`agentx-harness`，Apache-2.0）和它的轨迹，以及 [TraceLab](https://github.com/uw-syfi/TraceLab) 的语料（CC BY 4.0）。

MLPerf 是 MLCommons Association 的商标。本项目和 DeepSeek、NVIDIA、MLCommons、SemiAnalysis 以及上面提到的其他项目都没有关联；`benchmark/` 里的结果都不是这些 benchmark 的官方结果（见 [benchmark/README.md](benchmark/README.md)）。

## mx-llama.cpp / llama.cpp

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## DeepSeek

```
MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
