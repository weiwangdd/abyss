# 技术报告

`report.pdf` 是到 2026-11-10 为止的优化过程（补丁到 `0046`，包括第 0–19 层张量并行的原型、和上游 llama.cpp 的对比、三个标准 benchmark），源文件是 `report.tex`。版式参考 DeepSeek-V4.1-Flash 的技术报告，图表用 pgfplots 画，数据直接写在源文件里，出处见仓库根目录的 `CLAUDE.md`（各项的日期、脚本和结果；10-25 以前的也在 `docs/scheduler.md`），标准 benchmark 的见 `benchmark/README.md`。

## 编译

```bash
brew install tectonic
cd report
tectonic -X compile report.tex
```

第一次编译会下载宏包，之后离线也能编译；宏包下载过之后加 `--only-cached`（`tectonic -X compile --only-cached report.tex`）不再联网检查，几秒就编完，不加时曾经卡了约 12 分钟。字体用 macOS 自带的 Palatino、宋体（Songti SC）、Helvetica Neue 和 Menlo，在其他系统上要改 `report.tex` 开头的字体设置。

`figures/architecture.pdf` 由 `docs/architecture.svg` 转换而来（Chrome 无头模式打印成 PDF，页面尺寸 1500 × 1696 px、无边距）。架构图更新后要重新转换。

## README 的图

首页 README 里的两张图是报告图 1 的 (a)、(b)，标签改成了英文，源文件是 `readme-figures.tex`（每页一张图）。改了数据或标签后重新生成：

```bash
cd report
tectonic -X compile --only-cached readme-figures.tex
swiftc -O figpng.swift -o /tmp/figpng            # 把 PDF 的一页渲染成 PNG 并裁掉白边，只用 macOS 自带的 PDFKit
/tmp/figpng readme-figures.pdf 1 3 12 ../docs/images/decode.png
/tmp/figpng readme-figures.pdf 2 3 12 ../docs/images/prefill.png
```

参数依次是 PDF、页码、放大倍数（3 倍约 216 dpi）、四周留白的像素和输出文件。
