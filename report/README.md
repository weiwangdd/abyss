# 技术报告

`report.pdf` 是到 2026-11-10 为止的优化过程（补丁到 `0046`，包括第 0–19 层张量并行的原型、和上游 llama.cpp 的对比、三个标准 benchmark），源文件是 `report.tex`。版式参考 DeepSeek-V4.1-Flash 的技术报告，图表用 pgfplots 画，数据直接写在源文件里，出处见仓库根目录的 `CLAUDE.md`（各项的日期、脚本和结果；10-25 以前的也在 `docs/scheduler.md`），标准 benchmark 的见 `benchmark/README.md`。

`report-en.pdf` 是英文版，源文件 `report-en.tex`，由中文版逐段翻译（2026-11-10，内容和中文版相同，70 页）。两版要一起改：中文版改了哪一段，英文版的同一段也要改；章节、图表的 `\label` 两边相同。英文版用 `figures/architecture-en.pdf`，由 `docs/architecture-en.svg` 照中文版的办法转换。

## 编译

```bash
brew install tectonic
cd report
tectonic -X compile report.tex
tectonic -X compile report-en.tex   # 英文版
```

第一次编译会下载宏包，之后离线也能编译；宏包下载过之后加 `--only-cached`（`tectonic -X compile --only-cached report.tex`）不再联网检查，几秒就编完，不加时曾经卡了约 12 分钟。字体用 macOS 自带的 Palatino、宋体（Songti SC）、Helvetica Neue 和 Menlo，在其他系统上要改 `report.tex` 开头的字体设置。

`figures/architecture.pdf` 由 `docs/architecture.svg` 转换而来（Chrome 无头模式打印成 PDF，页面尺寸 1500 × 1696 px、无边距），英文版的 `figures/architecture-en.pdf` 由 `docs/architecture-en.svg` 同样转换。架构图更新后两版都要改、都要重新转换：把 SVG 放进一个只有 `@page { size: 1500px 1696px; margin: 0 }` 的 HTML 页面，再用 `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --no-pdf-header-footer --print-to-pdf=<输出.pdf> file://<页面.html>` 打印。

## README 的图

首页 README 的两张图是报告图 1 的 (a)、(b)：英文版 `README.md` 用英文标签，中文版 `README.zh-CN.md` 用中文标签。源文件是 `readme-figures.tex`（第 1、2 页英文，第 3、4 页中文，数据只写在两个宏里）。改了数据或标签后重新生成：

```bash
cd report
tectonic -X compile --only-cached readme-figures.tex
swiftc -O figpng.swift -o /tmp/figpng            # 把 PDF 的一页渲染成 PNG 并裁掉白边，只用 macOS 自带的 PDFKit
/tmp/figpng readme-figures.pdf 1 3 12 ../docs/images/decode.png
/tmp/figpng readme-figures.pdf 2 3 12 ../docs/images/prefill.png
/tmp/figpng readme-figures.pdf 3 3 12 ../docs/images/decode-zh.png
/tmp/figpng readme-figures.pdf 4 3 12 ../docs/images/prefill-zh.png
```

参数依次是 PDF、页码、放大倍数（3 倍约 216 dpi）、四周留白的像素和输出文件。
