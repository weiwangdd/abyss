# 技术报告

`report.pdf` 是到 2026-10-25 为止的优化过程，源文件是 `report.tex`。版式参考 DeepSeek-V4.1-Flash 的技术报告，图表用 pgfplots 画，数据直接写在源文件里，出处见 `docs/scheduler.md`。

## 编译

```bash
brew install tectonic
cd report
tectonic -X compile report.tex
```

第一次编译会下载宏包，之后离线也能编译。字体用 macOS 自带的 Palatino、宋体（Songti SC）、Helvetica Neue 和 Menlo，在其他系统上要改 `report.tex` 开头的字体设置。

`figures/architecture.pdf` 由 `docs/architecture.svg` 转换而来（Chrome 无头模式打印成 PDF，页面尺寸 1500 × 1696 px、无边距）。架构图更新后要重新转换。
