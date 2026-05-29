# 算子开发案例（Wiki）

本目录收录**端到端、可复现**的算子调优与排障记录，与 `kernels/manual/` 中的实现一一对应，便于在改 tiling、同步或依赖上游头文件时查阅。

## 案例列表

| 主题 | 文档 | 代码 |
|------|------|------|
| A5 TopK：直方图 tiling 从 256 提到 2048 | [a5-topk-hist-tiling-2048_zh.md](a5-topk-hist-tiling-2048_zh.md) | [`kernels/manual/a5/topk/`](../../../kernels/manual/a5/topk/) |

## 如何贡献新案例

1. 在 `kernels/manual/<platform>/<op>/` 保留可运行的 `run.sh`、README 与（如有）`perf/` 归档。
2. 在本目录新增 `*_zh.md`（建议同时提供英文 `*.md`）。
3. 在 [`docs/menu_ops_development.md`](../../menu_ops_development.md) 的「算子开发实践」中增加链接。
4. 在 [`docs/coding/opt_zh.md`](../opt_zh.md) 第 7 节「示例驱动的深度指南」中增加一行引用。
