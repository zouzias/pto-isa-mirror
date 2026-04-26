<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 手册与参考

本文档目录是 PTO ISA 的权威文档树。它将架构手册、指令集指南、家族契约和精确的指令参考分组整合在同一个位置。

## PTO ISA 中的文本汇编

本树是权威的 PTO ISA 手册。文本汇编拼写属于 PTO ISA 的语法层，而非第二份并行的架构手册。

- PTO ISA 定义了架构可见的语义、合法性、状态、排序、目标 profile 边界，以及 `pto.t*`、`pto.v*`、`pto.*` 及其他操作的可见行为
- PTO-AS 是用于编写这些操作和操作数的汇编拼写。它是 PTO ISA 的表达方式的一部分，而非具有不同语义的分立 ISA

如果问题是"PTO 程序在 CPU、A2/A3 和 A5 上的含义是什么？"，请留在本树中。如果问题是"这个操作的操作数形状或文本拼写是什么？"，请使用本树中语法与操作数相关的页面。

## 从这里开始

## 模型层次

阅读顺序与手册章节地图一致：先编程模型与机器模型，再语法与状态，再内存，最后是操作码参考。

- [编程模型](programming-model/tiles-and-valid-regions_zh.md)
- [机器模型](machine-model/execution-agents_zh.md)
- [语法与操作数](syntax-and-operands/assembly-model_zh.md)
- [类型系统](state-and-types/type-system_zh.md)
- [位置意图与合法性](state-and-types/location-intent-and-legality_zh.md)
- [内存模型](memory-model/consistency-baseline_zh.md)

- [指令集总览](instruction-surfaces/README_zh.md)
- [指令族](instruction-families/README_zh.md)
- [指令描述格式](reference/format-of-instruction-descriptions_zh.md)
- [Tile 指令集参考](tile/README_zh.md)
- [Vector 指令集参考](vector/README_zh.md)
- [标量与控制参考](scalar/README_zh.md)
- [其他与通信参考](other/README_zh.md)
- [通用约定](conventions_zh.md)

## 支持性参考

- [参考注释](reference/README_zh.md)（术语表、诊断、可移植性、规范来源）

## 兼容性重定向

`tile/`、`vector/`、`scalar/` 和 `other/` 下的分组指令集树是权威的 PTO ISA 路径。

部分旧的根级页面（如 `TADD_zh.md`、`TLOAD_zh.md`、`TMATMUL_zh.md`、`TDEQUANT_zh.md` 等）现仅作为兼容性重定向保留，以避免现有链接立即失效。新 PTO ISA 文档应链接到分组指令集路径，尤其是以下位置的独立 per-op 页面：

- `docs/isa/tile/ops/`
- `docs/isa/vector/ops/`
- `docs/isa/scalar/ops/`
