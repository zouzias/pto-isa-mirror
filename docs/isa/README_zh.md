<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 手册与参考

本文档目录是 PTO ISA 的权威文档树。它将架构手册、指令集指南、家族契约和精确的指令参考分组整合在同一个位置。

## PTO ISA 中的文本汇编

本树是权威的 PTO ISA 手册。文本汇编拼写属于 PTO ISA 的语法层，而非第二份并行的架构手册。

- PTO ISA 定义了架构可见的语义、合法性、状态、排序、目标 profile 边界，以及 tile、vector、scalar、communication、系统调度操作的可见行为
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

<<<<<<< HEAD
## 数据搬运 / 布局
- [TEXTRACT](TEXTRACT_zh.md) - 从源 Tile 中提取子 Tile。
- [TEXTRACT_FP](TEXTRACT_FP_zh.md) - 带 fp/缩放 Tile 的提取（向量量化参数）。
- [TIMG2COL](TIMG2COL_zh.md) - 用于类卷积工作负载的图像到列变换。
- [TINSERT](TINSERT_zh.md) - 在 (indexRow, indexCol) 偏移处将子 Tile 插入到目标 Tile 中。
- [TINSERT_FP](TINSERT_FP_zh.md) - 带 fp/缩放 Tile 的插入（向量量化参数）。
- [TFILLPAD](TFILLPAD_zh.md) - 复制 Tile 并在有效区域外使用编译时填充值进行填充。
- [TFILLPAD_INPLACE](TFILLPAD_INPLACE_zh.md) - 原地填充/填充变体。
- [TFILLPAD_EXPAND](TFILLPAD_EXPAND_zh.md) - 填充/填充时允许目标大于源。
- [TMOV](TMOV_zh.md) - 在 Tile 之间移动/复制，可选应用实现定义的转换模式。
- [TMOV_FP](TMOV_FP_zh.md) - 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 移动/转换到目标 Tile。
- [TRESHAPE](TRESHAPE_zh.md) - 将 Tile 重新解释为另一种 Tile 类型/形状，同时保留底层字节。
- [TTRANS](TTRANS_zh.md) - 使用实现定义的临时 Tile 进行转置。
- [TSUBVIEW](TSUBVIEW_zh.md) - 表达一个tile是另一个tile的subview。
- [TGET_SCALE_ADDR](TGET_SCALE_ADDR_zh.md) - 将输出tile的片上内存值绑定为扩展后的输入tile内存的值。
- [TCONCAT](TCONCAT_zh.md) - 将两个 Tile 沿列维度水平拼接。
=======
- [指令集总览](instruction-families/README_zh.md)
- [指令族](instruction-families/README_zh.md)
- [指令描述格式](reference/format-of-instruction-descriptions_zh.md)
- [Tile 指令集参考](tile/README_zh.md)
- [Vector 指令集参考](vector/README_zh.md)
- [标量与控制参考](scalar/README_zh.md)
- [通信指令集参考](comm/README_zh.md)
- [系统调度指令集参考](system/README_zh.md)
- [通用约定](conventions_zh.md)
>>>>>>> de8fc112 (Refactor duplicated kernel pipeline helpers)

## 支持性参考

- [参考注释](reference/README_zh.md)（术语表、诊断、可移植性、规范来源）

## 核间通信

- [TALLOC](TALLOC_zh.md) - 将 TPipe FIFO 槽位分配为一个 GlobalTensor 视图。
- [TPUSH](TPUSH_zh.md) - 将生产者 tile 推入 TPipe FIFO，用于 Cube-Vector 通信。
- [TPOP](TPOP_zh.md) - 从 TPipe FIFO 弹出消费者 tile/globalTensor，用于 Cube-Vector 通信。
- [TFREE](TFREE_zh.md) - 释放 TPipe 的 FIFO 空间；对于 TileData/GlobalTensor 的 TPOP 流程，该操作为空操作。

`tile/`、`vector/`、`scalar/`、`comm/` 和 `system/` 下的分组指令集树是权威的 PTO ISA 路径。

- `docs/isa/tile/ops/`
- `docs/isa/vector/ops/`
- `docs/isa/scalar/ops/`
- `docs/isa/comm/`
- `docs/isa/system/ops/`
