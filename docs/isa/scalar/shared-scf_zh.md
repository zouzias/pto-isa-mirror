# 共享结构化控制流

PTO 源程序使用共享的 MLIR `scf` 操作来表达 tile / vector 区域外围的循环、分支和 loop-carried state。这些形式属于文档化的源层指令集，但不是 PTO 自定义助记符。

## 摘要

共享结构化控制流给 PTO 提供了清晰且可分析的控制外壳，包括：

- `scf.for`
- `scf.if`
- `scf.while`
- `scf.condition`
- `scf.yield`

## 机制

`scf` 围绕 PTO 区域工作，而不是替代 PTO 指令。它用于：

- 在重复 tile / vector 工作外建立计数循环
- 跨迭代携带标量或 tile 状态
- 显式表达条件执行
- 让分析和 lowering 看见结构化控制流

## 约束

!!! warning "约束"
    - 控制流应保持结构化形式，除非需要更具体的架构可见机制
    - 区域携带值和分支结果必须通过 `scf.yield` 显式表示
    - `scf` 控制所需的谓词应来自共享标量指令，而不是未文档化的旁路状态

## 不允许的情形

!!! danger "不允许的情形"
    - 把 `scf` 写成 PTO 自定义助记符
    - 隐藏会影响 PTO 合法性的 loop-carried state
    - 用模糊 prose 代替结构化控制说明

## 相关页面

- [共享标量算术](./shared-arith_zh.md)
- [控制与配置](./control-and-configuration_zh.md)
