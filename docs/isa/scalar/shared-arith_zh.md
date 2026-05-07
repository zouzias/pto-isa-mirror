# 共享标量算术

PTO 源程序使用共享的 MLIR `arith` 指令集，在 tile 和 vector 区域周围完成标量算术设置。这些操作属于文档化的 PTO 源层指令集，但它们不是 PTO 自定义助记符。

## 摘要

共享标量算术提供：

- 常量
- 标量加减乘除与比较
- 标量 cast
- `select` 等控制前置计算

这些操作用于 tile / vector 区域外围的簿记，而不是替代 `pto.t*` 或 `pto.v*` 的 payload 运算。

## 机制

`arith` 值保持普通标量 SSA 形式，用于：

- 常量和循环边界
- offset、动态 shape 与 tail 计数
- `scf.if` / `scf.while` 的条件
- PTO 边界附近的标量宽度与类型调整

## 约束

!!! warning "约束"
    - 共享标量算术必须保持标量语义
    - 影响后续 PTO 合法性的比较、转换或宽度变化必须显式写出

## 不允许的情形

!!! danger "不允许的情形"
    - 用 `arith` 冒充向量或 tile payload 算术
    - 在 PTO 边界隐含 signedness、width change 或 `index` 转换

## 相关页面

- [共享结构化控制流](./shared-scf_zh.md)
- [控制与配置](./control-and-configuration_zh.md)
