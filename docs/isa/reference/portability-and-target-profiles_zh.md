# 可移植性与目标 Profile

PTO 的可移植性成立于虚拟 ISA 层，而不是每一种目标专属优化、支持子集或性能行为层。

## 可移植 PTO 契约

可移植 PTO 文档应描述：

- 合法程序的架构可见语义
- 必需的同步与可见性边
- tile、vector、scalar/control 与 communication 操作的含义

## Profile 缩窄

目标 profile 可以缩窄：

- 支持的数据类型
- 支持的 layout 或 tile role
- 支持的向量形式与流水线特性
- 支持的性能导向或不规则指令族

这些限制应写成目标 profile 限制，而不是把 PTO 本身重写成 profile 专属规则。

## 兼容性边界

跨 profile 的可移植代码应满足两层约束：

1. PTO 本身定义的合法性与语义约束
2. 所选 profile 额外施加的缩窄条件

如果某条程序只在单一 profile 上成立，应明确把该条件写进文档，而不是默认这类约束会从 backend 行为中被自行推断。

## 文档要求

- 架构层规则与 profile 层规则必须分开陈述。
- 当 profile 改变支持子集时，应说明“哪里被缩窄”，而不是只说“某目标不支持”。
- profile 特有性能、对齐或流水线细节可以记录，但不能伪装成通用 PTO 语义。

## 相关页面

- [规范来源](./source-of-truth_zh.md)
- [诊断与非法情形](./diagnostics-and-illegal-cases_zh.md)
