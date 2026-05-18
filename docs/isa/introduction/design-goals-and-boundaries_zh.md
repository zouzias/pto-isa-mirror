# 范围与边界

PTO ISA 规范的覆盖范围，以及它与相邻层之间的边界如下。

## PTO ISA 定义什么

PTO ISA 定义合法 PTO 程序的架构可见含义，包括：

- `pto.t*`、`pto.v*`、`pto.*` 及其他架构可见操作的语义
- tile、GlobalTensor、事件与显式同步的编程模型
- 让执行对程序、模拟器和 backend 可见的机器模型与内存顺序规则
- 在 CPU 仿真和受支持 Ascend NPU 目标之间保持稳定的合法指令集

如果两个受支持目标都接受同一个合法 PTO 程序，该程序的架构可见意义必须来自 PTO ISA，而不能由目标私自重定义。

## Target Profile 可以缩窄什么

PTO ISA 是稳定的，但不是无限制的。目标 profile 可以缩窄某个实现真正接受或高效支持的子集，例如：

- tile shape 或 tile rank
- 数据类型与 layout 组合
- 特定向量微指令形式
- 同步变体或 memory space
- 绑定到某一代硬件的指令子集

这些限制只缩窄目标子集，不改变 PTO ISA 本身的语义。

## PTO-AS 增加什么

PTO-AS 是 PTO ISA 的文本语法形式。它补充了：

- 指令名拼写
- 操作数顺序
- 属性和修饰符写法
- 解析、汇编和 round-trip 所需的文本约定

因此 PTO-AS 是 PTO ISA 的表达层，而不是另一套语义不同的架构。

## PTOBC 增加什么

PTOBC 是 PTO 程序的分发和传输形式，用于缓存、打包、跨工具传递 PTO 程序，而不是立刻坍缩成某一代硬件的专用二进制。

PTOBC 不重定义 ISA，它只是序列化承载 PTO 程序。

## PTO ISA 不冻结什么

本手册不会把所有编译器内部阶段或 backend lowering 细节冻结成公开契约。PTO ISA 不冻结：

- 编译器内部 IR 结构
- pass 顺序
- backend 专用调度策略
- 硬件私有流水线细节
- 原生硬件指令的二进制编码

这些内容属于编译器、汇编器、运行时和目标专属 backend 文档。

## 规范来源顺序

当边界不清楚时，按以下顺序裁决：

1. PTO ISA manual 与 per-op ISA 页面
2. 代码与验证器暴露出来的合法指令集
3. PTO-AS / PTOBC 文档中的语法和分发规则
4. backend profile 注释中的目标专属缩窄

如果某个 backend 依赖的行为不在这条权威链中，它就只是 backend 要求，而不是 PTO ISA 保证。

## 相关页面

- [什么是 PTO 虚拟 ISA](./what-is-pto-visa_zh.md)
- [规范来源](../reference/source-of-truth_zh.md)
- [可移植性与目标 Profile](../reference/portability-and-target-profiles_zh.md)
