# PTO ISA 术语表

本页收录 PTO ISA 文档中常用的术语和缩写。

## 硬件架构

| 术语 | 全称 | 释义 |
|------|------|------|
| A2A3 | — | 硬件后端代号，对应 A2 与 A3 代际 NPU |
| A5 | — | 硬件后端代号，对应 A5 代际 NPU |
| NPU | Neural Processing Unit | 神经网络处理单元，PTO 的目标加速设备 |
| GM | Global Memory | 全局内存（设备外存），通过 `GlobalTensor` 访问 |
| UB | Unified Buffer | 统一缓冲区，片上高速存储，Tile 数据的物理载体 |
| L1 | Level 1 Cache | L1 缓存，片上级缓存 |
| L0A/L0B/L0C | — | L0 级矩阵运算缓冲区（A/B 操作数与累加器） |

## 编程模型

| 术语 | 全称 | 释义 |
|------|------|------|
| Tile | — | 固定容量的二维片上缓冲区，PTO 指令的主要计算单元 |
| GlobalTensor | — | 全局内存的轻量级视图，带 shape/stride/layout 元数据 |
| Scalar | — | 用于参数化指令的立即数与枚举（舍入模式、比较模式等） |
| Event | — | 显式依赖 token，用于在不引入全局屏障的情况下表达顺序约束 |
| RecordEvent | — | C++ 接口返回类型，用于建立指令间依赖关系 |
| TileType | — | Tile 角色枚举（如 `Vec`、`Bias`、`Acc` 等），决定 Tile 在硬件中的位置 |
| Loc | — | Tile 位置属性，表示 Tile 位于哪个硬件缓冲区（如 UB、L1、L0C 等） |
| SFractal | — | 分形存储布局，Tile 在硬件缓冲区中的物理排布方式 |
| isRowMajor | — | Tile 布局属性，表示数据是否按行主序排列 |
| RowMajor | — | 行主序布局，数据按行连续存储 |
| ColMajor | — | 列主序布局，数据按列连续存储 |
| ValidRow / ValidCol | — | Tile 的有效行数 / 有效列数，定义计算结果的有效区域 |

## 编译器与 IR

| 术语 | 全称 | 释义 |
|------|------|------|
| SSA | Static Single Assignment | 静态单赋值，IR 的中间表示形式（AS Level 1） |
| DPS | Destination-Passing Style | 目标侧动态生成，显式资源绑定的低级 IR 形式（AS Level 2） |
| AS Level 1 | Assembly Level 1 | 汇编层级 1，对应 SSA 形式 |
| AS Level 2 | Assembly Level 2 | 汇编层级 2，对应 DPS 形式 |
| Lowering | — | 降阶，编译器将高级 IR 转换为低级形式的过程 |
| PTO_INST | — | PTO 指令标记宏，标注 C++ 内建接口 |
| TASSIGN | — | 资源绑定指令，将 Tile 绑定到指定内存偏移地址 |
| vbrcb | Vector Block Repeat Copy Broadcast | 向量块广播指令，用于行级数据广播 |

## 数据类型

| 术语 | 释义 |
|------|------|
| half | 半精度浮点数（16 位） |
| bfloat16_t | Brain Float 16 格式（16 位） |
| float | 单精度浮点数（32 位） |
| int8_t / uint8_t | 8 位有符号 / 无符号整数 |
| int16_t / uint16_t | 16 位有符号 / 无符号整数 |
| int32_t / uint32_t | 32 位有符号 / 无符号整数 |

## 开发模式

| 术语 | 释义 |
|------|------|
| PTO-Auto | 自动模式，编译器负责资源放置与调度 |
| PTO-Manual | 手动模式，开发者显式控制资源放置与同步 |
