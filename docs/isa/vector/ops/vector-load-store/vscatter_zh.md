# pto.vscatter

`pto.vscatter` 是 [向量加载存储指令集](../../vector-load-store_zh.md) 的一部分。

## 概述

`pto.vscatter` 执行带索引的 scatter 存储操作，将向量寄存器中的数据按索引分散写入统一缓冲区（UB）。

## 机制

`pto.vscatter` 属于 PTO 向量存储指令集。它将 UB 寻址、分布模式、掩码行为和对齐状态显式保持在 SSA 形式中，而不是隐藏在后端特定的下发过程中。

`pto.vscatter` 是 scatter 操作的向量内存/数据移动指令家族的一部分。该指令家族包括连续存储、双字存储、跨步存储和对齐状态存储等多种变体，所有变体都共享 UB 作为唯一有效操作数源的约束。

## 语法

```mlir
pto.vscatter %value, %dest, %offsets, %active_lanes
    : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vreg<NxI>, index
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `%value` | `!pto.vreg<NxT>` | 源向量寄存器 |
| `%dest` | `!pto.ptr<T, ub>` | UB 基地址指针 |
| `%offsets` | `!pto.vreg<NxI>` | 每 lane 的索引向量 |
| `%active_lanes` | `index` | 有效 lane 数 |

## 输入

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `%value` | `!pto.vreg<NxT>` | 源向量寄存器，N 为 lane 数，T 为元素类型 |
| `%dest` | `!pto.ptr<T, ub>` | UB 基地址指针，必须指向 UB 地址空间 |
| `%offsets` | `!pto.vreg<NxI>` | 每 lane 或每 block 的索引向量，用于计算存储地址 |
| `%active_lanes` | `index` | 界限有效请求的活跃 lane 数 |

## 预期输出

此操作写入 UB 内存，不返回 SSA 值。

| 结果 | 类型 | 说明 |
|------|------|------|
| （无） | — | 此操作写入 UB 内存，无 SSA 返回值 |

## 副作用

此操作写入 UB 可见内存和/或更新流式对齐状态。无状态的对齐形式在 SSA 形式中暴露其演进状态，但仍可能需要尾部刷新形式来完成流操作。

## 约束

!!! warning "约束"
    - 仅支持 `b8`、`b16` 和 `b32` 元素大小。
    - 索引向量必须使用此指令集支持的整数元素类型和布局。
    - 每个计算出的地址必须是元素对齐的。
    - 如果两个或多个索引发生别名，只有一次写入被保证，获胜 lane 是实现定义的。

## 异常

- 使用超出所需 UB 可见空间的地址或违反所选形式的对齐/分布约束是非法的。
- 被掩码遮蔽的 lane 或非活跃 block 不会使原本非法的地址变为合法，除非操作文本明确说明。
- 约束部分声明的任何额外非法性也是契约的一部分。

## 目标Profile限制

- **A5** 是当前手册中最详细的 concrete profile。A5 上 scatter 延迟约 **17 周期**（`Dtype: B16`）。
- **CPU 模拟** 和 **A2/A3** 目标可能支持更窄的子集，或在保持可见 PTO 契约的同时模拟行为。
- 依赖特定指令集类型列表、分布模式或融合形式的代码应将该依赖视为目标 profile 特定的，除非 PTO 手册明确说明跨目标可移植性。

## 延迟（A5）

| PTO 操作 | A5 指令 | 延迟 |
|----------|---------|------|
| `pto.vscatter` | `RV_VSCATTER` | **约 17 周期** (`Dtype: B16`) |

## 示例

### C 代码

```c
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

### MLIR SSA 形式

```mlir
// 使用 vci 生成索引，然后 scatter 写入
%idx = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
pto.vscatter %value, %ub_base, %idx, %c64
    : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vreg<64xi32>, index
```

## 详细说明

```c
// Scatter: 每个 lane i 将 value[i] 写入 UB[base + offsets[i]]
for (int i = 0; i < active_lanes; i++)
    UB[base + offsets[i] * sizeof(T)] = src[i];
```

### 使用场景

`pto.vscatter` 常用于以下场景：

1. **稀疏数据写入**：将稀疏矩阵或稀疏特征的非零元素写入 UB
2. **间接寻址存储**：基于预计算的索引将数据写入 UB 的任意位置
3. **索引排序后的数据重排**：先对索引排序，再按排序后的顺序 scatter 数据

### 对齐状态存储

对于需要流式写入的场景，请参考 `pto.vsta`、`pto.vstas`、`pto.vstar` 等对齐状态存储变体。

## 相关操作 / 指令集链接

- 指令集概述：[向量加载存储指令集](../../vector-load-store_zh.md)
- 上一操作：[pto.vsstb](./vsstb_zh.md)
- 下一操作：[pto.vsta](./vsta_zh.md)
