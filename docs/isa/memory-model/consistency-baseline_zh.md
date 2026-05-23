# 一致性基线

PTO 的内存模型建立在 **显式数据移动** 和 **显式顺序** 之上。基线保证刻意窄于“所有东西天然全局有序”。程序或所选指令集必须明确表达数据何时在不同阶段、不同指令集和不同 block 之间变得可见。

## 内存空间

PTO 定义三类架构上不同的内存空间：

| 空间 | 地址限定 | 作用域 | 可见性 |
| --- | --- | --- | --- |
| **GM** | `__gm__` | 全部 AI Core | 共享 |
| **UB** | `!pto.ptr<T, ub>` | 单个 AI Core | core-local |
| **Tile Buffer** | `!pto.tile_buf<...>` | 单个 AI Core | core-local，且带 pipeline 角色 |

它们不能互换。数据必须显式在这些空间之间移动。

## 顺序层级

PTO 定义三层顺序保证：

| 层级 | 含义 | 范围 | 如何建立 |
| --- | --- | --- | --- |
| **Program Order** | 同一 tile buffer 或向量寄存器中的程序顺序 | 单核心 | 隐式 |
| **Event Order** | 不同 pipeline / buffer 之间的顺序 | block 内 | `set_flag` / `wait_flag` 或 `RecordEvent` |
| **Barrier Order** | 多 block 之间的顺序 | grid 范围 | collective / barrier 类操作 |

### Program Order

同一 tile buffer 或同一向量寄存器中的操作按程序顺序排列，不需要额外同步。

### Event Order

不同 buffer 或不同 pipeline 之间的数据依赖必须通过事件建立：

```cpp
RecordEvent e0 = TLOAD(a, ga);
RecordEvent e1 = TLOAD(b, gb);
TMATMUL(c, a, b, e0, e1);
```

### Barrier Order

多 block 之间的同步需要 grid 范围的 barrier 或 collective：

```mlir
pto.tbroadcast %tensor, %src : ...
pto.twait
```

## PTO 不自动保证的内容

PTO **不会**自动保证：

| 不自动给出的保证 | 原因 |
| --- | --- |
| 每个跨 pipeline 写入立刻对所有消费者可见 | 需要显式事件 |
| vector register、tile buffer、GM 共用一套隐式 fence 模型 | 各空间有不同可见性 |
| 某目标更强的顺序具有可移植性 | profile 的更强保证不自动外推 |
| UB 写入不经显式搬运就能被 GM 读取 | UB→GM 需要显式移动 |
| GM 写入不经显式搬运就能被 UB 读取 | GM→UB 需要显式移动 |

## GM 可见性

经由 `TSTORE` 或 `copy_ubuf_to_gm` 写入 GM 的数据，在以下条件满足后才对其他 block 的 GM 读取可见：

1. 本 block 中之前的 store 已完成
2. 所需 `mem_bar` / `pipe_barrier` / collective 同步已建立
3. 运行时确认完成

跨 block 可见性的精确时间点是 implementation-defined，但顺序契约本身是架构保证。

## UB 可见性

UB 是 core-local 的，其他 core 无法直接看见。

在同一 core 内：

- UB 读会按程序顺序看见之前的 UB 写
- UB 读只有在对应 `wait_flag` 成功后，才应视为看见 `copy_gm_to_ubuf` 的结果

## Undefined / Unspecified / Implementation-defined

| 术语 | 含义 | 例子 |
| --- | --- | --- |
| **Undefined** | 行为故意不定义，任何结果都可能 | 读取 tile 域外元素 |
| **Unspecified** | ISA 不规定具体值或行为 | 精确 cycle 数 |
| **Implementation-defined** | 由实现定义并应文档化 | A5 上 denormal 的 FTZ 行为 |

## Target Refinement

CPU、A2/A3 和 A5 可以在实现细节和支持子集上不同，但基线文档必须明确哪些顺序事实是可移植的：

| 类别 | 可移植性 |
| --- | --- |
| 同一 tile buffer 内的程序顺序 | 可移植 |
| `set_flag` / `wait_flag` 建立的事件顺序 | 可移植 |
| `RecordEvent` 链接 | 可移植 |
| collective 带来的 barrier 顺序 | CPU 不适用，A2/A3 与 A5 支持 |
| UB→GM 的隐式可见性 | 不可移植；必须显式搬运 |
| A5 更强的顺序行为 | A5 专属，不可外推 |

## 不允许的情形

!!! danger "不允许的情形"
    - 把实现细节写成可移植内存模型
    - 用“通常有序”这种模糊语句替代明确顺序边
    - 把调度启发式混进内存模型保证
    - 在没有显式同步操作时声称跨 block 可见

## 相关页面

- [生产者-消费者排序](./producer-consumer-ordering_zh.md)
- [顺序与同步](../machine-model/ordering-and-synchronization_zh.md)
- [可移植性与目标 Profile](../reference/portability-and-target-profiles_zh.md)
