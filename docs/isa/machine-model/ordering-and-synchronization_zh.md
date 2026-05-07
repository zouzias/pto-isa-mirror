# 顺序与同步

PTO 不假设所有执行资源都隐式串行。只要数据或状态跨越指令集、流水线或共享资源流动，机器模型就必须把顺序显式写出来。同步原语、事件模型和 producer-consumer 顺序契约如下。

## 同步原语

PTO 按指令集划分四类同步原语：

### Tile 指令集原语

| 原语 | 语法 | 说明 |
| --- | --- | --- |
| `TSYNC` | `pto.tsync %events...` 或 `pto.tsync<Op>` | 等待显式 `RecordEvent`，或插入某类操作屏障 |
| `set_flag` | `pto.set_flag["SRC_PIPE","DST_PIPE","EVENT_ID"]` | 从一个 pipeline 向另一个 pipeline 发信号 |
| `wait_flag` | `pto.wait_flag["SRC_PIPE","DST_PIPE","EVENT_ID"]` | 等待既有事件 |

`TSYNC(events...)` 在每个 `RecordEvent` 上建立 happens-before 边；`TSYNC<Op>()` 为某类操作插入屏障。

### 向量指令集原语

| 原语 | 语法 | 说明 |
| --- | --- | --- |
| `set_flag` / `wait_flag` | `pto.set_flag[...]` / `pto.wait_flag[...]` | DMA 与向量计算之间的事件交接 |
| `mem_bar` | `pto.mem_bar` | GM↔UB 相关的内存栅栏 |

### DMA 原语

| 原语 | 语法 | 说明 |
| --- | --- | --- |
| `copy_gm_to_ubuf` | `pto.copy_gm_to_ubuf ...` | GM → UB |
| `copy_ubuf_to_gm` | `pto.copy_ubuf_to_gm ...` | UB → GM |
| `copy_ubuf_to_ubuf` | `pto.copy_ubuf_to_ubuf ...` | UB → UB |

DMA 与计算之间不存在隐式同步。

### 通信指令集原语

| 原语 | 说明 |
| --- | --- |
| `TBROADCAST` | 广播 |
| `TGET` / `TPUT` | 点对点通信 |
| `TWAIT` / `TTEST` | 跨 block 同步 |
| `TNOTIFY` / `TREDUCE` | 通知与归约 |

## 事件模型

PTO 使用事件驱动的同步模型。事件由三元组 `(src_pipe, dst_pipe, event_id)` 标识：

| 字段 | 含义 |
| --- | --- |
| `src_pipe` | 事件生产方 pipeline |
| `dst_pipe` | 事件消费方 pipeline |
| `event_id` | 事件槽编号 |

### 事件生命周期

```text
Producer -> set_flag -> event available -> wait_flag -> Consumer
```

事件一旦被设置，就对后续在同一三元组上等待的消费者可见。

### RecordEvent

Tile 操作的 C++ intrinsic 通常返回 `RecordEvent`，可作为后续操作的 `WaitEvents...`：

```cpp
RecordEvent e0 = TLOAD(a, ga);
RecordEvent e1 = TLOAD(b, gb);
TMATMUL(c, a, b, e0, e1);
```

这相当于在较高层表达显式事件链。

## 流水依赖图

AI Core 内的多个执行单元可以并行工作，但共享数据的地方必须显式建立顺序边：

```text
MTE2 (GM->UB) -> set_flag -> wait_flag -> Vector pipeline
Vector pipeline -> set_flag -> wait_flag -> MTE3 (UB->GM)
TLOAD -> RecordEvent / TSYNC -> Tile compute -> RecordEvent / TSYNC -> TSTORE
```

## 顺序规则

### Tile 指令顺序

1. 同一 tile buffer 内按程序顺序执行
2. 跨 tile buffer 或跨 pipeline 依赖通过事件建立
3. `RecordEvent` 或 `TSYNC` 决定后续 tile 操作何时可开始

### 向量指令顺序

1. `copy_gm_to_ubuf` 完成后，向量加载才能读取对应 UB 数据
2. 向量计算在同一 `SimdVecScopeOp` 内按程序顺序执行
3. 向量 store 完成后，`copy_ubuf_to_gm` 才能把结果搬回 GM

### GM 可见性

通过 `TSTORE` 或 `copy_ubuf_to_gm` 写入 GM 的数据，只在满足以下条件后对其他 block 可见：

1. 本 block 之前的 store 已完成
2. 需要的 `mem_bar` / `pipe_barrier` / 集合同步已建立
3. 运行时已确认完成

## 约束

!!! warning "约束"
    - 只要架构未自动保证顺序，就必须显式同步
    - 某目标可以更强，但文档不能依赖未声明的更强顺序
    - 向量流水同步规则与 tile 同步规则应分开陈述
    - `TSYNC` 作用于 tile-buffer 范围，不跨 tile buffer

## 不允许的情形

!!! danger "不允许的情形"
    - 把必须同步的路径写成“可省略”
    - 默认向量流水 hazard 会被 tile 规则自动覆盖
    - 把目标专属 barrier 写成架构普遍保证
    - 在 `copy_gm_to_ubuf` 完成前直接 `vlds`

## 相关页面

- [一致性基线](../memory-model/consistency-baseline_zh.md)
- [生产者-消费者排序](../memory-model/producer-consumer-ordering_zh.md)
- [Tile 同步与配置](../tile/sync-and-config_zh.md)
- [向量流水同步](../vector/pipeline-sync_zh.md)
- [标量流水同步](../scalar/pipeline-sync_zh.md)
