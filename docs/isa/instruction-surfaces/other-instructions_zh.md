# 其他指令集

“其他”指令集覆盖那些不适合归入 tile、vector 或 scalar/control 主干的可见操作，包括跨 NPU 通信、集合操作，以及若干支撑性非 ISA 操作。

## 指令集概览

| 类别 | 说明 | Availability |
|------|------|--------------|
| 通信与运行时 | 跨 NPU 点对点与集合通信 | A2/A3, A5 |
| 非 ISA 与支撑操作 | tile 序列、量化、释放与辅助语义 | All profiles |

### 通信与运行时

这组操作跨越并行组中的多个 NPU，需要 `ParallelGroup` 句柄：

| 类别 | 操作 |
|------|------|
| 集合广播 | `tbroadcast`、`tscatter`、`tgather` |
| 点对点 | `tget`、`tget_async`、`tput`、`tput_async` |
| 集合归约 | `treduce` |
| 通知协议 | `tnotify`、`ttest`、`twait` |

**CPU 模拟器**：通信类操作在 CPU 仿真路径上不可用。

### 非 ISA 与支撑操作

这组操作提供更高层的语义，不一定对应单条核心 ISA 指令：

| 类别 | 操作 |
|------|------|
| Tile 序列 | `talias`、`tconcat`、`taxpy` |
| 内存管理 | `tfree` |
| 量化 | `tquant`、`tdequant` |
| 计数 / 谓词辅助 | `tpop`、`tpush` |
| A5 专属 | `thistogram`、`tpack`、`trandom` |

## 共享约束

- 通信操作要求所有参与 rank 使用一致的并行组协议。
- 非根 rank 在广播 / scatter 等操作中必须准备好可写目标缓冲区。
- 支撑操作仍然要满足各自的类型、形状与 profile 约束。

## 不允许的情形

- 把 CPU 仿真路径当成跨 NPU 通信 profile。
- 在不支持的 profile 上使用 A5 专属支撑操作。
- 把 tile 序列 / 量化类支撑操作误当成普通逐元素 tile 指令。

## 相关页面

- [其他与通信参考](../other/README_zh.md)
- [通信与运行时](../other/communication-and-runtime_zh.md)
- [非 ISA 与支撑操作](../other/non-isa-and-supporting-ops_zh.md)
