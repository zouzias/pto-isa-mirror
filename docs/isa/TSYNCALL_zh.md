# TSYNCALL

## 简介

`TSYNCALL` 是跨核同步屏障，支持 A2/A3 和 A5 NPU 后端。

- **硬件模式**（默认）：`TSYNCALL()` 同步当前 kernel 内的 AIV cores，语义对齐 AscendC 无参硬件 `SyncAll()`。`TSYNCALL<false>()` 用于 AIC/AIV 混合同步场景（A2/A3 支持，A5 因 FFTS 不可用仅支持 AIV-only 硬件模式）。
- **软件模式**：`TSYNCALL<TSyncAllMode::Soft>(...)` 使用 GM workspace 轮询实现跨核 barrier，不依赖 FFTS 调度，同时支持 AIV-only 和 AIC+AIV 混合模式。

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`：

```cpp
template <bool IsAIVOnly = true>
PTO_INST void TSYNCALL();

template <TSyncAllMode Mode, bool IsAIVOnly = true>
PTO_INST void TSYNCALL(__gm__ int32_t *gmWorkspace, __ubuf__ int32_t *ubWorkspace, int32_t usedCores = 0);
```

## 参数

- `gmWorkspace`: 软件模式使用的 GM workspace。调用前需要初始化为 0。
- `ubWorkspace`: 软件模式使用的 UB scratch，容量至少为 `usedCores * 8 * sizeof(int32_t)`。
- `usedCores`: 参与软件 barrier 的 AIV core 数。为 0 时使用 `get_block_num()`。

## 约束

- 当前实现覆盖 A2/A3 和 A5 后端。
- 软件模式支持 AIV-only kernel 和 AIC+AIV 混合 kernel。
  - A2/A3 混合模式：AIC 核通过 `copy_cbuf_to_gm`（L1→GM DMA）直接写入 GM slot，AIV 核通过 UB workspace 写入。
  - A5 混合模式：AIC 核无法直接 DMA 写 GM，改为通过 intra_block 信号委托同 block 的 AIV subblock 0 代为执行 UB→GM 写入。
- 软件模式要求所有参与 core 以相同顺序进入同一组 barrier；每个参与 core 在 `gmWorkspace` 中占用 8 个 `int32_t`，用于按 cache line 隔离同步计数。
- 软件模式只提供 barrier 到达语义。若 barrier 前后还需要观察其他 GM 数据，调用方仍需保证对应数据的 cache 可见性。
- 软件模式轮询循环内置退避策略（超过阈值后插入 `pipe_barrier`）和超时保护（默认 1,000,000 次迭代上限），超时后 kernel 侧 break 退出，CPU 模拟器构建下触发断言。
- 硬件 AIV-only `TSYNCALL()` 需要 kernel ELF 中带 `.ascend.meta.<kernel>_mix_aiv` metadata，使 runtime 按 `KERNEL_TYPE_MIX_AIV_1_0` 调度。
- `TSYNCALL` 不返回 `RecordEvent`，也不作为 `Event<SrcOp, DstOp>` 的依赖操作使用。
- 在 auto 模式下与现有同步指令保持一致，不直接发射硬件同步。
