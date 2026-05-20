# dispatch_ffn_combine_v3 V4 PTO Showcase Design

## 0. 定位与边界

`dispatch_ffn_combine_v3` 当前已经是生产级 PTO 化实现：主链路使用 PTO `GlobalTensor` / `Tile` / `TASSIGN` / `TLOAD` / `TSTORE` / `TCVT` / `TEXPANDS` / vector arithmetic / `TMATMUL` / `TMATMUL_ACC` / `TSTORE_FP` / `TGET` / `TPUT` / `TNOTIFY` / `TWAIT`。V4 整改的目标不是把项目改成“纯 PTO”，也不是删除所有 AscendC/CANN substrate，而是让代码本身更清楚地展示 Parallel Tiling Operation 的编程模型优势。

V4 目标：

1. **数据流可读**：routing → remote gather → GMM1 → SwiGLU → GMM2 → combine → unpermute 的主链路在文件和函数边界上可见。
2. **任务流可读**：AIC 负责 GMM1/GMM2，AIV 负责 routing/dispatch/combine/restore 的角色边界清晰。
3. **同步流可读**：PTO comm signal 与 AscendC 本地 pipe/cross-core/cache substrate 的边界清晰。
4. **PTO seam 集中**：AscendC tensor/buffer 到 PTO global/tile view 的转换集中在少数 helper 中。
5. **保留项有归因**：未替换的 AscendC 接口明确属于 boundary adapter、coordination shell 或 substrate。

V4 不做：

- 不新建 `dispatch_ffn_combine_v4` 目录；先在当前 v3 项目内完成 showcase 整改。
- 不重写算法、tiling key、kernel ABI、host runtime、HCCL/ACL/MPI bootstrap。
- 不替换 `TQue/TBuf/TPipe` 生命周期。
- 不用 PTO event/comm primitive 强行替换 HardEvent、cross-core、cache coherence。
- 不恢复已清零的业务 `DataCopyPad` / `Duplicate`；可安全表达的 tail、stride、padded count、atomic store 和 UB fill 均走 PTO helper。

## 1. 文档分工

| 文档 | 职责 | 维护原则 |
| --- | --- | --- |
| `DESIGN.md` | V4 设计真值：PTO 数据流、任务流、同步流、primitive 映射、保留边界 | 设计或边界变化先更新这里 |
| `IMPLEMENTATION_PLAN.md` | 实施路线：阶段顺序、涉及文件、准出条件、验证节奏 | 只回答怎么做和何时验收 |
| `task.md` | 执行跟踪：V4 任务状态、验证结果、grep 台账、残留项 | 随开发进度更新 |
| `api_interface.md` | 非 PTO 直接依赖台账：boundary adapter / coordination shell / substrate | 保持与当前活树一致 |

## 2. V4 主链路

### 2.1 数据流

```text
input tokens / experts / probs
  -> routing + active mask
  -> token count / prefix / pre-rank sum
  -> PTO remote gather: TGET packed source payload into local expert-major input
  -> GMM1: PTO TMATMUL / TMATMUL_ACC
  -> SwiGLU / dequant / quant epilogue: PTO vector ops + TSTORE/TPUT where applicable
  -> GMM2: PTO TMATMUL / TMATMUL_ACC + TSTORE_FP
  -> combine return: local store or PTO TPUT to owner rank
  -> unpermute / weighted restore
  -> output
```

### 2.2 任务流

| Role | 当前入口 | V4 stage facade | 职责 |
| --- | --- | --- | --- |
| AIC | `GMM1`, `GMM2` | `RunGmm1Stage`, `RunGmm2Stage` | 专家维度 tile matmul、fixpipe store、GMM 间同步 |
| AIV | `DispatchAndCombine` | `RunRoutingStage`, `RunDispatchGatherStage`, `RunSwigluStage`, `RunCombineStage`, `RunRestoreStage` | routing/count、remote gather、SwiGLU、combine、restore |
| Remote window | `PtoRemoteWindow` | `PtoRemoteWindow` / stage context | remote signal、rank window、payload scratch |

### 2.3 同步流

| 同步类别 | PTO / AscendC 表达 | V4 边界 |
| --- | --- | --- |
| 跨 rank readiness | `pto::comm::TNOTIFY/TWAIT/TTEST` | 保留并作为 PTO comm 示例 |
| remote payload get/put | `pto::comm::TGET/TPUT` | 保留并显式标注为 PTO data movement |
| 本地 MTE/V/M/FIX pipe | `SetFlag/WaitFlag/PipeBarrier` thin bridge | 不改语义，只集中命名为 substrate bridge |
| cross-core 协作 | `CrossCoreSetFlag/CrossCoreWaitFlag` | 不替换，归类 coordination shell |
| cache coherence | `DataCacheCleanAndInvalid`, `dsb(DSB_DDR)`, `SyncAll` | 不替换，归类 remote-window substrate |

## 3. PTO primitive 映射

| PTO 模型 | 当前/目标代码位置 | 作用 |
| --- | --- | --- |
| `pto::GlobalTensor` + shape/stride | `op_kernel/utils/pto_global_view.hpp` | 统一 GM view，隐藏 AscendC `GlobalTensor` boundary seam |
| `pto::Tile` + `TASSIGN` | vector helper / epilogue / matmul helper | 把 AscendC local memory 地址绑定为 PTO tile |
| `TLOAD/TSTORE` | `pto_vector_ops.hpp`、soft-flag helper 和调用方 | 连续 GM↔UB 搬运，以及 soft-flag GM↔L1 Mat tile 搬运 |
| `TEXPANDS` | `pto_vector_ops.hpp` 和调用方 | UB tile 标量填充 / broadcast |
| `TCVT` | vector helper / epilogue | 类型转换 |
| `TADD/TADDS/TMUL/TMULS/TDIV/TABS/TEXP/TROWMAX` | epilogue / swiglu helper | vector compute |
| `TMATMUL/TMATMUL_ACC` | `block_mmad_preload_async_fixpipe_quant.hpp` | L0/L1 tile matmul |
| `TSTORE_FP` | `block_mmad_preload_async_fixpipe_quant.hpp` | int8 path accumulator fixpipe store |
| `TGET` | dispatch gather stage | remote read source payload |
| `TPUT` | combine/epilogue stage | remote write owner output |
| `TNOTIFY/TWAIT/TTEST` | `hccl_window.hpp` and stage sync | remote rank signal |

## 4. V4 结构整改

### 4.1 PTO helper 层

V4 将 PTO seam 分成三层：

| 文件 | 职责 |
| --- | --- |
| `op_kernel/utils/pto_global_view.hpp` | 只负责 PTO dynamic GM view adapter |
| `op_kernel/utils/pto_vector_ops.hpp` | 负责通用 PTO vector / matrix-row helper |
| `op_kernel/utils/pto_sync_bridge.hpp` | 负责命名化 AscendC local pipe bridge，不宣称纯 PTO |

### 4.2 Routing PTO adapter

`moe_v2_pto_sort.h` 中的本地 `PtoV2GlobalNd` / `MakeContiguousGlobal` / 通用 vector helper 需要收口到 V4 统一 helper。sort 专属 tile alias、常量和算法保留在 routing 内。

### 4.3 Stage facade

V4 使用 `op_kernel/stages/` 组织主链路 facade。每个 stage 是 header-only，保持 `PTO_DEVICE` inline，不改变 kernel ABI：

| Stage header | 职责 |
| --- | --- |
| `kernel_context.hpp` | 汇总执行上下文、阶段参数传递约定 |
| `routing_stage.hpp` | active mask、routing、local count |
| `dispatch_gather_stage.hpp` | count sync、remote gather、`TGET` |
| `gmm_stage.hpp` | GMM1/GMM2 high-level facade |
| `swiglu_stage.hpp` | GMM1 epilogue、SwiGLU、quant/dequant seam |
| `combine_stage.hpp` | CombineV1/V2 facade、local/remote return |
| `restore_stage.hpp` | unpermute / restore facade |

## 5. 未替换项与原因

| 未替换项 | 分类 | 原因 |
| --- | --- | --- |
| `kernel_operator.h`, `__aicore__`, `__gm__`, `GM_ADDR` | substrate | CANN kernel ABI 与编译属性，PTO 不替代这一层 |
| `LocalTensor`, `GlobalTensor` | substrate seam | PTO helper 通过它们获得地址/生命周期，不能一刀切删除 |
| `TPipe/TQue/TBuf` | substrate | PTO `Tile` 不是 queue/buffer allocator |
| `DataCopyPad` tail/pad/atomic | boundary adapter | 业务残留已清零；可安全展开的 tail、stride、padded count 和 atomic store 已走 `PtoLoadVector/PtoStoreVector/PtoStoreAtomicAddVector` |
| `SetFlag/WaitFlag/PipeBarrier/SyncAll` | local pipe substrate | HardEvent 生命周期不等价 PTO comm/event |
| `CrossCoreSetFlag/WaitFlag` | coordination shell | 本地跨核同步，不等价 remote signal |
| `DataCacheCleanAndInvalid`, `dsb` | cache substrate | remote window 可见性边界 |
| `GetBlockIdx/GetBlockNum/GetTaskRation` | runtime identity | PTO 不负责 runtime identity |
| `LoadData/Fixpipe/Gemm::helper::*` | matmul/fixpipe substrate | L1/L0/FIX 执行骨架，不能普通 `TLOAD/TSTORE` drop-in；`dispatch_policy_custom.hpp` 调用面已集中到 `substrate_bridge` |
| `dispatch_policy_custom.hpp` / `block_mmad_preload_async_fixpipe_quant.hpp` 中的 copy/fixpipe 壳 | matmul/fixpipe substrate | `dispatch_policy_custom.hpp` 的 `DataCopy/LoadData/LoadDataWithTranspose/Fixpipe` 仅保留在 PTO 命名 bridge body；`block_mmad_preload_async_fixpipe_quant.hpp` 的 soft-flag GM↔L1 copy 已走 PTO Mat `TLOAD/TSTORE`，其余 `PipeBarrier/CrossCoreSetFlag` 维持 coordination 归类 |

## 6. 通信与计算 PTO 化差距总表

### 6.1 通信相关

| 场景 | 当前状态 | PTO 覆盖情况 | 结论 / 后续动作 |
| --- | --- | --- | --- |
| remote data movement | dispatch gather / combine return 已走 `TGET/TPUT` | PTO 覆盖 | 已 PTO 化；当前未发现跨 rank payload 搬运绕过 PTO 的主路径 |
| remote signal / counter | token-ready / barrier counter 已走 `TNOTIFY/TWAIT` | PTO 覆盖 | 已 PTO 化；当前协议未用 `TTEST`，但不是缺口 |
| host ACL/HCCL/MPI bootstrap | host 侧仍负责 device/runtime、comm handle、remote window resource | PTO 不覆盖 host resource management | 保留 host runtime；不作为 kernel PTO 化缺口 |
| remote window 地址壳 | kernel 内仍有 `remoteWindow(...)` 地址计算和 context 传递 | PTO comm primitive 需要地址/view 输入 | 合理保留；不是实际 remote copy API |
| local cross-core sync | `CrossCoreSetFlag/CrossCoreWaitFlag` 保留 | 无清晰 PTO 1:1 替代；不等价 `TNOTIFY/TWAIT` | 保留为 coordination shell |
| cache coherence / visibility | `DataCacheCleanAndInvalid` / `dsb` 类语义保留 | 无清晰 PTO 1:1 替代 | 保留为 cache substrate |
| local pipe event | `SetFlag/WaitFlag/PipeBarrier/SyncAll` 仍在 helper body 内 | `TSYNC` 不能覆盖全部 HardEvent/pipe 生命周期 | 保留为 local pipe substrate；可做命名集中，不改语义 |

通信侧结论：跨 rank 数据搬运和 remote signal/counter 主链路已经 PTO 化；剩余项属于 host 建链、本地协调、cache 可见性或 pipe substrate。

### 6.2 计算相关

| 场景 | 当前状态 | PTO 覆盖情况 | 结论 / 后续动作 |
| --- | --- | --- | --- |
| vector compute / fill | `Cast/Add/Mul/Div/Abs/Exp/ReduceMax/Duplicate` 等已集中到 `pto_vector_ops.hpp` | PTO 覆盖 | 已 PTO 化；fill 走 `PtoFillVector/TEXPANDS` |
| regular GM-facing copy | 规整 GM↔UB vector load/store 已走 `TLOAD/TSTORE` helper；soft-flag GM↔L1 copy 已走 PTO Mat `TLOAD/TSTORE` helper | PTO 覆盖 | 已 PTO 化 |
| routing sort/gather | `moe_v2_pto_sort.h` 使用 PTO sort/gather/vector helper | PTO 覆盖 | 主要已 PTO 化 |
| matmul primitive | `PtoTileMmad` 使用 `TMATMUL/TMATMUL_ACC` | PTO 覆盖展示面 | PTO 化展示面已具备 |
| fixpipe store 展示面 | accumulator store path 使用 `TSTORE_FP` | PTO 部分覆盖 | 展示面已 PTO 化；底层 fixpipe 壳仍保留 |
| matmul/fixpipe substrate | `DataCopy/LoadData/LoadDataWithTranspose/Fixpipe/Gemm::*` 集中在 substrate 文件，其中 `dispatch_policy_custom.hpp` 调用面已收口到 `substrate_bridge` | PTO 只有部分方向，不是 drop-in | 当前不机械替换；保留 AscendC substrate body，但调用边界 PTO 命名化 |
| tail/pad boundary copy | `moe_v2_src_to_dst_and_gather.h`、`moe_token_unpermute.h`、`moe_v2_fullload_quant.h`、`dispatch_ffn_combine_kernel.hpp` 中可安全展开的连续/逐行 tail、stride、padded count copy 已改走 `PtoLoadVector/PtoStoreVector`；`op_kernel/` 下 `DataCopyPad/DataCopyExtParams/DataCopyPadExtParams` 已清零 | PTO 覆盖 | 通过显式 row stride 保持原 padding/stride 语义；build + small/large PASS |
| local atomic writeback | `moe_v2_expert_token_out.h` 的 expert count/cumsum 原子写回已改为 `TSTORE<AtomicAdd>` helper | PTO 覆盖 | 已验证 build + small/large PASS；后续同类规整 atomic store 可复用 helper |
| routing/quant buffer pipeline | `TQue/TBuf/TPipe/LocalTensor` 生命周期保留 | PTO `Tile` 不是 allocator/queue 替代 | 保留 AscendC substrate |
| kernel ABI/runtime identity | `kernel_operator.h/__aicore__/GM_ADDR/GetBlockIdx` 保留 | PTO 不覆盖 ABI/runtime identity | 合理保留 |

计算侧结论：vector/fill、规整 copy、soft-flag GM↔L1 copy、routing PTO helper、matmul primitive 展示面已经 PTO 化；local atomic writeback、连续单行 tail copy、unpermute tail、fullload row load/store、per-token row/scale store 与 expert-count padded copy 均已完成 PTO 收口；主要剩余差距集中在 matmul/fixpipe substrate body 和 AscendC buffer lifecycle。

### 6.3 后续推进优先级

| 优先级 | 候选点 | 原因 | 风险 |
| --- | --- | --- | --- |
| 1 | `moe_v2_expert_token_out.h` 的 `SetAtomicAdd/DataCopyPad/SetAtomicNone` | 已用 `TSTORE<AtomicAdd>` helper 收口规整 atomic store | 已完成；build + small/large PASS |
| 2 | `moe_v2_src_to_dst_and_gather.h` 的连续 tail load/store fallback | 已改走统一 `PtoLoadVector/PtoStoreVector`，覆盖非 32B 单行连续 copy | 已完成；build + small/large PASS |
| 3 | `moe_token_unpermute.h` / `moe_v2_fullload_quant.h` / `dispatch_ffn_combine_kernel.hpp` 中剩余 `DataCopyPad` boundary adapter | 已按连续/逐行语义展开为 PTO `PtoLoadVector/PtoStoreVector`；`op_kernel/` 下 `DataCopyPad/DataCopyExtParams/DataCopyPadExtParams` grep 清零 | 已完成；build + small/large PASS |
| 4 | 剩余 `Duplicate` fill | 已用 `PtoFillVector/TEXPANDS` 收口 | 已完成；build + small/large PASS |
| 5 | soft-flag GM↔L1 copy | 已用 PTO Mat `TLOAD/TSTORE` helper 收口 | 已完成；build + small/large PASS |
| 6 | `dispatch_policy_custom.hpp` substrate 调用面 | 已集中到 PTO 命名 `substrate_bridge`，不改变 L1/L0/FIX 语义 | 已完成；build + small/large PASS |
| 7 | local sync wrapper 集中到 `pto_sync_bridge.hpp` | 只改善命名和边界，不改变 HardEvent 语义 | 中低 |
| 8 | matmul/fixpipe substrate body 设计 | 残留最多，但牵涉 L1/L0/FIX 执行骨架 | 高；不建议机械替换 |
| 9 | `TPipe/TQue/TBuf/LocalTensor/GlobalTensor` 全替换 | 看似纯 PTO，但实际是运行 substrate | 很高；不建议作为目标 |

## 7. 正确性与性能策略

V4 是结构化整改，不做性能优化。所有阶段必须满足：

- build PASS；
- small case rank0/rank1 PASS；
- 关键阶段 large case rank0/rank1 PASS；
- 记录 kernel/e2e avg/min/max/std；
- 如出现 hang、mismatch 或异常回退，回退当前阶段，不扩散修改。

标准验证命令见 `IMPLEMENTATION_PLAN.md`。

## 8. 完成定义

V4 完成时应满足：

- PTO 数据流、任务流、同步流可从 stage 文件和函数名直接看出；
- 通用 PTO GM view/vector helper 不再分散复制；
- routing 的 PTO adapter 已接入统一 helper；
- 主 kernel 文件成为高层编排入口，而不是承载所有实现细节；
- 非 PTO 依赖全部落入明确边界并记录在 `api_interface.md`；
- small/large case 均 PASS。 
