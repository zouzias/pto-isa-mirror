# dispatch_ffn_combine_v3 API interface inventory

## 目的
这份清单只回答一件事：`dispatch_ffn_combine_v3` 当前还剩哪些 **PTO 之外的直接依赖接口**，以及它们分别属于哪一类：
- boundary adapter
- kernel coordination shell
- substrate / 宿主壳

它不再重复阶段任务状态；任务完成情况见 [task.md](task.md)。

## V4 showcase 口径

V4 在当前台账基础上新增一层展示性边界：

- **PTO 主模型**：`GlobalTensor/Shape/Stride/Tile/TASSIGN/TLOAD/TSTORE/TEXPANDS/TCVT/TMATMUL/TSTORE_FP/TGET/TPUT/TNOTIFY/TWAIT`，应在代码结构和文档中显性化。
- **PTO bridge**：AscendC `GlobalTensor/LocalTensor` 到 PTO view/tile 的转换，集中到 `pto_global_view.hpp` / `pto_vector_ops.hpp`。
- **AscendC substrate**：kernel ABI、queue/buffer 生命周期、本地 pipe/cross-core/cache coherence、带 stride/pad 语义的 boundary adapter，保留但必须有归因；规整 atomic store 和连续单行 tail copy 优先走 PTO helper。

V4 不是清零 AscendC API 的重写，`DESIGN.md` 和 `IMPLEMENTATION_PLAN.md` 是后续整改真值。B8 后 copy/fixpipe 区域的口径是：`dispatch_policy_custom.hpp` 的 `DataCopy/LoadData/LoadDataWithTranspose/Fixpipe` 只保留在 `substrate_bridge` body 内；`block_mmad_preload_async_fixpipe_quant.hpp` 的 soft-flag GM↔L1 copy 已走 PTO Mat `TLOAD/TSTORE`，其余 `PipeBarrier/CrossCoreSetFlag` 仍属于 coordination shell，不做 PTO drop-in 替换。

## 当前快照
- 这份清单对应当前活树，已覆盖 Task 6.1 ~ 6.5 的最终状态，并作为 V4 整改起点。
- `op_kernel/` 下 direct `Cast/Add/Adds/Mul/Muls/Div/Abs/Exp/ReduceMax` 已清零。
- routing 链 direct `PipeBarrier/SetWaitFlag/SyncAll` 已统一改走本地 PTO 风格 helper。
- 剩余 direct 非 PTO 接口主要集中在：
  1. 带 stride/pad 语义的 boundary adapter；
  2. cross-core / cache-coherence 这类 kernel coordination shell；
  3. matmul / fixpipe / Gemm substrate。

## PTO 公开映射参考
已确认存在并公开的 PTO 指令族：
- 数据搬运：`TLOAD` / `TSTORE` / `TMOV`
- 标量填充：`TEXPANDS`
- 类型转换：`TCVT`
- 向量算子：`TADD` / `TADDS` / `TMUL` / `TMULS` / `TDIV` / `TDIVS` / `TABS` / `TMAX` / `TMAXS`
- reduce：`TROWMAX` / `TCOLMAX`
- 同步/事件：`TASSIGN` / `TSYNC`
- matmul：`TMATMUL` / `TMATMUL_ACC`
- 通信：`TGET` / `TPUT` / `TNOTIFY` / `TWAIT`

仍需谨慎看待的 PTO 名字：
- `TSTORE_FP`
- `TMOV_FP`

它们虽然公开存在，但当前不应当成可以直接替换 v3 matmul/fixpipe substrate 的成熟 drop-in。

---

## 一、host/runtime 侧非 PTO 直接依赖

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| ACL runtime (`acl/acl.h`) | [main.cpp](main.cpp), [runtime_context.hpp](runtime_context.hpp) | No | 保留 | device/runtime 生命周期依赖，不是 PTO 替代范围。 |
| HCCL host bootstrap (`hccl/hccl_comm.h`, `hccl/hccl_types.h`) | [main.cpp](main.cpp), [runtime_context.hpp](runtime_context.hpp) | No | 保留 | host 侧建链与资源获取依赖；kernel 已不再直接触达。 |
| `HcomGetCommHandleByGroup` / `HcomGetL0TopoTypeEx` / `HcclAllocComResourceByTiling` | [runtime_context.hpp](runtime_context.hpp), [runtime_context.cpp](runtime_context.cpp) | No | 保留 | 仅允许停留在 host/runtime 层。 |

结论：host/runtime 的 HCCL/ACL 依赖仍然存在，但已经退回到正确边界。

---

## 二、AscendC substrate 依赖

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| `kernel_operator.h` | [op_kernel/dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp), [op_kernel/utils/dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp), [op_kernel/unpermute/moe_token_unpermute.h](op_kernel/unpermute/moe_token_unpermute.h) | No | 保留 | 整个 kernel substrate 总入口。 |
| `__aicore__`, `GM_ADDR` | 全部活 kernel 文件 | No | 保留 | kernel ABI / 编译属性。 |
| `LocalTensor`, `GlobalTensor` | 全部活 kernel 文件 | No | 保留 | PTO helper 仍直接消费 AscendC tensor substrate。 |
| `TPipe`, `TQue`, `TBuf` | routing / gather / unpermute / epilogue 文件 | No | 保留 | queue/buffer 生命周期仍是 AscendC substrate。 |
| `GetBlockIdx`, `GetBlockNum`, `GetTaskRation` | [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp), [hccl_window.hpp](op_kernel/utils/hccl_window.hpp) | No | 保留 | device builtins。 |
| `DataCacheCleanAndInvalid` | [hccl_window.hpp](op_kernel/utils/hccl_window.hpp) | No | 保留 | cache coherence 宿主壳。 |
| `SetL2CacheHint` | [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp) | No | 保留 | cache hint 调优接口。 |

结论：这些属于 PTO 当前仍依附的 AscendC substrate，不再作为 Stage 3b 未完成项处理。

---

## 三、业务/kernel 仍直接使用的非 PTO 接口

### 3.1 boundary adapter

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| `DataCopyPad` | 无业务残留 | Yes / Partial | 已清零 | `moe_v2_src_to_dst_and_gather.h`、`moe_token_unpermute.h`、`moe_v2_fullload_quant.h`、`dispatch_ffn_combine_kernel.hpp` 中可安全逐行/连续表达的 tail、stride、padded count copy 均已改走 `PtoLoadVector/PtoStoreVector`。 |
| `DataCopyPadExtParams` / `DataCopyExtParams` | 无业务残留 | Yes / Partial | 已清零 | 原参数结构不做 1:1 保留；已把对应语义展开为显式 row stride + PTO vector load/store。 |
| `SetAtomicAdd` / `SetAtomicNone` | 无业务残留 | Yes | 已用 PTO 收口 | `moe_v2_expert_token_out.h` 的 tensor atomic writeback 已改为 `PtoStoreAtomicAddVector` / `TSTORE<AtomicAdd>`。 |

### 3.2 向量计算接口

当前结论：**业务层 direct 向量算子已经清零。**

已收口到 PTO helper 的主入口：
- [moe_v2_pto_sort.h](op_kernel/moe_init_routing_quant_v2/moe_v2_pto_sort.h)
- [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
- [block_epilogue_pertoken_row.hpp](op_kernel/utils/block_epilogue_pertoken_row.hpp)
- [block_epilogue_pertoken_v2.hpp](op_kernel/utils/block_epilogue_pertoken_v2.hpp)
- [block_epilogue_pertoken_swiglu.hpp](op_kernel/utils/block_epilogue_pertoken_swiglu.hpp)

对应 PTO 面：
- `TCVT`
- `TADD/TADDS`
- `TMUL/TMULS`
- `TDIV`
- `TABS`
- `TROWMAX/TMAX`
- `TEXP`
- `TEXPANDS`

结论：向量计算已不再是当前活树里的 direct 非 PTO 残留面。

### 3.3 pipeline / sync / coordination

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| helper body 内部的 `PipeBarrier` / `SetFlag` / `WaitFlag` / `SyncAll` | [moe_v2_pto_sort.h](op_kernel/moe_init_routing_quant_v2/moe_v2_pto_sort.h), [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp), [block_epilogue_pertoken_row.hpp](op_kernel/utils/block_epilogue_pertoken_row.hpp), [block_epilogue_pertoken_v2.hpp](op_kernel/utils/block_epilogue_pertoken_v2.hpp), [block_epilogue_pertoken_swiglu.hpp](op_kernel/utils/block_epilogue_pertoken_swiglu.hpp), [hccl_window.hpp](op_kernel/utils/hccl_window.hpp) | Partial | 当前保留为 helper substrate | 业务调用面已经改成 PTO 风格 helper；剩下的是 wrapper 的底层实现。 |
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp), [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp) | No clear 1:1 | 当前保留 | 这是本地 cross-core 协作壳，不等价于 comm 侧 `TNOTIFY/TWAIT`。 |
| `DataCacheCleanAndInvalid` | [hccl_window.hpp](op_kernel/utils/hccl_window.hpp) | No | 当前保留 | remote-window cache coherence 壳。 |

结论：routing 链 direct sync 已完成 helper 化；当前剩余 direct sync 主要是 helper body 和 cross-core coordination shell。

---

## 四、matmul / fixpipe / Gemm substrate 残留

| 接口 / 依赖 | 位置 | PTO 是否有公开对应 | 结论 | 说明 |
| --- | --- | --- | --- | --- |
| `lib/matmul_intf.h` | [dispatch_ffn_combine.cpp](op_kernel/dispatch_ffn_combine.cpp) | No | 保留 | AscendC/Gemm substrate 头。 |
| `DataCopy` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp), [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp) | Yes / Partial | 已收口边界 | `dispatch_policy_custom.hpp` 中 direct call 仅保留在 `substrate_bridge::PtoDataCopy*` body；soft-flag GM↔L1 copy 已改走 PTO Mat `TLOAD/TSTORE` helper。 |
| `LoadData` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) | Partial | 已收口边界 | L1→L0 fragment copy 不做普通 `TLOAD` 直替，direct call 仅保留在 `substrate_bridge::PtoLoadDataL1ToL0*` body。 |
| `LoadDataWithTranspose` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) | Partial | 已收口边界 | 底层转置搬运面，direct call 仅保留在 `substrate_bridge::PtoLoadDataL1ToL0B` body。 |
| `Fixpipe` / `FixpipeParamsV220` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) | Partial | 已收口边界 | `TSTORE_FP/TMOV_FP` 方向存在但不做成熟 drop-in；direct call 仅保留在 `substrate_bridge::PtoFixpipeL0CToGm` body。 |
| `Gemm::Tile::*` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp), [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp) | Partial / No | 当前保留 | 已经通过本地 `Pto*` alias 收口，但底层仍是 Gemm substrate。 |
| `Gemm::helper::*` | [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) | No | 当前保留 | traits/helper 层，不是 PTO primitive。 |

结论：matmul/fixpipe 仍然是最重的硬残留，但已经从业务 kernel seam 中隔离出来。

---

## 五、按“还能不能继续改”重新归类

### A. 如需继续推进，优先级最高
- helper body 对 `TSYNC/TASSIGN` 的进一步吸收
- matmul/fixpipe substrate body 的进一步设计

### B. 当前已经完成收口，不再作为 Stage 3b 未完成项
- direct 向量算子和 `Duplicate` fill
- routing 链 direct `PipeBarrier/SetWaitFlag/SyncAll`
- kernel local `SyncAll<true>()` 业务调用点
- soft-flag GM↔L1 copy
- `dispatch_policy_custom.hpp` matmul/fixpipe substrate 调用点散落问题

### C. 当前明确属于 substrate / 宿主壳
- `kernel_operator.h`
- `__aicore__`, `GM_ADDR`
- `LocalTensor`, `GlobalTensor`
- `TPipe`, `TQue`, `TBuf`
- `GetBlockIdx`, `GetBlockNum`, `GetTaskRation`
- `DataCacheCleanAndInvalid`
- `SetL2CacheHint`
- `CrossCoreSetFlag`, `CrossCoreWaitFlag`
- `lib/matmul_intf.h`
- `Gemm::helper::*`

---

## 六、按文件看当前还剩哪些直接接口

### [op_kernel/dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
- `CrossCoreSetFlag` / `CrossCoreWaitFlag`
- `SetL2CacheHint`
- 本地 sync helper 的底层实现
- per-token row/scale 与 expert-count padded copy 已改为显式 row stride + PTO `PtoLoadVector/PtoStoreVector`
- 剩余 `Duplicate` fill 已改走 `kernel_detail::PtoFillVector/TEXPANDS`

### [op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h](op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h)
- 本文件 `DataCopyPad` 已清零
- 输入 tail load / 输出 tail store 已改走 PTO `PtoLoadVector/PtoStoreVector`

### [op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h](op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h)
- 本文件 `DataCopyPad` / `SetAtomicAdd` / `SetAtomicNone` 已清零
- expert count/cumsum 原子写回已改走 PTO `PtoStoreAtomicAddVector`

### [op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h](op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h)
- `DataCopyPad` 已清零
- `LoadXRows` 多行 load 已展开为逐行 PTO `PtoLoadVector`
- `StoreExpandedXRow` 单行 store 已改为 PTO `PtoStoreVector`

### [op_kernel/unpermute/moe_token_unpermute.h](op_kernel/unpermute/moe_token_unpermute.h)
- `DataCopyPad` / `DataCopyPadExtParams` 已清零
- token slice load/store 已统一走 PTO `PtoLoadVector/PtoStoreVector`

### [op_kernel/utils/hccl_window.hpp](op_kernel/utils/hccl_window.hpp)
- `DataCacheCleanAndInvalid`
- local barrier/sync wrapper body
- 核心 remote signal 已是 `pto::comm::TNOTIFY/TWAIT`

### [op_kernel/utils/dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp)
- `substrate_bridge::PtoDataCopyGmToL1*`
- `substrate_bridge::PtoDataCopyL1ToFp`
- `substrate_bridge::PtoLoadDataL1ToL0*`
- `substrate_bridge::PtoFixpipeL0CToGm`
- `Gemm::Tile::*`
- `Gemm::helper::*`

### [op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp)
- soft-flag GM↔L1 copy 已改为 PTO Mat `PtoLoadSoftFlagL1/PtoStoreSoftFlagL1`
- `PipeBarrier`
- `CrossCoreSetFlag`
- matmul event shell

---

## 七、直接结论
1. Stage 3b 范围内，**最该继续做的 direct 业务接口已经做完**：向量算子、routing/local sync 和业务层 `DataCopyPad` 均已收口。
2. 当前树上的非 PTO 直接依赖，主量已经不是业务 seam，而是：
   - cross-core / cache coordination shell；
   - matmul / fixpipe / Gemm substrate。
3. 如果后续还要继续收口，优先顺序应当是：
   1. helper body 对 `TSYNC/TASSIGN` 的进一步吸收；
   2. 单独处理 matmul/fixpipe substrate body。
