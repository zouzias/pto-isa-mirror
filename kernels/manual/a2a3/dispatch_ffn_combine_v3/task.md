# dispatch_ffn_combine_v3 task list

## V4 PTO showcase 整改目标

V4 在 V3 已经 PTO 化的生产实现基础上，继续做结构化整改：让代码本身更清楚展示 Parallel Tiling Operation 的数据流、任务流、同步流和 primitive 映射。V4 不追求纯 PTO，不删除必要 AscendC substrate。

## V4 当前状态

| 任务 | 状态 | 结果 / 验收 |
| --- | --- | --- |
| V4-B0 文档与任务基线 | 已完成 | 新增 `DESIGN.md`、`IMPLEMENTATION_PLAN.md`，更新 `task.md` / `api_interface.md`。 |
| V4-B1 统一 PTO bridge / vector ops | 已完成 | 新增 `op_kernel/utils/pto_vector_ops.hpp`，收口重复 vector helper；build + small PASS。 |
| V4-B2 收口 routing PTO adapter | 已完成 | `moe_v2_pto_sort.h` 复用统一 PTO helper；small/large PASS。 |
| V4-B3 主 kernel stage facade 拆分 | 已完成 | 新增 `op_kernel/stages/` facade，AIC/AIV 主路径显式表达 GMM / routing / gather / SwiGLU / combine / restore；small/large PASS。 |
| V4-B4 copy/fixpipe substrate 边界清理 | 已完成 | 评估完成，无额外代码改动；`DataCopy/LoadData/LoadDataWithTranspose/Fixpipe` 继续归类为 matmul/fixpipe substrate。 |
| V4-B5 验证与 showcase 收口 | 已完成 | README/API/DESIGN/task 台账已更新；B4/B5 未改 kernel 代码，沿用 V4-B3 small/large PASS。 |
| V4-B6 atomic/tail PTO 实验 | 已完成 | `moe_v2_expert_token_out.h` 原子写回改用 `TSTORE<AtomicAdd>`；`moe_v2_src_to_dst_and_gather.h` 连续单行 tail load/store 改走 `TLOAD/TSTORE` helper；build + small/large PASS。 |
| V4-B7 数据搬移 PTO 收口 | 已完成 | `unpermute` tail、`fullload` row load/store、per-token row/scale store、expert-count padded copy 均改走 PTO `PtoLoadVector/PtoStoreVector`；`op_kernel/` 下 `DataCopyPad/DataCopyExtParams/DataCopyPadExtParams/SetAtomic*` 清零；small/large PASS。 |
| V4-B8 剩余填充/soft-flag/substrate bridge PTO 收口 | 已完成 | `Duplicate` 改走 `PtoFillVector/TEXPANDS`；soft-flag GM↔L1 copy 改走 PTO Mat `TLOAD/TSTORE`；`dispatch_policy_custom.hpp` matmul/fixpipe substrate 调用集中到 `substrate_bridge`；build + small/large PASS。 |

## V4 不动边界

- 不改 `TQue/TBuf/TPipe` 生命周期。
- 不替换 HardEvent、CrossCore、cache coherence 语义。
- `DataCopyPad` 业务调用已清零；已确认规整 atomic store、连续单行 tail copy、逐行 stride copy 和 padded count copy 可走 PTO helper。
- `Duplicate` 业务调用已清零；规整 UB fill 改走 `PtoFillVector/TEXPANDS`。
- 不改 runtime identity、kernel ABI、tiling key、host ACL/HCCL/MPI bootstrap。
- 不把 `dispatch_policy_custom.hpp` 的 `DataCopy/LoadData/Fixpipe` 机械替成 PTO primitive；仅把 substrate 调用面集中到 PTO 命名 bridge。

## V4 验收命令

small case：

```bash
bash kernels/manual/a2a3/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

large case：

```bash
bash kernels/manual/a2a3/dispatch_ffn_combine_v3/run.sh \
  --soc ascend910_93 \
  --world-size 2 \
  --m 4097 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 8194
```

---

## 目标
- 完成 Stage 3b 的功能迁移，优先把已有等价面的实现收口到 PTO 风格。
- HCCL 仅保留 host 侧建链和 remote GM window 获取职责，kernel 侧只保留 PTO-neutral remote-window 语义。
- 先完成功能覆盖和依赖收口，再单独做性能优化。

## 当前结论
- Task 1 已完成：runtime -> tiling -> kernel 的 HCCL 语义已经压缩成 remote-window 壳，kernel 侧不再直接消费 host HCCL API。
- Task 6.1 已完成：`op_kernel/` 下 direct `Cast/Add/Adds/Mul/Muls/Div/Abs/Exp/ReduceMax` 已清零；向量计算统一收口到本地 PTO helper。
- Task 6.2 已完成：routing 链、epilogue 链和 kernel 本地同步链上的 direct `PipeBarrier/SetWaitFlag/SyncAll` 已收口到 PTO 风格 helper；剩余 direct `CrossCore*` 属于跨核协作壳，不再算业务层漏改。
- Task 6.3 已完成：`DataCopyPad` / `DataCopyPadExtParams` 只剩 boundary adapter，不再散落在业务 compute seam 内部。
- Task 6.4 已完成：matmul / fixpipe / Gemm 残留已经稳定收敛到 [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp) 和 [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp)。
- Task 6.5 已完成：v3 的非 PTO 直接依赖已经分成 boundary adapter、kernel coordination shell、substrate 三类，并同步到 [api_interface.md](api_interface.md)。
- V4-B6 已完成：expert token count/cumsum 原子写回已用 PTO `TSTORE<AtomicAdd>`，`src_to_dst_and_gather` 的连续单行 tail load/store 已用 PTO `TLOAD/TSTORE` helper，build + small/large PASS。
- V4-B7 已完成：`unpermute` 非对齐 tail、`fullload` 多行 load/单行 store、per-token row/scale store、expert-count padded copy 均已改走 PTO `PtoLoadVector/PtoStoreVector`；`op_kernel/` 下 `DataCopyPad/DataCopyExtParams/DataCopyPadExtParams/SetAtomic*` grep 为 0。
- V4-B8 已完成：`Duplicate` 业务调用已改走 PTO `PtoFillVector/TEXPANDS`；soft-flag GM↔L1 copy 已改走 PTO Mat `TLOAD/TSTORE`；`dispatch_policy_custom.hpp` 的 matmul/fixpipe substrate 调用已集中到 `substrate_bridge`，build + small/large PASS。

## 最新验证
- build
  - `cmake --build kernels/manual/a2a3/dispatch_ffn_combine_v3/build --target dispatch_ffn_combine_v3 -j16`
  - V4-B1 后 PASS
  - V4-B3 删除旧 `DispatchAndCombine` 主体后 PASS
  - V4-B4/B5 未改 kernel 代码，沿用 V4-B3 验证结果
- small case
  - `bash kernels/manual/a2a3/dispatch_ffn_combine_v3/run.sh --soc ascend910_93 --world-size 2 --m 16 --k 128 --n 128 --topk 2 --experts 2 --max-output-size 32`
  - V4-B1 后 kernel `63.41 us`, e2e `157.06 us`, rank0/rank1 PASS
  - V4-B2 后 kernel `49.76 us`, e2e `153.00 us`, rank0/rank1 PASS
  - V4-B3 删除旧主体后 kernel `30.01 us`, e2e `144.16 us`, rank0/rank1 PASS
  - V4-B4/B5 未改 kernel 代码，沿用 V4-B3 验证结果
  - V4-B6 atomic/tail PTO 实验后 kernel `46.64 us`, e2e `139.69 us`, rank0/rank1 PASS
  - V4-B7 数据搬移 PTO 收口后 kernel `49.98 us`, e2e `154.72 us`, rank0/rank1 PASS
  - V4-B7 最终清理后复验 kernel `22.46 us`, e2e `112.85 us`, rank0/rank1 PASS
  - V4-B8 剩余填充/soft-flag/substrate bridge PTO 收口后 kernel `35.55 us`, e2e `115.00 us`, rank0/rank1 PASS
- large case
  - `bash kernels/manual/a2a3/dispatch_ffn_combine_v3/run.sh --soc ascend910_93 --world-size 2 --m 4097 --k 128 --n 128 --topk 2 --experts 2 --max-output-size 8194`
  - V3 收口点 kernel `29.29 us`, e2e `116.56 us`, rank0/rank1 PASS
  - V4-B2 后 kernel `43.20 us`, e2e `149.26 us`, rank0/rank1 PASS
  - V4-B3 删除旧主体后 kernel `35.18 us`, e2e `119.46 us`, rank0/rank1 PASS
  - V4-B4/B5 未改 kernel 代码，沿用 V4-B3 验证结果
  - V4-B6 atomic/tail PTO 实验后 kernel `92.76 us`, e2e `244.53 us`, rank0/rank1 PASS
  - V4-B7 数据搬移 PTO 收口后 kernel `54.78 us`, e2e `162.57 us`, rank0/rank1 PASS
  - V4-B7 最终清理后复验 kernel `49.70 us`, e2e `144.65 us`, rank0/rank1 PASS
  - V4-B8 剩余填充/soft-flag/substrate bridge PTO 收口后 kernel `29.94 us`, e2e `117.63 us`, rank0/rank1 PASS

## 任务完成表

| 任务 | 状态 | 结果 |
| --- | --- | --- |
| Task 1 runtime/kernel 侧 HCCL 语义压成 remote-window 壳 | 已完成 | host/runtime 保留 HCCL bootstrap；kernel 只消费 remote-window + PTO comm 语义。 |
| Task 2.1 GM-facing copy 收口 | 已完成 | 业务 kernel 中 direct GM-facing `DataCopy` 已清零。 |
| Task 2.2 带 padding / atomic 的系统边界 copy 收口 | 已完成 | 剩余 `DataCopyPad` 仅保留在 boundary adapter。 |
| Task 2.3 `fullload_quant` compute seam | 已完成 | 维持 hot-path adapter，避免大 case 稳定性回退。 |
| Task 3.1 向量 helper 收敛 | 已完成 | direct 向量算子已经收口到 PTO helper。 |
| Task 3.2 pipeline/sync helper 收敛 | 已完成 | routing / epilogue / kernel local sync 已 helper 化。 |
| Task 4.1 Gemm/matmul 外围壳隔离 | 已完成 | matmul/fixpipe 残留集中到 substrate 文件。 |
| Task 4.2 Gemm 壳里的具体残留接口 | 已完成 | `Gemm::Tile::*` / `Gemm::helper::*` / `Fixpipe` 已稳定归类为 substrate。 |
| Task 5 残留非-PTO 依赖台账 | 已完成 | 当前活树的残留已经重新归档。 |
| Task 6.1 向量算子 PTO 化 | 已完成 | `op_kernel/` 下 direct 向量算子 grep 为 0。 |
| Task 6.2 pipeline/sync helper PTO 化 | 已完成 | routing 链 direct sync 已清空；kernel local `SyncAll<true>()` 已收口到 `kernel_detail::PtoSyncAll<true>()`。 |
| Task 6.3 boundary adapter 再评估 | 已完成 | 剩余 `DataCopyPad` 已确认属于 boundary adapter，不继续混入业务 seam。 |
| Task 6.4 substrate 壳精细拆分 | 已完成 | `dispatch_policy_custom.hpp` 与 `block_mmad_preload_async_fixpipe_quant.hpp` 成为唯一主集中点。 |
| Task 6.5 AscendC substrate 与宿主壳台账 | 已完成 | 当前剩余依赖的职责边界已经固定。 |
| V4-B6 atomic/tail PTO 实验 | 已完成 | `TSTORE<AtomicAdd>` 替换 expert count/cumsum 原子写回；`TLOAD/TSTORE` 替换 `src_to_dst_and_gather` 的连续单行 tail fallback。 |
| V4-B7 数据搬移 PTO 收口 | 已完成 | `unpermute` tail、`fullload` row load/store、per-token row/scale store、expert-count padded copy 均改走 PTO `PtoLoadVector/PtoStoreVector`。 |
| V4-B8 剩余填充/soft-flag/substrate bridge PTO 收口 | 已完成 | `PtoFillVector/TEXPANDS` 覆盖剩余 fill；soft-flag GM↔L1 copy 改走 PTO Mat `TLOAD/TSTORE`；matmul/fixpipe substrate 调用集中到 `substrate_bridge`。 |

## 残留非-PTO 依赖台账

### 1. boundary adapter 暂留
- [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
  - `DataCopyPad` 已清零
  - 说明：per-token row/scale 与 expert-count padded copy 已展开为显式 row stride + PTO `PtoLoadVector/PtoStoreVector`。
- [moe_v2_src_to_dst_and_gather.h](op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h)
  - 已清零本文件的 `DataCopyPad`
  - 说明：输入 tail load / 输出 tail store 是连续单行 copy，已改走 `PtoLoadVector/PtoStoreVector`。
- [moe_v2_expert_token_out.h](op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h)
  - 已清零本文件的 `DataCopyPad`, `SetAtomicAdd`, `SetAtomicNone`
  - 说明：atomic 写回 adapter 已改走 `PtoStoreAtomicAddVector` / `TSTORE<AtomicAdd>`。
- [moe_v2_fullload_quant.h](op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_quant.h)
  - `DataCopyPad` 已清零
  - 说明：`LoadXRows` 多行 load 和 `StoreExpandedXRow` 单行 store 已改走 PTO helper。
- [moe_token_unpermute.h](op_kernel/unpermute/moe_token_unpermute.h)
  - `DataCopyPad`, `DataCopyPadExtParams` 已清零
  - 说明：尾块 load/store 已统一走 PTO helper。

### 2. kernel coordination shell 暂留
- [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
  - `CrossCoreSetFlag`, `CrossCoreWaitFlag`, `SetL2CacheHint`
  - 说明：跨核协作和 cache hint 壳；当前没有清晰 public PTO 1:1 对应面。
- [hccl_window.hpp](op_kernel/utils/hccl_window.hpp)
  - `DataCacheCleanAndInvalid`
  - 说明：remote-window cache coherence 壳；核心 notify/wait 已是 `pto::comm::TNOTIFY/TWAIT`。
- 本地 helper 定义
  - [moe_v2_pto_sort.h](op_kernel/moe_init_routing_quant_v2/moe_v2_pto_sort.h)
  - [dispatch_ffn_combine_kernel.hpp](op_kernel/dispatch_ffn_combine_kernel.hpp)
  - [block_epilogue_pertoken_row.hpp](op_kernel/utils/block_epilogue_pertoken_row.hpp)
  - [block_epilogue_pertoken_v2.hpp](op_kernel/utils/block_epilogue_pertoken_v2.hpp)
  - [block_epilogue_pertoken_swiglu.hpp](op_kernel/utils/block_epilogue_pertoken_swiglu.hpp)
  - [hccl_window.hpp](op_kernel/utils/hccl_window.hpp)
  - 说明：这些文件内部仍用 AscendC 原语实现 wrapper，但业务调用面已经统一转到 PTO 风格 helper。

### 3. substrate / 宿主壳暂留
- [dispatch_policy_custom.hpp](op_kernel/utils/dispatch_policy_custom.hpp)
  - `substrate_bridge::PtoDataCopyGmToL1*`, `PtoDataCopyL1ToFp`, `PtoLoadDataL1ToL0*`, `PtoFixpipeL0CToGm`, `Gemm::Tile::*`, `Gemm::helper::*`
  - 说明：matmul/fixpipe substrate 的集中实现面；direct `DataCopy/LoadData/LoadDataWithTranspose/Fixpipe` 仅保留在 bridge body 内，不散落在调用点。
- [block_mmad_preload_async_fixpipe_quant.hpp](op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp)
  - `PipeBarrier`, `CrossCoreSetFlag`, matmul event shell
  - 说明：主算外围 shell，已从业务 kernel 中隔离出来；soft-flag GM↔L1 copy 已改走 PTO Mat `PtoLoadSoftFlagL1/PtoStoreSoftFlagL1`。
- 通用 AscendC substrate
  - `kernel_operator.h`, `__aicore__`, `GM_ADDR`, `LocalTensor`, `GlobalTensor`, `TPipe`, `TQue`, `TBuf`
  - 说明：当前 PTO 仍依附这层 substrate 运行。

## 性能记录（只记录，不优化）

| 检查点 | small kernel / e2e | large kernel / e2e | 说明 |
| --- | --- | --- | --- |
| swiglu 向量 seam 收口后 | `54.30 us / 147.36 us` | `34.80 us / 131.66 us` | direct 向量热段转 PTO helper 的首个稳定点。 |
| 第一批 sync helper 收口后 | `24.18 us / 125.36 us` | `28.90 us / 108.74 us` | routing / epilogue / kernel helper 首轮收口。 |
| 本轮 routing + kernel sync 收口后 | `29.00 us / 114.37 us` | `29.29 us / 116.56 us` | 6.2 完成后的当前稳定点。 |
| V4-B3 stage facade 删除旧主体后 | `30.01 us / 144.16 us` | `35.18 us / 119.46 us` | 主 kernel 已显式拆成 routing / gather / GMM / SwiGLU / combine / restore stage；B4/B5 未改 kernel 代码。 |
| V4-B6 atomic/tail PTO 实验后 | `46.64 us / 139.69 us` | `92.76 us / 244.53 us` | `TSTORE<AtomicAdd>` 覆盖 expert count/cumsum 原子写回；`TLOAD/TSTORE` 覆盖 `src_to_dst_and_gather` 连续单行 tail fallback。 |
| V4-B7 数据搬移 PTO 收口后 | `49.98 us / 154.72 us` | `54.78 us / 162.57 us` | `DataCopyPad/DataCopyExtParams/DataCopyPadExtParams/SetAtomic*` 在 `op_kernel/` 下清零；剩余数据搬移主量为 PTO helper 或 matmul/fixpipe substrate。 |
| V4-B8 剩余填充/soft-flag/substrate bridge 收口后 | `35.55 us / 115.00 us` | `29.94 us / 117.63 us` | `Duplicate` 清零；soft-flag GM↔L1 改 PTO Mat `TLOAD/TSTORE`；`dispatch_policy_custom` substrate 调用集中到 PTO 命名 bridge。 |

## V4 最终验收 checklist
- [x] `op_kernel/utils/pto_global_view.hpp` / `pto_vector_ops.hpp` 集中 PTO bridge 和 vector helper
- [x] `moe_v2_pto_sort.h` 复用统一 PTO helper，sort 专属 tile/algorithm 保持本地
- [x] `op_kernel/stages/` 显式表达 AIC/AIV 任务流和 routing/gather/GMM/SwiGLU/combine/restore 数据流
- [x] copy/fixpipe substrate 已复核；不做 `DataCopy/LoadData/LoadDataWithTranspose/Fixpipe` PTO drop-in 替换
- [x] README/API/DESIGN/task 台账已同步 V4 showcase 口径
- [x] V4-B3 删除旧主体后 build 成功
- [x] V4-B3 删除旧主体后 small case PASS
- [x] V4-B3 删除旧主体后 large case PASS
- [x] V4-B6 atomic/tail PTO 实验后 build 成功
- [x] V4-B6 atomic/tail PTO 实验后 small case PASS
- [x] V4-B6 atomic/tail PTO 实验后 large case PASS
- [x] V4-B7 数据搬移 PTO 收口后 build 成功
- [x] V4-B7 数据搬移 PTO 收口后 small case PASS
- [x] V4-B7 数据搬移 PTO 收口后 large case PASS
- [x] V4-B8 剩余填充/soft-flag/substrate bridge 收口后 build 成功
- [x] V4-B8 剩余填充/soft-flag/substrate bridge 收口后 small case PASS
- [x] V4-B8 剩余填充/soft-flag/substrate bridge 收口后 large case PASS

## 最终验收 checklist
- [x] HCCL/HCOM 直接 API 仅保留在 host runtime 层
- [x] 业务 kernel 中 direct GM-facing `DataCopy` 已清零
- [x] `DataCopyPad/DataCopyExtParams/DataCopyPadExtParams/SetAtomic*` 已在 `op_kernel/` 下清零
- [x] `op_kernel/` 下 direct 向量算子已清零
- [x] routing / epilogue / kernel local sync 已收口到 PTO 风格 helper
- [x] matmul/fixpipe 残留已集中隔离到 substrate 文件
- [x] build 成功
- [x] small case PASS
- [x] large case PASS
- [x] 剩余非-PTO 依赖均已给出明确归因
