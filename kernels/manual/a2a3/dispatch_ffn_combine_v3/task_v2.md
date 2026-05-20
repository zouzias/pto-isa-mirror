# dispatch_ffn_combine_v3 task_v2 list

## 目标
- 在 `task.md` 已完成的 Stage 3b 基础上，继续减少 v3 device 侧对 AscendC 公共 API 的直接暴露。
- 优先把能用 PTO 公共模型表达的 `GlobalTensor / LocalTensor / DataCopy / DataCopyPad / signal wait-notify` 收口到 `include/pto/**` 的类型和原语。
- 不机械删除 `kernel_operator.h`；只有当对应文件不再直接使用 AscendC substrate 时才移除 include。

## 适用环境
- 当前项目默认：CANN 8.5.0 / A3 / `ascend910_93`。
- 本阶段只改 `kernels/manual/a2a3/dispatch_ffn_combine_v3/` 内部。
- 不做性能优化；每个功能性替换都必须 build + small case 验证，关键阶段再跑 large case。

## 边界规则
- host runtime / ACL / HCCL bootstrap / tiling 平台头暂不替换。
- `TPipe / TQue / TBuf` 没有 PTO 1:1 API；只能在具体路径重构成 `pto::Tile + TASSIGN + event` 后再移除。
- `SetFlag / WaitFlag / PipeBarrier / SyncAll` 不能一刀切；只有 PTO primitive 相邻链路才能改成 `RecordEvent / TSYNC`，跨核或底层 pipe seam 先保留。
- `GetBlockIdx / GetBlockNum / GetTaskRation / GetTPipePtr` 属于 AscendC substrate 或 runtime identity，暂不作为第一批替换对象。

## 当前基线
- `task.md` 已记录：direct 向量算子已清零，普通 GM-facing `DataCopy` 已清零，`DataCopyPad` 已收口为 boundary adapter，matmul/fixpipe 已集中隔离到 substrate 文件。
- 本轮新关注点：进一步减少 `kernel_operator.h` 暴露面，并把已经能由 PTO 类型表达的 helper 签名继续向 PTO 公共 API 收口。

## 任务完成表

| 任务 | 状态 | 结果 |
| --- | --- | --- |
| Task 0 重新建立 AscendC 直接依赖基线 | 已完成 | 已完成源码基线统计，依赖集中在 tensor/view、pipe queue、sync/barrier、copy/fill/vector、runtime identity、matmul/fixpipe substrate 六类。 |
| Task 1 `kernel_operator.h` include 分层台账 | 已完成 | 直接 include 仍集中在 kernel entry、routing base、unpermute、substrate helper；多数组件通过上层头传递 AscendC 类型，暂不能机械删除。 |
| Task 2 PTO GlobalTensor 边界收口 | 进行中 | 把已存在的 `MakeContiguousGlobal` / dynamic shape / stride helper 收拢成更少的 PTO-facing adapter；业务 helper 优先接收 PTO view。 |
| Task 3 PTO Tile 替代 LocalTensor 的低风险 seam | 未开始 | 只挑已经由 PTO `TLOAD/TSTORE/TMATMUL` 消费的局部 buffer seam；不碰 routing TQue 生命周期。 |
| Task 4 剩余 `DataCopyPad` 再拆分 | 未开始 | 能表达为连续 PTO tile load/store 的改成 `TLOAD/TSTORE`；真正 padding/tail boundary 继续留 adapter 并记录原因。 |
| Task 5 通信 ready/barrier 协议 PTO 化复查 | 未开始 | `hccl_window.hpp` 已有 `TNOTIFY/TWAIT`；检查是否还有 GM signal 可改成 `TTEST/TWAIT`。 |
| Task 6 `RecordEvent/TSYNC` 替换试点 | 未开始 | 只在相邻操作已 PTO 化的局部链路上替换显式 `SetFlag/WaitFlag`；不动跨核和底层 pipe seam。 |
| Task 7 `TPipe/TQue/TBuf` 重构候选评估 | 未开始 | 列出最小可迁移路径；只有找到可验证的 `Tile + TASSIGN` 结构后才动代码。 |
| Task 8 `kernel_operator.h` include shrink | 未开始 | 完成前序替换后删除不再需要的直接 include / using；保持 standalone build 通过。 |
| Task 9 回归验证与台账更新 | 未开始 | 干净 build、small case PASS；阶段完成后跑 large case，并更新本文件。 |

## Task 0 直接依赖基线

| 类别 | 命中数 | 主要文件 |
| --- | ---: | --- |
| Tensor/view | 767 | `op_kernel/moe_init_routing_quant_v2/moe_v2_pto_sort.h` (100)<br>`op_kernel/dispatch_ffn_combine_kernel.hpp` (97)<br>`op_kernel/utils/block_epilogue_pertoken_swiglu.hpp` (63) |
| AscendC include/ns | 526 | `op_kernel/dispatch_ffn_combine_kernel.hpp` (154)<br>`op_kernel/utils/dispatch_policy_custom.hpp` (84)<br>`op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp` (77) |
| Pipe/queue/buf | 419 | `op_kernel/moe_init_routing_quant_v2/moe_v2_gather_dynamic_quant.h` (73)<br>`op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h` (70)<br>`op_kernel/moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h` (42) |
| Sync/barrier | 282 | `op_kernel/dispatch_ffn_combine_kernel.hpp` (77)<br>`op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp` (45)<br>`op_kernel/utils/block_epilogue_pertoken_v2.hpp` (38) |
| Scalar Get/Set | 70 | `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h` (12)<br>`op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h` (9)<br>`op_kernel/moe_init_routing_quant_v2/moe_v2_gather_dynamic_quant.h` (8) |
| Copy/fill/vector | 61 | `op_kernel/moe_init_routing_quant_v2/moe_v2_src_to_dst_and_gather.h` (11)<br>`op_kernel/dispatch_ffn_combine_kernel.hpp` (7)<br>`op_kernel/moe_init_routing_quant_v2/moe_v2_expert_token_out.h` (7) |
| Runtime identity | 22 | `op_kernel/utils/dispatch_policy_custom.hpp` (7)<br>`op_kernel/utils/hccl_window.hpp` (4)<br>`op_kernel/dispatch_ffn_combine_kernel.hpp` (2) |
| Cube/fixpipe substrate | 9 | `op_kernel/utils/dispatch_policy_custom.hpp` (9) |

## Task 1 `kernel_operator.h` include 分层台账

| 分类 | 文件 | 处理策略 |
| --- | --- | --- |
| 入口/launch 直接依赖 | `op_kernel/dispatch_ffn_combine.cpp`, `op_kernel/dispatch_ffn_combine.h`, `op_kernel/dispatch_ffn_combine_kernel.hpp` | 暂保留；先减少业务 helper 签名里的 AscendC 类型。 |
| routing base 直接依赖 | `moe_v2_common.h`, `moe_v2_fullload_quant_base.h`, `moe_v2_gather_out.h`, `moe_v2_gather_quant.h`, `moe_v2_mrgsort*.h`, `moe_v2_sort_base.h` | 暂保留；这些文件仍拥有 `TPipe/TQue/GlobalTensor/LocalTensor` 生命周期。 |
| unpermute 直接依赖 | `op_kernel/unpermute/moe_token_unpermute.h` | 暂保留；`TPipe/TQue/DataCopyPad` 还没 PTO 化。 |
| substrate/helper 直接依赖 | `dispatch_policy_custom.hpp`, `hccl_window.hpp`, `get_tensor_addr.hpp`, `layout3d.hpp` | `dispatch_policy_custom.hpp`/`hccl_window.hpp` 保留；两个小工具后续可检查是否通过上层头间接化。 |
| transitive AscendC user | `block_epilogue_pertoken*.hpp`, `block_mmad_preload_async_fixpipe_quant.hpp`, `moe_v2_pto_sort.h` 等 | 优先改这些 PTO seam 的签名与 adapter，减少 `AscendC::GlobalTensor/LocalTensor` 外露。 |

## 第一批执行顺序
1. Task 0：用 grep 生成直接依赖基线，写入本文件。已完成。
2. Task 1：按文件给 `kernel_operator.h` 分层。已完成。
3. Task 2：先处理已有 PTO bridge 的重复 `GlobalTensor` adapter，不碰算法路径。
4. Task 4：挑 `dispatch_ffn_combine_kernel.hpp` 中最清晰的剩余 `DataCopyPad` seam 做试点。
5. Task 9：build + small case 验证。

## 验证命令

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:$LD_LIBRARY_PATH
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
export MPI_RUNNER=mpirun
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

## 当前进度
- 进度：2/9
- 当前任务：Task 2 PTO GlobalTensor 边界收口。
