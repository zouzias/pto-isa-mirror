# dispatch_ffn_combine_v3 V4 PTO Showcase Implementation Plan

## 1. 文档定位

本文回答 V4 整改“按什么顺序做、每个阶段改哪些文件、何时验收”。V4 设计真值见 `DESIGN.md`；执行状态和验证结果见 `task.md`；非 PTO 依赖台账见 `api_interface.md`。

| 文档 | 职责 |
| --- | --- |
| `DESIGN.md` | V4 PTO showcase 设计真值 |
| `IMPLEMENTATION_PLAN.md` | 阶段路线和验收节奏 |
| `task.md` | 任务跟踪和验证记录 |
| `api_interface.md` | 非 PTO 依赖归类台账 |

## 2. 开发原则

1. **先展示结构，后考虑性能**：V4 是代码组织和 PTO seam 整改，不做主动性能优化。
2. **不追求纯 PTO**：AscendC substrate 保留，但必须边界清晰。
3. **低风险先行**：先收口 helper，再收口 routing adapter，最后拆主 kernel stage。
4. **语义不变**：不改 tiling key、kernel ABI、launch args、AIC/AIV role、HardEvent/CrossCore/cache coherence。
5. **阶段验收**：每个代码阶段至少 build + small PASS；关键阶段 large PASS。

## 3. 当前起点

已完成 V3 收口：

- PTO GM view helper 已集中到 `op_kernel/utils/pto_global_view.hpp`。
- 主链路已使用 PTO `TLOAD/TSTORE/TCVT/TMATMUL/TSTORE_FP/TGET/TPUT/TNOTIFY/TWAIT`。
- 剩余 AscendC 依赖已归类为 boundary adapter、coordination shell、substrate。

V4 待解决问题：

- PTO 主链路被 `dispatch_ffn_combine_kernel.hpp` 大文件和多个 detail namespace 淹没。
- `PtoLoadVector/PtoStoreVector/PtoCastVector/PtoMulVector` 等 helper 分散重复。
- `moe_v2_pto_sort.h` 仍有本地 PTO adapter。
- 主路径缺少 stage facade，难以直接展示 Parallel Tiling Operation 的数据流/任务流/同步流。

## 4. 阶段路线

### B0：文档与任务基线

目标：建立 V4 设计真值和执行跟踪。

文件：

```text
DESIGN.md
IMPLEMENTATION_PLAN.md
task.md
api_interface.md
```

准出：

- `DESIGN.md` 说明 V4 是 PTO showcase remediation，不是纯 PTO 重写。
- `IMPLEMENTATION_PLAN.md` 给出 B0-B5 路线。
- `task.md` 新增 V4 执行跟踪，不删除 V3 历史。
- `api_interface.md` 新增 V4 非 PTO 依赖边界。

### B1：统一 PTO bridge / vector ops

目标：集中通用 PTO helper，降低重复实现。

文件：

```text
op_kernel/utils/pto_global_view.hpp
op_kernel/utils/pto_vector_ops.hpp
op_kernel/utils/pto_sync_bridge.hpp        # 可选
op_kernel/dispatch_ffn_combine_kernel.hpp
op_kernel/utils/block_epilogue_pertoken_v2.hpp
op_kernel/utils/block_epilogue_pertoken_row.hpp
op_kernel/utils/block_epilogue_pertoken_swiglu.hpp
```

要求：

- `pto_global_view.hpp` 只放 GM view adapter。
- `pto_vector_ops.hpp` 放通用 vector / matrix-row helper。
- 保留现有 `TileElems`，不统一改默认值。
- `pto_sync_bridge.hpp` 若创建，只做 AscendC local pipe bridge，不替换 HardEvent 语义。

准出：

- 重复 vector helper 明显减少。
- build PASS。
- small case PASS。

### B2：收口 routing PTO adapter

目标：让 routing PTO sort 复用统一 helper。

文件：

```text
op_kernel/moe_init_routing_quant_v2/moe_v2_pto_sort.h
op_kernel/moe_init_routing_quant_v2/moe_v2_pto_adapter.h   # 可选
```

要求：

- 保持 `MoeInitRoutingQuantV2::pto_detail` 调用面。
- sort 专属 tile alias/常量/算法保留在 `moe_v2_pto_sort.h`。
- 不改 routing `TQue/TBuf/TPipe` 生命周期。
- 不改 tail/pad/atomic boundary adapter。

准出：

- 本地 `PtoV2GlobalNd/MakeContiguousGlobal` 被统一 helper 替代或清晰转发。
- small case PASS。
- large case PASS。

### B3：主 kernel stage facade 拆分

目标：让数据流/任务流/同步流在文件结构中显性化。

文件：

```text
op_kernel/dispatch_ffn_combine_kernel.hpp
op_kernel/stages/kernel_context.hpp
op_kernel/stages/routing_stage.hpp
op_kernel/stages/dispatch_gather_stage.hpp
op_kernel/stages/gmm_stage.hpp
op_kernel/stages/swiglu_stage.hpp
op_kernel/stages/combine_stage.hpp
op_kernel/stages/restore_stage.hpp
```

要求：

- header-only 拆分，不新增 device `.cpp`。
- 保持 `Params`、tiling key、kernel name、launch ABI、block dim、AIC/AIV role 不变。
- 先移动/命名，不重写算法。
- 保持 sync flag 和 cross-core 顺序不变。

准出：

- `dispatch_ffn_combine_kernel.hpp` 成为高层 orchestrator。
- stage 名称体现 routing → gather → GMM1 → SwiGLU → GMM2 → combine → restore。
- small case PASS。
- large case PASS。

### B4：copy/fixpipe substrate 边界清理（可选）

目标：改善 substrate 可读性，不做 PTO drop-in 替换。

文件：

```text
op_kernel/utils/dispatch_policy_custom.hpp
op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp
op_kernel/utils/copy_policy_private.hpp      # 可选
```

要求：

- 不机械替换 `DataCopy/LoadData/LoadDataWithTranspose/Fixpipe`。
- 只抽私有 helper 或补边界命名。
- 保持 `PtoTileMmad/TMATMUL/TSTORE_FP` 行为不变。

准出：

- build PASS。
- small/large PASS。
- `api_interface.md` 明确该区域仍是 substrate。

### B5：验证与 showcase 收口

目标：完成 V4 结果记录。

文件：

```text
task.md
api_interface.md
README.md
DESIGN.md
```

要求：

- 记录每阶段 build/small/large 验证结果。
- 记录 grep 台账。
- README 增加 V4 PTO 模型说明。
- api_interface 更新最终剩余非 PTO 依赖。

## 5. 验证命令

### 环境

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:${LD_LIBRARY_PATH:-}
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
export MPI_RUNNER=mpirun
```

### small case

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

### large case

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

### grep 台账

```bash
rg -n "PtoShapeDyn|PtoStrideDyn|PtoV2GlobalNd|MakeContiguousGlobal" \
  kernels/manual/a2a3/dispatch_ffn_combine_v3/op_kernel

rg -n "namespace kernel_detail|namespace row_detail|namespace swiglu_detail|namespace pto_detail" \
  kernels/manual/a2a3/dispatch_ffn_combine_v3/op_kernel

rg -n "DataCopyPad|SetAtomicAdd|CrossCoreSetFlag|CrossCoreWaitFlag|DataCacheCleanAndInvalid" \
  kernels/manual/a2a3/dispatch_ffn_combine_v3/op_kernel
```

## 6. 完成定义

- V4 文档齐备。
- 通用 PTO helper 集中。
- routing PTO adapter 收口。
- 主 kernel stage facade 成型。
- 明确保留的 AscendC substrate 都有台账。
- small/large PASS。
