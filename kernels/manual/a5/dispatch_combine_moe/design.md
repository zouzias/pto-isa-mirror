# dispatch_combine_moe no-AscendC 化设计

## 目标

让 `kernels/manual/a5/dispatch_combine_moe` 的当前项目源码不再直接依赖 Ascend C 编程面，并且不再直接使用已有 PTO 仓可替代的底层 runtime/compiler builtin。用户已明确约束：**不允许修改 `include/pto/**`**，因此本设计只复用 PTO 仓内已经存在的公共头、后端头、custom sync、comm primitive、kernel metadata、配置表/宏模式。

## 确认边界

1. **不能改 `include/pto/**`**：不新增 PTO runtime/kernel facade，不补公共 wrapper。
2. **device 源码 no-AscendC**：当前目录下 device 源码最终 gate 不允许出现 `AscendC::`、`using namespace AscendC`、裸 `AscendC` 类型名、大写 `ASCENDC` 关键字、直接 `#include "kernel_operator.h"` / `#include <kernel_operator.h>`。
3. **device 源码不直接使用可替代的底层 builtin**：`get_block_idx/get_block_num/get_subblockid/get_subblockdim/block_idx`、`pipe_barrier`、`set_flag/wait_flag`、`set_intra_block/wait_intra_block/ffts_cross_core_sync`、`dcci/dsb`、`icache_preload` 等，必须优先替换为 PTO 仓现有接口或现有模式。
4. **host 侧也要替换 `platform_ascendc`**：`platform_ascendc::PlatformAscendCManager`、`GetCoreNumAiv/GetCoreNumAic/CalcTschBlockDim` 不再保留；改为 PTO 仓已有的 config/table 化资源描述方式。
5. **允许保留 kernel ABI 必要语法**：`__global__`、`__gm__`、`AICORE`、`__in__/__out__` 这类 PTO 现有 kernel 示例已经使用的 ABI/地址空间语法可保留；它们不是本任务要清理的 `AscendC::` 编程面。
6. **算法不降级**：保持 mixed AIC/AIV MegaMoE 主线、PTO tile/vector/comm pipeline、HCCL remote window 数据流，不回退到 naive 单 AIV correctness path。

## PTO 仓内替代依据

### Kernel entry / metadata

- `include/pto/pto-inst.hpp` 是统一入口。
- `include/pto/common/kernel_meta.hpp` 已提供 `PTO_SYNCALL_AIV_KERNEL_META`、`PTO_SYNCALL_MIX_AIC_KERNEL_META`、`PTO_SYNCALL_AIC_KERNEL_META`。
- `tests/npu/a2a3/src/st/testcase/syncall/syncall_kernel.cpp` 和 `tests/npu/a2a3/src/st/testcase/syncall/syncall_mix_1_1_kernel.cpp` 证明可用 `<pto/pto-inst.hpp>` + `__global__ AICORE` + PTO metadata，不直接 include `kernel_operator.h`。

### Core identity / mixed participant index

- `include/pto/npu/a5/SyncAll.hpp`、`include/pto/npu/a2a3/SyncAll.hpp` 已有 `pto::SYNCALL_GET_MIX_AIC_BLOCKS()`、`pto::SYNCALL_GET_MIX_AIV_RATIO()`、`pto::SYNCALL_GET_MIX_PARTICIPANT_IDX()`、`pto::SYNCALL_GET_MIX_PARTICIPANT_COUNT()`。
- 当前项目禁止直接写 `get_block_idx/get_block_num/get_subblockid/get_subblockdim/block_idx`，统一通过上述 PTO 函数推导：
  - AIC logical idx：`pto::SYNCALL_GET_MIX_PARTICIPANT_IDX()` 在 cube 分支即 AIC block idx。
  - AIV logical idx：`pto::SYNCALL_GET_MIX_PARTICIPANT_IDX() - pto::SYNCALL_GET_MIX_AIC_BLOCKS()`。
  - AIV participant count：`pto::SYNCALL_GET_MIX_AIC_BLOCKS() * pto::SYNCALL_GET_MIX_AIV_RATIO()`。
  - mixed participant count：`pto::SYNCALL_GET_MIX_PARTICIPANT_COUNT()`。

### Sync / event / barrier

- `include/pto/common/pto_instr.hpp` 提供 `pto::SYNCALL`、`pto::TSYNC`。
- `include/pto/common/event.hpp` 提供 `pto::PtoSetWaitFlag<SrcPipe, DstPipe>()`。
- `include/pto/npu/a5/TSync.hpp`、`include/pto/npu/a2a3/TSync.hpp` 提供 `pto::Event<SrcOp, DstOp, AutoToken, EventID>`。
- `include/pto/npu/a5/custom/TSync_Custom.hpp`、`include/pto/npu/a2a3/custom/TSync_Custom.hpp` 提供 `pto::TSync_Custom<pto::SyncOpType::..., pto::SyncOpType::TLOAD>`，可替代当前直接 `CrossCoreSetFlag/CrossCoreWaitFlag` 的 C2V/V2C handoff。
- 当前项目禁止直接写 `pipe_barrier/set_flag/wait_flag/set_intra_block/wait_intra_block/ffts_cross_core_sync`，统一通过上述 PTO event/sync/custom sync 表达。

### Cache / visibility / communication signal

- `include/pto/comm/comm_types.hpp` 已有 `pto::comm::Signal` / `Signal2D`。
- `include/pto/comm/pto_comm_inst.hpp` 已有 `pto::comm::TNOTIFY/TWAIT/TTEST/TGET/TPUT`。
- `include/pto/npu/a5/SyncAll.hpp`、`include/pto/npu/a2a3/SyncAll.hpp` 中已有 `pto::SYNCALL_SOFT_DCCI`、`pto::SYNCALL_SOFT_DCCI_RANGE`、`pto::SYNCALL_SOFT_GM_LOAD` 等 PTO 命名空间函数。
- `kernels/manual/a5/gemm_ar/ready_queue.hpp` 和 `kernels/manual/a5/gemm_ar/comm_kernel.cpp` 展示了 queue + `pto::comm::Signal/TTEST/TNOTIFY/TWAIT` 的 doorbell/poll 模式；本项目应优先用 PTO signal/comm primitive 表达 remote-window 可见性和跨 rank/跨 stage 通知。

### Host platform / blockDim / core count

- `kernels/manual/a5/gemm_ar/gemm_ar_config.h` 使用 `CONFIG_COMPUTE_BLOCK_NUM`、`CONFIG_COMM_BLOCK_NUM` 等配置宏/常量描述 block 数，而不是 host 查询 `PlatformAscendCManager`。
- `kernels/manual/a2a3/gemm_ar/main.cpp` 中 HCCL/MC2 comm resource 采用手工 tiling/config 结构填充，例如 `commBlockNum = 48U`，不是依赖 `platform_ascendc`。
- 当前项目 host 侧改为本目录内 `PtoDeviceResources` / `DispatchCombineMoeResourceConfig` 表：由 `run.sh` / CMake / `cfg.soc_version` 选择 A5/A3 默认资源，支持环境变量或 CMake 宏覆盖。默认按 mixed 1:2 模式计算：
  - `aicBlocks = CONFIG_MIX_AIC_BLOCKS`；
  - `aivRatio = CONFIG_MIX_AIV_RATIO`；
  - `aivNum = aicBlocks * aivRatio`；
  - `blockDim = aicBlocks * (1 + aivRatio)`；
  - A5 默认可按当前 mixed 1:2 baseline 设为 `aicBlocks=20, aivRatio=2, aivNum=40, blockDim=60`，并允许 build/run 覆盖。

## 当前触点盘点

| 类别 | 代表文件 | 当前问题 | 替代抓手 |
|---|---|---|---|
| kernel ABI / task 分派 | `op_kernel/dispatch_combine_moe.cpp` | `kernel_operator.h`、`using namespace AscendC`、`REGISTER_TILING_DEFAULT`、`KERNEL_TASK_TYPE` | `<pto/pto-inst.hpp>`、`PTO_SYNCALL_MIX_AIC_KERNEL_META`、`__global__ AICORE` |
| mixed AIC/AIV core identity | `dispatch_combine_moe_kernel.hpp`、epilogue、routing/unpermute | `AscendC::GetBlockIdx/GetBlockNum/GetTaskRation` 与 raw `get_block_idx/get_subblockid` | `pto::SYNCALL_GET_MIX_*` 系列 |
| pipe / local event | `moe_pto_utils.hpp`、MMAD、epilogue、routing | `AscendC::HardEvent`、`SetFlag/WaitFlag`、raw flag | `pto::PtoSetWaitFlag`、`pto::Event`、`pto::TSYNC` |
| cross-core handoff | GMM/SwiGLU/combine interlock | `CrossCoreSetFlag/CrossCoreWaitFlag` | `pto::TSync_Custom` / `pto::SYNCALL<SyncCoreType::Mix>` |
| cache / GM visibility | `hccl_window.hpp`、queue-like publish path | `DataCacheCleanAndInvalid`、raw `dcci/dsb` | `pto::comm::Signal/TNOTIFY/TWAIT/TTEST`、`pto::SYNCALL_SOFT_*` |
| routing/unpermute pipe object | `token_reorder/routing/*.h/*.cpp` | `AscendC::TPipe` / `FetchEventID` | 固定 event id + `pto::PtoSetWaitFlag` / `pto::Event`，去除 `TPipe*` 生命周期参数 |
| host platform query | `op_host/tiling_builder.cpp` | `platform_ascendc::PlatformAscendCManager` | 本目录资源配置表 + gemm_ar 风格 `CONFIG_*` 覆盖 |

## 推荐设计

### 1. 静态 gate 先落地

源码扫描范围：`op_kernel/`、`op_host/`、`CMakeLists.txt`、`run.sh`、必要 host/kernel glue；排除 `build/`、`out/`、`.cache/`、自动生成文件。

最终禁止项：

```text
AscendC::
using namespace AscendC
# include kernel_operator.h
platform_ascendc::
PlatformAscendCManager
GetCoreNumAiv / GetCoreNumAic / CalcTschBlockDim
get_block_idx / get_block_num / get_subblockid / get_subblockdim / block_idx
pipe_barrier / set_flag / wait_flag
set_intra_block / wait_intra_block / ffts_cross_core_sync
dcci / dsb / DataCacheCleanAndInvalid
icache_preload
```

允许项：`__global__`、`__gm__`、`AICORE`、`PTO_INTERNAL`、`PTO_SYNCALL_*`、`pto::...`、现有 PTO custom sync 头。

### 2. Kernel entry PTO 化

- `dispatch_combine_moe.cpp` include `<pto/pto-inst.hpp>`，不 include `kernel_operator.h`。
- 使用 `extern "C" __global__ AICORE void dispatch_combine_moe(...)`。
- 使用 `PTO_SYNCALL_MIX_AIC_KERNEL_META(dispatch_combine_moe, 1, 2)` 或等价现有 PTO metadata 表达 mixed AIC/AIV。
- `REGISTER_TILING_DEFAULT` / `KERNEL_TASK_TYPE` 不保留；tiling key 分支若必须保留，转为普通 tiling data 字段检查，不依赖 AscendC 宏。

### 3. Core identity PTO 化

新增当前项目内小 helper（只调用 PTO 现有函数，不调用 raw builtin）：

```text
PtoMixAicBlocks()      -> pto::SYNCALL_GET_MIX_AIC_BLOCKS()
PtoMixAivRatio()       -> pto::SYNCALL_GET_MIX_AIV_RATIO()
PtoMixParticipantIdx() -> pto::SYNCALL_GET_MIX_PARTICIPANT_IDX()
PtoAivLogicalIdx()     -> participantIdx - aicBlocks
PtoAivLogicalCount()   -> aicBlocks * aivRatio
```

所有当前 `get_block_idx() + get_subblockid() * get_block_num()`、`GetBlockIdx/GetBlockNum/GetTaskRation` 都通过该 helper 替换。

### 4. Sync/event PTO 化

- `PtoPipeBarrier<Pipe>` 不再映射到非法的 V/M 单 pipe `pto::TSYNC<pto::Op::VECTOR/TMATMUL>`；MTE2/MTE3/ALL 仍使用 PTO `TSYNC`，V/M/FIX/MTE1/S 等 pipe 使用 PTO `PtoSetWaitFlag<SrcPipe, DstPipe>` 形成合法等待闭环。
- `PtoSetFlag/PtoWaitFlag<AscendC::HardEvent>` 改为 `pto::PtoSetWaitFlag<SrcPipe, DstPipe>` 或 `pto::Event<SrcOp, DstOp, false, EVENT_ID>`。
- GMM/SwiGLU/combine 跨 AIC/AIV handoff 改为 `pto::TSync_Custom`，按方向使用 `TSTORE_C2GM/TSTORE_V2GM/TMOV_C2UB/TINSERT_V2L1 -> TLOAD`。
- 所有 `icache_preload` 删除，除非能用已有 PTO prefetch primitive (`TPREFETCH` / `TPREFETCH_ASYNC`) 表达同等语义；不能等价时作为性能风险记录，不用 raw builtin。

### 5. Remote window / cache visibility PTO 化

- `PtoRemoteWindow` 中删除 `AscendC::GlobalTensor` 和 `DataCacheCleanAndInvalid`。
- token-ready、barrier、summary、doorbell 统一使用 `pto::comm::Signal/TNOTIFY/TWAIT/TTEST`。
- 必须做 cache line acquire/release 时，使用 PTO namespace 中已有 `SYNCALL_SOFT_DCCI/SYNCALL_SOFT_DCCI_RANGE/SYNCALL_SOFT_GM_LOAD`，不直接写 `dcci/dsb`。
- 对照 `gemm_ar/ready_queue.hpp` 的 publish/acquire 模式，但当前项目代码不复制 raw builtin；只吸收其 queue/doorbell 结构，落到 PTO signal + PTO soft dcci helper。

### 6. Routing / unpermute PTO 化

- 删除所有 `using namespace AscendC`。
- 去掉 `AscendC::TPipe*` 参数；构造对象不再依赖 `TPipe` 生命周期。
- `HardEvent::X_Y` 改为 pipe pair / `pto::Op` pair / fixed event id。
- `PtoSetWaitFlag<HardEvent>` 全部改为 `pto::PtoSetWaitFlag<PIPE_X, PIPE_Y>` 或 `pto::Event`。
- core 分片逻辑统一使用 `PtoAivLogicalIdx()/PtoAivLogicalCount()`。

### 7. Host platform PTO 化

- `tiling_builder.cpp` 删除 `#include "tiling/platform/platform_ascendc.h"`。
- 新增当前项目内资源配置（例如 `op_host/pto_resource_config.hpp` 或置于 `tiling_builder.cpp` 匿名命名空间）：
  - 由 `cfg.soc_version` / `DISPATCH_COMBINE_MOE_TARGET_ARCH` / CMake 宏选择 A5/A3；
  - 默认 A5 mixed 1:2：`aicBlocks=20`、`aivRatio=2`、`aivNum=40`、`blockDim=60`；
  - 支持 `CONFIG_MIX_AIC_BLOCKS`、`CONFIG_MIX_AIV_RATIO`、`CONFIG_DISPATCH_COMBINE_MOE_AIV_NUM`、`CONFIG_DISPATCH_COMBINE_MOE_BLOCK_DIM` 覆盖，风格对齐 `gemm_ar_config.h`。
- `FillInitRoutingTiling()` 使用资源配置给出的 `aivNum`。
- `launchConfig.blockDim` 使用资源配置给出的 `blockDim`。

### 8. 验收设计

- 静态 gate：禁止项全部为 0，且确认 `include/pto/**` 没有修改。
- 编译 gate：A5 compile-only 优先；如果 A3 分支保留，追加 A3 compile。
- Runtime gate：仅在 A3 runtime 可用且阶段适合时执行 smoke；必须设置超时，不无限等待。
- 收尾代码检视：检查 PTO 替代是否真实、算法/overlap 未降级、host/resource 配置是否可覆盖、sync 是否存在卡死风险。

## 风险与对策

| 风险 | 对策 |
|---|---|
| `pto::SYNCALL_GET_MIX_*` 是后端实现函数而非高层文档 API | 它位于 `include/pto/**` 现有 PTO namespace，且用户禁止改 include 但允许复用；实施中通过编译 gate 验证可见性 |
| `TSync_Custom` 覆盖不了所有现有 cross-core flag 语义 | 先按 C2V/V2C handoff 分类替换；特殊 flag 单独代码检视，不退回 raw builtin |
| host 资源表默认值与平台查询结果不一致 | 使用 gemm_ar 风格 `CONFIG_*` 可覆盖；默认值按当前 mixed 1:2 baseline 写入，compile/static gate 后再按 runtime 结果校正 |
| 删除 `icache_preload` 可能影响性能 | 先评估是否可用 `TPREFETCH/TPREFETCH_ASYNC` 替代；不能等价时记录性能风险并在代码检视中重点关注，不直接保留 builtin |
| raw builtin 从 gemm_ar 样例复制进当前项目 | 禁止。gemm_ar 只作为模式参考；当前项目落地必须用 PTO namespace 或配置表替代 |

## 任务拆分

1. 静态 gate 与触点清单。
2. kernel entry no-`kernel_operator.h` + PTO metadata spike。
3. host resource config 替换 `platform_ascendc`。
4. core identity helper 替换 `GetBlockIdx/get_block_idx` 系列。
5. `moe_pto_utils.hpp` 改为 PTO event/sync helper。
6. MMAD/vector/epilogue helper 替换 HardEvent/barrier。
7. routing/unpermute 去 `TPipe`、HardEvent、raw core idx。
8. remote window cache/signal PTO 化。
9. compile/static gate。
10. 收尾代码检视。
