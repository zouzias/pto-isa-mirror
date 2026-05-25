# dispatch_combine_tile 设计文档

## 1. 项目定位

`dispatch_combine_tile` 是 A2/A3 manual kernel 目录下的 PTO 风味 MoE dispatch/combine 示例。它对齐
`/mnt/data/ntlab/zy/code/zhangyuan/shmem_zy/examples/dispatch_gmm_combine` 的通信重排语义，但实现侧不继承
该项目的任何 SHMEM 接口。第一版去掉：

- GMM1/GMM2；
- SwiGLU；
- int8 dynamic quant 和 per-token scale；
- fused mix-kernel overlap。

第一版必须是真实多卡多进程通信版本。每个 host 进程对应一个 rank，绑定一个 device；host 侧 HCCL/MPI/bootstrap
参考本仓 `kernels/manual/a2a3/gemm_ar` 的组织方式；device 侧 payload 和 signal 用 PTO comm primitive 表达。

目标链路：

```text
DispatchCombineTileDispatch
  inputA[local M,K] + expertIdx[local M,topK]
    -> local peer window packedA / peerTokenPerExpert / expandedRowIdx
    -> PTO TGET peer packedA
    -> workspace.dispatchedA[local experts, source-rank-major]
    -> workspace dispatch metadata

host prepare expertOutput
  workspace.dispatchedA
    -> expertOutput

DispatchCombineTileCombine
  expertOutput + workspace metadata
    -> PTO TPUT owner peerWindow.ptrD
    -> PTO TWAIT all peer returns
    -> unpermute + probs weighted sum
    -> outputC[local M,K]
```

## 2. 设计依据

算法语义来自既有 `dispatch_gmm_combine` 的 routing/restore 逻辑，只借鉴数据重排结果，不借鉴通信 API：

1. routing 先把本 rank token 按 global expert 打包到本 rank peer memory；
2. 每个 rank 把本 rank `localTokenPerExpert` row 推送到所有 peer 的 `peerTokenPerExpert` count matrix；
3. 所有 rank 等待本地 count matrix 收齐所有 source rank 的 count；
4. 当前 rank 只拉取属于自己 local experts 的 token，布局为 `local expert -> source rank -> rows`；
5. expert compute 后，当前 rank 把每个 source rank 的结果反向写回 source rank `ptrD`；
6. source rank 等待所有 peer 写回完成后，用 `expandedRowIdx + probs` 做 unpermute/weighted restore。

HCCL/MPI host 侧组织参考本仓 `kernels/manual/a2a3/gemm_ar`：

- `comm_mpi.h` 风格的多进程启动、rank/size/barrier；
- host 侧 ACL/HCCL 初始化、device 绑定、window 资源申请；
- `HcclDeviceContext` 风格的 device-visible window context，MESH 路径直接使用 HCCL 返回 context，RING 路径按
  `gemm_ar` 从 HCCL resource 中提取 `windowsIn[]` 后拷到 device；
- kernel launch 后用 host barrier 做迭代级对齐；
- device kernel 内用 PTO `TGET/TPUT/TNOTIFY/TWAIT` 完成 payload/signal。

明确不以 `dispatch_gmm_combine_v2` 和 `dispatch_ffn_combine_v3` 作为设计依据。

## 3. 非目标

- 不实现 GMM1/GMM2。
- 不实现 SwiGLU。
- 不实现 int8 dynamic quant routing。
- 不依赖 `moe_init_routing_quant_v2`、Catlass、SHMEM 示例代码或 AscendC 算子库。
- 不做单进程多 rank 模拟作为第一版验收路径。
- 不设计 `--case all` 这类隐藏 shape 参数组合。

## 4. 硬约束

### 4.1 Kernel 数量

device 侧计算 kernel 只能有两个：

| Kernel | 职责 |
| --- | --- |
| `DispatchCombineTileDispatch` | routing metadata、peer publication、PTO TGET dispatch gather |
| `DispatchCombineTileCombine` | PTO TPUT combine return、TWAIT、unpermute、`probs` 加权输出 |

中间不新增 `PrepareExpertOutput` device kernel。第一版 `expertOutput` 由 host 在每个 rank 本地生成，用于模拟外部
expert compute。

### 4.2 纯 PTO device 风格

kernel 侧代码必须只依赖 PTO 公开接口和 C/C++ 基础控制流。允许 include：

```cpp
#include <pto/pto-inst.hpp>
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/constants.hpp>  // 如确有常量需要
```

禁止在 kernel 文件中出现：

```text
kernel_operator.h
AscendC::
LocalTensor
GlobalTensor from AscendC namespace
TQue / TBuf / TPipe
DataCopy / DataCopyPad
Catlass
aclshmem / shmem
```

SHMEM 接口在本项目中完全禁用，包括 host 和 device 两侧。禁止的不只是 device 调用，也包括 host bootstrap、
内存分配、barrier 和 remote pointer 查询：

```text
aclshmem_init / aclshmem_malloc / aclshmem_ptr / aclshmem_barrier_all / aclshmem_free
aclshmemx_* / shmem_* / symmetricPtr
```

远端内存只能使用 `gemm_ar` 风格的 HCCL window：

```text
host:
  MPI bootstrap / HcclCommInitRootInfo
  HcclAllocComResourceByTiling
  HcclDeviceContext{rankId, rankNum, winSize, windowsIn[]}
  peerWindow 从本 rank windowsIn[rankId] 中按固定 offset 切片

device:
  remotePtr = ctx->windowsIn[peerRank] + (localPtr - ctx->windowsIn[myRank])
  GlobalTensor(remotePtr + fieldOffset, shape, stride)
  PTO TGET / TPUT / TNOTIFY / TWAIT
```

地址映射 helper 只允许做 `windowsIn[] + offset` 计算；不能封装任何数据搬运或同步语义。

PTO 数据面要求：

- GM 边界使用 `pto::GlobalTensor` 表达 shape/stride/layout。
- 本地 payload 行搬运使用 `pto::Tile<TileType::Vec>` + `TASSIGN` + `TLOAD/TSTORE`。
- 跨 rank payload 搬运使用 PTO `TGET/TPUT`。
- 跨 rank readiness 使用 PTO `TNOTIFY/TWAIT/TTEST`。
- GM offset 只允许出现在构造 `GlobalTensor` 或 workspace/window view 的边界 helper 中。
- metadata 可以在 correctness 第一版中使用受控标量 GM 读写；一旦 metadata 进入 tile 批量搬运，也必须封装成
  `GlobalTensor` + `Tile`。

### 4.3 HCCL window 使用规则

`gemm_ar` 的关键规则在本项目中保持不变：

1. 所有会被 peer rank 访问的 buffer 必须放在 HCCL RDMA window 内；只被本 rank 访问的 buffer 使用普通
   `aclrtMalloc`。
2. 每个 rank 的 window layout 必须完全一致，`peerWindow` 在所有 rank 上使用同一个 window offset。remote pointer
   helper 依赖同 offset 映射，不能把 `aclrtMalloc` 出来的普通 GM 指针传给 remote pointer helper。
3. host 必须检查 `peerWindowOffset + peerWindowBytes <= HcclDeviceContext.winSize`，并在 run.sh 中按 shape 自动估算
   `HCCL_BUFFSIZE`。
4. kernel launch 参数里的 `peerWindow` 必须是本 rank `windowsIn[rankId] + peerWindowOffset`，`hcclCtx` 必须是
   device-visible `HcclDeviceContext*`。
5. device 侧 helper 只允许类似下面的逻辑：

   ```text
   HcclRemotePtr(ctx, localWindowPtr, peerRank):
     localBase = ctx->windowsIn[ctx->rankId]
     offset = localWindowPtr - localBase
     return ctx->windowsIn[peerRank] + offset
   ```

   `localWindowPtr` 必须指向本 rank HCCL window 内部；helper 不做通信、不做同步、不查询 runtime。
6. payload 可见性由 PTO protocol 保证：写完 window payload 后，必须先完成本地 PTO pipeline drain，再发布
   `TNOTIFY`。`gemm_ar` 中的 `pipe_barrier(PIPE_ALL) + dsb(DSB_DDR) -> TNOTIFY` 只作为 memory-ordering 语义参考；
   本项目代码不能直接引入 AscendC 依赖，如需显式 fence，必须封装成 PTO-local helper 或使用 PTO comm primitive
   已提供的可见性语义。

### 4.4 Host 责任

host 侧必须提供完整可运行示例：

- 解析显式 shape 参数；
- 按 `-pes` 启动/参与多进程通信；
- 为每个进程确定 `rank` 和 `device`；
- 初始化 ACL/HCCL/MPI runtime，方式参考 `gemm_ar`，不使用 SHMEM runtime；
- 分配本 rank input/output/workspace/expertOutput，并从 HCCL window 固定 offset 切出 peerWindow；
- launch 两个 kernel；
- 在两个 kernel 之间准备 `expertOutput`；
- 拷回并校验结果；
- 打印 dispatch/combine/total e2e 时间。

ACL/HCCL/MPI 属于 host/runtime 边界，不进入 device kernel 纯 PTO 约束。

### 4.5 单算子内部并发与流水保真约束

PTO 化不能把原算法退化成单核串行 correctness 版。第一版明确拆成 dispatch 和 combine 两个独立算子，由 CPU
顺序 launch；两个算子之间不设计 device overlap。性能约束只针对每个算子内部的并发粒度、双缓冲、预取和内存对齐：

| 原算法行为 | PTO 版要求 |
| --- | --- |
| routing/pack 多核处理 token | 多 AIV block 对连续 token shard 两遍处理：先计数，再基于 per-block prefix 稳定 pack |
| count row 推送到所有 peer 后跨 rank 等待 | 用 PTO `TPUT` 推送 count row 到 peer HCCL window，再用 `TNOTIFY/TWAIT/TTEST` 等待，不能改成 host barrier |
| peer 维度由 `coreIdx + n * coreNum` 分摊 | dispatch gather 和 combine return 都按 source/destination rank round-robin 分给 AIV blocks |
| GM->UB->GM copy 双缓冲 | 本地 copy、`TGET`、`TPUT` 都用 PTO tile staging，默认 ping-pong tile，不写裸地址 memcpy |
| expert-major group loop | 保留 expert-major 顺序和 segment 边界；去掉 GMM 后不引入无意义全核 barrier，但 remote payload 发布前必须完成本地 pipe/DDR 可见性 |
| combine 写回后全 rank 同步再 restore | 每个 destination rank 写完发往某 owner 的所有 local-expert 段后再通知；owner `TWAIT` 收齐所有 peer 后才能 restore |
| token unpermute 多核处理 | restore 按 token row / hidden chunk 分摊到 AIV blocks，不能由单 block 串行遍历全部 `[M,K]` |

本地同步边界使用 PTO public 能力表达：

- 同 core pipe 依赖：PTO primitive 的 `RecordEvent`/wait-events，或 PTO event helper。
- 全 AIV 阶段 barrier：`pto::SYNCALL<SyncCoreType::AIVOnly>()`；如需要 soft barrier，则使用 workspace GM
  control block + PTO `SYNCALL` overload。
- 细粒度 producer-consumer：workspace/window 中的 GM signal、counter 或 ready queue，配合
  `pto::comm::TNOTIFY/TWAIT/TTEST`。

禁止把原 `CrossCoreSetFlag/CrossCoreWaitFlag` 直接照搬到本项目；如某个原同步点没有 PTO public API 的一等替代，
必须先在设计中归类为 local coordination substrate，并用 PTO wrapper/GM signal 保持同等 ready 粒度。

## 5. 参数与 shape

第一版所有 shape 和工程运行参数都由命令行显式传入，不提供 `--case all`。原 SHMEM 项目的 `-N/-dataType/-weightNz/-transB`
属于 GMM/量化路径，本项目没有 GMM/SwiGLU/量化，不保留这些参数。

### 5.1 Shape 参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `-pes` / `--pes` / `--nranks` | rank 数，也记为 `EP` | `2` |
| `-M` / `--tokens` | 每个 rank token 数 | `64` |
| `-K` / `--hidden` | hidden size，也是 dispatch/combine 列维度 | `7168` |
| `-topK` / `--topk` | 每 token 路由 expert 数 | `8` |
| `-expertPerPe` / `--experts-per-rank` | 每 rank local expert 数 | `2` |
| `--max-output-size` | 每 rank dispatch 输出行容量；`0` 表示 `EP * M * topK` | `0` |

### 5.2 Kernel 性能参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `-aivBlocks` / `--aiv-blocks` | dispatch/combine kernel 使用的 AIV block 数；`0` 表示 host 按设备能力选择 | `0` |
| `-tileCols` / `--tile-cols` | hidden 维度每次 PTO Vec tile 搬运列数 | `1024` |
| `--row-chunk` | 大 segment 二级分摊时每次处理的 row 数；`0` 表示按实现默认 | `0` |
| `--metadata-pad` | `int32` metadata expert 维度 padding 粒度 | `16` |

### 5.3 Runtime/通信参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `-r` / `--run-mode` | 第一版只接受 `npu`；CPU/sim 不作为多卡验收路径 | `npu` |
| `-v` / `--soc-version` | CMake `SOC_VERSION` | `Ascend910B1` |
| `-device-base` / `--device-base` / `--first-device` | rank0 绑定的 device id | `0` |
| `--ndevices` | 可用 device 数，用于校验 `deviceBase + pes <= ndevices` | `pes` |
| `--mpi-bin` | `mpirun` 所在目录；空则按 `gemm_ar` 搜索常见路径 | 空 |
| `--hccl-buffsize-mb` | 手动指定 `HCCL_BUFFSIZE`；`0` 表示 run.sh 按 shape 自动抬高 | `0` |
| `--keep-hccl-shm` | 不清理 `/dev/shm/sem.hccl*` 和 IPC 残留，调试用 | `0` |
| `--rank-from-mpi` | host binary 从 MPI 获取 rank/size | `1` |
| `--rank` / `--nranks` | 单进程调试 fallback；不作为验收路径 | unset |

### 5.4 数据、校验与计时参数

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `-debug` / `--debug` | dump/verify 粒度 | `0` |
| `-iters` / `--iters` | 计时迭代次数 | `1` |
| `-warmup` / `--warmup` | 计时前 warmup 迭代次数 | `1` |
| `--seed` | host 生成输入数据的 base seed，rank 侧用 `seed + rank` | `1234` |
| `--data-dir` | 输入/输出/debug dump 目录 | `./out` |
| `--gen-data` | `1` 表示 host 现场生成输入；`0` 表示从 `data-dir` 读取 | `1` |
| `--verify` | 是否执行 CPU golden 校验 | `1` |
| `--rtol` / `--atol` | `outputC` 校验容差 | `1e-2` / `1e-2` |
| `--skip-build` | 只运行已编译 binary | `0` |
| `--clean-build` | 运行前删除 build 目录 | `1` |

### 5.5 原 SHMEM 参数映射

| 原 SHMEM run 参数 | PTO 项目处理 |
| --- | --- |
| `-pes` | 保留，作为 MPI rank 数 |
| `-fnpu` | 替换为 `--device-base/--first-device` |
| `-ipport` | 删除；HCCL root info 通过 MPI broadcast，不走 SHMEM init attr |
| `-M` | 保留 |
| `-K` | 保留 |
| `-N` | 删除；本项目没有 GMM1/GMM2 中间维度 |
| `-expertPerPe` | 保留 |
| `-dataType` | 删除；第一版固定 `inputA/expertOutput/outputC` 为 half，`probs` 为 float |
| `-weightNz` | 删除；本项目没有权重 |
| `-transB` | 删除；本项目没有 GMM 权重矩阵 |
| hardcoded `topK=8` | 改为显式 `-topK/--topk` |
| hardcoded `ubMoveNum=3584` | 替换为显式 `--tile-cols/--row-chunk` |
| hardcoded `maxOutputSize=m*topK*2` | 替换为 `--max-output-size`，默认 `EP*M*topK` |
| SHMEM 对称内存大小 | 替换为按 shape 自动估算 HCCL window / `HCCL_BUFFSIZE` |

派生量：

```text
EP = pes
expertPerRank = expertPerPe
expertNum = EP * expertPerRank
expandedRowsPerRank = M * topK
maxOutputSize = (maxOutputSizeArg == 0) ? EP * M * topK : maxOutputSizeArg
aivBlocks = (aivBlocksArg == 0) ? runtime/default AIV block count : aivBlocksArg
```

每个 rank 只持有本 rank 的 `[M,K]` 输入和 `[M,K]` 输出；跨 rank 输入通过 peer window 通信获得。

## 6. 每 rank 输入输出接口

### 6.1 Host-visible GM 输入

| Buffer | dtype | shape | 说明 |
| --- | --- | --- | --- |
| `inputA` | `half` | `[M, K]` | 本 rank token |
| `expertIdx` | `int32` | `[M, topK]` | 本 rank token 的 global expert id，范围 `[0, expertNum)`；非法 id 跳过 |
| `probs` | `float` | `[M, topK]` | 本 rank combine restore 权重 |
| `expertOutput` | `half` | `[maxOutputSize, K]` | host 在 dispatch 后为本 rank local experts 准备，行顺序必须等于 `dispatchedA` |

`expertOutput` 第一版生成规则：

```text
expertOutput[row, col] = dispatchedA[row, col]
```

这等价于 identity expert compute。后续接入真实 expert compute 时，只要保持行顺序不变即可替换该输入。

### 6.2 Host-visible GM 输出

| Buffer | dtype | shape | 说明 |
| --- | --- | --- | --- |
| `outputC` | `half` | `[M, K]` | 本 rank最终 unpermute + weighted sum 输出 |

debug 模式下 host 还会 dump workspace/window 中的阶段性 buffer。

### 6.3 data-dir 文件约定

当 `--gen-data=1` 时，rank0 生成所有 rank 输入和 CPU golden，再通过 MPI barrier 保证各 rank 可读；当
`--gen-data=0` 时，run.sh/host 直接读取已有文件。

```text
${dataDir}/rank_${rank}_inputA.bin          half[M,K]
${dataDir}/rank_${rank}_expertIdx.bin       int32[M,topK]
${dataDir}/rank_${rank}_probs.bin           float[M,topK]
${dataDir}/rank_${rank}_golden_outputC.bin  half[M,K]
${dataDir}/rank_${rank}_outputC.bin         half[M,K]
```

debug dump 可选：

```text
${dataDir}/rank_${rank}_localTokenPerExpert.bin
${dataDir}/rank_${rank}_peerTokenPerExpert.bin
${dataDir}/rank_${rank}_cumsumPerExpert.bin
${dataDir}/rank_${rank}_packedA_head.bin
${dataDir}/rank_${rank}_dispatchedA_head.bin
${dataDir}/rank_${rank}_ptrD_head.bin
```

## 7. GM buffer 逻辑布局

第一版使用两个 GM 区域：

- `workspace`：普通 device GM，只对本 rank 可见，用于本 rank dispatch 结果、metadata 和 verify dump。
- `peerWindow`：HCCL/MPI bootstrap 建立的跨 rank 可访问 window，每个 rank 暴露相同 layout，用于 remote payload 和
  signal。

所有 offset 由 host 和 device 使用同一套公式计算。

### 7.1 workspace

```text
workspace
  +-- localTokenPerExpert int32[expertNum]
  +-- blockTokenPerExpert int32[numAivBlocks * expertNum]
  +-- blockPrefixPerExpert int32[numAivBlocks * expertNum]
  +-- cumsumPerExpert     int32[EP * expertNum]
  +-- dispatchOffset      int32[expertPerRank]
  +-- prevSumBeforeRank   int32[EP * expertPerRank]
  +-- localSync           int32[syncSlots]
  +-- floatScratch        float[aivBlocks * tileCols] # optional restore accumulation scratch
  +-- dispatchedA         half[maxOutputSize * K]
  +-- ptrDLocal           half[M * topK * K]       # optional debug mirror
```

字段语义：

| 字段 | 生产方 | 消费方 | 说明 |
| --- | --- | --- | --- |
| `localTokenPerExpert[expert]` | Dispatch kernel | peerWindow / verifier | 本 rank 发往每个 global expert 的 token 数 |
| `blockTokenPerExpert[block, expert]` | Dispatch token shard | Dispatch prefix/pack | 每个 AIV block 的本地计数，用于并行 routing |
| `blockPrefixPerExpert[block, expert]` | Dispatch prefix | Dispatch pack | 当前 block 在该 expert 内的起始 cursor，保证跨 block stable pack |
| `cumsumPerExpert[src, expert]` | Dispatch kernel | Dispatch/Combine/verifier | 每个 source rank 内按 global expert 的 prefix end，即 `sum_{e<=expert}` |
| `dispatchOffset[localExpert]` | Dispatch kernel | Combine kernel | local expert 在本 rank `dispatchedA` 中的起始 row |
| `prevSumBeforeRank[src, localExpert]` | Dispatch kernel | Combine kernel | 当前 local expert 内，source rank 子段之前的累计 rows |
| `localSync` | Dispatch/Combine kernel | Dispatch/Combine kernel | PTO `SYNCALL` soft workspace 或细粒度 GM signal/counter |
| `floatScratch` | Combine kernel | Combine kernel | half×float restore 需要 float 累加时的 block-local scratch |
| `dispatchedA[row, col]` | Dispatch kernel | host prepare / verifier | 本 rank local experts 的 expert-major dispatch 输出 |
| `ptrDLocal[flat, col]` | Combine kernel | verifier | 本 rank `peerWindow.ptrD` 的 debug mirror，可选 |

### 7.2 peerWindow

```text
peerWindow
  +-- peerTokenPerExpert  int32[EP * expertNum]
  +-- expandedRowIdx      int32[M * topK]
  +-- packedA             half[M * topK * K]
  +-- ptrD                half[M * topK * K]
  +-- countReadySignal    int32[EP]
  +-- combineDoneSignal   int32[EP]
```

字段语义：

| 字段 | 写方 | 读方 | 说明 |
| --- | --- | --- | --- |
| `peerTokenPerExpert[src, expert]` | source rank dispatch via `TPUT` | owner rank dispatch/combine | 本 rank 收齐的所有 source rank count matrix |
| `expandedRowIdx` | owner rank dispatch | owner rank combine / verifier | 本 rank restore 需要的 row mapping |
| `packedA` | owner rank dispatch | destination rank dispatch via `TGET` | 本 rank按 global expert packed 的 token payload |
| `ptrD` | destination rank combine via `TPUT` | owner rank combine restore | combine return 的远端写回目标 |
| `countReadySignal[src]` | source rank dispatch notify | peers dispatch wait | count/payload publication ready |
| `combineDoneSignal[dst]` | destination rank combine notify | owner rank combine restore wait | `dst` rank 已把发往本 rank 的所有 local-expert 段写入 `ptrD` |

`peerWindow` 是第一版真实通信的唯一跨 rank device buffer。它不是 SHMEM symmetric memory；host 从 HCCL
`windowsIn[rankId]` 中切出本 rank 的 window region，device kernel 通过 `HcclDeviceContext.windowsIn[]` 计算 peer
window 地址，再构造 `GlobalTensor` view 后调用 PTO comm primitive。

remote address 计算采用 `gemm_ar` 的 `HcclDeviceContext` 模型：host 生成 device-visible context，其中包含
`windowsIn[rank]`。本项目可以实现一个不 include AscendC 的薄 helper：

```text
RemoteWindowPtr(ctx, localPtr, peerRank):
  localBase = ctx->windowsIn[ctx->rankId]
  offset = localPtr - localBase
  return ctx->windowsIn[peerRank] + offset
```

该 helper 只做 window 地址映射，不搬运数据、不发信号；实际 payload/signal 必须继续通过 PTO `TGET/TPUT/TNOTIFY/TWAIT`
完成。

## 8. Routing/restore 算法伪代码

### 8.1 Dispatch kernel

```text
myRank = rank passed by host
blockId = this AIV block id
blockNum = launched AIV block count

clear local workspace metadata
clear peerWindow.peerTokenPerExpert / expandedRowIdx / packedA ready region

# 1. local routing count, parallel by contiguous token shard.
#    Contiguous shards preserve token order when blockPrefixPerExpert is accumulated by block id.
tokenBegin, tokenEnd = contiguous shard of [0, M) assigned to blockId
for token in tokenBegin..tokenEnd-1:
  for slot in 0..topK-1:
    expert = expertIdx[token, slot]
    if expert invalid:
      peerWindow.expandedRowIdx[token * topK + slot] = -1
      continue
    blockTokenPerExpert[blockId, expert]++

SYNCALL(AIVOnly)

# 2. prefix over blocks and experts
for expert assigned to blockId stride blockNum:
  sum = 0
  for b in 0..blockNum-1:
    blockPrefixPerExpert[b, expert] = sum
    sum += blockTokenPerExpert[b, expert]
  localTokenPerExpert[expert] = sum

SYNCALL(AIVOnly)

# 3. prefix over global experts in source publication order
packedExpertOffset[0] = 0
single/control block or expert-parallel scan computes:
  packedExpertOffset[expert] = sum_{e < expert} localTokenPerExpert[e]

SYNCALL(AIVOnly)

# 4. local routing pack, stable token/slot order inside each expert
local per-block cursor[expertNum] starts from blockPrefixPerExpert[blockId, expert]
for token in tokenBegin..tokenEnd-1:
  for slot in 0..topK-1:
    expert = expertIdx[token, slot]
    if expert invalid:
      continue
    packedRow = packedExpertOffset[expert] + cursor[expert]++
    peerWindow.expandedRowIdx[token * topK + slot] = packedRow
    peerWindow.packedA[packedRow, :] = inputA[token, :]

# Publish my count row to every peer's count matrix.
for dst in blockId, blockId + blockNum, ... < EP:
  TPUT remote(dst).peerTokenPerExpert[myRank, :]
       <- workspace.localTokenPerExpert[:]
  pipeline/drain and DDR barrier
  TNOTIFY(remote(dst).countReadySignal[myRank], 1, Set)

# 5. wait all peer publication
for src in blockId, blockId + blockNum, ... < EP:
  TWAIT(local peerWindow.countReadySignal[src], 1, GE)

SYNCALL(AIVOnly)

# 6. build prefix metadata from local peerWindow.peerTokenPerExpert
for src assigned to blockId stride blockNum:
  cumsumPerExpert[src, expert] = sum_{e <= expert} peerWindow.peerTokenPerExpert[src, e]

single/control block computes dispatchOffset[localExpert]
for localExpert assigned to blockId stride blockNum:
  globalExpert = myRank * expertPerRank + localExpert
  prevSumBeforeRank[src, localExpert] = sum_{r < src} peerWindow.peerTokenPerExpert[r, globalExpert]

SYNCALL(AIVOnly)

# 7. PTO TGET payload for my local experts
for localExpert in 0..expertPerRank-1:
  globalExpert = myRank * expertPerRank + localExpert
  for src in blockId, blockId + blockNum, ... < EP:
    rows = peerWindow.peerTokenPerExpert[src, globalExpert]
    srcStart = (globalExpert == 0) ? 0 : cumsumPerExpert[src, globalExpert - 1]
    dstStart = dispatchOffset[localExpert] + prevSumBeforeRank[src, localExpert]
    for row chunk / hidden chunk:
      TGET workspace.dispatchedA[dstStart:dstStart+rows, :]
           <- remote(src).packedA[srcStart:srcStart+rows, :]
      using ping-pong Vec staging tiles

# Dispatch kernel ends after all local expert payloads are pulled into workspace.dispatchedA.
# Host barrier may align iterations, but it is not part of the device communication protocol.
```

### 8.2 Combine kernel

```text
myRank = rank passed by host
blockId = this AIV block id
blockNum = launched AIV block count

# 1. return expertOutput to owner/source ranks
for src in blockId, blockId + blockNum, ... < EP:
  for localExpert in 0..expertPerRank-1:
    globalExpert = myRank * expertPerRank + localExpert
    rows = peerWindow.peerTokenPerExpert[src, globalExpert]
    srcStart = dispatchOffset[localExpert] + prevSumBeforeRank[src, localExpert]
    dstStart = (globalExpert == 0) ? 0 : cumsumPerExpert[src, globalExpert - 1]
    for row chunk / hidden chunk:
      TPUT remote(src).ptrD[dstStart:dstStart+rows, :]
           <- expertOutput[srcStart:srcStart+rows, :]
      using ping-pong Vec staging tiles
  pipeline/drain and DDR barrier
  TNOTIFY remote(src).combineDoneSignal[myRank], 1, Set

SYNCALL(AIVOnly)

# 2. wait all peer returns into my ptrD
for peer in blockId, blockId + blockNum, ... < EP:
  TWAIT(peerWindow.combineDoneSignal[peer], 1, GE)

SYNCALL(AIVOnly)

# 3. restore local output
for token in blockId, blockId + blockNum, ... < M:
  clear outputC[token, :]
  for slot in 0..topK-1:
    row = peerWindow.expandedRowIdx[token * topK + slot]
    if row < 0:
      continue
    for hidden chunk:
      outputC[token, :] += probs[token, slot] * peerWindow.ptrD[row, :]
      using Vec tile weighted-add pipeline
```

### 8.3 单算子内部性能编排

`DispatchCombineTileDispatch` 和 `DispatchCombineTileCombine` 是两个独立算子，由 CPU 顺序 launch；两者之间不设计
device overlap，也不把 host 侧 `expertOutput` 准备计入算子内部 overlap。性能设计只看每个算子内部的并发、预取、
双缓冲和内存对齐。

#### Dispatch 算子内部

1. **routing/count 阶段**
   - AIV block 按连续 token range 分片，保证 expert 内稳定顺序。
   - 每个 block 只写自己的 `blockTokenPerExpert[block, expert]`，避免多 block 对同一 counter 原子争用。
   - `expertIdx` 按连续 GM 访问；`inputA` 只有在 pack 阶段读取，减少 count 阶段带宽。
   - `blockTokenPerExpert/blockPrefixPerExpert/localTokenPerExpert` 以 64B 对齐，expert 维度 padding 到 8 或 16 个
     `int32`，避免多个 block 写相邻 cache line 时互相污染。

2. **pack 阶段**
   - 每个 block 使用 `blockPrefixPerExpert` 作为该 expert 的初始 cursor，token range 连续，因此跨 block 合并后仍是
     token-major stable order。
   - `inputA[token, :] -> peerWindow.packedA[packedRow, :]` 按 hidden chunk 搬运；chunk 使用双 Vec tile ping-pong：
     当前 chunk `TLOAD` 输入，上一 chunk `TSTORE` 到 window。
   - `tileCols` 默认 1024 half 元素，要求 `tileCols * sizeof(half)` 64B 对齐；如果 UB 占用过高，按
     `tileCols=512/1024/2048` 做显式参数调优。
   - `peerWindow.packedA` row stride 等于 `K`，host 要保证 row 起点至少 64B 对齐；默认 shape `K=7168` 满足。

3. **count exchange 阶段**
   - 按 peer rank round-robin 分给 AIV blocks：`dst = blockId + n * blockNum`。
   - count row 用 `TPUT` 写入 `remote(dst).peerTokenPerExpert[myRank, :]`，不是所有 rank 再 `TGET`。
   - 小 metadata 使用一块独立 Vec tile staging；count row padding 到 64B，避免 signal 与 count 共 cache line。
   - `TNOTIFY countReadySignal[myRank]` 只在 count row `TPUT` 完成后发出；信号数组与 payload 区分离 64B 对齐。

4. **payload gather 阶段**
   - expert-major 外循环保持与原算法一致，内层 peer rank 按 block shard 并行。
   - 每个 `(localExpert, srcRank)` segment 按 row chunk × hidden chunk 搬运，使用 PTO `TGET` ping-pong overload 或等价
     双 Vec tile staging。
   - 对 `rows == 0` 的 segment 直接跳过，避免空 `TGET`。
   - 如果某个 peer segment 很大，优先切 hidden chunk；如果 rows 很大，再切 row chunk，保持单次 `GlobalTensor` view
     尺寸稳定。
   - `workspace.dispatchedA` 按 local expert 连续布局，`dispatchOffset` 64B 对齐到 row 粒度；后续 `expertOutput`
     直接按同一行序消费。

#### Combine 算子内部

1. **return TPUT 阶段**
   - peer rank round-robin 分给 AIV blocks，保留原算法按 `coreIdx + n * coreNum` 分摊通信的思路。
   - 每个 block 对自己负责的 peer，按 local expert 顺序写回，segment 起点由
     `cumsumPerExpert[src, globalExpert - 1]` 和 `dispatchOffset/prevSumBeforeRank` 计算。
   - `expertOutput -> remote(src).ptrD` 使用 PTO `TPUT` ping-pong staging；同一个 peer 的所有 local-expert segment
     完成后才 `TNOTIFY combineDoneSignal[myRank]`。
   - 不使用 atomic add：每个 `ptrD[row, :]` 只有 owner expert rank 写一次，restore 时再做 `probs` 加权。

2. **wait/restore 阶段**
   - 每个 block 先等待自己负责的 peer signal，再 `SYNCALL(AIVOnly)`，确保本 rank `ptrD` 全部可见后 restore。
   - restore 按连续 token range 分片；每个 block 独占输出 token row，避免 outputC 原子累加。
   - 对每个 token，先清零输出 row，再遍历 `topK`，对 `ptrD[row, :]` 做 hidden chunk vector 加权累加。
   - 如果 half × float 路径需要 float 累加，使用 workspace float scratch row/chunk；scratch 按 block 分片，避免跨 block
     共享。
   - `ptrD` 和 `outputC` row stride 都是 `K`，默认 row 起点 64B 对齐；hidden chunk 与 `tileCols` 对齐。

#### L0/L1/UB 与 cache 策略

- 该版本没有 GMM/SwiGLU，不使用 L0 matmul hot path；L0 对齐只作为后续接 expert compute 的行序契约保留。
- 热路径主要是 GM/window/UB Vec tile 搬运；L1 不作为必须 staging 层，除非后续实测 `TGET/TPUT` 对大 segment 需要
  L1 预取。第一版不引入手写 L1 cache path，避免和 PTO comm primitive 语义冲突。
- UB 采用双 buffer：metadata tile 独立于 payload tile；payload ping/pong tile 不与 weighted-add tile 复用同一地址段。
- `peerWindow` 字段按 64B 对齐，`packedA/ptrD/dispatchedA/expertOutput/outputC` row 起点按 64B 对齐；signal 独立
  cache line。
- 大 shape 下优先增加 `aivBlocks` 覆盖 peer shards 和 token shards；小 `EP` 场景 peer shards 不足时，payload segment
  内再按 row chunk 二级分摊，避免只有少数 block 工作。

## 9. Dispatch 输出行布局

每个 rank 的 `dispatchedA` 只保存本 rank local experts 收到的 token，使用 expert-major + source-rank-major 布局：

```text
this rank:
  local expert 0:
    source rank 0 rows
    source rank 1 rows
    ...
    source rank EP-1 rows
  local expert 1:
    source rank 0 rows
    source rank 1 rows
    ...
    source rank EP-1 rows
```

global expert 到 owner rank / local expert 的映射：

```text
ownerRank = expertId / expertPerRank
localExpert = expertId % expertPerRank
```

子段公式：

```text
expertRows(dst, localExpert) =
  sum_{src=0}^{EP-1} peerTokenPerExpert[src, dst * expertPerRank + localExpert]

dispatchOffset(dst, localExpert) =
  sum_{e=0}^{localExpert-1} expertRows(dst, e)

prevSumBeforeRank(dst, src, localExpert) =
  sum_{r=0}^{src-1} peerTokenPerExpert[r, dst * expertPerRank + localExpert]

segmentStart(dst, src, localExpert) =
  dispatchOffset(dst, localExpert) + prevSumBeforeRank(dst, src, localExpert)
```

在本 rank kernel 内，`dst == myRank`。公式保留 `dst` 是为了 CPU golden 和跨 rank解释一致。

## 10. Combine 与 restore 语义

Combine kernel 的输入 `expertOutput[row, :]` 必须与本 rank `dispatchedA[row, :]` 行顺序一一对应。

Combine return 阶段按 dispatch 的反向映射写回 owner rank 的 `peerWindow.ptrD`：

```text
ptrD[src, token * topK + slot, :] =
  expertOutput[dst, segmentStart(dst, src, localExpert) + i, :]
```

Restore 阶段只处理本 rank 原始 token：

```text
outputC[token, col] =
  sum_{slot=0}^{topK-1} probs[token, slot] * peerWindow.ptrD[token * topK + slot, col]
```

非法 expert id 或 inactive 路由如果后续加入，统一表现为：

```text
expandedRowIdx = -1
ptrD contribution = 0
```

第一版不设计 active mask 输入；如需覆盖 inactive token，可通过非法 expert id 模拟跳过。

## 11. Kernel API 草案

公共 shape 结构：

```cpp
struct DispatchCombineTileShape {
    uint32_t ep;
    uint32_t m;
    uint32_t k;
    uint32_t topK;
    uint32_t expertPerRank;
    uint32_t expertNum;
    uint32_t maxOutputSize;
    uint32_t aivBlocks;
    uint32_t tileCols;
    uint32_t rowChunk;
    uint32_t metadataPad;
};
```

Dispatch kernel：

```cpp
__global__ AICORE void DispatchCombineTileDispatch(
    DispatchCombineTileShape shape,
    uint32_t myRank,
    GM_ADDR inputA,
    GM_ADDR expertIdx,
    GM_ADDR peerWindow,
    GM_ADDR hcclCtx,
    GM_ADDR workspace);
```

Combine kernel：

```cpp
__global__ AICORE void DispatchCombineTileCombine(
    DispatchCombineTileShape shape,
    uint32_t myRank,
    GM_ADDR expertOutput,
    GM_ADDR probs,
    GM_ADDR outputC,
    GM_ADDR peerWindow,
    GM_ADDR hcclCtx,
    GM_ADDR workspace);
```

说明：

- `GM_ADDR` 只作为 kernel ABI 边界类型出现。
- `peerWindow` 是本 rank window base。
- `hcclCtx` 是 host 按 `gemm_ar` 模式准备的 device-visible `HcclDeviceContext`，用于计算 peer window 地址。
- kernel body 内必须尽快把 GM pointer 转换为 typed pointer 和 `pto::GlobalTensor` view。
- 每个进程只处理自己的 `myRank`，不遍历模拟所有 rank。
- `shape.aivBlocks` 必须等于实际 launch 的 AIV block 数，用于 token shard、peer shard 和 workspace
  `blockTokenPerExpert/blockPrefixPerExpert` 索引。
- `shape.tileCols` 控制 `TLOAD/TSTORE/TGET/TPUT` 的 hidden chunk 列数；host 负责校验非零且不超过 PTO Vec tile
  staging 能力。
- `shape.rowChunk` 控制大 segment 的 row 维二级分摊；`0` 表示 kernel 使用默认策略。
- `shape.metadataPad` 必须与 host workspace/window offset 计算一致，用于 metadata expert 维度 padding。

## 12. PTO tile 数据面设计

### 12.0 Kernel 文件结构与类型别名

`dispatch_combine_tile_kernel.cpp` 只放 PTO kernel 和 device helper。建议结构：

```text
dispatch_combine_tile_kernel.cpp
  includes: pto/pto-inst.hpp, pto/comm/pto_comm_inst.hpp, cstdint
  structs:
    HcclDeviceContext
    DispatchCombineTileShape
    WorkspaceLayout
    PeerWindowLayout
    WorkspaceView
    PeerWindowView
  helpers:
    AlignUp / PtrAdd / RemotePtr
    MakeGlobal1D / MakeGlobal2D / MakeSignal
    CopyRowHalf / RemoteGetRowsHalf / RemotePutRowsHalf
    PublishCountRows / WaitCountRows
    BuildPrefixMetadata
    DispatchPayloadGather
    CombineReturn
    RestoreOutput
  kernels:
    DispatchCombineTileDispatch
    DispatchCombineTileCombine
```

PTO alias：

```cpp
using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using GlobalHalf = pto::GlobalTensor<half, ShapeDyn, StrideDyn, pto::Layout::ND>;
using GlobalI32 = pto::GlobalTensor<int32_t, ShapeDyn, StrideDyn, pto::Layout::ND>;
using GlobalFloat = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
using VecHalfTile = pto::Tile<pto::TileType::Vec, half, 1, kTileCols, pto::BLayout::RowMajor, -1, -1>;
using VecI32Tile = pto::Tile<pto::TileType::Vec, int32_t, 1, kMetaCols, pto::BLayout::RowMajor, -1, -1>;
```

`kTileCols/kMetaCols` 第一版可以用 compile-time 上限实现，再用 `shape.tileCols` 控制实际 `GlobalTensor` view
长度；如果 PTO tile 类型必须静态列数，run.sh/CMake 通过 `-DCONFIG_TILE_COLS` 固化，host 同步校验 `tileCols`
等于编译值。

### 12.0.1 统一 layout 计算

host 和 kernel 必须共享同一套 offset 公式。实现时建议将 layout 结构只包含 offset 和 bytes，不含指针：

```cpp
struct WorkspaceLayout {
    uint64_t localTokenPerExpert;
    uint64_t blockTokenPerExpert;
    uint64_t blockPrefixPerExpert;
    uint64_t cumsumPerExpert;
    uint64_t dispatchOffset;
    uint64_t prevSumBeforeRank;
    uint64_t localSync;
    uint64_t floatScratch;
    uint64_t dispatchedA;
    uint64_t ptrDLocal;
    uint64_t totalBytes;
};

struct PeerWindowLayout {
    uint64_t peerTokenPerExpert;
    uint64_t expandedRowIdx;
    uint64_t packedA;
    uint64_t ptrD;
    uint64_t countReadySignal;
    uint64_t combineDoneSignal;
    uint64_t totalBytes;
};
```

layout 规则：

- 所有字段起点按 64B 对齐。
- `expertNumPadded = AlignUp(expertNum, metadataPad)`。
- `peerTokenPerExpert` shape 为 `[EP, expertNumPadded]`。
- `blockTokenPerExpert/blockPrefixPerExpert` shape 为 `[aivBlocks, expertNumPadded]`。
- `packedA/ptrD` row stride 为 `K`，row 数分别为 `M * topK`。
- `dispatchedA/expertOutput` row stride 为 `K`，row 数为 `maxOutputSize`。
- `floatScratch` 只有 half×float restore 需要 float 累加时分配，shape 至少 `[aivBlocks, tileCols]`。

host 用该 layout 计算 allocation/window bytes；kernel 用同一字段 offset 构造 view。禁止在 stage 主流程里散落手写
magic offset。

### 12.0.2 View 与 remote pointer helper

device helper：

```text
LocalWorkspaceView(workspaceBase, layout)
LocalPeerWindowView(peerWindowBase, layout)
RemotePeerWindowView(ctx, peerWindowBase, peerRank, layout)
```

`RemotePeerWindowView` 只做：

```text
remoteBase = ctx->windowsIn[peerRank] + (peerWindowBase - ctx->windowsIn[ctx->rankId])
```

然后对 `remoteBase + fieldOffset` 构造 `GlobalTensor` 或 signal view。它不调用 `TGET/TPUT/TNOTIFY/TWAIT`，也不做
cache/fence。

### 12.1 Local row copy helper

本地 payload 行搬运统一走 row tile helper：

```text
CopyRowHalf(dstBase, dstRowStride, srcBase, srcRowStride, rowLen)
```

实现要求：

- 每次处理一个 hidden chunk。
- chunk 使用 `Tile<TileType::Vec, half, 1, TileCols, RowMajor, -1, -1>`。
- source/destination 都构造成 `GlobalTensor<half, ShapeDyn, StrideDyn, Layout::ND>`。
- 用 `TLOAD` / `TSTORE` 完成 GM -> UB -> GM。
- pack 阶段使用两个 Vec tile：

  ```text
  for hidden chunk h:
    cur = ping/pong
    TLOAD(cur, inputA[token, h:h+cols])
    TSTORE(peerWindow.packedA[packedRow, h:h+cols], cur)
  ```

  后续可替换为 PTO event 链，第一版至少保持 ping-pong tile 地址不重叠。

### 12.2 Remote row helper

跨 rank payload 搬运统一走 PTO comm helper：

```text
TGET remote peerWindow.packedA[src] -> local workspace.dispatchedA
TPUT local expertOutput -> remote peerWindow.ptrD[src]
TNOTIFY/TWAIT signal readiness
```

实现要求：

- remote source/destination 也必须先封装成 `GlobalTensor` view。
- staging buffer 使用 `Tile<TileType::Vec, half, 1, TileCols, RowMajor, -1, -1>`。
- 大 row 按 hidden chunk 分段，必要时使用 PTO comm ping-pong overload。
- 不允许在 kernel 内调用 HCCL/SHMEM/AscendC 通信 API；HCCL 只提供 `HcclDeviceContext.windowsIn[]` 地址能力。
- `TGET/TPUT` 的实现映射：

  ```text
  GlobalHalf dst(localPtr + rowStart*K + colStart, shape=[rows, cols], stride=[K, 1])
  GlobalHalf src(remotePtr + rowStart*K + colStart, shape=[rows, cols], stride=[K, 1])
  pto::comm::TGET(dst, src, pingTile, pongTile)
  pto::comm::TPUT(remoteDst, localSrc, pingTile, pongTile)
  ```

- `rows` 很大时，outer loop 先切 `rowChunk`，inner loop 切 `tileCols`，避免单个 `GlobalTensor` view 过大导致
  backend 编译或调度不可控。

### 12.2.1 Metadata TPUT/TWAIT helper

count row exchange 用 PTO comm，不做裸 GM 远端写：

```text
PublishCountRow(dstRank):
  local GlobalI32 workspace.localTokenPerExpert[0:expertNumPadded]
  remote GlobalI32 remote(dstRank).peerTokenPerExpert[myRank, 0:expertNumPadded]
  TPUT(remoteCountRow, localCountRow, metaPingTile, metaPongTile)
  TNOTIFY(remote(dstRank).countReadySignal[myRank], 1, Set)

WaitCountRows():
  TWAIT(peerWindow.countReadySignal[src], 1, GE)
```

combine done 同理使用 signal view：

```text
TNOTIFY(remote(src).combineDoneSignal[myRank], 1, Set)
TWAIT(peerWindow.combineDoneSignal[peer], 1, GE)
```

signal view 用 `GlobalTensor<int32_t>` shape `[1]` 或 `pto::comm::Signal`，但调用点必须显式是
`pto::comm::TNOTIFY/TWAIT`。

### 12.3 Weighted add helper

restore 使用：

```text
AddWeightedRow(outputRow, ptrDRow, prob, K)
```

第一版可分两步实现：

1. `TLOAD` output 和 ptrD row 到 Vec tile。
2. `TMULS(ptrDTile, ptrDTile, prob)` 后 `TADD(outTile, outTile, ptrDTile)`，再 `TSTORE`。

如果 half + float 权重的 PTO 指令限制导致类型路径复杂，第一版允许 output 累加 buffer 在 workspace 中使用 float，
最终再转 half 输出；该调整必须在设计文档和 verifier tolerance 中同步记录。

### 12.4 Metadata

metadata 的 token counting 和 prefix 第一版允许使用标量循环，因为它是 correctness scaffold，且数据量远小于 payload。
但标量 GM offset 必须集中到 metadata helper 中，不能散落在 stage 主循环。

实现边界：

- token counting 写 `blockTokenPerExpert[blockId, expert]`，不使用原子。
- prefix helper 只在 metadata GM 区读写 int32，不访问 payload。
- `localTokenPerExpert` 在 publish count 前必须完整写好并经过 `SYNCALL(AIVOnly)`。
- `cumsumPerExpert[src, expert]` 使用 padded expert stride，但只对 `[0, expertNum)` 有效。

### 12.5 Dispatch kernel stage 映射

`DispatchCombineTileDispatch` 内部建议按 stage 函数组织：

```text
InitViews(shape, workspace, peerWindow, hcclCtx)
ClearDispatchState()
CountLocalRoutes()
BuildBlockPrefixAndLocalCounts()
BuildPackedExpertOffset()
PackLocalRowsToWindow()
PublishCountRows()
WaitCountRows()
BuildPrefixMetadata()
GatherLocalExpertPayload()
```

其中：

- `ClearDispatchState` 只清当前迭代会读的 metadata/signal；大 payload 可按写入覆盖，不强制全清。
- `CountLocalRoutes` 和 `PackLocalRowsToWindow` 使用连续 token shard。
- `PublishCountRows` 和 `GatherLocalExpertPayload` 使用 peer rank shard。
- `GatherLocalExpertPayload` 必须只拉取 `ownerRank == myRank` 的 global experts。

### 12.6 Combine kernel stage 映射

`DispatchCombineTileCombine` 内部建议按 stage 函数组织：

```text
InitViews(shape, workspace, peerWindow, hcclCtx)
ClearCombineState()
ReturnExpertRowsToOwners()
WaitAllCombineDone()
RestoreOutputRows()
```

其中：

- `ClearCombineState` 至少清 `combineDoneSignal` 和 `outputC` 目标 row；`ptrD` 可按写入覆盖。
- `ReturnExpertRowsToOwners` 使用 peer rank shard；每个 peer 的所有 local expert segment 完成后只发一次 notify。
- `WaitAllCombineDone` 先各 block 等自己负责的 peer，再 `SYNCALL(AIVOnly)`。
- `RestoreOutputRows` 使用连续 token shard，每个 block 独占 token row。

## 13. 开发文档分工

本项目当前使用两个 md 文件指导后续开发：

| 文件 | 作用 | 更新时机 |
| --- | --- | --- |
| `DESIGN.md` | 算法、接口、约束、内存布局、性能编排和验收定义的设计基线 | 只有设计决策变化时更新 |
| `IMPLEMENTATION_PLAN.md` | 实现任务拆分、任务状态、阶段验证方式和执行反馈记录 | 每完成或阻塞一个任务时更新 |

开发时先以 `DESIGN.md` 判断实现是否符合目标语义，再以 `IMPLEMENTATION_PLAN.md` 跟踪当前任务、执行命令、
验证结果和遗留风险。不要新增单独 `TODO.md`，除非后续范围扩大到多个独立计划。

## 14. Host 执行流程

### 14.1 PTO 化项目初始化方案

项目初始化不是简单复制原 SHMEM 示例。第一版目录结构按 `gemm_ar` 的 manual PTO/HCCL 工程组织建立，
同时保留原 `dispatch_gmm_combine` 的 shape、数据文件、debug dump 和 verify 语义。

#### 14.1.1 来源映射

| 来源 | 继承内容 | PTO 化处理 |
| --- | --- | --- |
| `gemm_ar/CMakeLists.txt` | Bisheng/CMake、host executable + vector kernel shared library、PTO include 优先级、HCCL/ACL link | 目标改为 `dispatch_combine_tile` 和 `dispatch_combine_tile_kernel`；不新增 cube kernel |
| `gemm_ar/run.sh` | CANN 环境搜索、MPI 搜索、`mpirun -n`、`HCCL_BUFFSIZE` 自动抬高、build/run 一体 | 参数替换为 dispatch/combine 显式 shape；不提供 `--case all` |
| `gemm_ar/comm_mpi.h` | dlopen MPI、rank/size/bcast/barrier/finalize | 直接作为 host 多进程 bootstrap 基础 |
| `gemm_ar/hccl_context.h` + `main.cpp` HCCL 初始化 | `HcclDeviceContext`、MESH/RING window 提取、`windowsIn[]` device-visible context | 保留 HCCL window 地址模型；peerWindow 从 `windowsIn[rank]` 固定 offset 切片 |
| 原 `dispatch_gmm_combine/scripts/run.sh` | `-pes/-M/-K/-expertPerPe`、out/data 目录、生成输入、验证输出 | 删除 SHMEM `ipport/fnpu` 启动；用 `mpirun` 和 `--device-base` 替代 |
| 原 `dispatch_gmm_combine/main.cpp` | rank-local input/output、routing metadata、workspace/window 规模、warmup/iters/profile 输出 | 删除 Catlass/AscendC/SHMEM/GMM/SwiGLU/量化；只保留 dispatch/combine 语义和计时维度 |

明确不复制：

- `aclshmem_add_fusion_example`、`aclshmem_*`、`shmem_*`、`symmetricPtr`；
- Catlass、AscendC kernel API、`moe_init_routing_quant_v2`；
- 原 `for idx ... & wait` 多进程启动方式；
- 原 GMM/权重/量化参数和相关 utils 依赖。

#### 14.1.2 初始化后目录与接口边界

初始化阶段需要建立后续 PTO 实现会直接使用的接口骨架：

```text
dispatch_combine_tile/
  DESIGN.md                    # 设计基线
  IMPLEMENTATION_PLAN.md        # 任务、状态、反馈
  CMakeLists.txt                # gemm_ar 风格 build，单 host exe + 单 vec kernel so
  run.sh                        # gemm_ar 风格 mpirun/HCCL_BUFFSIZE/build/run
  common.h                      # shared ABI: shape/layout/runtime flags
  args.h                        # host 参数解析与校验
  layout.h                      # host/kernel 共享 offset 公式入口
  golden.h                      # host deterministic data + CPU golden 接口
  hccl_context.h                # HcclDeviceContext 与 host HCCL window context
  comm_mpi.h                    # MPI dlopen wrapper
  kernel_launchers.h            # 两个 kernel 的 host launch ABI
  dispatch_combine_tile_kernel.cpp
  main.cpp
```

`common.h` 初始化时先定义 ABI 名称和字段，不要求实现完整逻辑：

```cpp
struct DispatchCombineTileShape;
struct WorkspaceLayout;
struct PeerWindowLayout;
struct DispatchCombineTileRuntimeConfig;
```

`kernel_launchers.h` 初始化时必须固定后续实现使用的两个 launch wrapper 名称：

```text
LaunchDispatchCombineTileDispatch(...)
LaunchDispatchCombineTileCombine(...)
```

`dispatch_combine_tile_kernel.cpp` 初始化时必须固定两个 device kernel 名称：

```text
DispatchCombineTileDispatch
DispatchCombineTileCombine
```

后续 Task 1 只在这些已固定接口上补编译实现；Task 2 补参数/layout；Task 3 补 MPI/HCCL；Task 5 以后补 PTO
tile 数据面。初始化阶段不得引入第三个 device kernel、临时 SHMEM 路径或和上述 ABI 不一致的替代入口。

#### 14.1.3 初始化主流程骨架

`main.cpp` 初始化时按后续完整流程放置阶段函数声明或空实现，保证实现任务可以逐段填充：

```text
ParseArgs
InitMpiAndRank
BindDeviceContinuous
InitHcclWindowContext
ComputeLayouts
AllocateLocalBuffers
SlicePeerWindow
GenerateOrLoadData
RunDispatch
PrepareExpertOutputIdentity
RunCombine
VerifyAndDump
Cleanup
```

`run.sh` 初始化时按 `gemm_ar` 模式预留这些阶段：

```text
source CANN env
locate MPI/mpirun
parse explicit shape/runtime/debug args
estimate HCCL_BUFFSIZE from peerWindowBytes
build unless skip-build
mpirun -n ${PES} ./dispatch_combine_tile ...
```

原始项目的 `out/` 文件约定映射到本项目 `--data-dir`，并由 host/golden 代码负责生成和校验。初始化阶段只建立
目录参数和文件命名契约，不复制原 Python 量化/GMM 数据生成脚本。

### 14.2 工程产物

项目目录第一版至少包含：

```text
dispatch_combine_tile/
  CMakeLists.txt
  run.sh
  main.cpp
  hccl_context.h
  comm_mpi.h
  kernel_launchers.h
  dispatch_combine_tile_kernel.cpp
```

构建产物：

```text
build/dispatch_combine_tile
```

`CMakeLists.txt` 参考 `gemm_ar`，但目标名改为 `dispatch_combine_tile`；kernel 文件只 include PTO headers，不 include
AscendC/Catlass/SHMEM。

如 PTO Vec tile 列数需要静态模板参数，CMake 暴露：

```text
-DCONFIG_TILE_COLS=${TILE_COLS}
-DCONFIG_META_COLS=${META_COLS}
```

host binary 启动时校验运行参数 `--tile-cols/--metadata-pad` 与编译常量一致。

### 14.3 Host 主流程

```text
parse args
source CANN set_env.sh or require ASCEND_CANN_PATH/ASCEND_HOME_PATH
locate mpirun from --mpi-bin or gemm_ar-style MPI_SEARCH_DIRS
clean build dir if clean-build=1; cmake/make unless skip-build=1
clean/create data-dir
generate per-rank inputA/expertIdx/probs if gen-data=1
compute CPU golden metadata/output for verify path
CommMpiInit / get rank and size
validate size == pes
device = deviceBase + rank
aclInit / aclrtSetDevice / create stream
initialize HCCL/bootstrap resources following gemm_ar pattern
choose aivBlocks and tileCols explicitly or from device capability
compute workspaceBytes and peerWindowBytes
check HCCL winSize and HCCL_BUFFSIZE cover peerWindowBytes

allocate inputA / expertIdx / probs / outputC / workspace / expertOutput
slice peerWindow from windowsIn[rankId] at fixed offset
prepare device-visible HcclDeviceContext / window context

for each iteration:
  clear workspace / peerWindow / outputC
  host barrier

  time dispatch:
    DispatchCombineTileDispatch<<<aivBlocks...>>>
    aclrtSynchronizeStream
    host barrier

  prepare expertOutput = workspace.dispatchedA identity
  host barrier

  time combine:
    DispatchCombineTileCombine<<<aivBlocks...>>>
    aclrtSynchronizeStream
    host barrier

  copy outputC and optional debug buffers back
  verify against CPU golden

free resources / CommMpiFinalize
```

第一版允许 host 通过 D2H/H2D 准备 `expertOutput`，因为该路径是验证 helper，不计入 dispatch/combine kernel 语义。
host barrier 只用于进程级阶段对齐和 window 复用保护，不替代 device kernel 内部的 `TWAIT`。

## 15. 计时与验收

必须打印：

```text
dispatch_e2e_us=...
prepare_host_us=...
combine_e2e_us=...
total_e2e_us=...
```

定义：

| 指标 | 起点 | 终点 | 说明 |
| --- | --- | --- | --- |
| `dispatch_e2e_us` | host barrier 后、launch dispatch 前 | dispatch stream sync + host barrier 后 | 包含 dispatch kernel 和跨 rank 等待 |
| `prepare_host_us` | dispatch 后准备 expertOutput 前 | expertOutput device buffer ready | host 验证 helper 时间，不算核心 kernel |
| `combine_e2e_us` | host barrier 后、launch combine 前 | combine stream sync + host barrier 后 | 包含 combine kernel、device `TWAIT` 和跨 rank 等待 |
| `total_e2e_us` | dispatch launch 前 | combine stream sync 后 | dispatch + prepare + combine |

必须 verify：

| debug | 验证内容 |
| --- | --- |
| `0` | `outputC` vs CPU golden |
| `1` | debug 0 + `localTokenPerExpert/peerTokenPerExpert/cumsumPerExpert/dispatchOffset` |
| `2` | debug 1 + `blockTokenPerExpert/blockPrefixPerExpert/packedA/dispatchedA/ptrD` 前若干行 |

性能验收除打印 e2e 外，还必须在 debug 或 verbose 日志中输出：

```text
aiv_blocks=...
tile_cols=...
peer_window_bytes=...
workspace_bytes=...
dispatch_peer_shards=...
combine_peer_shards=...
```

验收时检查：

- dispatch/combine launch 的 AIV block 数大于 1 时，peer 维度确实由多个 block 分摊；
- routing pack 使用连续 token shard + block prefix，不能退化为单 block pack；
- `TGET/TPUT` 路径使用 ping-pong staging tile 或等价 PTO comm double-buffer overload；
- combine restore 按 token shard 并行执行。

## 16. run.sh 参数与默认命令

`run.sh` 一次只运行一组显式 shape，不提供 `--case all`。

默认命令：

```bash
bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 -debug 0 -iters 5 -warmup 3
```

`run.sh` 必须用 `mpirun -n ${pes}` 启动对应数量的 host 进程，每个进程获得独立 MPI rank 和 device id。不要继承原
SHMEM 项目的 `ipport + for loop + background process` 启动方式。多进程启动、MPI 搜索、HCCL root info broadcast
参考 `gemm_ar`；device kernel 侧仍只能使用 PTO comm primitive。

host binary 参数建议固定为长参数，run.sh 负责把短参数展开，例如：

```bash
mpirun -n ${PES} ./dispatch_combine_tile \
  --rank-from-mpi 1 \
  --first-device ${DEVICE_BASE} \
  --tokens ${M} --hidden ${K} --topk ${TOPK} --experts-per-rank ${EXPERT_PER_PE} \
  --max-output-size ${MAX_OUTPUT_SIZE} \
  --aiv-blocks ${AIV_BLOCKS} --tile-cols ${TILE_COLS} --row-chunk ${ROW_CHUNK} \
  --metadata-pad ${METADATA_PAD} \
  --data-dir ${DATA_DIR} --seed ${SEED} --debug ${DEBUG} \
  --warmup ${WARMUP} --iters ${ITERS} --verify ${VERIFY} --rtol ${RTOL} --atol ${ATOL}
```

host binary 内部从 MPI 获取 `rank/size`；只有调试单进程时才允许显式 `--rank/--nranks`，但该路径不作为第一版验收。

`run.sh` 必须按显式 shape 估算并设置 `HCCL_BUFFSIZE`，覆盖：

```text
peerWindowBytes =
  align(peerTokenPerExpert bytes)
  + align(expandedRowIdx bytes)
  + align(packedA bytes)
  + align(ptrD bytes)
  + align(signal bytes)
  + safety margin
```

不得依赖手工预置一个固定 HCCL window 大小来跑默认 shape。

`run.sh` 启动前必须打印完整参数摘要：

```text
RUN_MODE=...
SOC_VERSION=...
PES=... DEVICE_BASE=... NDEVICES=...
M=... K=... TOPK=... EXPERT_PER_PE=... MAX_OUTPUT_SIZE=...
AIV_BLOCKS=... TILE_COLS=... ROW_CHUNK=...
HCCL_BUFFSIZE=... DATA_DIR=...
WARMUP=... ITERS=... VERIFY=... DEBUG=...
```

建议验收命令：

```bash
# 小规模 layout 排查
bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 -debug 2

# 常规正确性
bash run.sh -pes 2 -M 128 -K 256 -topK 2 -expertPerPe 2 -debug 1
bash run.sh -pes 2 -M 128 -K 256 -topK 4 -expertPerPe 2 -debug 1

# split 风格默认 shape
bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 -debug 0

# 较大 shape
bash run.sh -pes 2 -M 1024 -K 2048 -topK 8 -expertPerPe 2 -debug 0
```

## 17. Shape 校验

host 必须拒绝以下输入：

- `pes == 0`
- `runMode != npu`
- `M == 0`
- `K == 0`
- `topK == 0`
- `expertPerPe == 0`
- `aivBlocks == 0` 且 host 无法从 device capability 推导有效 block 数
- `tileCols == 0`
- `K % tileCols != 0` 第一版拒绝；后续可增加 tail tile 支持
- `rowChunk != 0` 且 `rowChunk` 过小导致 row chunk 调度开销明显大于 payload 搬运
- `expertNum != pes * expertPerPe` 溢出
- `maxOutputSizeArg != 0` 且小于 `EP * M * topK`；第一版不做 capacity/drop
- `M * topK`、`EP * M * topK`、`workspace bytes`、`peerWindow bytes` 溢出 `size_t`
- `maxOutputSize * K` 超过 device allocation 能力
- `peerWindow bytes` 超过 HCCL `winSize`
- `deviceBase + pes > ndevices`
- `pes` 大于 HCCL/MPI bootstrap 支持的 rank 数
- 找不到 CANN 环境、`mpirun` 或 HCCL 初始化失败
- `dataDir` 不可创建/不可写；`genData=0` 时缺少任一 rank 输入文件
- `verify=1` 且无法生成或读取 CPU golden

kernel 内可以保留轻量防御：

```text
if shape field is zero: return
if expert id invalid: mark expandedRowIdx = -1 and skip
```

## 18. 第一版完成定义

第一版实现必须满足：

1. `run.sh` 能从项目目录一键编译并运行默认 shape。
2. 默认 `-pes 2` 路径真实启动 2 个 host 进程并跨 2 个 device/window 通信，不能退化为单进程模拟。
3. device 侧只有 `DispatchCombineTileDispatch` 和 `DispatchCombineTileCombine` 两个计算 kernel。
4. host/device 代码都不调用 SHMEM/aclshmem 接口；kernel 文件不 include AscendC/Catlass/SHMEM device 依赖。
5. 本地 payload 行搬运通过 PTO `GlobalTensor` / `Tile` / `TLOAD` / `TSTORE` 表达，跨 rank payload 搬运通过 PTO
   `TGET/TPUT` 表达。
6. 跨 rank readiness 通过 PTO `TNOTIFY/TWAIT` 表达。
7. dispatch/combine 内部保留原算法的并发分工：连续 token shard 并行 routing、peer rank round-robin 通信、
   ping-pong PTO staging、并行 restore，不能提交单 block 串行版本作为第一版。
8. 打印 `dispatch_e2e_us`、`prepare_host_us`、`combine_e2e_us`、`total_e2e_us`。
9. `debug=0` 至少验证 `outputC`，`debug=2` 能验证 `blockTokenPerExpert/blockPrefixPerExpert/packedA/dispatchedA/ptrD`
   的布局。
