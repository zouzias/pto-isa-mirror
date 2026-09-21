# TPUT_ASYNC

## 简介

`TPUT_ASYNC`是异步远程写原语。默认`IMMEDIATE`模式会发布一次从本地GM到远端GM的传输并返回
对应`AsyncEvent`。`DEFER`模式只暂存远程写，在`SubmitAsyncPutBatch`前不会启动传输；暂存过程可能
等待队列资源，随后返回无效占位Event。

数据流：

`srcGlobalData（本地 GM）` → DMA引擎 → `dstGlobalData（远端 GM）`

## 模板参数

- `engine`：
    - `DmaEngine::SDMA`（默认）
    - `DmaEngine::URMA`（Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510）
    - `DmaEngine::RDMA`（Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510；当前网卡平台仅支持 HNS1825）

> **注意（SDMA路径）**
> `TPUT_ASYNC` 配合 `DmaEngine::SDMA` 目前**仅支持扁平连续的逻辑一维tensor**。
> 当前SDMA异步实现不支持非一维或非连续布局。

## C++内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// A2/A3
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session,
                               AsyncPutMode mode = AsyncPutMode::IMMEDIATE,
                               WaitEvents &... events);

// A5，peer来自session.destRankId
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session,
                               AsyncPutMode mode = AsyncPutMode::IMMEDIATE,
                               uint32_t jettyIndex = 0U,
                               WaitEvents &... events);

// A5，显式peer
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, uint32_t peer,
                               AsyncPutMode mode = AsyncPutMode::IMMEDIATE,
                               uint32_t jettyIndex = 0U,
                               WaitEvents &... events);
```

默认`IMMEDIATE`保持标准立即提交并返回对应完成Event。`DEFER`只暂存远程写并返回无效占位Event
（`handle == 0`）。传入前置Event时必须先显式写出`mode`；A5还必须在Event前显式写出
`jettyIndex`，包括`0U`。

## 提交模式

| 模式 | 行为 | 返回值 |
|---|---|---|
| `AsyncPutMode::IMMEDIATE`（默认） | 立即发布本次远程写。 | 本次已提交操作对应的有效完成`AsyncEvent`。 |
| `AsyncPutMode::DEFER` | 将本次远程写暂存到当前Session，不推进硬件可见Producer，也不敲Send Doorbell。 | `handle == 0`的无效占位Event，不得用于判断完成。 |

一次或多次成功的非零Defer调用组成当前聚合Batch。使用`SubmitAsyncPutBatch`发布该Batch。只有
`SubmitAsyncPutBatch`返回的Event表示Batch完成。

## SubmitAsyncPutBatch辅助函数

`SubmitAsyncPutBatch`是`PTO_INTERNAL`辅助函数，不是独立PTO指令：

```cpp
// A2/A3 SDMA及A5绑定peer形式
template <DmaEngine engine = DmaEngine::SDMA>
PTO_INTERNAL AsyncEvent SubmitAsyncPutBatch(const AsyncSession& session);

// A5显式peer形式
template <DmaEngine engine = DmaEngine::SDMA>
PTO_INTERNAL AsyncEvent SubmitAsyncPutBatch(
    const AsyncSession& session, uint32_t peer,
    uint32_t jettyIndex = 0U);
```

A2/A3 SDMA使用仅含Session的形式。A5 URMA中，仅含Session的形式发布首次成功Defer绑定的`peer`和
`jettyIndex`；显式peer形式要求其值与Batch中所有Defer完全一致。有效返回Event覆盖Batch中全部
非零远程写；调用`Wait/Test`前应先检查`event.valid()`。

`AsyncSession` 是引擎无关的会话对象。使用 `BuildAsyncSession<engine>()` 构建一次后，传递给所有异步调用和事件等待。模板参数 `engine` 在编译期选择DMA后端，使代码对未来引擎（CCU等）保持前向兼容。

## AsyncSession构建

使用 `include/pto/comm/async_common/async_event_impl.hpp` 中的 `BuildAsyncSession`。
该函数有两个重载——分别用于SDMA和URMA，参数列表不同。

### SDMA构建（默认）

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    AsyncSession &session,
                                    uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scratchTile` | — | 用于SDMA控制元数据的UB scratch tile（参见 [scratchTile的作用](#scratchtile的作用)）。|
| `workspace` | — | 由主机侧 `SdmaWorkspaceManager` 分配的GM指针。|
| `session` | — | 输出的 `AsyncSession` 对象。|
| `syncId` | `0` | MTE3/MTE2管道同步事件ID（0-7）。若kernel在相同ID上使用了其他管道屏障，则需覆盖此值。|
| `baseConfig` | `{kDefaultSdmaBlockBytes, 0, 1}` | `{block_bytes, comm_block_offset, queue_num}`。适用于大多数单队列传输场景。|
| `channelGroupIdx` | `kAutoChannelGroupIdx` | SDMA通道组索引。默认内部使用 `get_block_idx()` 映射到当前AI Core。多block、并发或自定义通道映射场景下需覆盖此值。|

### URMA构建（仅NPU_ARCH 3510）

> URMA（User-level RDMA Memory Access）是Ascend 950PR/Ascend 950DT（NPU_ARCH 3510）上的硬件加速RDMA传输引擎。
> URMA要求CANN Toolkit **>= 9.1.0**。

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace,
                                    uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

| 参数 | 说明 |
|---|---|
| `workspace` | 由主机侧 `UrmaWorkspaceManager` 分配的GM指针。|
| `destRankId` | 此会话通信的远端PE rank id。对于 `TPUT_ASYNC`，这是数据写入的目标rank。|
| `session` | 输出的 `AsyncSession` 对象。|

URMA不需要 `scratchTile`——轮询通过 `ld_dev`/`st_dev` 硬件原语直接操作。

### RDMA构建（仅NPU_ARCH 3510）

RDMA面向经典跨机场景组网，当前网卡平台仅支持 HNS1825。

```cpp
#ifdef PTO_RDMA_SUPPORTED
template <DmaEngine engine, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile,
                                    __gm__ uint8_t *workspace,
                                    uint32_t myPe,
                                    AsyncSession &session,
                                    uint32_t syncId = 0);
#endif
```

| 参数 | 说明 |
|---|---|
| `scratchTile` | 用于 WQE/CQE 控制数据的 UB/Vec tile，至少 64 字节。|
| `workspace` | Host侧 RDMA 初始化流程返回的 GM 指针。|
| `myPe` | 本地 rank id，用于选择已注册的本地内存区域。|
| `session` | 输出的 `AsyncSession` 对象。|
| `syncId` | MTE/Scalar 同步事件 ID，取值范围为 0-7。|

构建一次不绑定 peer 的Session，再通过显式 `peer` 参数选择目标rank：

```cpp
comm::AsyncSession session;
if (comm::BuildAsyncSession<comm::DmaEngine::RDMA>(scratchTile, rdmaWorkspace, myPe, session, syncId)) {
    auto event = comm::TPUT_ASYNC<comm::DmaEngine::RDMA>(dstG, srcG, session, peer);
    (void)event.Wait(session);
}
```

## 约束

- `GlobalSrcData::RawDType == GlobalDstData::RawDType`
- `GlobalSrcData::layout == GlobalDstData::layout`
- SDMA和URMA路径均要求源tensor为**扁平连续的逻辑一维**
- SDMA workspace必须是由主机侧 `SdmaWorkspaceManager` 分配的有效GM指针
- URMA workspace必须是由主机侧 `UrmaWorkspaceManager` 分配的有效GM指针
- Session和workspace的生命周期必须覆盖相关Event的完成阶段
- URMA仅在NPU_ARCH 3510（Ascend 950PR/Ascend 950DT）上可用
- URMA要求CANN Toolkit **>= 9.1.0**
- 传给 `UrmaWorkspaceManager::Init()` 的对称数据缓冲区必须由大页内存支撑（使用 `ACL_MEM_MALLOC_HUGE_ONLY` 分配）。底层MR注册要求大页背景；`ACL_MEM_MALLOC_HUGE_FIRST` 在小尺寸分配时可能静默回退到4KB小页，导致注册失败
- RDMA仅在Ascend 950PR/Ascend 950DT（NPU_ARCH 3510）上可用，当前网卡平台仅支持 HNS1825
- RDMA源、目的tensor必须是扁平连续的逻辑一维，且本地和远端完整传输范围必须位于Host初始化阶段注册的内存区域内
- 单次RDMA传输不能超过 `0x7fffffff` 字节

若不满足一维连续要求，当前实现返回无效async event（`handle == 0`）。

### Defer聚合Batch约束

- 聚合模式支持A2/A3的`DmaEngine::SDMA`和A5的`DmaEngine::URMA`，不支持A5 SDMA聚合。
- A2/A3接口不包含`peer`和`jettyIndex`；A5保留绑定peer与显式peer两种形式。
- 首次成功的非零Defer启动Batch；零长度Defer为no-op。
- 每次Defer和最终Submit必须使用相同Session及Engine。一个URMA Batch还必须保持相同的`peer`和
  `jettyIndex`。
- 首次Defer到Submit之间，不得在同一Channel Group或Jetty上插入普通`TPUT_ASYNC`、
  `TGET_ASYNC`、`TPUT_ASYNC_NOTIFY`、另一个聚合Batch或`TPREFETCH_ASYNC`。
- Submit前不得复制、重建、销毁Session或转移其所有权。
- 所有源数据范围、Session和Workspace必须保持有效，直至Submit Event完成。
- Defer或Submit在发布前检测到参数、资源或容量错误时，会丢弃当前暂存Batch，不会将其发布。
  Defer始终返回无效占位Event，因此该返回值不能区分成功与失败；调用方必须在首次Defer前满足全部
  前置条件。
- Submit要求当前Batch至少包含一次成功的非零Defer。空Submit返回无效Event，并在启用断言的构建
  中报告调用契约错误。
- Submit后Session隐藏Batch状态被清零，可以开始新Batch；这不表示上一Batch已经完成。
- 同一Batch中不同远端目的范围不得重叠，不得利用Defer顺序表达写间依赖。

### Defer聚合Batch资源限制

- 源、目的Tensor必须具有相同RawDType和Layout，必须都是扁平连续的逻辑一维Tensor；每次非零传输
  的源、目的指针均不得为空。
- Tensor元素数、字节数和地址范围计算必须能用`uint64_t`表示；目的Tensor元素容量不得小于源
  Tensor元素数。
- 接口不校验通信内存归属、地址边界、范围重叠或目标Peer所有权，调用方必须保证这些条件。
- 每次SDMA Defer消耗`ceil(transferBytes / blockBytes)`个数据SQE，并基于整个Batch累计SQE索引
  在`queueNum`个队列间轮转分配；每个队列分配到的数量必须小于该队列SQ深度。
- SDMA会在源、目的Tensor基地址上增加`commBlockOffset`。底层分配必须无溢出地覆盖
  `[base + commBlockOffset, base + commBlockOffset + transferBytes)`。
- URMA要求有效的注册Workspace、Peer内存注册和Jetty；`peer`必须小于Workspace Rank数，且
  `jettyIndex < session.qpCount`。
- 完整URMA源范围必须位于`UrmaWorkspaceManager::Init()`注册的本地通信缓冲区内，完整目的范围
  必须位于所选Peer的已注册通信缓冲区内。
- URMA将一次远程写拆分为每个最大256 MiB的WQE。完整Batch必须同时适配所选Jetty的WQ和CQ深度。
  更大工作负载应拆分为多个Batch；接口不会自动提交部分Batch。

## scratchTile的作用

`scratchTile` **不是**用于存放用户数据负载的暂存缓冲区。
它被转换为 `TmpBuffer`，用作临时UB工作区，用于：

- 写入/读取SDMA控制字（flag、sq_tail、channel_info）
- 轮询事件完成标志
- 完成时提交队列尾部

实际数据负载直接在GM缓冲区之间传输；`scratchTile` 仅用于控制和同步元数据。

## scratchTile类型与大小约束

- 必须是 `pto::Tile` 类型
- 必须是UB/Vec tile（`ScratchTile::Loc == TileType::Vec`）
- 可用字节数至少为 `sizeof(uint64_t)`（8字节）

推荐使用：`Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256Byte）。

## 完成语义（Quiet语义）

对于`IMMEDIATE`模式，不同引擎的底层完成机制不同，但用户侧的quiet语义行为一致：

- **SDMA**：每次`TPUT_ASYNC`都会提交数据传输SQE和用于标记本次操作完成的flag SQE。对其返回的Event调用`Wait`或`Test`时，通过轮询对应flag判断该次`TPUT_ASYNC`是否完成；完成后也能保证同一Session中此前提交的所有SDMA操作均已完成。
- **URMA**：`TPUT_ASYNC` 立即提交RDMA WRITE WQE并敲门铃。`Wait` 通过轮询Completion Queue（CQ）等待所有预期的CQE被消费。
- **RDMA**：`TPUT_ASYNC` 向 `peer` 选择的队列提交RDMA WRITE WQE；不同peer/queue分别跟踪完成状态。

- `event.Wait(session)` —阻塞，直到**自上次Wait以来所有已发出的异步操作**全部完成

这意味着多次 `TPUT_ASYNC` 调用后，只需对最后一个返回的 `AsyncEvent` 调用一次 `Wait`，即可等待所有pending操作完成（类似shmem的quiet语义）。

RDMA操作涉及不同peer时，必须分别等待每个peer的最后一个Event。

同一Session最多可有64个未完成操作，超过后提交可能产生背压。

wait成功后，所有已发出的 `dstGlobalData` 写入均已全部完成。

对于`DEFER`模式，每次`TPUT_ASYNC`返回的占位Event均为无效Event。对该占位Event执行`Wait`或
`Test`不能证明传输已经发布或完成；只能等待`SubmitAsyncPutBatch`返回的有效Event。Submit只负责
发布暂存任务，不表示Batch已经完成；Submit Event的Wait/Test成功后才表示发送端完成并允许复用
源数据。

聚合Batch不是事务。接收端可能逐步观察到不同写，错误发生后已完成的写不会回滚，公共契约也不
定义各写之间的完成顺序。Submit Event不会自动通知接收端；接收端消费仍需使用`TNOTIFY/TWAIT`
等应用协议，并执行平台要求的Cache可见性处理。

只有Wait/Test成功才能确认完成。失败结果不会取消已提交传输；此时不得复用源数据，也不得释放或
重建Session、Workspace、Scratch Tile、Channel Group或Jetty。

## SDMA并发与Session所有权

- 同一个Session不能被多个执行流并发使用。
- 共用同一Channel Group的操作必须共享同一个Session。
- 并发Kernel或Kernel内多个独立Session必须使用隔离的Channel Group。
- 重新构建Session或复用Channel Group前，必须先完成此前所有Event。

### Defer聚合Batch并发与Session所有权

- 同一Session上的Build/Rebuild、Defer、Submit、Wait和Test必须串行执行。
- 不同Session不代表物理资源不同。从Build到最后一个Event完成期间，不得将另一Session映射到相同
  SDMA Channel、URMA WQ或URMA CQ。
- URMA `PER_PEER`模式下，同一Peer共享同一物理WQ/CQ，调用必须串行；不同Peer使用不同队列。
- URMA `SHARED_POOL`模式下，物理Jetty为`session.qpIdxBase + jettyIndex`。同一AIV上的Session
  若该物理索引相同则发生别名，即使Peer不同也会冲突。
- 同一物理Channel Group、WQ或CQ上的Wait/Test不得与Defer、Submit或另一个Wait/Test并发执行。
- 只有Session使用的所有物理队列上的未完成Event均已完成后，才能重建或销毁Session及通信资源。
- 释放URMA通信资源前，还必须同步所有使用该通信Context的Host Stream，确保后续Stream任务不再
  引用该Context。

## 示例

### 单次传输

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimplePut(__gm__ T *remoteDst, __gm__ T *localSrc,
                                 __gm__ uint8_t *sdmaWorkspace)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc,  shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```

### 批量传输（Quiet语义）

```cpp
template <typename T>
__global__ AICORE void BatchPut(__gm__ T *remoteDstBase, __gm__ T *localSrc,
                                __gm__ uint8_t *sdmaWorkspace, int nranks)
{
    using ShapeDyn  = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT        = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT srcG(localSrc, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session)) {
        return;
    }

    comm::AsyncEvent lastEvent;
    for (int rank = 0; rank < nranks; ++rank) {
        GT dstG(remoteDstBase + rank * 1024, shape, stride);
        lastEvent = comm::TPUT_ASYNC(dstG, srcG, session);
    }
    (void)lastEvent.Wait(session);  // 一次 Wait 等待所有 pending 操作
}
```

### 聚合Batch传输

```cpp
template <typename GT>
AICORE void AggregatePutSdma(
    GT& dst0, GT& src0, GT& dst1, GT& src1,
    const comm::AsyncSession& session)
{
    (void)comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(
        dst0, src0, session, comm::AsyncPutMode::DEFER);
    (void)comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(
        dst1, src1, session, comm::AsyncPutMode::DEFER);

    auto batchEvent =
        comm::SubmitAsyncPutBatch<comm::DmaEngine::SDMA>(session);
    if (!batchEvent.valid() || !batchEvent.Wait(session)) {
        return;
    }
}
```

Defer返回的无效Event只是占位值，不是成功状态。暂存前必须保证参数和Batch容量合法；只有有效的
Submit Event可用于完成判断。

A5 URMA显式peer形式保持原有peer位置，并要求每次Defer和Submit使用相同的`peer`与`jettyIndex`：

```cpp
(void)comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    dst0, src0, session, peer, comm::AsyncPutMode::DEFER, jettyIndex);
(void)comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    dst1, src1, session, peer, comm::AsyncPutMode::DEFER, jettyIndex);
auto batchEvent = comm::SubmitAsyncPutBatch<comm::DmaEngine::URMA>(
    session, peer, jettyIndex);
```

### URMA示例（NPU_ARCH 3510）

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

template <typename T>
__global__ AICORE void SimplePutUrma(__gm__ T *remoteDst, __gm__ T *localSrc,
                                     __gm__ uint8_t *urmaWorkspace, uint32_t destRankId)
{
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;

    ShapeDyn shape(1, 1, 1, 1, 1024);
    StrideDyn stride(1024, 1024, 1024, 1024, 1);
    GT dstG(remoteDst, shape, stride);
    GT srcG(localSrc, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(urmaWorkspace, destRankId, session)) {
        return;
    }

    auto event = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(dstG, srcG, session);
    (void)event.Wait(session);
}
```
