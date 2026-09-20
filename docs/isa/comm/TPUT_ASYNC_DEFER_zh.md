# TPUT_ASYNC_DEFER

## 简介

`TPUT_ASYNC_DEFER`准备一次从本地GM到远端GM的异步写。它把一个或多个传输描述写入后端发送队列，但不推进硬件可见Producer，也不敲Send Doorbell，因此该传输任务尚未发布给硬件。一次或多次成功的非零长度`TPUT_ASYNC_DEFER`组成当前Batch。随后调用[`SubmitAsyncPutBatch`](SUBMIT_ASYNC_PUT_BATCH_zh.md)统一推进相应Producer并敲Send Doorbell，把Batch发布给硬件异步执行，同时返回一个最终Event。`TPUT_ASYNC_DEFER`本身不返回Event。

数据方向：

`srcGlobalData（本地GM）` → DMA引擎 → `dstGlobalData（远端GM）`

## 模板参数

`engine`在编译期选择DMA后端：

| 引擎 | 平台限制 | 说明 |
|---|---|---|
| `DmaEngine::SDMA`（默认） | A2/A3 | 使用SDMA Channel Group。 |
| `DmaEngine::URMA` | Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510 | 使用由`peer`和`jettyIndex`选择的一条Jetty。 |

当前仅支持标准`TPUT_ASYNC`数据传输，且不支持A5 `DmaEngine::SDMA`。

## C++内建接口

声明于`include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <
    DmaEngine engine = DmaEngine::SDMA,
    typename GlobalDstData,
    typename GlobalSrcData>
PTO_INST void TPUT_ASYNC_DEFER(
    GlobalDstData& dstGlobalData,
    GlobalSrcData& srcGlobalData,
    const AsyncSession& session,
    uint32_t peer,
    uint32_t jettyIndex = 0U);
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `dstGlobalData` | Payload在远端GM中的目的GlobalTensor。 |
| `srcGlobalData` | Payload在本地GM中的源GlobalTensor；传输大小由其Shape和元素类型决定。 |
| `session` | 由`BuildAsyncSession<engine>()`成功构建的会话，同时保存当前Batch状态。 |
| `peer` | URMA目标rank；SDMA不使用此参数，远端地址由`dstGlobalData`确定。 |
| `jettyIndex` | 当前AIV根据URMA Workspace配置可用的Jetty范围内，从0开始的索引；默认值为0，SDMA忽略此参数。 |

该接口返回`void`，不会为单次远程写创建`AsyncEvent`，也不接受前置`WaitEvents`。

## 远程写语义

一次非零Defer向Batch加入一次远程写：

1. 根据Session配置校验Session、Tensor、非空指针和Batch容量。
2. 把TPUT传输描述写入发送队列。
3. 不推进硬件可见Producer，也不敲Send Doorbell。

零长度源Tensor是no-op，不创建Batch。

Defer不保证立即返回。接口可能等待准备传输任务所需的资源可用，但仍不会敲Send Doorbell或向硬件发布
当前Batch。

## AsyncSession构建

使用与引擎对应的`BuildAsyncSession`重载。

### SDMA构建（默认）

```cpp
comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
    scratchTile, workspace, session, syncId, baseConfig, channelGroupIdx);
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `scratchTile` | — | SDMA控制数据使用的UB scratch tile。 |
| `workspace` | — | 由`SdmaWorkspaceManager`初始化的GM Workspace。 |
| `session` | — | 输出的`AsyncSession`。 |
| `syncId` | `0` | 流水同步事件ID。 |
| `baseConfig` | 默认SDMA配置 | 块大小、通信偏移和队列数。 |
| `channelGroupIdx` | 自动选择 | 选择Session引用的SDMA Channel Group。 |

`scratchTile`是SDMA控制数据使用的临时UB空间，可用空间至少为`sizeof(uint64_t)`（8字节），推荐使用`Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256字节）。在该Session关联的全部Event完成前，`scratchTile`必须保持有效。`workspace`由`SdmaWorkspaceManager`初始化。

### URMA构建（仅NPU_ARCH 3510）

```cpp
comm::BuildAsyncSession<comm::DmaEngine::URMA>(
    workspace, session);
```

| 参数 | 说明 |
|---|---|
| `workspace` | 由`UrmaWorkspaceManager`初始化的GM Workspace。 |
| `session` | 输出的`AsyncSession`。 |

URMA不需要scratch tile，Workspace由`UrmaWorkspaceManager`初始化。只有构建成功的Session才能用于
Defer、Submit和Event完成检查。

## 约束

- 第一次成功的非零Defer开始一个Batch；后续Defer继续向该Batch追加远程写。
- 同一Batch的所有Defer和最终Submit必须使用同一个Session及相同的`engine`。
- URMA同一Batch必须始终使用相同的`peer`和`jettyIndex`。
- 非零Defer后必须调用`SubmitAsyncPutBatch`；接口不提供主动取消Batch的操作。
- Defer或Submit在发布前检测到参数、资源或容量错误时，当前Batch的暂存描述和Batch状态会被丢弃，
  此前暂存的内容不会发布。Defer返回`void`，不直接报告该失败；若执行继续且此后没有新的成功Defer，
  Submit的失败路径返回无效Event。
- Defer到Submit之间，不能在同一Channel Group或Jetty上插入普通`TPUT_ASYNC`、
  `TGET_ASYNC`、`TPUT_ASYNC_NOTIFY`、另一Batch或`TPREFETCH_ASYNC`。
- 第一次成功的非零Defer到Submit之间，不得复制、重建、销毁Session或转移其所有权。
- 不同远程写的目的范围不得重叠，也不能依赖Defer调用顺序表达写入之间的数据依赖。
- 每个源地址及其内容必须保持有效且不变，直到Submit返回的最终Event完成。

### Tensor与资源要求

- 源和目的Tensor的`RawDType`必须相同。
- 源和目的Tensor的Layout必须相同。
- 源和目的Tensor必须是flat contiguous 1D。
- 目的Tensor的元素容量不得小于源Tensor。
- 非零传输的源和目的指针必须有效，总元素数和总字节数不得溢出`uint64_t`。
- 接口不会验证地址是否属于通信内存、是否越界、是否重叠或是否属于目标Peer；这些条件由调用方保证。
- SDMA workspace和Channel Group必须由有效的`SdmaWorkspaceManager`环境提供。
- SDMA每次Defer占用`ceil(传输字节数 / blockBytes)`个数据SQE。分配按整个Batch的累计SQE序号
  在`queueNum`条队列间轮转，不会在每次Defer时从队列0重新开始；Batch分配到每条队列的数据SQE数
  必须小于该队列的SQ深度。
- SDMA会把`commBlockOffset`同时加到源和目的Tensor基址；底层内存必须覆盖
  `[基址 + commBlockOffset, 基址 + commBlockOffset + 传输字节数)`，并保证区间起止地址计算不溢出。
- URMA workspace、Peer内存注册和Jetty必须由有效的`UrmaWorkspaceManager`环境提供。
- URMA要求`peer`小于Workspace中的Rank数，且`jettyIndex < session.qpCount`。
- 当前AIV只有一条可用Jetty时，仅支持`jettyIndex == 0`；存在多条可用Jetty时，可以选择其可用
  范围内的任一索引。
- URMA的完整源范围必须位于本Rank传给`UrmaWorkspaceManager::Init()`的通信缓冲区内，完整目的范围
  必须位于所选Peer的通信缓冲区内。通信缓冲区必须使用当前CANN/HCCL支持注册的Device内存。
- URMA把一次远程写按最多256 MiB拆分为WQE；整个Batch的WQE总数必须同时不超过所选Jetty的
  WQ和CQ深度。
- Batch大小必须符合Session配置限制。更大的工作负载应拆分为多个Batch；接口不会自动提交部分Batch。
  此前尚未完成的任务可能使Defer等待资源，但当前Batch本身仍不得超过上述总容量限制。

## 完成语义

Defer不返回Event，也不表示传输完成。最终`AsyncEvent`只由`SubmitAsyncPutBatch`创建；该Event成功
完成后，才表示Batch中所有非零远程写完成。在此之前，所有源范围、Session、Workspace、SDMA
scratch tile和通信资源都必须保持有效。

公共语义不保证各次远程写按照Defer调用顺序完成。调用方不得依赖该顺序建立数据依赖，接收端也可能
逐步观察到Batch中的不同写入。

## 并发与Session所有权

- `AsyncSession`是单所有者、非线程安全对象，不得通过复制Session建立新的所有者。
- 同一Session上的Build/Rebuild、Defer、Submit、Wait和Test必须由一个执行流串行调用。
- 不同Session不代表物理资源相互独立。共享同一SDMA Workspace的A2/A3并发Session必须使用一致的
  `queueNum`配置，并选择不重叠的物理Channel范围；某Session使用的范围为
  `[channelGroupIdx * queueNum, channelGroupIdx * queueNum + queueNum)`。
- 从Build选中物理资源开始，到暂存Batch已Submit且该资源上的最后一个Event完成前，不得Build/Rebuild
  另一个映射到相同SDMA Channel、URMA WQ或URMA CQ的Session。
- URMA `PER_PEER`模式下，同一Peer的调用访问同一物理WQ/CQ，必须串行；不同Peer使用不同队列。
- URMA `SHARED_POOL`模式下，物理Jetty索引为`session.qpIdxBase + jettyIndex`。同一AIV构建的Session
  选择相同相对`jettyIndex`时会别名；并发调用必须选择不同物理Jetty，即使Peer不同，相同Jetty仍会冲突。
- 同一物理Channel Group、WQ或CQ上的Wait/Test不能与Defer、Submit或其他Wait/Test并发执行。
- 并发远端目的地址范围不得重叠。
- Submit后可由同一所有者串行构造下一Batch；只有该Session产生的全部未完成Event均已完成后，
  才能重建或销毁Session及其通信资源。

## 示例

### SDMA

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

__global__ AICORE void BatchPutSdma(
    __gm__ int32_t* remoteDst, __gm__ int32_t* localSrc,
    __gm__ uint8_t* sdmaWorkspace, __gm__ uint32_t* status)
{
    constexpr uint32_t kElementsPerWrite = 16U;
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<int32_t, ShapeDyn, StrideDyn, Layout::ND>;
    using ScratchTile =
        Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;

    status[0] = 0U;
    ShapeDyn shape(1, 1, 1, 1, kElementsPerWrite);
    StrideDyn stride(
        kElementsPerWrite, kElementsPerWrite, kElementsPerWrite,
        kElementsPerWrite, 1);
    GT dst0(remoteDst, shape, stride);
    GT src0(localSrc, shape, stride);
    GT dst1(remoteDst + kElementsPerWrite, shape, stride);
    GT src1(localSrc + kElementsPerWrite, shape, stride);

    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x0);
    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(
            scratchTile, sdmaWorkspace, session)) {
        return;
    }

    constexpr uint32_t kIgnoredPeer = 0U;
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::SDMA>(
        dst0, src0, session, kIgnoredPeer);
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::SDMA>(
        dst1, src1, session, kIgnoredPeer);

    comm::AsyncEvent event =
        comm::SubmitAsyncPutBatch<comm::DmaEngine::SDMA>(
            session, kIgnoredPeer);
    if (!event.valid()) {
        return;
    }
    if (!event.Wait(session)) {
        return;
    }
    status[0] = 1U;
}
```

SDMA为保持接口一致而接收`peer`和`jettyIndex`，但忽略二者。

### URMA

```cpp
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/pto_tile.hpp>

using namespace pto;

__global__ AICORE void BatchPutUrma(
    __gm__ int32_t* remoteDst, __gm__ int32_t* localSrc,
    __gm__ uint8_t* urmaWorkspace, __gm__ uint32_t* status,
    uint32_t peer)
{
    constexpr uint32_t kElementsPerWrite = 16U;
    using ShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using StrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
    using GT = GlobalTensor<int32_t, ShapeDyn, StrideDyn, Layout::ND>;

    status[0] = 0U;
    ShapeDyn shape(1, 1, 1, 1, kElementsPerWrite);
    StrideDyn stride(
        kElementsPerWrite, kElementsPerWrite, kElementsPerWrite,
        kElementsPerWrite, 1);
    GT dst0(remoteDst, shape, stride);
    GT src0(localSrc, shape, stride);
    GT dst1(remoteDst + kElementsPerWrite, shape, stride);
    GT src1(localSrc + kElementsPerWrite, shape, stride);

    comm::AsyncSession session;
    if (!comm::BuildAsyncSession<comm::DmaEngine::URMA>(
            urmaWorkspace, session)) {
        return;
    }

    constexpr uint32_t kJettyIndex = 0U;
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::URMA>(
        dst0, src0, session, peer, kJettyIndex);
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::URMA>(
        dst1, src1, session, peer, kJettyIndex);

    comm::AsyncEvent event =
        comm::SubmitAsyncPutBatch<comm::DmaEngine::URMA>(
            session, peer, kJettyIndex);
    if (!event.valid()) {
        return;
    }
    if (!event.Wait(session)) {
        return;
    }
    status[0] = 1U;
}
```

`localSrc`和`remoteDst`必须位于URMA注册的通信缓冲区内。示例选择相对索引0；存在多条可用Jetty时，可以选择可用范围内的任一索引，但同一Batch中的所有Defer和最终Submit必须使用相同索引。

`Wait`返回`true`后，示例才将`status[0]`置为1。返回`false`时状态保持为0并退出，但这不表示传输已取消；Host应停止继续使用相关通信资源，并按运行环境的失败处理流程清理通信上下文。
