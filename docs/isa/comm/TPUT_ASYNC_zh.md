# TPUT_ASYNC

## 简介

`TPUT_ASYNC`是异步远程写原语。默认会发布一次从本地GM到远端GM的传输并返回对应
`AsyncEvent`。A2/A3 SDMA和A5 URMA也可以通过`AsyncSession::submitMode`暂存多次远程写并聚合
提交，无需改变`TPUT_ASYNC`函数签名。

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
                               WaitEvents &... events);

// A5，peer来自session.destRankId
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session,
                               WaitEvents &... events);

// A5，显式peer
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, uint32_t peer,
                               WaitEvents &... events);
```

调用签名以及`peer`位置保持不变。调用方在`BuildAsyncSession`后配置Session中的聚合参数；
前置`WaitEvents...`继续使用原有调用形式。

## 提交模式

```cpp
session.submitMode = AsyncSubmitMode::DEFER;
session.batchSize = 16U;
```

- `AsyncSubmitMode::IMMEDIATE`（默认）：先提交相同Session、相同引擎中的pending Batch，再提交
  当前远程写；返回Event保持标准单次操作范围。若需确认前序逻辑Batch完成，必须另行等待最后一个
  Defer Event。
- `AsyncSubmitMode::DEFER`：暂存当前远程写，返回非零的未来完成目标快照；只有包含该目标的物理
  Batch提交后，才能对其调用`Wait/Test`。
- `AsyncSubmitMode::DEFER_AND_SUBMIT`：暂存当前远程写并提交剩余物理Batch；返回Event可以立即
  用于`Wait/Test`。

`batchSize`表示一个物理Batch最多包含多少次非空Defer调用。达到阈值后自动提交。
`batchSize == UINT32_MAX`等价于关闭阈值自动提交，此时应把最后一次真实远程写切换为
`DEFER_AND_SUBMIT`，或随后执行相同引擎的立即操作。`batchSize == 0`表示关闭Batch，使用
`DEFER`或`DEFER_AND_SUBMIT`属于调用契约错误。

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
- A2/A3接口不包含`peer`；A5保留绑定peer与显式peer两种形式。
- 首次成功的非零Defer启动Batch；零长度Defer为no-op。
- 同一逻辑Batch中的每次Defer必须使用相同Session和Engine；一个URMA逻辑Batch还必须保持相同
  `peer`。
- 使用相同Session的同引擎立即`TPUT_ASYNC`、`TGET_ASYNC`或`TPUT_ASYNC_NOTIFY`会先提交pending
  Put描述符；`TPREFETCH_ASYNC`仅在复用该外部SDMA Session时执行此行为。
- 该隐式提交只建立本端发布顺序。A5 URMA在一条Jetty上发布的Notify不是其他Jetty上Defer写的远端
  完成栅栏；若要把signal作为全部Batch Payload远端可见的依据，必须先等待Batch Event或使用明确
  的顺序协议。
- A5 `TPUT_ASYNC<SDMA>`使用同步MTE fallback，不会提交或修改pending URMA Batch。
- Session存在pending工作时，不得复制、重建、销毁Session或转移其所有权。
- 所有源数据范围、Session和Workspace必须保持有效，直至对应Event完成。
- Defer在发布前检测到参数、资源或容量错误时，会丢弃当前尚未提交的物理Batch，不会将其发布。
- 自动提交物理Batch后会清零staged计数，但该物理Batch仍属于逻辑完成前缀；后续Defer会开始新的
  物理Batch。
- 同一Batch中不同远端目的范围不得重叠，不得利用Defer顺序表达写间依赖。
- 物理Batch提交前，不得对Defer Event调用`Wait/Test`，也不得把它作为前置Event传入。本版本不
  增加运行时提交状态检查，由调用方保证该约束。

### Defer聚合Batch资源限制

- 源、目的Tensor必须具有相同RawDType和Layout，必须都是扁平连续的逻辑一维Tensor；每次非零传输
  的源、目的指针均不得为空。
- Tensor元素数、字节数和地址范围计算必须能用`uint64_t`表示；目的Tensor元素容量不得小于源
  Tensor元素数。
- 接口不校验通信内存归属、地址边界、范围重叠或目标Peer所有权，调用方必须保证这些条件。
- 每次SDMA Defer消耗`ceil(transferBytes / blockBytes)`个数据SQE，并基于当前物理Batch累计SQE索引
  在`queueNum`个队列间轮转分配；每个队列分配到的数量必须小于该队列SQ深度。
- SDMA会在源、目的Tensor基地址上增加`commBlockOffset`。底层分配必须无溢出地覆盖
  `[base + commBlockOffset, base + commBlockOffset + transferBytes)`。
- URMA要求有效的注册Workspace和Peer内存注册；`peer`必须小于Workspace Rank数。
- 完整URMA源范围必须位于`UrmaWorkspaceManager::Init()`注册的本地通信缓冲区内，完整目的范围
  必须位于所选Peer的已注册通信缓冲区内。
- URMA将一次远程写拆分为每个最大256 MiB的WQE，并确定性轮询分配到`session.qpCount`条Jetty。
  每个物理Batch必须适配全部参与Jetty的WQ和CQ深度。

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

- 对`IMMEDIATE`调用返回的Event执行`event.Wait(session)`，会阻塞到本次操作及其完成目标覆盖的
  同Session先前操作均已完成。

这意味着连续执行多次`IMMEDIATE TPUT_ASYNC`后，只需等待最后一个返回的`AsyncEvent`，即可等待其
覆盖的pending操作完成（类似shmem的quiet语义）。`DEFER` Event则采用下文所述的前缀快照语义。

RDMA操作涉及不同peer时，必须分别等待每个peer的最后一个Event。

同一Session最多可有64个未完成操作，超过后提交可能产生背压。

wait成功后，所有已发出的 `dstGlobalData` 写入均已全部完成。

对于`DEFER`模式，每次非空调用都会返回未来完成目标快照。包含该快照的物理Batch提交前，不支持
调用`Wait/Test`；提交后，中间Event可检查对应Defer前缀，最后一次调用返回的Event覆盖完整逻辑
前缀。Wait/Test成功后达到发送端完成条件，并允许复用对应源数据。

聚合Batch不是事务。接收端可能逐步观察到不同写，错误发生后已完成的写不会回滚，公共契约也不
定义各写之间的完成顺序。最终Event不会自动通知接收端；接收端消费仍需使用`TNOTIFY/TWAIT`
等应用协议，并执行平台要求的Cache可见性处理。

只有Wait/Test成功才能确认完成。失败结果不会取消已提交传输；此时不得复用源数据，也不得释放或
重建Session、Workspace、Scratch Tile、Channel Group或Jetty。

## SDMA并发与Session所有权

- 同一个Session不能被多个执行流并发使用。
- 共用同一Channel Group的操作必须共享同一个Session。
- 并发Kernel或Kernel内多个独立Session必须使用隔离的Channel Group。
- 重新构建Session或复用Channel Group前，必须先完成此前所有Event。

### Defer聚合Batch并发与Session所有权

- 同一Session上的Build/Rebuild、指令调用、Wait和Test必须串行执行。
- 不同Session不代表物理资源不同。从Build到最后一个Event完成期间，不得将另一Session映射到相同
  SDMA Channel、URMA WQ或URMA CQ。
- URMA `PER_PEER`模式下，同一Peer共享同一物理WQ/CQ，调用必须串行；不同Peer使用不同队列。
- URMA `SHARED_POOL`模式下，Batch Session拥有
  `[session.qpIdxBase, session.qpIdxBase + session.qpCount)`范围内的Jetty。即使Peer不同，只要
  范围重叠，Session仍会别名。
- 同一物理Channel Group、WQ或CQ上的Wait/Test不得与Defer、提交或另一个Wait/Test并发执行。
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
    comm::AsyncSession& session)
{
    session.batchSize = UINT32_MAX;
    session.submitMode = comm::AsyncSubmitMode::DEFER;
    auto firstEvent =
        comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(dst0, src0, session);
    session.submitMode = comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
    auto batchEvent =
        comm::TPUT_ASYNC<comm::DmaEngine::SDMA>(dst1, src1, session);
    if (!batchEvent.valid() || !batchEvent.Wait(session)) {
        return;
    }
}
```

`firstEvent`是有效的未来快照，但第二次调用提交物理Batch前不得等待它。提交后两个Event都可以
等待；`batchEvent`覆盖两次写。

A5 URMA显式peer形式保持原有peer位置，并要求完整逻辑Batch使用相同`peer`。WQE会轮询分配到
Session拥有的Jetty范围：

```cpp
session.batchSize = UINT32_MAX;
session.submitMode = comm::AsyncSubmitMode::DEFER;
auto firstEvent = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    dst0, src0, session, peer);
session.submitMode = comm::AsyncSubmitMode::DEFER_AND_SUBMIT;
auto batchEvent = comm::TPUT_ASYNC<comm::DmaEngine::URMA>(
    dst1, src1, session, peer);
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
