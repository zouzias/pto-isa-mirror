# TPUT_ASYNC_SUBMIT

## 简介

`TPUT_ASYNC_SUBMIT`发布当前`AsyncSession`中由[`TPUT_ASYNC_DEFER`](TPUT_ASYNC_DEFER_zh.md)准备的全部远程写。它推进硬件可见Producer，按照后端要求敲Send Doorbell，并返回一个覆盖整个Batch的`AsyncEvent`。

调用关系：

`TPUT_ASYNC_DEFER × N（暂存）` → `TPUT_ASYNC_SUBMIT（发布）` → `Wait/Test（完成）`

Submit返回表示Batch已经交给硬件执行，不表示传输已经完成。调用方必须使用返回的Event判断完成。

## 模板参数

`engine`在编译期选择DMA后端：

| 引擎 | 平台限制 | 说明 |
|---|---|---|
| `DmaEngine::SDMA`（默认） | A2/A3 | 通过一个SDMA Channel Group提交Batch。 |
| `DmaEngine::URMA` | Ascend 950PR/Ascend 950DT，仅NPU_ARCH 3510 | 通过一条Jetty提交Batch。 |

当前仅支持标准`TPUT_ASYNC`数据传输，且不支持A5 `DmaEngine::SDMA`。

## C++内建接口

声明于`include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <DmaEngine engine = DmaEngine::SDMA>
PTO_INST AsyncEvent TPUT_ASYNC_SUBMIT(
    const AsyncSession& session,
    uint32_t peer = UINT32_MAX,
    uint32_t jettyIndex = 0U);
```

## 参数说明

| 参数 | 说明 |
|---|---|
| `session` | 保存当前非空Batch的`AsyncSession`，必须与此前Defer使用的Session相同。 |
| `peer` | URMA必填，用于指定目标Rank，且必须与本Batch所有Defer一致；SDMA忽略此参数。省略时默认值为`UINT32_MAX`。 |
| `jettyIndex` | 当前AIV根据URMA Workspace配置可用的Jetty范围内，从0开始的索引，必须与本Batch所有Defer一致；默认值为0，SDMA忽略此参数。 |

返回值为最终`AsyncEvent`。有效Event覆盖该Batch中的全部非零长度远程写，不为单次远程写提供
独立完成状态。调用Wait/Test前必须先检查`event.valid()`；无效Event不得用于判断Batch成功完成。

## Batch提交语义

Submit要求当前Batch至少包含一次成功的非零长度Defer。

- **SDMA**：推进参与队列的Producer，敲所需的Send Doorbell，并为整个Batch返回一个Event。
- **URMA**：推进`jettyIndex`所选Jetty的Producer，敲该Jetty的Send Doorbell，并为整个Batch返回
  一个Event。

Submit返回有效Event后，Session可以开始构造下一Batch，但这不表示上一Batch已经完成。

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

- 空Batch不能Submit。
- Submit必须与本Batch所有Defer使用同一个Session和`engine`。
- URMA必须与本Batch所有Defer使用相同的`peer`和`jettyIndex`。
- Defer或Submit在发布前检测到参数、资源或容量错误时，当前Batch的暂存描述和Batch状态会被丢弃，
  此前暂存的内容不会发布；Submit的失败路径返回无效Event。Batch不会自动部分提交。
- 当前AIV只有一条可用Jetty时，仅支持`jettyIndex == 0`；存在多条可用Jetty时，可以选择其可用
  范围内的任一索引。
- `peer`必须小于URMA Workspace中的Rank数，且`jettyIndex < session.qpCount`。
- Event完成前，所有源范围必须保持有效且内容不变。
- Event完成前，不得释放或重建Session、Workspace、SDMA scratch tile、SDMA Channel Group或
  URMA Jetty。
- 重建或销毁Session及其通信资源前，必须确认该Session使用的每个活动物理队列上的最后一个Event
  均已完成。同一物理队列的后续Event覆盖此前提交，但一个Peer或Jetty上的Event不覆盖其他独立队列。
- 释放URMA通信资源前，必须完成每个活动Peer和Jetty的最后一个Event，并同步使用该通信上下文的Host
  Stream。
- Tensor、地址范围、SDMA `commBlockOffset`、SQ容量以及URMA注册内存和WQ/CQ容量必须满足
  [`TPUT_ASYNC_DEFER`](TPUT_ASYNC_DEFER_zh.md)中的约束。
- 不支持的Engine在编译期被拒绝。其他参数和通信资源必须满足对应`BuildAsyncSession`建立的要求。

## 完成语义

- `event.Wait(session)`阻塞等待Batch完成。
- `event.Test(session)`非阻塞检测Batch是否完成。
- 调用Wait/Test前必须确认`event.valid()`为`true`。
- Wait/Test必须使用提交该Batch的同一个Session。
- Event完成表示Batch中的全部Payload传输完成，源范围可以复用。
- Batch不是事务。接收端可能逐步观察到不同远程写，错误时也不会回滚已经完成的远端写。
- 公共语义不定义各次远程写之间的完成顺序。
- 最终Event是发送端完成对象，不会自动通知远端。远端消费数据时，调用方仍需建立
  `TNOTIFY`/`TWAIT`等协议，并按照平台规则处理接收端Cache可见性。

`Wait`只有返回`true`才表示Batch已完成。返回`false`表示尚未确认完成，且不会取消已经提交的传输；
此时不得复用源数据，也不得释放或重建Session、Workspace及通信资源。

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
- Submit后可由同一所有者串行构造下一Batch，但Session中的资源容量和提交顺序仍由调用方负责。

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

    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::SDMA>(
        dst0, src0, session);
    comm::TPUT_ASYNC_DEFER<comm::DmaEngine::SDMA>(
        dst1, src1, session);

    comm::AsyncEvent event =
        comm::TPUT_ASYNC_SUBMIT<comm::DmaEngine::SDMA>(session);
    if (!event.valid()) {
        return;
    }
    if (!event.Wait(session)) {
        return;
    }
    status[0] = 1U;
}
```

SDMA忽略`peer`和`jettyIndex`，可省略`peer`；URMA必须显式传入合法的`peer`。

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
        comm::TPUT_ASYNC_SUBMIT<comm::DmaEngine::URMA>(
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

`localSrc`和`remoteDst`必须位于URMA注册的通信缓冲区内。示例选择相对索引0；存在多条可用Jetty时，可以选择可用范围内的任一索引，但同一Batch中的所有Defer和Submit必须使用相同索引。

`Wait`返回`true`后，示例才将`status[0]`置为1。返回`false`时状态保持为0并退出，但这不表示传输已取消；Host应停止继续使用相关通信资源，并按运行环境的失败处理流程清理通信上下文。
