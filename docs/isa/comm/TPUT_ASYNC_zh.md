# pto.tput_async

## 概要

`pto.tput_async` 发起一次异步远程写，并立即返回一个 `AsyncEvent` 句柄。

## 语义

已核实的 public API 通过 `AsyncSession` 发起异步操作，并返回 `AsyncEvent`。

提交并不意味着完成；调用方必须后续通过 `event.Wait(session)` 或 `event.Test(session)` 检查完成状态。

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`。

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
PTO_INST AsyncEvent TPUT_ASYNC(GlobalDstData &dstGlobalData, GlobalSrcData &srcGlobalData,
                               const AsyncSession &session, WaitEvents &... events);
```

## Session 构建

已核实的 helper API 提供了 `BuildAsyncSession` 重载：

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
PTO_INTERNAL bool BuildAsyncSession(ScratchTile &scratchTile, __gm__ uint8_t *workspace,
                                    AsyncSession &session, uint32_t syncId = 0,
                                    const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                                    uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

```cpp
#ifdef PTO_URMA_SUPPORTED
template <DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace, uint32_t destRankId,
                                    AsyncSession &session);
#endif
```

## 约束

!!! warning "约束"
    - 提交前必须先构建 `AsyncSession`。
    - 传入 wrapper 的事件 token 会在异步提交开始前被先行等待。
    - 完成状态必须通过返回的 `AsyncEvent` 和同一逻辑 session 家族进行检查。
    - 引擎特定的合法性约束依赖具体实现；public wrapper 本身不重复陈述所有后端传输限制。

## `AsyncEvent` 完成接口

已核实的 public 完成接口包括：

```cpp
bool AsyncEvent::Wait(const AsyncSession &session) const;
bool AsyncEvent::Test(const AsyncSession &session) const;
```

在已核实实现中，若 `handle == 0`，这两个方法会立即返回成功。

## 示例

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void example_put_async(auto &dstG, auto &srcG, const comm::AsyncSession &session) {
    auto event = comm::TPUT_ASYNC(dstG, srcG, session);
    (void)event.Wait(session);
}
```
