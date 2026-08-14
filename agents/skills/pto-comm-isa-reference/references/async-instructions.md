# 异步通信指令详解（TPUT_ASYNC / TGET_ASYNC / BuildAsyncSession）

## TPUT_ASYNC — 异步远程写

启动 GM→GM DMA 传输，立即返回 `AsyncEvent`。

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
AsyncEvent TPUT_ASYNC(GlobalDstData &dst, GlobalSrcData &src,
                      const AsyncSession &session, WaitEvents&... events);

// A5 显式 peer 重载（URMA/RDMA 使用；SDMA 忽略）
AsyncEvent TPUT_ASYNC(GlobalDstData &dst, GlobalSrcData &src,
                      const AsyncSession &session, uint32_t peer,
                      WaitEvents&... events);
```

## TGET_ASYNC — 异步远程读

启动远端 GM→本地 GM DMA 传输。

```cpp
template <DmaEngine engine = DmaEngine::SDMA,
          typename GlobalDstData, typename GlobalSrcData, typename... WaitEvents>
AsyncEvent TGET_ASYNC(GlobalDstData &dst, GlobalSrcData &src,
                      const AsyncSession &session, WaitEvents&... events);

// A5 显式 peer 重载（URMA/RDMA 使用；SDMA 忽略）
AsyncEvent TGET_ASYNC(GlobalDstData &dst, GlobalSrcData &src,
                      const AsyncSession &session, uint32_t peer,
                      WaitEvents&... events);
```

---

## BuildAsyncSession — 构建异步会话

### SDMA 构建（默认）

```cpp
template <DmaEngine engine = DmaEngine::SDMA, typename ScratchTile>
bool BuildAsyncSession(ScratchTile &scratchTile, __gm__ uint8_t *workspace,
                       AsyncSession &session,
                       uint32_t syncId = 0,
                       const sdma::SdmaBaseConfig &baseConfig = {sdma::kDefaultSdmaBlockBytes, 0, 1},
                       uint32_t channelGroupIdx = sdma::kAutoChannelGroupIdx);
```

| 参数 | 说明 |
|------|------|
| `scratchTile` | 用于 SDMA 控制元数据的 UB scratch tile（非数据负载），推荐 `Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>`（256B） |
| `workspace` | 由 Host 侧 `SdmaWorkspaceManager` 分配的 GM 指针 |
| `syncId` | MTE3/MTE2 管道同步事件 ID（0-7），避免与 kernel 内其他管道屏障冲突 |
| `baseConfig` | `{block_bytes, comm_block_offset, queue_num}`，默认适用于单队列场景 |
| `channelGroupIdx` | SDMA 通道组索引，默认使用 `get_block_idx()` 映射 |

### URMA 构建（仅 Ascend950 / NPU_ARCH 3510）

```cpp
bool BuildAsyncSession(__gm__ uint8_t *workspace, uint32_t destRankId, AsyncSession &session);
```

### RDMA 构建（HNS1825，仅 Ascend950 / NPU_ARCH 3510）

```cpp
template <DmaEngine engine, typename ScratchTile>
bool BuildAsyncSession(ScratchTile &scratchTile, __gm__ uint8_t *workspace,
                       uint32_t myPe,
                       AsyncSession &session, uint32_t syncId = 0);

template <DmaEngine engine, typename ScratchTile>
bool BuildAsyncSession(ScratchTile &scratchTile, __gm__ uint8_t *workspace,
                       uint32_t destRankId, uint32_t myPe,
                       AsyncSession &session, uint32_t syncId = 0);
```

`workspace` 由 Host 侧 `rdma::RdmaWorkspaceManager` 创建，`myPe` 为本地 rank。第一个重载配合
`TPUT_ASYNC/TGET_ASYNC(..., session, peer)` 跨 peer 复用；第二个重载通过 `destRankId` 绑定远端 rank，
兼容原有调用。HNS1825 要求 UB/Vec `scratchTile` 至少 64B，`syncId` 范围为 `[0, 7]`。
Host 必须按 `Preflight → Init → Kernel → Finalize` 管理生命周期。

---

## 异步约束

- **仅支持扁平连续的逻辑一维 tensor**（非一维返回无效 event）
- SDMA workspace 必须由 Host 侧 `SdmaWorkspaceManager` 分配
- URMA workspace 必须由 Host 侧 `UrmaWorkspaceManager` 分配
- URMA 需要大页内存（`ACL_MEM_MALLOC_HUGE_ONLY`），小页分配导致注册失败
- `scratchTile` 仅用于控制元数据，不是数据暂存缓冲
- RDMA 必须在 CMake 配置前通过 `PTO_RDMA_BACKEND=HNS_1825` 使能，当前仅支持 Ascend950 /
  NPU_ARCH 3510
- RDMA 的本地和远端完整传输范围都必须位于 `RdmaWorkspaceManager::Init` 注册的通信缓冲区
- HNS1825 单次传输最多 `0x7fffffff` 字节，scratch 至少 64B

---

## 完成语义（Quiet 语义）

- `event.Wait(session)` 阻塞直到该 event 对应的提交范围完成
- SDMA 同一 session 连续提交时，只需等待最后一个 event；URMA/RDMA 只覆盖同一 peer/QP，不同 peer/QP
  必须分别等待
- 类似 shmem 的 quiet 语义

---

## 完整示例

```cpp
// 构建会话
using ScratchTile = Tile<TileType::Vec, uint8_t, 1, comm::sdma::UB_ALIGN_SIZE>;
ScratchTile scratchTile;
TASSIGN(scratchTile, 0x0);

comm::AsyncSession session;
if (!comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, sdmaWorkspace, session)) {
    return;
}

// 批量传输 + 一次 Wait
comm::AsyncEvent lastEvent;
for (int rank = 0; rank < nranks; ++rank) {
    GT dstG(remoteDst + rank * size, shape, stride);
    lastEvent = comm::TPUT_ASYNC(dstG, srcG, session);
}
(void)lastEvent.Wait(session);  // 等待所有 pending 操作完成
```
