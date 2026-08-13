# 多 Block 调度与地址管理

## 远端地址获取

通信算子需要知道远端 NPU 的 GM 地址。常用方式：

### 方式 1：通过 HCCL 通信窗口

```cpp
// Device 侧：计算远端地址
inline __gm__ half *GetRemotePtr(__gm__ DeviceContext *ctx, __gm__ half *local_ptr, int remote_rank)
{
    ptrdiff_t offset = reinterpret_cast<__gm__ uint8_t *>(local_ptr) -
                       reinterpret_cast<__gm__ uint8_t *>(ctx->windowsIn[ctx->myRank]);
    return reinterpret_cast<__gm__ half *>(
        reinterpret_cast<__gm__ uint8_t *>(ctx->windowsIn[remote_rank]) + offset);
}
```

### 方式 2：通过 ParallelGroup（集合通信）

```cpp
GlobalData tensors[NRANKS];
for (int r = 0; r < nranks; ++r) {
    tensors[r] = GlobalData(remote_addrs[r]);
}
comm::ParallelGroup<GlobalData> group(tensors, nranks, my_rank);
```

### 方式 3：通过 RDMA 注册 MR（仅 A5）

Host 先用 `RdmaWorkspaceManager` 注册每个 rank 的通信缓冲区并交换 Device
虚拟地址。Device 侧从 workspace 读取 peer MR base，再叠加应用定义的逻辑
offset；不要求各 rank 的通信缓冲区具有相同虚拟地址：

```cpp
uint64_t peerBase = comm::rdma::PeerMrBaseAddr(rdmaWorkspace, peerRank);
if (peerBase == 0) {
    return;
}
__gm__ T *remotePtr =
    reinterpret_cast<__gm__ T *>(peerBase + remoteOffsetBytes);
```

`remoteOffsetBytes` 及完整远端访问范围必须落在 peer 注册 MR 内；本地操作数
范围也必须落在本地注册 MR 内。当前该路径要求构建阶段选择
`PTO_RDMA_BACKEND=HNS_1825`。

### 地址对齐要求

- 所有 GM 地址必须满足 32 字节对齐
- Signal 地址必须 4 字节对齐
- TPUT_ASYNC/TGET_ASYNC 的 workspace 由专用 Manager 管理，无需额外对齐
- RDMA 的本地和远端完整 payload 范围都必须位于对应的已注册 MR 内

---

## Block 分配策略

```cpp
int block_idx = get_block_idx();
CommKernel<<<COMM_BLOCK_NUM, nullptr, stream>>>(..., COMM_BLOCK_NUM);
```

### Row-level 均分（推荐）

消除 ±1 不均衡：

```cpp
int total_rows = tile_count * ROWS_PER_TILE * (nranks - 1);
int rows_per_block = (total_rows + num_comm_blocks - 1) / num_comm_blocks;
int row_start = block_idx * rows_per_block;
int row_end = min((block_idx + 1) * rows_per_block, total_rows);

while (cur_row < row_end) {
    // 从 flat row index 恢复 (tile, rank, row_in_tile)
}
```

### Tile-level 均分

```cpp
int tiles_per_block = (total_tiles + num_blocks - 1) / num_blocks;
int my_start = block_idx * tiles_per_block;
int my_end = min(my_start + tiles_per_block, total_tiles);
```

### Block 角色分化

在 barrier 等场景中，block 0 承担特殊角色：

```cpp
if (block_idx == 0) {
    // block 0：执行跨 rank 信号通知/等待，完成后设置本地广播标志
} else {
    // 其他 block：等待本地广播标志
}
```

---

## Tiling 策略

### UB Tile 大小计算

```cpp
static constexpr size_t TILE_UB_BYTES = ((BASE_M * BASE_N * sizeof(half) + 1023) / 1024) * 1024;
```

### 维度对齐

```cpp
static constexpr uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
static constexpr uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

static constexpr uint32_t G_M = AlignUp(ORIG_M, BASE_M);
static constexpr uint32_t G_N = AlignUp(ORIG_N, BASE_N);
static constexpr uint32_t NUM_TILES = (G_M / BASE_M) * (G_N / BASE_N);
```

### UB Buffer 规划

```
UB 布局（乒乓模式）：
┌──────────────┬──────────────┐
│  pingTile    │  pongTile    │
│  0x0         │  TILE_UB     │
│  BASE_M ×    │  BASE_M ×    │
│  BASE_N ×    │  BASE_N ×    │
│  sizeof(T)   │  sizeof(T)   │
└──────────────┴──────────────┘
```

对齐要求：
- Tile 行对齐到 32 字节（RowMajor 时 cols × sizeof(T) 需 32B 对齐）
- Tile 间间隔至少 TILE_UB_BYTES（避免重叠）
