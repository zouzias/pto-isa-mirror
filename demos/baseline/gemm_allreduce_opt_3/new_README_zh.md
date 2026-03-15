# GEMM AllReduce 优化版 — ReduceScatter + AllGather 方案

本项目是 GEMM + AllReduce 融合算子的优化实现。相比原方案（TPUT\<AtomicAdd\> 广播），本方案采用 **ReduceScatter + AllGather** 通信算法，配合增大 K 维度和减少通信 block 数量，将 **Time Saved 从 3.6% 提升至 20.2%**。

## 优化概述

| 项目 | 原方案 | 优化后 |
|------|--------|--------|
| 通信算法 | TPUT\<AtomicAdd\> 广播到所有 rank | ReduceScatter + AllGather (plain TPUT) |
| K 维度 | 4096 | **16384** |
| COMM_BLOCK_NUM | 40 | **20** |
| 每 rank TPUT 次数 | 14,336 | **3,584** (降低 4×) |
| 每 rank 通信数据量 | 1.75 GB | **448 MB** (降低 ~4×) |
| 通信时间 | ~31,330 us | **~9,945 us** |
| Time Saved | 3.6% | **20.2%** |
| Speedup | 1.034× | **1.253×** |

## 数据并行模式

### 矩阵分割策略（K 维切分）

全局矩阵乘法：C\[M×N\] = A\[M×K\] × B\[K×N\]

```
默认参数: M=16384, K=16384, N=4096, 8 ranks

K 维按 rank 均分，每个 rank 持有:
  A_part [M × (K/r)] = [16384 × 2048]
  B_part [(K/r) × N] = [2048 × 4096]

每个 rank 独立计算:
  C_rank [M × N] = A_part × B_part = [16384 × 4096]

AllReduce 目标:
  C_final = C_rank_0 + C_rank_1 + ... + C_rank_7
```

### 数学正确性

```
A × B = [A_0 | A_1 | ... | A_r-1] × [B_0; B_1; ...; B_r-1]
      = A_0 × B_0 + A_1 × B_1 + ... + A_r-1 × B_r-1
```

K 维切分后各 rank 的部分积之和等于完整矩阵乘结果。

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Compute Stream (Cube, 32 blocks)          Comm Stream (Vec, 20 blocks)         │
│                                                                                  │
│  ┌──────────────────────┐                                                        │
│  │ Compute Tile → TSTORE│──┐                                                     │
│  │ pipe_barrier(PIPE_ALL)│  │                                                    │
│  │ Enqueue tile_idx      │  │ Ready Queue                                        │
│  └──────────────────────┘  │                                                     │
│  ┌──────────────────────┐  │         ┌──────────────────────────────────────┐    │
│  │ Compute Tile → TSTORE│──┼────────→│ Phase 1: ReduceScatter TPUT          │    │
│  └──────────────────────┘  │         │   按 tile 只发给 owner rank          │    │
│  ┌──────────────────────┐  │         │ Phase 2: Barrier                     │    │
│  │ Compute Tile → TSTORE│──┘         │ Phase 3: Local TREDUCE (owned tiles) │    │
│  └──────────────────────┘            │ Phase 4: AllGather TPUT              │    │
│         ...                          │   把归约结果广播给所有 rank           │    │
│                                      │ Phase 5: Barrier                     │    │
│                                      └──────────────────────────────────────┘    │
│                                                                                  │
│  关键特性:                                                                        │
│  - 双流并行：Cube 计算流 + Vector 通信流，计算与通信重叠执行                         │
│  - 按 tile 信号：计算完一个 tile 即可被通信侧消费，无需等全部算完                     │
│  - 通信量降低 4×：ReduceScatter 只发给 owner，AllGather 只由 owner 广播              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 通信算法详解：ReduceScatter + AllGather

### 输出矩阵的 Tile 划分

```
C_rank / C_final [M × N] = 16384 × 4096

Tile 大小: 128 × 256 (128KB per tile)
  M 方向: 16384/128 = 128 块
  N 方向: 4096/256  = 16 块
  总 tile 数: 128 × 16 = 2048

     ni →  0    1    2   ...   15
  mi ↓
    0    │  0  │  1  │  2  │ ... │ 15  │     tile_idx = mi × 16 + ni
    1    │ 16  │ 17  │ 18  │ ... │ 31  │
   ...   │     │     │     │     │     │
   127   │2032 │2033 │2034 │ ... │2047 │

Owner 分配 (round-robin):
  owner(tile_idx) = tile_idx % 8
  → Rank 0 拥有: tile 0, 8, 16, ..., 2040 (共 256 个)
  → Rank 1 拥有: tile 1, 9, 17, ..., 2041 (共 256 个)
  → ...
```

### Phase 1: ReduceScatter — 每 tile 只发给 owner

通信 block 轮询 Ready Queue，取到就绪的 tile 后判断 owner：

```
以 tile_idx = 42 为例，owner = 42 % 8 = 2 (Rank 2)

原方案 (Broadcast):                    新方案 (ReduceScatter):
  Rank 0 → Rank 1,2,3,4,5,6,7           Rank 0 → Rank 2 (仅 owner)
  （7 次 TPUT）                          （1 次 TPUT）

所有 rank 对 tile 42 的操作:
  Rank 0 → Rank 2  (1 次 TPUT)
  Rank 1 → Rank 2  (1 次 TPUT)
  Rank 2   本地已有，不发送 (0 次)
  Rank 3 → Rank 2  (1 次 TPUT)
  ...
  Rank 7 → Rank 2  (1 次 TPUT)

  tile 42 全局 TPUT 次数: 7 (vs 原方案 56)
```

每个 rank 的 ReduceScatter 发送量：

```
2048 tiles 中:
  256 个 tile 的 owner 是自己 → 不发 (0 TPUT)
  1792 个 tile → 各发 1 次 TPUT

  每 rank 发送: 1792 × 128KB = 224 MB (vs 原方案 1.75 GB)
```

### Phase 2: Barrier

`ShmemDeviceQuiet()` + `ShmemDeviceBarrierAll()`，保证所有 ReduceScatter 写入对全部 rank 可见。

### Phase 3: 本地归约（仅 owned tiles）

每个 rank **只对自己拥有的 256 个 tile** 做 8-way 求和：

```
以 Rank 0 为例，归约 tile 0:

  输入 (8 份):
    shmem_output[tile 0]     ← 本 rank 自己算的
    recv_buffers[1][tile 0]  ← Rank 1 发来的
    recv_buffers[2][tile 0]  ← Rank 2 发来的
    ...
    recv_buffers[7][tile 0]  ← Rank 7 发来的
  ──────────────────────────
  输出:
    reduced_output[tile 0]   ← 完整 AllReduce 结果

  归约方法: TREDUCE_PINGPONG (半高 64×256 子 tile)
  归约 tile 数: 256 (vs 原方案 2048，降低 8×)
```

**TREDUCE_PINGPONG** 使用交替的 EVENT_ID1/EVENT_ID2，避免 V 流水线事件计数器溢出（历史优化中发现的关键问题）。

### Phase 4: AllGather — owner 广播归约结果

每个 rank 把自己归约好的 256 个 tile，TPUT 到所有其他 rank 的 `reduced_output`：

```
Rank 0 广播 tile 0 的最终结果:

  reduced_output[tile 0] ─→ Rank 1 reduced_output[tile 0]
                          ─→ Rank 2 reduced_output[tile 0]
                          ─→ ...
                          ─→ Rank 7 reduced_output[tile 0]

  每个 owned tile 发 7 次 TPUT
  每 rank 发送: 256 × 7 × 128KB = 224 MB
```

### Phase 5: Barrier

`ShmemDeviceQuiet()` + `ShmemDeviceBarrierAll()` + `pipe_barrier(PIPE_ALL)`。

AllReduce 完成，每个 rank 的 `reduced_output` 都包含完整的 C\_final\[M×N\]。

### 通信量对比总览

```
               ┌────────────────────────┬──────────────────────────────┐
               │    原方案               │    新方案                     │
               │  Broadcast Scatter      │  ReduceScatter + AllGather   │
┌──────────────┼────────────────────────┼──────────────────────────────┤
│ 每 tile 发送 │ 7 个 remote rank       │ RS: 1 个 owner               │
│ 目标         │                        │ AG: owned tile 发 7 rank     │
├──────────────┼────────────────────────┼──────────────────────────────┤
│ TPUT 次数    │ 2048×7 = 14,336       │ 1,792 + 1,792 = 3,584       │
│ /rank        │                        │ (降低 4×)                    │
├──────────────┼────────────────────────┼──────────────────────────────┤
│ 通信数据量   │ 1.75 GB               │ 224 + 224 = 448 MB           │
│ /rank        │                        │ (降低 ~4×)                   │
├──────────────┼────────────────────────┼──────────────────────────────┤
│ 归约 tile 数 │ 2048 (全部)            │ 256 (仅 owned, 降低 8×)     │
│ /rank        │                        │                              │
├──────────────┼────────────────────────┼──────────────────────────────┤
│ 实测 Comm    │ ~31,330 us            │ ~9,945 us (降低 ~3.2×)       │
└──────────────┴────────────────────────┴──────────────────────────────┘
```

## 数据流全貌

```
                    每个 Rank 本地
  A_part[M,K/r] × B_part[K/r,N] = C_rank[M,N]
          │                    │
          ▼                    ▼
  ┌──────────────┐      ┌──────────────┐
  │ GEMM Compute │      │ shmem_output │  2048 tiles, 128×256 each
  │ 32 blocks    │ ──▶  │ [M×N]        │
  └──────────────┘      └──────┬───────┘
          │                    │
          │ 入队 tile_idx       │ ReduceScatter: 每 tile 只发给 owner
          ▼                    ▼
  ┌──────────────┐      ┌──────────────┐
  │ Ready Queue  │      │ recv_buffers │  按 slot 收别人发来的 tile
  │ (tile 信号)  │      │[nranks × MN] │
  └──────────────┘      └──────┬───────┘
                               │ Barrier
                               ▼
                        ┌──────────────┐
                        │ TREDUCE      │  仅对 owned tiles: 8 份 Sum
                        │ (PINGPONG)   │
                        └──────┬───────┘
                               │ AllGather: owner 把结果广播到所有 rank
                               ▼
                        ┌──────────────┐
                        │reduced_output│  每 rank 得到完整 C_final [M×N]
                        │ [M×N]        │
                        └──────────────┘
```

## 对称堆内存布局

每个 rank 在对称堆上分配三块内存：

| 缓冲区 | 大小 | 用途 |
|---------|------|------|
| `shmem_output` | M×N×4B = 256 MB | 本地 GEMM 结果 C\_rank |
| `recv_buffers` | nranks × M×N×4B = 2048 MB | 按 rank 分 slot，接收 ReduceScatter 数据 |
| `reduced_output` | M×N×4B = 256 MB | AllReduce 最终结果 C\_final |

```
recv_buffers 逻辑布局 (以 Rank 0 为例):
┌────────────────┬────────────────┬─────┬────────────────┐
│ slot 0 (rank0) │ slot 1 (rank1) │ ... │ slot 7 (rank7) │
│ [M×N] 本地数据 │ [M×N] rank1送来│     │ [M×N] rank7送来│
└────────────────┴────────────────┴─────┴────────────────┘
注：slot 0 = 本 rank 自身数据，从 shmem_output 直接读取，不通过 recv_buffers
```

## 计算内核

### 核心参数

| 参数 | 值 | 说明 |
|------|-----|------|
| G_M | 16384 | 矩阵 M 维度 |
| G_K | 16384 | 矩阵 K 维度（全局） |
| G_N | 4096 | 矩阵 N 维度 |
| G_BASE_M | 128 | Tile M 维度 |
| G_BASE_K | 64 | Tile K 维度 |
| G_BASE_N | 256 | Tile N 维度 |
| G_STEP_KA/KB | 4 | L1 缓存 K-slice 数（4× 减少 DMA 频率） |
| COMPUTE_BLOCK_NUM | 32 | 计算 block 数 |
| k_per_rank | K/nranks = 2048 | 每 rank 的 K 维度 |

### 计算流程

每个计算 block 负责一组 tile（2048 / 32 = 64 个），对每个 tile：

1. **K-loop**：遍历 K/r / baseK = 2048/64 = 32 次迭代
   - 每 4 次迭代做一次 TLOAD（L1 缓存优化，4× 减少 GM→L1 DMA）
   - 每次迭代 TEXTRACT + TMATMUL\_ACC 累加
2. **TSTORE**：写入 `shmem_output` 对应位置
3. **pipe\_barrier(PIPE\_ALL)**：确保 GM 写入完成
4. **MultiBlockEnqueueFast**：入队 `tile_idx`，通知通信侧

### 两级双缓冲流水线

```
时间 →
L1 (MTE2):  [TLOAD A0,B0]              [TLOAD A1,B1]              ...
L0 (MTE1):       [EXT k0] [EXT k1] [EXT k2] [EXT k3] [EXT k0'] ...
Cube (M):             [MUL k0] [ACC k1] [ACC k2] [ACC k3] [MUL k0'] ...
                      ↑ 三级流水线完全并行 ↑
```

## 三项优化措施及收益

### 1. ReduceScatter + AllGather 通信算法

- **原方案**：每个 tile 广播到所有 7 个 remote rank（TPUT 14,336 次/rank，1.75 GB/rank）
- **新方案**：ReduceScatter 只发给 owner（1,792 次）+ AllGather 由 owner 广播（1,792 次），总 3,584 次/rank，448 MB/rank
- **收益**：通信时间从 ~31ms 降至 ~10ms

### 2. K 维度从 4096 增至 16384

- 每 rank 计算量：2 × M × (K/r) × N
  - 原：2 × 16384 × 512 × 4096 = 137 GFLOP → ~0.7 ms
  - 新：2 × 16384 × 2048 × 4096 = 550 GFLOP → ~1.7 ms
- **收益**：计算在总时间中占比增大，流水线重叠带来的 Time Saved 更显著

### 3. COMM_BLOCK_NUM 从 40 降至 20

- 减少通信 block 占用的 AI Core 资源
- **收益**：降低 Cube/Vector core 之间的资源竞争，流水线计算效率从 12.3% 提升到 59.9%

## 性能结果

```
================================================================
  Matrix dimensions: M=16384, K=16384, N=4096
  Ranks: 8
  Compute blocks: 32, Comm blocks: 20
  Algorithm: ReduceScatter + AllGather
================================================================

  Compute Kernel (pure):     1707.5 us, 160981 GFLOPS

  Sequential (no overlap):   11747.1 us total
    Compute:  1802.0 us (152542 GFLOPS)
    Comm:     9945.1 us (201.1 GB/s)

  Pipelined (with overlap):  9374.4 us total
    Compute:  2852.3 us (96370 GFLOPS, 59.9% of pure)
    Comm:     9374.4 us (213.3 GB/s)

  Performance Comparison:
    Speedup:     1.253×
    Time Saved:  2372.7 us (20.198%)
```

## 构建与运行

```bash
# 一条命令：生成数据 + 构建 + 运行
./run_performance_test.sh

# 指定参数
./run_performance_test.sh --nranks 8 --first-device 0

# 仅运行不重编译
./run_performance_test.sh --skip-build
```

## 文件结构

```
gemm_allreduce_opt_3/
├── CMakeLists.txt              # 构建配置
├── new_README_zh.md            # 本文档（中文）
├── new_README.md               # 本文档（英文）
├── README_zh.md                # 旧版文档（中文）
├── run_performance_test.sh     # 性能测试入口脚本
├── main.cpp                    # 主程序（双流启动逻辑）
├── gemm_compute_kernel.cpp     # 计算内核（Cube, 32 blocks）
├── comm_kernel.cpp             # 通信内核（Vec, 20 blocks）
│                                 含 ReduceScatter + AllGather 逻辑
│                                 及 host 侧 launch / 验证 / 性能统计
├── ready_queue.hpp             # 多 block 无锁队列
├── input/                      # 输入数据 (gen_data.py 生成)
├── output/                     # 输出数据 / golden
└── scripts/
    └── gen_data.py             # 数据生成脚本 (K=16384)
```

## 关键技术点

### 1. Plain TPUT 代替 TPUT\<AtomicAdd\>

新方案使用 plain TPUT（无原子操作），通过 ReduceScatter 的 owner 分配机制避免了写冲突：

- 每个 tile 在 ReduceScatter 阶段只被发送到唯一的 owner rank
- 不同 rank 写入 owner 的 `recv_buffers` 中不同 slot（按发送方 rank 编号），无竞争
- AllGather 阶段每个 tile 也只有 owner 写，其他 rank 只读

### 2. TREDUCE_PINGPONG 避免事件计数器溢出

- V 流水线的事件计数器只有 2-bit 容量
- 普通 TREDUCE 在循环中 EVENT_ID1 持续累积导致溢出
- TREDUCE_PINGPONG 交替使用 EVENT_ID1 和 EVENT_ID2，每个计数器最大值为 1

### 3. Ready Queue 无锁信号机制

- 每个计算 block 一个独立队列（单生产者单消费者）
- 通信侧用 TTEST/TWAIT 硬件指令轮询
- 实现逐 tile 的计算-通信流水

## 参考

- [PTO 通信指令集](../../../include/pto/comm/)
- [TPUT 指令文档](../../../include/pto/comm/TPut.hpp)
- [TREDUCE_PINGPONG 指令文档](../../../include/pto/comm/TReduce.hpp)
