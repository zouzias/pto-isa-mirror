# PTO-ISA 实现 Tile 级通算融合高性能 GEMM+AllReduce Kernel 指南

## 概述

本文档介绍如何使用 PTO-ISA 实现一个 tile 级通算融合的高性能 GEMM+AllReduce kernel。该方案通过以下核心技术实现计算与通信的深度重叠：

1. **Cube/Vector 单元并发**：Cube 单元执行 GEMM，Vector 单元执行 AllReduce 通信
2. **Tile 级流水线**：在 tile 粒度上实现计算与通信的 overlap
3. **双缓冲（Double Buffering）**：L1、L0A/L0B、UB 多级缓冲实现流水线

## 目录

1. [硬件架构背景](#1-硬件架构背景)
2. [高性能 GEMM 实现](#2-高性能-gemm-实现)
3. [AllReduce 通信实现](#3-allreduce-通信实现)
4. [Cube/Vector 并发架构](#4-cubevector-并发架构)
5. [Tile 级通算融合实现](#5-tile-级通算融合实现)
6. [完整示例代码](#6-完整示例代码)
7. [性能调优指南](#7-性能调优指南)

---

## 1. 硬件架构背景

### 1.1 AI Core 架构

每个 AI Core 包含：
- **1 个 Cube 单元**：用于矩阵计算（GEMM），执行 TMATMUL 指令
- **2 个 Vector 单元**：用于向量计算和通信操作，执行 AllReduce 等集合通信

```
┌─────────────────────────────────────────────────────┐
│                    AI Core                          │
├─────────────────────┬───────────────────────────────┤
│     Cube 单元       │       Vector 单元 (x2)        │
│   ┌─────────────┐   │   ┌─────────────────────────┐ │
│   │             │   │   │  SubBlock 0  SubBlock 1 │ │
│   │    GEMM     │   │   │  ┌───────┐  ┌───────┐   │ │
│   │  (TMATMUL)  │   │   │  │AllRed │  │AllRed │   │ │
│   │             │   │   │  │uce    │  │uce    │   │ │
│   └─────────────┘   │   │  └───────┘  └───────┘   │ │
│                     │   └─────────────────────────┘ │
└─────────────────────┴───────────────────────────────┘
         ↑                          ↑
         │                          │
    __DAV_CUBE__               __DAV_VEC__
    编译时宏                    编译时宏
```

### 1.2 内存层次

```
┌──────────────────────────────────────────────────────────┐
│                    Global Memory (GM)                     │
│  ┌────────────────────────────────────────────────────┐  │
│  │              Symmetric Heap (RDMA)                  │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                            ↓ TLOAD
┌──────────────────────────────────────────────────────────┐
│                     L1 Buffer (~1MB)                      │
│  用于 GEMM 数据分块缓存 (aMatTile, bMatTile)             │
└──────────────────────────────────────────────────────────┘
                            ↓ TEXTRACT / TMOV
┌───────────────────────┬──────────────────────────────────┐
│   L0A Buffer (64KB)   │      L0B Buffer (64KB)           │
│   存放左矩阵 tile     │      存放右矩阵 tile             │
└───────────────────────┴──────────────────────────────────┘
                            ↓ TMATMUL
┌──────────────────────────────────────────────────────────┐
│                    L0C Buffer (256KB)                     │
│                    存放输出累加 tile                      │
└──────────────────────────────────────────────────────────┘
                            ↓ TSTORE
┌──────────────────────────────────────────────────────────┐
│              Unified Buffer (UB) (~256KB)                 │
│           向量计算缓冲 / 通信中间结果                     │
└──────────────────────────────────────────────────────────┘
```

---

## 2. 高性能 GEMM 实现

### 2.1 核心流水线

高性能 GEMM 的核心是构建四级流水线：

```
TLOAD (GM→L1) → TEXTRACT (L1→L0A/L0B) → TMATMUL (Cube计算) → TSTORE (L0C→GM)
```

### 2.2 分块策略

#### 参数定义

| 参数 | 含义 | 参考值 |
|------|------|--------|
| `baseM` | L0A tile 的 M 维度 | 128 |
| `baseK` | L0A/L0B tile 的 K 维度 | 64 |
| `baseN` | L0B tile 的 N 维度 | 256 |
| `stepKa` | L1 缓存的 K 块数 (A矩阵) | 4 |
| `stepKb` | L1 缓存的 K 块数 (B矩阵) | 4 |
| `singleCoreM` | 单核处理的 M 维度 | 1536 |
| `singleCoreN` | 单核处理的 N 维度 | 1024 |
| `singleCoreK` | 单核处理的 K 维度 | 6144 |

#### L0 Buffer 容量约束

- L0A/L0B 使用 32KB ping-pong 分割
- L0A tile: `baseM × baseK × sizeof(half) ≤ 32KB`
  - 例: 128 × 64 × 2 = 16KB ✓
- L0B tile: `baseK × baseN × sizeof(half) ≤ 32KB`
  - 例: 64 × 256 × 2 = 32KB ✓

### 2.3 Tile 类型定义

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// L1 缓冲 Tile（用于 TLOAD 存储从 GM 加载的数据）
using TileMatA = Tile<TileType::Mat, half, baseM, baseK * stepKa, 
                      BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>;
using TileMatB = Tile<TileType::Mat, half, baseK * stepKb, baseN, 
                      BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>;

// L0A/L0B Tile（用于 TMATMUL 输入）
using LeftTile = TileLeft<half, baseM, baseK, baseM, baseK>;
using RightTile = TileRight<half, baseK, baseN, baseK, baseN>;

// L0C Tile（用于 TMATMUL 输出累加）
using ResTile = TileAcc<float, baseM, baseN, baseM, baseN>;

// 双缓冲数组
TileMatA aMatTile[2];
TileMatB bMatTile[2];
LeftTile aTile[2];
RightTile bTile[2];
ResTile cTile;
```

### 2.4 Buffer 地址分配

```cpp
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;  // 32KB per ping-pong slot

// L1 Buffer 分配
TASSIGN(aMatTile[0], 0x0);
TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(half));
TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * 2 * sizeof(half));
TASSIGN(bMatTile[1], 0x0 + baseM * baseK * stepKa * 2 * sizeof(half) + 
                          baseK * baseN * stepKb * sizeof(half));

// L0A/L0B 双缓冲分配
TASSIGN(aTile[0], 0x0);                      // L0A ping
TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES);  // L0A pong
TASSIGN(bTile[0], 0x0);                      // L0B ping
TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES);  // L0B pong

// L0C 分配
TASSIGN(cTile, 0x0);
```

### 2.5 同步事件机制

PTO-ISA 使用 `set_flag` / `wait_flag` 在不同流水线阶段之间同步：

```cpp
// PIPE 类型说明：
// - PIPE_MTE2: GM → L1 数据搬运
// - PIPE_MTE1: L1 → L0 数据搬运
// - PIPE_M:    Cube 矩阵计算
// - PIPE_MTE3: L0C → GM 数据回写
// - PIPE_FIX:  定点计算（格式转换）

// 同步示例：等待 TLOAD 完成后再执行 TEXTRACT
TLOAD(aMatTile[bufIdx], gmA);
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
TEXTRACT(aTile[bufIdx], aMatTile[bufIdx], 0, 0);
```

### 2.6 完整 GEMM 流水线核心代码

```cpp
template <typename T, typename U, typename S>
AICORE inline void ProcessKIteration(
    uint32_t kIter, uint32_t i, uint32_t j,
    __gm__ U *currentSrc0, __gm__ S *currentSrc1,
    TileMatA aMatTile[2], TileMatB bMatTile[2],
    LeftTile aTile[2], RightTile bTile[2], 
    ResTile &cTile,
    uint8_t &mte2DBFlag, uint8_t &mte1DBFlag)
{
    const uint32_t kModstepKa = kIter % stepKa;

    // === TLOAD 阶段：GM → L1 ===
    if (kModstepKa == 0) {
        GlobalDataSrcA gmA(currentSrc0 + i * singleCoreK * baseM + kIter * baseK);
        GlobalDataSrcB gmB(currentSrc1 + j * singleCoreK * baseN + kIter * baseK);

        // 等待上一次 TEXTRACT 完成，可复用 L1 buffer
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;
    }

    const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;
    
    // 等待 TMATMUL 完成，可复用 L0 buffer
    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);

    // === TEXTRACT 阶段：L1 → L0A/L0B ===
    if (kModstepKa == 0) WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModstepKa * baseK);

    if (kModstepKa == 0) WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);

    if ((kIter + 1) % stepKa == 0) {
        SetFlag<PIPE_MTE1, PIPE_MTE2>(currMte2Idx);
    }

    // === TMATMUL 阶段：Cube 计算 ===
    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    
    if (kIter == 0) {
        TMATMUL(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    } else {
        TMATMUL_ACC(cTile, cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    }
    
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;
}

// === TSTORE 阶段：L0C → GM ===
template <typename T>
AICORE inline void StoreResult(ResTile &cTile, __gm__ T *currentDst, uint32_t i, uint32_t j)
{
    SetFlag<PIPE_M, PIPE_FIX>(0);
    WaitFlag<PIPE_M, PIPE_FIX>(0);
    
    GlobalDataOut dstGlobal(currentDst + i * baseM * n + j * baseN);
    TSTORE(dstGlobal, cTile);
    
    SetFlag<PIPE_FIX, PIPE_M>(0);
    WaitFlag<PIPE_FIX, PIPE_M>(0);
}
```

---

## 3. AllReduce 通信实现

### 3.1 shmem 通信模型

PTO-ISA 使用 OpenSHMEM 风格的单边通信模型：

```cpp
// 获取当前 rank
int my_rank = shmem_my_pe();
int nranks = shmem_n_pes();

// 获取远程 rank 的内存地址映射
__gm__ T* remote_ptr = (__gm__ T*)shmem_ptr(local_ptr, remote_pe);

// 对称内存分配（所有 rank 上地址相同）
void* symm_buf = shmem_malloc(size);
```

### 3.2 ParallelGroup 构建

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

template <typename T, int kRows, int kCols>
AICORE void SetupParallelGroup(
    __gm__ T *shmem_buffer,
    int my_rank, int nranks)
{
    using ShapeDyn = Shape<1, 1, 1, kRows, kCols>;
    using StrideDyn = Stride<1, 1, 1, kCols, 1>;
    using Global = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    
    ShapeDyn shape(1, 1, 1, kRows, kCols);
    StrideDyn stride(1, 1, 1, kCols, 1);
    
    // 构建所有 rank 的 GlobalTensor 数组
    Global tensors[16];
    for (int i = 0; i < nranks && i < 16; ++i) {
        __gm__ T* rank_i_ptr = (__gm__ T*)shmem_ptr(shmem_buffer, i);
        tensors[i] = Global(rank_i_ptr, shape, stride);
        tensors[i].SetRank(i);
    }
    
    // 创建 ParallelGroup
    pto::comm::ParallelGroup<Global> pg(tensors, nranks, my_rank);
}
```

### 3.3 TREDUCE_PINGPONG 实现

使用 ping-pong 双缓冲的 Reduce 操作：

```cpp
template <typename T, int kTRows, int kTCols>
AICORE void RunAllReduceOnCore(
    __gm__ T *input, 
    __gm__ T *output, 
    __gm__ T *shmem,
    int core_idx,
    int nranks, int my_rank)
{
    using ShapeDyn = Shape<1, 1, 1, kTRows, kTCols>;
    using StrideDyn = Stride<1, 1, 1, kTCols, 1>;
    using Global = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using TileData = Tile<TileType::Vec, T, kTRows, kTCols, BLayout::RowMajor, -1, -1>;

    // 计算分区偏移
    const int64_t row_offset = static_cast<int64_t>(core_idx) * kTRows * kTCols;
    
    // 分配 UB tile（三个用于 ping-pong）
    TileData src0Tile(kTRows, kTCols);
    TileData src1Tile(kTRows, kTCols);
    TileData dstTile(kTRows, kTCols);
    
    constexpr size_t tileBytes = kTRows * kTCols * sizeof(T);
    constexpr size_t alignedTileBytes = ((tileBytes + 31) / 32) * 32;
    
    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, alignedTileBytes);
    TASSIGN(dstTile, alignedTileBytes * 2);

    // 构建 ParallelGroup
    ShapeDyn shape(1, 1, 1, kTRows, kTCols);
    StrideDyn stride(1, 1, 1, kTCols, 1);
    
    Global tensors[16];
    for (int i = 0; i < nranks; ++i) {
        __gm__ T* rank_i_base = (__gm__ T*)shmem_ptr(shmem, i);
        __gm__ T* rank_i_partition = rank_i_base + row_offset;
        tensors[i] = Global(rank_i_partition, shape, stride);
        tensors[i].SetRank(i);
    }
    
    pto::comm::ParallelGroup<Global> pg(tensors, nranks, my_rank);
    Global dstGlobal(output + row_offset, shape, stride);

    // 全局同步：确保所有 rank 数据就绪
    ShmemDeviceBarrierAll();
    
    // 执行 ping-pong Reduce Sum
    pto::comm::TREDUCE_PINGPONG(pg, dstGlobal, src0Tile, src1Tile, dstTile, 
                                comm::ReduceOp::Sum);
}
```

### 3.4 TREDUCE_PINGPONG 原理

ping-pong 双缓冲实现计算与通信重叠：

```
时间 →
┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐
│Load ││Load ││Load ││Load ││Store│
│rank0││rank1││rank2││rank3││     │
├─────┤├─────┤├─────┤├─────┤└─────┘
│     ││Add  ││Add  ││Add  │
│     ││0+1  ││+2   ││+3   │
└─────┘└─────┘└─────┘└─────┘

使用 ping-pong 优化后：
┌──────────┬──────────┬──────────┬──────────┐
│Load rank0│Load rank1│Load rank2│Load rank3│  ← 预取
├──────────┼──────────┼──────────┼──────────┤
│          │ Add 0+1  │ Add +2   │ Add +3   │  ← 计算与下一次Load重叠
│          │Load rank2│Load rank3│ Store    │
└──────────┴──────────┴──────────┴──────────┘
```

---

## 4. Cube/Vector 并发架构

### 4.1 编译时条件分支

使用 `--cce-aicore-arch=dav-c220` 编译选项启用混合架构：

```cmake
target_compile_options(${NAME}_kernel PRIVATE 
    ${CMAKE_CCE_COMPILE_OPTIONS} 
    --cce-aicore-arch=dav-c220   # 混合 Cube/Vector 架构
    -DMEMORY_BASE -std=c++17)
```

在 kernel 中使用编译时宏区分执行路径：

```cpp
// 检测编译时宏
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#include "pto/comm/pto_comm_inst.hpp"
#else
constexpr bool DAV_VEC = false;
#endif
```

### 4.2 Kernel 结构

```cpp
template <typename T, uint32_t gemmBaseM, uint32_t gemmBaseK, uint32_t gemmBaseN,
          int arRows, int arCols>
__global__ AICORE void MatmulAllReduceConcurrentKernel(
    // GEMM inputs/outputs
    __gm__ half *gemmSrcA, __gm__ half *gemmSrcB, __gm__ float *gemmDstC,
    // AllReduce inputs/outputs
    __gm__ T *arInput, __gm__ T *arOutput, __gm__ T *arShmem,
    // Configuration
    int total_blocks, int gemm_blocks, int nranks, uint64_t fftsConfig)
{
    // 初始化 FFTS 配置（通信必需）
    util_set_ffts_config(fftsConfig);
    ShmemDeviceBarrierAll();
    
    const int block_idx = get_block_idx();
    const int my_rank = shmem_my_pe();
    
    // ================================================================
    // Cube 单元执行路径：GEMM
    // ================================================================
    if constexpr (DAV_CUBE) {
        int gemm_core_idx = block_idx;
        
        RunGEMMOnCore<float, half, half, gemmBaseM, gemmBaseK, gemmBaseN>(
            gemmSrcA, gemmSrcB, gemmDstC, gemm_core_idx);
        
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    
    // ================================================================
    // Vector 单元执行路径：AllReduce
    // ================================================================
    if constexpr (DAV_VEC) {
        int allreduce_core_idx = block_idx * get_subblockdim() + get_subblockid();
        
        // 核内同步
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        
        RunAllReduceOnCore<T, arRows, arCols>(
            arInput, arOutput, arShmem, allreduce_core_idx, nranks, my_rank);
        
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
    }
    
    ShmemDeviceBarrierAll();
}
```

---

## 5. Tile 级通算融合实现

### 5.1 Catlass/Catcoc 框架

`shmem/examples/matmul_allreduce` 展示了更高级的 tile 级融合方案，基于 Catlass（类似 CUTLASS）框架：

```cpp
// 核心组件
using BlockMmad = Catlass::Gemm::Block::BlockMmad<...>;           // GEMM 计算块
using BlockEpilogueReduceScatter = CommEpilogue::Block::...;       // ReduceScatter 通信
using BlockEpilogueAllGather = CommEpilogue::Block::...;           // AllGather 通信

// 融合 Kernel
using MatmulAllReduceKernel = DGemm::Kernel::MatmulAllReduce<
    BlockMmad,
    BlockEpilogueReduceScatter,
    BlockEpilogueAllGather,
    BlockMmadScheduler,
    BlockEpilogueScheduler,
    WORKSPACE_STAGES
>;
```

### 5.2 通信间隔（COMM_INTERVAL）

控制每多少个 GEMM tile 执行一次通信：

```cpp
constexpr uint32_t COMM_INTERVAL = 3;  // 每 3 个 GEMM tile 触发一次通信
```

### 5.3 AIC/AIV 分工

MatmulAllReduce 使用模板特化分离 Cube（AIC）和 Vector（AIV）执行路径：

```cpp
// AIC (Cube) 执行 GEMM
template <>
CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params) {
    // GEMM 计算循环
    for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
        uint32_t stageId = commIdx % WORKSPACE_STAGES;
        
        // 等待上一轮 AIV 完成
        if (commIdx >= WORKSPACE_STAGES) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);
        }
        
        // 执行多个 GEMM tile
        for (uint32_t blockIdxInComm = ...; blockIdxInComm < blockPerComm; ...) {
            blockMmad(gmA, layoutA, gmB, layoutB, gmSymmetric, layoutC, actualBlockShape);
        }
        
        // 通知 AIV 可以开始通信
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
    }
}

// AIV (Vector) 执行通信
template <>
CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params) {
    for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
        uint32_t stageId = commIdx % WORKSPACE_STAGES;
        
        // 等待 AIC 完成 GEMM
        Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
        
        // 跨 rank 同步
        aclshmemx_barrier_all_vec();
        
        // ReduceScatter
        AscendC::SetAtomicAdd<ElementD>();
        reduceScatter(...);
        AscendC::SetAtomicNone();
        
        aclshmemx_barrier_all_vec();
        
        // AllGather
        allGather(...);
        
        aclshmemx_barrier_all_vec();
        
        // 通知 AIC 可以复用 workspace
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
    }
}
```

### 5.4 Workspace 双缓冲

使用 WORKSPACE_STAGES 实现 GEMM 与通信的流水线：

```
时间 →
┌────────────────┬────────────────┬────────────────┐
│  GEMM Tile 0-2 │  GEMM Tile 3-5 │  GEMM Tile 6-8 │  ← Cube 单元
│  → Stage 0     │  → Stage 1     │  → Stage 0     │
├────────────────┼────────────────┼────────────────┤
│                │  ReduceScatter │  ReduceScatter │  ← Vector 单元
│                │    Stage 0     │    Stage 1     │
│                │  AllGather     │  AllGather     │
│                │    Stage 0     │    Stage 1     │
└────────────────┴────────────────┴────────────────┘
```

---

## 6. 完整示例代码

### 6.1 简化版 GEMM + AllReduce

```cpp
#include <pto/pto-inst.hpp>
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

// ============================================================================
// 配置参数
// ============================================================================
constexpr uint32_t TOTAL_BLOCK_NUM = 4;
constexpr uint32_t GEMM_BLOCK_NUM = TOTAL_BLOCK_NUM;
constexpr uint32_t GEMM_BASE_M = 128;
constexpr uint32_t GEMM_BASE_K = 64;
constexpr uint32_t GEMM_BASE_N = 128;
constexpr int AR_ROWS = 16;
constexpr int AR_COLS = 256;

// ============================================================================
// Cube 路径：GEMM 实现
// ============================================================================
#ifdef __DAV_C220_CUBE__

template <typename T, typename U, typename S, 
          uint32_t baseM, uint32_t baseK, uint32_t baseN>
AICORE inline void RunGEMMOnCore(
    __gm__ U *srcA, __gm__ S *srcB, __gm__ T *dstC,
    int core_idx)
{
    using NDValidShapeA = TileShape2D<U, baseM, baseK>;
    using NDsingleCoreShapeA = BaseShape2D<U, baseM, baseK>;
    using GlobalDataSrcA = GlobalTensor<U, NDValidShapeA, NDsingleCoreShapeA>;

    using NDValidShapeB = TileShape2D<U, baseK, baseN, Layout::DN>;
    using NDsingleCoreShapeB = BaseShape2D<U, baseK, baseN, Layout::DN>;
    using GlobalDataSrcB = GlobalTensor<U, NDValidShapeB, NDsingleCoreShapeB, Layout::DN>;

    using NDValidShapeC = TileShape2D<T, baseM, baseN>;
    using NDWholeShapeC = BaseShape2D<T, baseM, baseN>;
    using GlobalDataOut = GlobalTensor<T, NDValidShapeC, NDWholeShapeC>;

    // 计算分区偏移
    uint64_t offsetA = static_cast<uint64_t>(core_idx) * baseM * baseK;
    uint64_t offsetB = static_cast<uint64_t>(core_idx) * baseK * baseN;
    uint64_t offsetC = static_cast<uint64_t>(core_idx) * baseM * baseN;

    // 定义 Tile
    using TileMatA = Tile<TileType::Mat, U, baseM, baseK, 
                          BLayout::ColMajor, baseM, baseK, SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, S, baseK, baseN, 
                          BLayout::RowMajor, baseK, baseN, SLayout::ColMajor>;
    using LeftTile = TileLeft<U, baseM, baseK, baseM, baseK>;
    using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;
    using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    LeftTile aTile;
    RightTile bTile;
    ResTile cTile;

    // 分配 Buffer
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + baseM * baseK * sizeof(U));
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    // 构建 Global Tensor
    GlobalDataSrcA gmA(srcA + offsetA);
    GlobalDataSrcB gmB(srcB + offsetB);
    GlobalDataOut dstGlobal(dstC + offsetC);

    // TLOAD: GM → L1
    TLOAD(aMatTile, gmA);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TLOAD(bMatTile, gmB);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    
    // TMOV: L1 → L0
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(bTile, bMatTile);
    
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    
    // TMATMUL: Cube 计算
    TMATMUL(cTile, aTile, bTile);
    
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    
    // TSTORE: L0C → GM
    TSTORE(dstGlobal, cTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

#endif  // __DAV_C220_CUBE__

// ============================================================================
// Vector 路径：AllReduce 实现
// ============================================================================
#ifdef __DAV_C220_VEC__

template <typename T, int kTRows, int kTCols>
AICORE inline void RunAllReduceOnCore(
    __gm__ T *input, __gm__ T *output, __gm__ T *shmem,
    int core_idx, int nranks, int my_rank)
{
    using ShapeDyn = Shape<1, 1, 1, kTRows, kTCols>;
    using StrideDyn = Stride<1, 1, 1, kTCols, 1>;
    using Global = GlobalTensor<T, ShapeDyn, StrideDyn, Layout::ND>;
    using TileData = Tile<TileType::Vec, T, kTRows, kTCols, BLayout::RowMajor, -1, -1>;

    const int64_t row_offset = static_cast<int64_t>(core_idx) * kTRows * kTCols;
    
    // 分配 ping-pong Tile
    TileData src0Tile(kTRows, kTCols);
    TileData src1Tile(kTRows, kTCols);
    TileData dstTile(kTRows, kTCols);
    
    constexpr size_t tileBytes = kTRows * kTCols * sizeof(T);
    constexpr size_t alignedTileBytes = ((tileBytes + 31) / 32) * 32;
    
    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, alignedTileBytes);
    TASSIGN(dstTile, alignedTileBytes * 2);

    ShapeDyn shape(1, 1, 1, kTRows, kTCols);
    StrideDyn stride(1, 1, 1, kTCols, 1);
    
    // 构建 ParallelGroup
    Global tensors[16];
    for (int i = 0; i < nranks; ++i) {
        __gm__ T* rank_i_ptr = (__gm__ T*)shmem_ptr(shmem, i);
        tensors[i] = Global(rank_i_ptr + row_offset, shape, stride);
        tensors[i].SetRank(i);
    }
    
    pto::comm::ParallelGroup<Global> pg(tensors, nranks, my_rank);
    Global dstGlobal(output + row_offset, shape, stride);

    ShmemDeviceBarrierAll();
    
    // 执行 Reduce Sum
    pto::comm::TREDUCE_PINGPONG(pg, dstGlobal, src0Tile, src1Tile, dstTile, 
                                comm::ReduceOp::Sum);
}

#endif  // __DAV_C220_VEC__

// ============================================================================
// 主 Kernel
// ============================================================================
template <typename T>
__global__ AICORE void GemmAllReduceConcurrentKernel(
    __gm__ half *gemmA, __gm__ half *gemmB, __gm__ float *gemmC,
    __gm__ T *arInput, __gm__ T *arOutput, __gm__ T *arShmem,
    int nranks, uint64_t fftsConfig)
{
    util_set_ffts_config(fftsConfig);
    ShmemDeviceBarrierAll();
    
    const int block_idx = get_block_idx();
    const int my_rank = shmem_my_pe();
    
    // Cube 执行 GEMM
    if constexpr (DAV_CUBE) {
        RunGEMMOnCore<float, half, half, GEMM_BASE_M, GEMM_BASE_K, GEMM_BASE_N>(
            gemmA, gemmB, gemmC, block_idx);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    
    // Vector 执行 AllReduce
    if constexpr (DAV_VEC) {
        int ar_core_idx = block_idx * get_subblockdim() + get_subblockid();
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
        
        RunAllReduceOnCore<T, AR_ROWS, AR_COLS>(
            arInput, arOutput, arShmem, ar_core_idx, nranks, my_rank);
        
        AscendC::PipeBarrier<PIPE_ALL>();
        aclshmemi_barrier_core_soft();
    }
    
    ShmemDeviceBarrierAll();
}
```

---

## 7. 性能调优指南

### 7.1 GEMM 性能优化要点

| 优化项 | 说明 | 建议 |
|--------|------|------|
| **核划分** | 将 C 矩阵按 M×N 分块到多核 | 使用 2D 划分（如 4×6 = 24 核）|
| **Base Block 选择** | 选择最大化计算密度的 tile | [128, 256, 64] 适合 FP16 |
| **L1 缓存** | 使用 stepKa/stepKb 多加载几个 K 块 | stepKa = stepKb = 4 |
| **双缓冲** | L1、L0A/L0B 都使用 ping-pong | 每级 2 个 buffer |
| **同步最小化** | 只在真正的依赖点 wait_flag | 避免冗余同步 |

### 7.2 通信性能优化要点

| 优化项 | 说明 | 建议 |
|--------|------|------|
| **Tile 大小** | 选择合适的通信粒度 | 32×256 或更大 |
| **COMM_INTERVAL** | 控制计算与通信比例 | 3-5 个 GEMM tile |
| **Ping-pong** | 使用 TREDUCE_PINGPONG | 重叠通信与规约计算 |
| **对称内存** | 使用 shmem_malloc 分配 | 避免地址转换开销 |

### 7.3 利用率指标解读

| 指标 | 含义 | 优化方向 |
|------|------|----------|
| TMATMUL Ratio ↓ + TLOAD Ratio ↑ | 内存受限 | 增加数据复用、优化 L1 缓存 |
| TEXTRACT Ratio ↑ | L1→L0 搬运成为瓶颈 | 增大 stepKa/stepKb |
| TSTORE Ratio ↓ | 输出写回占比小 | 正常，GEMM 特性 |

### 7.4 调试技巧

1. **启用 cce::printf**：编译时添加 `-d` 选项
2. **使用 GetSystemCycle()**：精确测量各阶段耗时
3. **检查验证结果**：每个 rank 独立验证后再合并

---

## 附录：参考实现

| 示例 | 路径 | 说明 |
|------|------|------|
| 高性能 GEMM | `kernels/manual/a2a3/gemm_performance/` | 纯 GEMM 优化 |
| MatMul+AllReduce 并发 | `kernels/manual/a2a3/matmul_allreduce_concurrent/` | Cube/Vector 并发 |
| Catlass MatMul+AllReduce | `shmem/examples/matmul_allreduce/` | Tile 级深度融合 |
| 通信测试 | `tests/comm/st/testcase/treduce_perf_test/` | TREDUCE 性能基准 |

---

## 版本历史

| 日期 | 变更 |
|------|------|
| 2026-02-05 | 初始版本，整合 GEMM + AllReduce 实现指南 |
