# PTO-ISA 实现 Tile 级通算融合高性能 GEMM+AllReduce Kernel 指南

## 概述

本文档介绍 `gemm_ar` 示例如何使用 PTO-ISA 实现一个 **tile 级通算融合**的高性能 GEMM+AllReduce kernel。该方案的核心技术：

1. **Cube/Vector 单元并发**：同一 kernel 内，Cube 单元执行 GEMM，Vector 单元执行 AllReduce
2. **TNOTIFY/TWAIT tile 级同步**：Cube 每完成一个 tile 就通过 shmem flag 通知 Vec，Vec 逐 tile 等待并执行 AllReduce
3. **对称内存（Symmetric Heap）**：GEMM 结果直接写入 shmem，各 rank 的 Vec 通过 `shmem_ptr()` 远程读取其他 rank 的数据做 reduce
4. **多级双缓冲（Double Buffering）**：L1、L0A/L0B 两级 ping-pong 实现 GEMM 流水线

### 文件结构

```
gemm_ar/
├── gemm_config.h              # 全局参数（矩阵维度、tile 尺寸、核数）
├── gemm_ar_kernel.cpp         # Kernel 实现（Cube GEMM + Vec AllReduce）
├── common.hpp                 # Shmem 初始化/同步工具函数
├── main.cpp                   # Host 端：多进程启动、数据搬运、kernel launch
├── CMakeLists.txt             # 构建配置
├── run.sh                     # 一键编译运行脚本
└── scripts/gen_data.py        # 生成输入数据和 golden 参考
```

## 目录

1. [硬件架构背景](#1-硬件架构背景)
2. [参数配置](#2-参数配置)
3. [高性能 GEMM 实现（Cube 路径）](#3-高性能-gemm-实现cube-路径)
4. [AllReduce 通信实现（Vec 路径）](#4-allreduce-通信实现vec-路径)
5. [Tile 级通算融合：TNOTIFY/TWAIT](#5-tile-级通算融合tnotifytwait)
6. [Kernel 入口与 Host 端](#6-kernel-入口与-host-端)
7. [构建与运行](#7-构建与运行)
8. [性能调优指南](#8-性能调优指南)

---

## 1. 硬件架构背景

### 1.1 AI Core 双单元架构

每个 AI Core 包含两个可并行执行的计算单元，通过编译时宏 `__DAV_CUBE__` / `__DAV_VEC__` 区分：

```
┌──────────────────────────────────────────────────────┐
│                     AI Core                          │
├────────────────────────┬─────────────────────────────┤
│      Cube 单元         │     Vector 单元 (×2)        │
│   (__DAV_CUBE__)       │     (__DAV_VEC__)           │
│  ┌──────────────────┐  │  ┌───────────────────────┐  │
│  │     TMATMUL      │  │  │ SubBlock0  SubBlock1  │  │
│  │   矩阵乘累加     │  │  │ ┌────────┐ ┌────────┐ │  │
│  │                  │  │  │ │TREDUCE │ │TREDUCE │ │  │
│  │  GEMM tile 计算  │  │  │ │PINGPONG│ │PINGPONG│ │  │
│  └──────────────────┘  │  │ └────────┘ └────────┘ │  │
│                        │  └───────────────────────┘  │
└────────────────────────┴─────────────────────────────┘
```

- **Cube**：执行 TMATMUL/TMATMUL_ACC，处理 `[baseM, baseK] × [baseK, baseN]` 的矩阵乘
- **Vector (×2 SubBlocks)**：每个 SubBlock 处理 AllReduce 的一半行（`baseM/2 × baseN`）

### 1.2 内存层次

```
┌───────────────────────────────────────────────────────┐
│             Global Memory (GM) / Symmetric Heap       │
│  ┌─────────────────────────────────────────────────┐  │
│  │  GEMM 输出区: shmem[0 .. 1<<28)                 │  │
│  │  同步 flag 区: shmem[1<<28 ..)                   │  │
│  └─────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────┘
                        ↓ TLOAD (GM→L1)
┌───────────────────────────────────────────────────────┐
│                  L1 Buffer (~1MB)                      │
│  aMatTile[2]: [baseM, baseK*stepKa] × 2 (ping-pong)  │
│  bMatTile[2]: [baseK*stepKb, baseN] × 2 (ping-pong)  │
└───────────────────────────────────────────────────────┘
                        ↓ TEXTRACT (L1→L0)
┌──────────────────────┬────────────────────────────────┐
│  L0A (64KB total)    │  L0B (64KB total)              │
│  aTile[2]: 32KB each │  bTile[2]: 32KB each           │
│  (ping-pong)         │  (ping-pong)                   │
└──────────────────────┴────────────────────────────────┘
                        ↓ TMATMUL (Cube)
┌───────────────────────────────────────────────────────┐
│                 L0C Buffer (256KB)                     │
│              cTile: [baseM, baseN]                     │
└───────────────────────────────────────────────────────┘
                        ↓ TSTORE (L0C→GM)
                   写回 Symmetric Heap
```

---

## 2. 参数配置

所有参数在 `gemm_config.h` 中定义，Host 和 Kernel 共享：

```cpp
// gemm_config.h
constexpr uint32_t GEMM_M = 6144;          // 全局 M 维度
constexpr uint32_t GEMM_K = 6144;          // 全局 K 维度
constexpr uint32_t GEMM_N = 6144;          // 全局 N 维度

constexpr uint32_t SINGLE_CORE_M = 1536;   // 单核分到的 M
constexpr uint32_t SINGLE_CORE_K = 6144;   // 单核分到的 K（不分割）
constexpr uint32_t SINGLE_CORE_N = 1024;   // 单核分到的 N

constexpr uint32_t BLOCK_DIM = 24;         // AI Core 数量

constexpr uint32_t BASE_M = 128;           // tile M 维度
constexpr uint32_t BASE_K = 64;            // tile K 维度
constexpr uint32_t BASE_N = 256;           // tile N 维度

constexpr uint32_t STEP_KA = 4;            // L1 A 矩阵一次加载的 K 块数
constexpr uint32_t STEP_KB = 4;            // L1 B 矩阵一次加载的 K 块数
```

### 关键派生参数

| 参数 | 计算公式 | 值 | 含义 |
|------|----------|------|------|
| `mLoop` | `SINGLE_CORE_M / BASE_M` | 12 | 每核 M 方向 tile 数 |
| `nLoop` | `SINGLE_CORE_N / BASE_N` | 4 | 每核 N 方向 tile 数 |
| `kLoop` | `SINGLE_CORE_K / BASE_K` | 96 | 每个 tile 的 K 迭代次数 |
| 每核 tile 总数 | `mLoop × nLoop` | 48 | 每核需计算的 tile 总数 |
| 核布局 | `GEMM_M / SINGLE_CORE_M × GEMM_N / SINGLE_CORE_N` | 4×6=24 | 2D 核划分 |

### L0 Buffer 容量约束

- L0A/L0B 各 64KB，使用 32KB ping-pong 分割
- L0A tile: `BASE_M × BASE_K × sizeof(half) = 128 × 64 × 2 = 16KB ≤ 32KB` ✓
- L0B tile: `BASE_K × BASE_N × sizeof(half) = 64 × 256 × 2 = 32KB ≤ 32KB` ✓

### L1 Buffer 容量约束

- A 双缓冲: `2 × BASE_M × BASE_K × STEP_KA × sizeof(half) = 2 × 128 × 64 × 4 × 2 = 128KB`
- B 双缓冲: `2 × BASE_K × BASE_N × STEP_KB × sizeof(half) = 2 × 64 × 256 × 4 × 2 = 256KB`
- 总计: 384KB ≤ 1MB ✓

---

## 3. 高性能 GEMM 实现（Cube 路径）

整个 Cube 路径在 `#ifdef __DAV_CUBE__` 内编译。

### 3.1 工作分区 — InitGMOffsets

每个 core 拥有 C 矩阵的一个 `[SINGLE_CORE_M, SINGLE_CORE_N]` 块，读取对应的 A panel 和 B panel：

```cpp
// gemm_ar_kernel.cpp — InitGMOffsets
constexpr uint32_t mIter = m / singleCoreM;       // = 4 (M方向核数)
uint32_t mIterIdx = get_block_idx() % mIter;       // 当前核的 M 索引
uint32_t nIterIdx = get_block_idx() / mIter;       // 当前核的 N 索引

// 注意：GEMM 结果写入 shmem（对称堆），不是普通 GM
currentDst = reinterpret_cast<__gm__ T*>(shmem) + gmOffsetC;
currentSrc0 = src0 + gmOffsetA;   // A 矩阵在普通 GM
currentSrc1 = src1 + gmOffsetB;   // B 矩阵在普通 GM
```

**关键设计**：`currentDst` 指向 **shmem**（对称堆），而非普通 device memory。这样 GEMM 结果直接写到对称堆中，后续 Vec 路径通过 `shmem_ptr()` 可跨 rank 远程访问。

### 3.2 Tile 类型定义

```cpp
constexpr uint32_t L0_PINGPONG_BYTES = 32 * 1024;  // 32KB

// L1 缓冲（双缓冲，每次加载 stepKa 个 K 块）
using TileMatA = Tile<TileType::Mat, U, baseM, baseK * stepKa,
                      BLayout::ColMajor, baseM, baseK * stepKa, SLayout::RowMajor>;
using TileMatB = Tile<TileType::Mat, S, baseK * stepKb, baseN,
                      BLayout::RowMajor, baseK * stepKb, baseN, SLayout::ColMajor>;

// L0A/L0B（双缓冲，单个 K 块尺寸）
using LeftTile  = TileLeft<U, baseM, baseK, baseM, baseK>;
using RightTile = TileRight<S, baseK, baseN, baseK, baseN>;

// L0C 累加器
using ResTile = TileAcc<T, baseM, baseN, baseM, baseN>;
```

### 3.3 Buffer 地址分配

```cpp
// L1: A ping/pong + B ping/pong
TASSIGN(aMatTile[0], 0x0);
TASSIGN(aMatTile[1], 0x0 + baseM * baseK * stepKa * sizeof(U));
TASSIGN(bMatTile[0], 0x0 + baseM * baseK * stepKa * 2 * sizeof(U));
TASSIGN(bMatTile[1], 0x0 + baseM * baseK * stepKa * 2 * sizeof(U)
                          + baseK * baseN * stepKb * sizeof(U));

// L0A: ping(0~32KB) / pong(32KB~64KB)
TASSIGN(aTile[0], 0x0);
TASSIGN(aTile[1], 0x0 + L0_PINGPONG_BYTES);
// L0B: 同理
TASSIGN(bTile[0], 0x0);
TASSIGN(bTile[1], 0x0 + L0_PINGPONG_BYTES);
// L0C
TASSIGN(cTile, 0x0);
```

### 3.4 四级流水线 — ProcessKIteration

每个 tile `(i, j)` 需要遍历 `kLoop = 96` 次 K 迭代，实现四级双缓冲流水：

```
TLOAD (GM→L1)  →  TEXTRACT (L1→L0)  →  TMATMUL (Cube)  →  TSTORE (L0C→GM)
     ↑ 每 stepKa=4 次        ↑ 每次              ↑ 每次          ↑ K循环结束后
     ↑ 加载一批到L1          ↑ 切片出一个baseK    ↑ 累加         ↑ 写回 shmem
```

同步事件机制使用 `SetFlag` / `WaitFlag`：

| 同步点 | 含义 |
|--------|------|
| `PIPE_MTE2 → PIPE_MTE1` | TLOAD 完成，可以 TEXTRACT |
| `PIPE_MTE1 → PIPE_MTE2` | TEXTRACT 完成，可以复用 L1 buffer |
| `PIPE_MTE1 → PIPE_M` | TEXTRACT 完成，可以 TMATMUL |
| `PIPE_M → PIPE_MTE1` | TMATMUL 完成，可以复用 L0 buffer |
| `PIPE_M → PIPE_FIX` | TMATMUL 完成，可以 TSTORE |
| `PIPE_FIX → PIPE_M` | TSTORE 完成，可以开始下一个 tile |

```cpp
template <...>
AICORE inline void ProcessKIteration(
    uint32_t kIter, uint32_t i, uint32_t j,
    __gm__ U *currentSrc0, __gm__ S *currentSrc1,
    TileMatA aMatTile[2], TileMatB bMatTile[2],
    LeftTile aTile[2], RightTile bTile[2],
    ResTile &cTile,
    uint8_t &mte2DBFlag, uint8_t &mte1DBFlag)
{
    const uint32_t kModstepKa = kIter % stepKa;

    // ── TLOAD: 每 stepKa 次迭代加载一次 ──
    if (kModstepKa == 0) {
        WaitFlag<PIPE_MTE1, PIPE_MTE2>(mte2DBFlag);
        TLOAD(aMatTile[mte2DBFlag], gmA);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
        TLOAD(bMatTile[mte2DBFlag], gmB);
        SetFlag<PIPE_MTE2, PIPE_MTE1>(1);
        mte2DBFlag ^= 1;
    }

    // ── TEXTRACT: 从 L1 切出当前 K 块到 L0 ──
    WaitFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    if (kModstepKa == 0) WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);
    TEXTRACT(aTile[mte1DBFlag], aMatTile[currIdx], 0, kModstepKa * baseK);
    if (kModstepKa == 0) WaitFlag<PIPE_MTE2, PIPE_MTE1>(1);
    TEXTRACT(bTile[mte1DBFlag], bMatTile[currIdx], (kIter % stepKb) * baseK, 0);

    // ── TMATMUL: 矩阵乘累加 ──
    SetFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    WaitFlag<PIPE_MTE1, PIPE_M>(mte1DBFlag);
    if (kIter == 0) TMATMUL(cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    else            TMATMUL_ACC(cTile, cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
    SetFlag<PIPE_M, PIPE_MTE1>(mte1DBFlag);
    mte1DBFlag ^= 1;
}
```

### 3.5 StoreResult — 写回 shmem

K 循环完成后，将 L0C 中的 `[baseM, baseN]` tile 写回到 shmem 中对应位置：

```cpp
template <...>
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

## 4. AllReduce 通信实现（Vec 路径）

整个 Vec 路径在 `#ifdef __DAV_VEC__` 内编译，由 `RunAllReduceE2E` 实现。

### 4.1 AllReduce Tile 尺寸

每个 AI Core 有 2 个 Vector SubBlock，每个处理 tile 的一半行：

```cpp
// AllReduce tile: [baseM/2, baseN] = [64, 256]
// 每个 SubBlock 处理 64 行，两个 SubBlock 合并覆盖完整的 baseM=128 行
using NDValidShapeC = TileShape2D<T, baseM / 2, baseN, Layout::ND>;
using NDWholeShapeC = BaseShape2D<T, m, n, Layout::ND>;
using GlobalDataAR  = GlobalTensor<T, NDValidShapeC, NDWholeShapeC, Layout::ND>;
using TileDataAR    = Tile<TileType::Vec, T, baseM / 2, baseN, BLayout::RowMajor, -1, -1>;
```

### 4.2 UB Ping-Pong 分配

`TREDUCE_PINGPONG` 需要 3 个 UB tile：

```cpp
TileDataAR accTile(baseM / 2, baseN);    // 累加缓冲
TileDataAR recvTile(baseM / 2, baseN);   // 接收缓冲
TileDataAR dstTile(baseM / 2, baseN);    // 输出缓冲

TASSIGN(accTile,  0x0);
TASSIGN(recvTile, 0x0 + baseM / 2 * baseN * sizeof(T));
TASSIGN(dstTile,  0x0 + baseM / 2 * baseN * sizeof(T) * 2);
```

### 4.3 ParallelGroup 构建与跨 rank 数据访问

对于每个 tile `(i, j)`，需要构建包含所有 rank 对应位置数据的 `ParallelGroup`：

```cpp
// 当前 SubBlock 在 singleCore 区域内的偏移
auto tile_offset = i * baseM * n + get_subblockid() * baseM / 2 * n + j * baseN;
// 完整偏移 = core 偏移 + tile 偏移
auto total_offset = gmOffsetC + tile_offset;

// 构建跨 rank 的 ParallelGroup
GlobalDataAR tensors[16];
for (int r = 0; r < n_ranks && r < 16; ++r) {
    // shmem_ptr: 获取 rank r 的对称堆基址映射
    __gm__ T *rank_r_shmem = reinterpret_cast<__gm__ T*>(shmem_ptr(shmem, r));
    __gm__ T *dataPtr = rank_r_shmem + total_offset;
    tensors[r] = GlobalDataAR(dataPtr);
}
pto::comm::ParallelGroup<GlobalDataAR> pg(tensors, n_ranks, my_rank);
```

**关键**：`shmem_ptr(shmem, r)` 是核心 API，它将本地 shmem 地址映射到 rank `r` 的远程地址。每个 rank 的 GEMM 结果都写在自己的 shmem 中，通过 `shmem_ptr` 可以跨 rank 读取。

### 4.4 TREDUCE_PINGPONG

```cpp
GlobalDataAR dstGlobal(out_addr + tile_offset);
pto::comm::TREDUCE_PINGPONG(pg, dstGlobal, accTile, recvTile, dstTile, comm::ReduceOp::Sum);
```

执行流程：

```
时间 →
┌──────────┬──────────┬──────────┬──────────┬──────┐
│Load rank0│Load rank1│Load rank2│Load rank3│Store │
│→ accTile ││→ recvTile│→ recvTile│→ recvTile│→ out │
├──────────┼──────────┼──────────┼──────────┤      │
│          │ acc += r1│ acc += r2│ acc += r3│      │
└──────────┴──────────┴──────────┴──────────┴──────┘

Ping-pong 优化：Load 与 Add 流水重叠
┌──────────┬────────────┬────────────┬────────────┐
│Load r0   │Load r1     │Load r2     │Load r3     │ ← recvTile
│→accTile  │→recvTile   │→recvTile   │→recvTile   │
│          │acc+=recv   │acc+=recv   │acc+=recv   │ ← accTile
│          │            │            │Store acc   │ ← dstGlobal
└──────────┴────────────┴────────────┴────────────┘
```

---

## 5. Tile 级通算融合：TNOTIFY/TWAIT

### 5.1 核心思想

Cube 和 Vec 在同一个 kernel 中并行执行。Cube 每完成一个 tile 的 TSTORE，就通过 `TNOTIFY` 在 shmem flag 区设置标志；Vec 通过 `TWAIT` 轮询该标志，flag 到达后立即开始该 tile 的 AllReduce。

```
时间 →
Cube:  ┌─Tile 0─┐ ┌─Tile 1─┐ ┌─Tile 2─┐ ┌─Tile 3─┐ ...
       │K loop  │ │K loop  │ │K loop  │ │K loop  │
       │TSTORE  │ │TSTORE  │ │TSTORE  │ │TSTORE  │
       └──┬─────┘ └──┬─────┘ └──┬─────┘ └──┬─────┘
          │TNOTIFY    │TNOTIFY   │TNOTIFY   │TNOTIFY
          ↓           ↓          ↓          ↓
Vec:      ┌──TWAIT──┐ ┌─TWAIT─┐ ┌──TWAIT──┐ ┌─TWAIT─┐
          │TREDUCE  │ │TREDUCE│ │TREDUCE  │ │TREDUCE│
          │PINGPONG │ │PING.. │ │PINGPONG │ │PING.. │
          └─────────┘ └───────┘ └─────────┘ └───────┘
```

### 5.2 shmem Flag 布局

Flag 存放在 shmem 的高地址区域 `shmem + (1UL << 28)`，256MB 偏移处：

```
shmem 布局:
┌──────────────────────────────────────┐ offset 0
│         GEMM 输出数据区              │
│    M × N × sizeof(float)            │
│    = 6144 × 6144 × 4 = 144MB        │
├──────────────────────────────────────┤ offset 1<<28 = 256MB
│         TNOTIFY/TWAIT flag 区        │
│    每个 core 有 mLoop×nLoop 个 flag  │
│    共 BLOCK_DIM × 48 × sizeof(int32) │
└──────────────────────────────────────┘
```

Flag 索引方式（每个 flag 是一个 `int32_t`）：

```cpp
// Cube 端写 flag:
// block_idx * mLoop * nLoop + j * mLoop + i
__gm__ int32_t *flagPtr = shmemFlag + get_block_idx() * mLoop * nLoop + j * mLoop + i;
```

### 5.3 Cube 端 — TNOTIFY

每个 tile `(i, j)` 完成 StoreResult + PipeBarrier 后，设置对应 flag：

```cpp
// gemm_ar_kernel.cpp — RunGemmE2E 内循环
for (uint32_t i = 0; i < mLoop; i++) {
    for (uint32_t j = 0; j < nLoop; j++) {
        // K 循环
        for (uint32_t kIter = 0; kIter < kLoop; kIter++) {
            ProcessKIteration<...>(kIter, i, j, ...);
        }
        // 写回 shmem
        StoreResult<...>(cTile, currentDst, i, j);
        AscendC::PipeBarrier<PIPE_ALL>();

        // ★ 通知 Vec: tile (i,j) 已就绪
        {
            using GSignal = GlobalTensor<int32_t, ShapeDyn, StrideDyn, Layout::ND>;
            __gm__ int32_t *flagPtr = shmemFlag
                + get_block_idx() * mLoop * nLoop + j * mLoop + i;
            GSignal flagSignal(flagPtr, shape, stride);
            comm::TNOTIFY(flagSignal, 1, comm::NotifyOp::Set);
        }
    }
}
```

### 5.4 Vec 端 — TWAIT

Vec 在处理每个 tile 前，先等待**所有 rank** 的对应 flag 都被设置：

```cpp
// gemm_ar_kernel.cpp — RunAllReduceE2E 内循环
for (uint32_t i = 0; i < mLoop; i++) {
    for (uint32_t j = 0; j < nLoop; j++) {
        if (is_overlap) {
            // ★ 等待所有 rank 的 Cube 完成此 tile
            for (uint32_t r = 0; r < n_ranks; r++) {
                __gm__ int32_t *flagPtr =
                    reinterpret_cast<__gm__ int32_t*>(shmem_ptr(shmemFlag, r))
                    + get_block_idx() * mLoop * nLoop + j * mLoop + i;
                GSignal flagSignal(flagPtr, shape, stride);
                comm::TWAIT(flagSignal, 1, comm::WaitCmp::EQ);
            }
        }

        // AllReduce this tile
        // ... 构建 ParallelGroup, TREDUCE_PINGPONG ...
    }
}
```

**同步量分析**：
- 每个 tile 需要等 `n_ranks` 个 flag = 4 次 TWAIT
- 每核 48 tiles → **每核 192 次 TWAIT**
- 每次 TWAIT 轮询远程 shmem flag，有非零延迟

### 5.5 Overlap 时序示例（2 rank 简化）

```
Rank 0 Cube: ┌─T0─┐ ┌─T1─┐ ┌─T2─┐ ┌─T3─┐ ...
             └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
                │N      │N      │N      │N       (TNOTIFY)

Rank 1 Cube: ┌─T0─┐ ┌─T1─┐ ┌─T2─┐ ┌─T3─┐ ...
             └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
                │N      │N      │N      │N       (TNOTIFY)

Rank 0 Vec:  ────┌W0┐┌──AR0──┐┌W1┐┌──AR1──┐ ...  (TWAIT→TREDUCE)
Rank 1 Vec:  ────┌W0┐┌──AR0──┐┌W1┐┌──AR1──┐ ...

N = TNOTIFY, W = TWAIT, AR = TREDUCE_PINGPONG
Vec 必须同时等到 Rank0 和 Rank1 的 flag 才能开始 AR
```

---

## 6. Kernel 入口与 Host 端

### 6.1 GemmAllReduce Kernel

单一 `__global__ AICORE` kernel 同时包含 Cube 和 Vec 路径：

```cpp
// gemm_ar_kernel.cpp
template <typename T, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, ...>
__global__ AICORE void GemmAllReduce(
    __gm__ uint8_t *out,     // AllReduce 最终输出（普通 GM）
    __gm__ uint8_t *src0,    // A 矩阵
    __gm__ uint8_t *src1,    // B 矩阵
    __gm__ uint8_t *shmem,   // 对称堆基址
    bool is_overlap)          // 是否启用 tile 级 overlap
{
#ifdef __DAV_CUBE__
    RunGemmE2E<float, half, half, float, blockDim, m, k, n, ...>(
        reinterpret_cast<__gm__ float*>(out),
        reinterpret_cast<__gm__ half*>(src0),
        reinterpret_cast<__gm__ half*>(src1),
        shmem, is_overlap);
#endif

#ifdef __DAV_VEC__
    RunAllReduceE2E<float, m, n, singleCoreM, singleCoreN, baseM, baseN>(
        shmem, out, is_overlap);
#endif
}
```

**注意**：
- Cube 将 GEMM 结果写入 `shmem`
- Vec 从所有 rank 的 `shmem` 读取，reduce 后写入 `out`
- `is_overlap=true` 时启用 TNOTIFY/TWAIT；`is_overlap=false` 时 Vec 跳过 TWAIT

### 6.2 Host 端 Kernel Launch

```cpp
// gemm_ar_kernel.cpp — Host 可见的 launch 函数
template <typename T>
void LaunchGEMME2E(uint8_t *out, uint8_t *src0, uint8_t *src1,
                   uint8_t *shmem, void *stream, bool is_overlap)
{
    GemmAllReduce<T, BLOCK_DIM, GEMM_M, GEMM_K, GEMM_N,
                  SINGLE_CORE_M, SINGLE_CORE_K, SINGLE_CORE_N,
                  BASE_M, BASE_K, BASE_N,
                  STEP_M, STEP_KA, STEP_KB, STEP_N>
        <<<BLOCK_DIM, nullptr, stream>>>(out, src0, src1, shmem, is_overlap);
}
```

### 6.3 多进程执行模型

Host 端使用 `fork()` 为每个 rank 创建独立进程：

```cpp
// main.cpp — 多进程启动
bool RunGemmMultiProcess(int n_ranks, int n_devices, ...) {
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid == 0) {
            // 子进程：初始化 shmem → 分配内存 → launch kernel → 验证
            bool ok = RunGemmKernel(first_rank_id + r, n_ranks, ...);
            _exit(ok ? 0 : 1);
        }
        pids.push_back(pid);
    }
    // 父进程等待所有子进程
    for (pid_t p : pids) waitpid(p, &status, 0);
}
```

### 6.4 单 Rank 执行流程

```cpp
bool RunGemmKernel(int rank_id, int n_ranks, ...) {
    // 1. 初始化 ACL
    shmem_set_conf_store_tls(false, nullptr, 0);
    aclInit(nullptr);
    aclrtSetDevice(device_id);
    aclrtCreateStream(&stream);

    // 2. 初始化 Shmem
    ShmemEnv env{rank_id, n_ranks, "tcp://127.0.0.1:8778"};
    ShmemInitFromEnv(env);

    // 3. 分配内存
    aclrtMalloc(&src0Device, aFileSize, ...);
    aclrtMalloc(&src1Device, bFileSize, ...);
    aclrtMalloc(&dstDevice,  cFileSize, ...);
    uint8_t *shmemDevice = (uint8_t*)ShmemCalloc((1UL << 28), sizeof(uint32_t));

    // 4. 加载输入数据并搬到 device
    ReadFile("../input/x1_gm.bin", ...);
    aclrtMemcpy(src0Device, ..., ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 全局同步后 launch
    ShmemBarrierAll();
    LaunchGEMME2E<uint16_t>(dstDevice, src0Device, src1Device,
                            shmemDevice, stream, /*is_overlap=*/true);
    aclrtSynchronizeStream(stream);

    // 6. 结果回传并验证
    aclrtMemcpy(dstHost, ..., ACL_MEMCPY_DEVICE_TO_HOST);
    // golden = A * B, result = AllReduce(A*B across ranks) = n_ranks * A*B
    // 验证时 result / n_ranks == golden
}
```

---

## 7. 构建与运行

### 7.1 环境要求

```bash
# 必须设置的环境变量
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export SHMEM_HOME_PATH=<path-to-shmem>
```

### 7.2 编译选项

`CMakeLists.txt` 中的关键编译选项：

```cmake
# Kernel 编译选项
target_compile_options(${NAME}_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220      # ★ 启用 Cube/Vector 混合架构
    -DMEMORY_BASE                    # 启用 device 端 shmem_ptr 等 API
    -std=c++17
)

# Kernel 链接
target_link_libraries(${NAME}_kernel PRIVATE shmem runtime)
target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

# Host 链接
target_link_libraries(${NAME} PRIVATE
    ${NAME}_kernel
    ascendcl shmem stdc++ m pthread ...
)
```

**关键**：`--cce-aicore-arch=dav-c220` 编译器会将 kernel 分别编译成 Cube 版本（定义 `__DAV_CUBE__`）和 Vec 版本（定义 `__DAV_VEC__`），运行时两个版本在同一个 AI Core 上并行执行。

### 7.3 运行

```bash
# 基本运行
bash run.sh -r npu -v Ascend910B4

# 带 debug 输出
bash run.sh -r npu -v Ascend910B4 -d
```

### 7.4 数据生成

```bash
cd scripts
python gen_data.py
# 生成：
#   ../input/x1_gm.bin   — A 矩阵 (fp16)
#   ../input/x2_gm.bin   — B 矩阵 (fp16)
#   ../output/golden.bin  — 参考输出 (fp32)
```

---

## 8. 性能调优指南

### 8.1 GEMM 优化要点

| 优化项 | 当前配置 | 说明 |
|--------|----------|------|
| 核划分 | 4×6 = 24 核 | C 矩阵按 M×N 2D 划分 |
| Base tile | [128, 64, 256] | 最大化 Cube 计算密度 |
| L1 step | stepKa=stepKb=4 | 减少 TLOAD 次数（每 4 个 K 块一次） |
| 双缓冲 | L1 + L0A/L0B 均双缓冲 | 隐藏数据搬运延迟 |
| TMATMUL_ACC | 首次 TMATMUL + 后续 ACC | 避免 L0C 清零开销 |

### 8.2 Overlap 同步开销

当前 `gemm_ar` 实现是 **per-tile** 同步 —— 每个 tile 一次 TNOTIFY + `n_ranks` 次 TWAIT：

- TNOTIFY：48 次（每核 48 tiles，每 tile 一次）
- TWAIT：48 × 4 = **192 次**（每 tile 等 4 个 rank 的 flag）
- 每次 TWAIT 轮询远程 shmem flag，有非零延迟

这是最细粒度的同步方式，提供最大的 overlap 机会，但同步开销也最大。

**可能的优化方向 —— 分组同步（Grouped TNOTIFY/TWAIT）**：

将多个 tile 合成一组，每组只做一次 TNOTIFY/TWAIT，减少同步次数：

| 分组策略 | TNOTIFY 次数 | TWAIT 次数 | 同步减少 | 代价 |
|----------|-------------|-----------|----------|------|
| per-tile（当前） | 48 | 192 | — | 同步开销大 |
| per-row (GROUP=4) | 12 | 48 | 4× | Vec 需多等 ~0.26ms |
| per-chunk (GROUP=12) | 4 | 16 | 12× | Vec 需多等 ~0.8ms |
| all-at-once (GROUP=48) | 1 | 4 | 48× | 无 tile 级 overlap |

粒度与 overlap 的权衡：
- 粒度越细 → Vec 越早开始（更多 overlap）→ 但同步开销越大
- 粒度越粗 → 同步开销越小 → 但 Vec 要等更久才能开始
- 当 GEMM 远快于 AllReduce 时（如 3ms vs 8ms），适当放粗粒度对 overlap 影响很小

> **注**：目前 `gemm_ar` 和 `gemm_ar_performance` 均使用 per-tile 同步。分组同步是一个可探索的优化方向，需修改 kernel 中的 TNOTIFY/TWAIT 逻辑。

### 8.3 AllReduce 开销分析

AllReduce 时间 = TWAIT 等待 + TREDUCE_PINGPONG 计算

- **TWAIT 开销**：每次轮询远程 shmem flag，延迟取决于互联带宽和延迟
- **TREDUCE_PINGPONG 开销**：= 远程 TLOAD × (n_ranks-1) + 本地 Vec Add × (n_ranks-1) + TSTORE
- Tile 大小 `[64, 256] × sizeof(float) = 64KB`，每个 tile reduce 需搬移 `64KB × 4 ranks = 256KB`

### 8.4 Overlap 效率上限

在 GEMM 远快于 AllReduce 时，overlap 效率受限于两者的时间比：

```
Overlap 隐藏的 AR 时间 ≤ GEMM 总时间

如果 GEMM = 3ms, AR = 9ms:
  - Non-overlap 总时间 = 3 + 9 = 12ms
  - Overlap 理论最优 = max(3, 9) = 9ms (隐藏 3ms AR)
  - AR 被隐藏比例 = 3/9 = 33%
  - 加速比 = 12/9 = 1.33×
```

要提高 overlap 效率：
1. **增大 GEMM 耗时**：增大矩阵规模或 K 维度
2. **减小 AR 耗时**：增大 AllReduce tile 尺寸（减少 tile 数从而减少 TWAIT 次数）
3. **减少 rank 数**：更少的 rank 意味着更少的远程读取

### 8.5 调试技巧

| 方法 | 说明 |
|------|------|
| `cce::printf` | 编译时加 `-d` 选项启用，可在 Cube/Vec 内打印 |
| `is_overlap=false` | 跳过 TWAIT，Vec 直接做 AllReduce（需先确保 GEMM 全部完成） |
| 分步验证 | 先验证纯 GEMM 正确性，再加 AllReduce |
| `gemm_ar_performance` | 性能版本，支持 warmup、多次迭代、overlap vs non-overlap 对比 |

---

## 附录 A：shmem 通信工具（common.hpp）

`common.hpp` 封装了 shmem API，支持两种后端：

| 函数 | ASCEND_SHMEM | CANN_SHMEM |
|------|-------------|------------|
| `ShmemInit(env)` | `shmem_set_attr` + `shmem_init_attr` | `aclshmemx_init_attr` |
| `ShmemMalloc(bytes)` | `shmem_malloc` | `shmem_malloc` |
| `ShmemCalloc(count, size)` | `shmem_calloc` | `shmem_calloc` |
| `ShmemBarrierAll()` | `shmem_barrier_all` | `aclshmem_barrier_all` |
| `ShmemFinalize()` | `shmem_finalize` | `shmem_finalize` |
| `ShmemPtr(ptr, pe)` [device] | `shmem_ptr` | `aclshmem_ptr` |

通过编译宏 `ASCEND_SHMEM` 或 `CANN_SHMEM` 选择后端（CMake 自动检测）。

## 附录 B：参考实现

| 示例 | 路径 | 说明 |
|------|------|------|
| Tile 级融合 GEMM+AR | `kernels/manual/a2a3/gemm_ar/` | 本文档对应的实现 |
| 性能测试版本 | `kernels/manual/a2a3/gemm_ar_performance/` | 支持 warmup、多次迭代、overlap/non-overlap 对比 |

---

## 版本历史

| 日期 | 变更 |
|------|------|
| 2026-02-05 | 初始版本 |
| 2026-02-06 | 根据实际 gemm_ar 代码重写，修正架构描述、同步机制、完整代码示例 |
