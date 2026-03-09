# GEMM AllReduce Demo - Device-Side Computation and Communication

This Demo implements a GEMM + AllReduce operator where **all computation and communication execute on the Device side (NPU)** without Host involvement. It uses `TPUT<AtomicAdd>` to implement remote atomic addition, directly completing AllReduce operations on the Device side.

## Data Parallel Mode

This implementation supports **Data Parallel Mode**:

### Matrix Partitioning Strategy

Assuming total input matrices:
- **Matrix A**: M × K (default: 16384 × 4096)
- **Matrix B**: K × N (default: 4096 × 4096)

For r ranks, matrices are split along the **K dimension**:
- **Each rank's A matrix**: M × (K/r) (e.g., 16384 × 512 when r=8)
  - A matrix is split along **columns** (K dimension is the column dimension)
- **Each rank's B matrix**: (K/r) × N (e.g., 512 × 4096 when r=8)
  - B matrix is split along **rows** (K dimension is the row dimension)

**Constraint**: K must be divisible by the number of ranks (`G_K % n_ranks == 0`)

### Computation Flow

Each rank computes:
```
C_part = A_part[M × (K/r)] × B_part[(K/r) × N] = C_part[M × N]
```

### AllReduce Operation

All ranks' outputs are summed via AllReduce:
```
C_final = sum(C_part_0 + C_part_1 + ... + C_part_r-1) = C_final[M × N]
```

### Mathematical Correctness

Since matrix multiplication satisfies:
```
A × B = [A_0 | A_1 | ... | A_r-1] × [B_0; B_1; ...; B_r-1]
       = A_0 × B_0 + A_1 × B_1 + ... + A_r-1 × B_r-1
```

Therefore, the data parallel mode result matches the full matrix multiplication result.

## Architecture Design

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│   Compute Stream (Cube)              Communication Stream (Vec)              │
│                                                                              │
│   ┌─────────────────┐                                                        │
│   │ Compute Tile 0  │──┐                                                     │
│   └─────────────────┘  │                                                     │
│   ┌─────────────────┐  │ Queue     ┌─────────────────────────────────────┐   │
│   │ Compute Tile 1  │──┼──────────│ Poll Queue (TTEST/TWAIT)             │   │
│   └─────────────────┘  │          │ → TPUT<AtomicAdd> to all ranks      │   │
│   ┌─────────────────┐  │          │   (including self)                   │   │
│   │ Compute Tile 2  │──┘          │ → reduced_output accumulates sum     │   │
│   └─────────────────┘             │ → No explicit reduce phase needed   │   │
│          ...                      └─────────────────────────────────────┘   │
│                                                                              │
│   Key Features:                                                                 │
│   - All computation and communication execute on Device side (AICORE identifier) │
│   - Uses TPUT<AtomicAdd> to implement remote atomic addition, no extra reduce    │
│   - Compute kernel signals comm kernel via queue after completing each tile     │
│   - Comm kernel polls queues and immediately performs TPUT<AtomicAdd> when ready │
│   - All ranks' data are directly accumulated to reduced_output via atomic add   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Innovation: TPUT<AtomicAdd> for Device-Side AllReduce

Traditional AllReduce implementations require:
1. Each rank sends data to remote ranks
2. Receive data from remote ranks
3. Perform reduce operation locally

This implementation uses **TPUT<AtomicAdd>** directly:
- Each rank atomically adds its GEMM result to **all ranks'** `reduced_output` (including itself) via `TPUT<AtomicAdd>`
- All operations complete on Device side without Host involvement
- Avoids explicit reduce phase, simplifying implementation and improving performance

## Core Components

### 1. Compute Kernel (`gemm_compute_kernel.cpp`)

- **Architecture**: Cube-only (dav-c220-cube)
- **AICORE Identifier**: All computation executes on Device side
- **Function**: Executes GEMM computation (C = A × B)
- **Signaling Mechanism**: After completing each tile, enqueues tile index to `MultiBlockQueueSet`
- **Multi-block Support**: Multiple compute blocks execute in parallel, each with its own queue (no lock contention)
- **Tile Processing**: Uses **two-level double buffering** K-loop optimization for L1 cache reuse and full pipeline overlap
  - **L1-level Double Buffering** (`mte2DBFlag`): Loads every 4 K iterations, caching 4 K-slices
  - **L0-level Double Buffering** (`mte1DBFlag`): Uses TEXTRACT to extract single K-slice from L1 cache
  - **Performance Optimization**: 4x reduction in GM→L1 DMA frequency, three-stage pipeline (MTE2/MTE1/Cube) fully parallel

### 2. Communication Kernel (`comm_kernel.cpp`)

- **Architecture**: Vector-only (dav-c220-vec)
- **AICORE Identifier**: All communication executes on Device side
- **Function**: 
  1. **Polling Phase**: Uses `TTEST`/`TWAIT` to poll compute kernel queues, detecting ready tiles
  2. **Atomic Add Phase**: Uses `TPUT<AtomicAdd>` to atomically add tile data to all ranks' `reduced_output`
     - Each rank adds its data to all ranks (including itself)
     - After completion, each rank's `reduced_output` contains the sum of all ranks' data
  3. **Synchronization Phase**: Uses `ShmemDeviceBarrierAll()` to ensure all atomic operations complete
- **Multi-block Support**: Communication blocks share queue polling tasks, improving parallelism

### 3. Queue Mechanism (`ready_queue.hpp`)

- **Design**: One independent queue per compute block (single producer)
- **Advantages**: No atomic operation overhead, O(B) polling instead of O(N) flag polling
- **Synchronization**: Uses PTO `TTEST`/`TWAIT` instructions for hardware-friendly synchronization
- **Data Structure**: `MultiBlockQueueSet` manages multiple `PerBlockQueue`

## Data Flow

### AllReduce Data Flow (Using TPUT<AtomicAdd>)

```
Rank 0                                    Rank 1
┌────────────────────┐                   ┌────────────────────┐
│ Compute: C0 = A0×B │                   │ Compute: C1 = A1×B │
│         ↓         │                    │         ↓         │
│ Queue: tile ready  │                   │ Queue: tile ready  │
│         ↓         │                    │         ↓         │
│ TPUT<AtomicAdd>    │                   │ TPUT<AtomicAdd>    │
│ C0 → reduced_out  │                   │ C1 → reduced_out  │
│ (all ranks)        │                   │ (all ranks)        │
│         ↓         │                    │         ↓         │
│ reduced_output =   │                   │ reduced_output =   │
│ C0 + C1 (sum)     │                   │ C0 + C1 (sum)     │
└────────────────────┘                   └────────────────────┘
```

### Key Data Structures

- **`shmem_output`**: Local GEMM result for each rank (symmetric heap memory)
  - Size: `M × N × sizeof(float)`
  - Stores each rank's partial result `C_part = A_part × B_part`
- **`reduced_output`**: Final AllReduce result (symmetric heap memory)
  - Size: `M × N × sizeof(float)`
  - Initialized to 0
  - Accumulates all ranks' data via `TPUT<AtomicAdd>`
  - After completion, each rank's `reduced_output` contains the same result (sum of all ranks' data)
  - **Note**: `TPUT<AtomicAdd>` writes directly to `reduced_output`, no intermediate buffer needed
- **`recv_buffers`**: Minimal debug buffer (1MB, for debugging only)
  - **Not used in actual communication**: Actual communication uses `TPUT<AtomicAdd>` to write directly to `reduced_output`
  - Memory optimization: Reduced from `nranks × outputSize` to 1MB, enabling 8+ rank tests

## 性能分析逻辑

### 性能测试流程

性能测试包含以下阶段：

#### 1. Warmup 阶段
```cpp
for (int i = 0; i < WARMUP_ITERS; ++i) {
    resetState();  // 重置所有状态
    // 启动 compute 和 comm kernels（并行）
    // 同步等待完成
}
```
- **目的**: 预热硬件，稳定性能
- **默认迭代次数**: 5 次

#### 2. Compute-Only 基准测试
```cpp
// 只运行 compute kernel，测量纯计算时间
auto compute_start = std::chrono::high_resolution_clock::now();
launchGemmCompute(...);
aclrtSynchronizeStream(computeStream);
auto compute_end = std::chrono::high_resolution_clock::now();
compute_time_us = duration_cast<microseconds>(compute_end - compute_start);
```
- **目的**: 测量纯 GEMM 计算时间，作为性能参考
- **用途**: 计算 GFLOPS，评估计算性能

#### 3. Sequential 执行测试
```cpp
for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
    resetState();
    auto seq_start = std::chrono::high_resolution_clock::now();
    
    // 先执行 compute
    launchGemmCompute(...);
    aclrtSynchronizeStream(computeStream);
    
    // 再执行 comm
    launchGemmComm(...);
    aclrtSynchronizeStream(commStream);
    
    auto seq_end = std::chrono::high_resolution_clock::now();
    sequential_times_us.push_back(duration);
}
```
- **目的**: 测量串行执行时间（计算完成后再通信，无重叠）
- **默认迭代次数**: 100 次
- **统计指标**: 平均值、最小值、最大值、标准差

#### 4. Pipelined 执行测试
```cpp
for (int iter = 0; iter < MEASURE_ITERS; ++iter) {
    resetState();
    auto pipe_start = std::chrono::high_resolution_clock::now();
    
    // 同时启动 compute 和 comm（并行执行）
    launchGemmCompute(...);
    launchGemmComm(...);
    
    aclrtSynchronizeStream(computeStream);
    aclrtSynchronizeStream(commStream);
    
    auto pipe_end = std::chrono::high_resolution_clock::now();
    pipelined_times_us.push_back(duration);
}
```
- **目的**: 测量流水线执行时间（计算和通信重叠）
- **默认迭代次数**: 100 次
- **统计指标**: 平均值、最小值、最大值、标准差

#### 5. Performance Metrics Calculation

```cpp
// GFLOPS calculation (Data Parallel Mode)
// Per-rank FLOPS: 2 * M * (K/r) * N, where r is the number of ranks
uint32_t k_per_rank = G_K / n_ranks;
double gemm_flops_per_rank = 2.0 * M * k_per_rank * N;  // Per-rank floating point operations
double gemm_flops_total = 2.0 * M * K * N;  // Total system floating point operations

// Per-rank GFLOPS (actual computation performance of each rank)
double compute_gflops = gemm_flops_per_rank / (compute_time_us * 1e-6) / 1e9;

// Total system GFLOPS (total throughput of all ranks combined)
double seq_gflops = gemm_flops_total / (seq_avg * 1e-6) / 1e9;
double pipe_gflops = gemm_flops_total / (pipe_avg * 1e-6) / 1e9;

// Speedup calculation
double speedup = seq_avg / pipe_avg;
double time_saved = seq_avg - pipe_avg;
double time_saved_percent = (time_saved / seq_avg) * 100.0;
```

**Note**: In data parallel mode, each rank only computes part of the K dimension (K/r), therefore:
- **Per-rank FLOPS** = `2 * M * (K/r) * N`: Used to evaluate single rank's computation performance
- **Total system FLOPS** = `2 * M * K * N`: Used to evaluate total system throughput

### Performance Report Output

The following performance data will be output after execution:

```
[PERF] Performance Results (n_iters=100):

  Component Breakdown (Reference - Per Rank):
    Compute Kernel:    46.650 us (min: 45.123, max: 48.234, std: 0.567)
    Compute GFLOPS:    12345.678 GFLOPS (avg), 12500.123 GFLOPS (best) [per rank, M×K/r×N]

  Sequential Execution (Compute -> Comm, no overlap):
    Total Time:      268.322 us (min: 253.661, max: 281.971, std: 5.829)
    Compute Time:     46.789 us (min: 45.234, max: 48.567, std: 0.623)
    Compute GFLOPS:   12340.123 GFLOPS (compute-only efficiency)
    Comm Time:       221.533 us (min: 208.427, max: 233.404, std: 4.206)
    Comm Data:        0.512 GB/rank (per rank sends to all 8 ranks)
    Comm Bandwidth:   2310.456 GB/s
    Throughput:       12005.052 GFLOPS (total efficiency, M×K×N)

  Pipelined Execution (Compute || Comm, with overlap):
    Total Time:      243.533 us (min: 230.852, max: 267.842, std: 5.574)
    Compute Done:     46.823 us (min: 45.456, max: 48.123, std: 0.589)
    Compute GFLOPS:   12338.234 GFLOPS (compute-only efficiency)
    Comm Done:        243.456 us (min: 230.789, max: 267.234, std: 5.623)
    Comm Data:        0.512 GB/rank (per rank sends to all 8 ranks)
    Comm Bandwidth:   2103.789 GB/s
    Throughput:       13227.075 GFLOPS (total efficiency, M×K×N)

  Performance Comparison:
    Speedup:           1.102x (Pipelined vs Sequential)
    Time Saved:        24.790 us (9.239%)
    Overlap Time:      24.856 us
    Overlap Efficiency: 11.234%

  Compute Efficiency Analysis:
    Pure Compute:       12345.678 GFLOPS
    Sequential Compute: 12340.123 GFLOPS (99.955% of pure)
    Pipelined Compute:  12338.234 GFLOPS (99.940% of pure)
```

**Performance Metrics Explanation**:
- **Compute GFLOPS (per rank)**: Based on `2 * M * (K/r) * N`, evaluates single rank's computation performance
- **Throughput (total)**: Based on `2 * M * K * N`, evaluates total system throughput
- **Comm Data**: Data sent per rank = `M × N × sizeof(float) × n_ranks` (sent to all ranks, including self)

### Performance Analysis Metrics

| Metric | Description |
|--------|-------------|
| **Compute Kernel** | Pure GEMM computation time, used as performance upper bound reference |
| **Sequential Execution** | Serial execution time (compute→comm), no overlap |
| **Pipelined Execution** | Pipelined execution time (compute\|\|comm), with overlap |
| **Speedup** | Speedup ratio of pipelined vs sequential |
| **Time Saved** | Time saved by pipelining (absolute value and percentage) |
| **Throughput (GFLOPS)** | Throughput based on total execution time |
| **Std Dev** | Standard deviation, reflects performance stability |

### Performance Optimization Highlights

1. **Compute-Communication Overlap**: Hide communication latency through dual-stream parallel execution
2. **Fine-grained Synchronization**: Use queue mechanism to achieve tile-level fine-grained synchronization
3. **Device-Side Execution**: All operations execute on Device side, avoiding Host-Device data transfer overhead
4. **Atomic Operation Optimization**: Use `TPUT<AtomicAdd>` to directly implement AllReduce, avoiding additional reduce phase

## Parameter Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| G_M | 16384 | Matrix M dimension (global) |
| G_K | 4096 | Matrix K dimension (global, will be split across ranks) |
| G_N | 4096 | Matrix N dimension (global) |
| G_BASE_M | 128 | Tile M dimension |
| G_BASE_K | 64 | Tile K dimension |
| G_BASE_N | 256 | Tile N dimension |
| COMPUTE_BLOCK_NUM | 32 | Number of compute blocks |
| COMM_BLOCK_NUM | 40 | Number of communication blocks |
| WARMUP_ITERS | 5 | Warmup iteration count |
| MEASURE_ITERS | 100 | Performance measurement iteration count |

**Data Parallel Mode Parameters** (calculated at runtime):
- `k_per_rank = G_K / n_ranks`: K dimension size per rank
- Input per rank: `A_part[M, k_per_rank]` and `B_part[k_per_rank, N]`
- Output per rank: `C_part[M, N]` (full size, but contains only partial computation result)

## 构建与运行

```bash
# 设置环境变量
source /path/to/set_env.sh
export SHMEM_HOME_PATH=/path/to/shmem

# 构建并运行
./run.sh -r npu -v Ascend910B1

# 或手动构建
mkdir build && cd build
cmake -DCMAKE_C_COMPILER=bisheng -DCMAKE_CXX_COMPILER=bisheng \
      -DRUN_MODE=npu -DSOC_VERSION=Ascend910B1 ..
make -j16
./gemm_allreduce
```

## 文件结构

```
gemm_allreduce/
├── CMakeLists.txt           # 构建配置
├── README.md                # 本文档
├── run.sh                   # 构建运行脚本
├── main.cpp                 # 主程序（双流启动逻辑）
├── gemm_compute_kernel.cpp  # 计算内核（Cube，AICORE）
├── comm_kernel.cpp          # 通信内核（Vec，AICORE）
├── ready_queue.hpp          # 队列机制
├── common.hpp               # 公共定义
├── input/                   # 输入数据
├── output/                  # 输出数据
└── scripts/
    └── gen_data.py          # 数据生成脚本
```

## 与 gemm_allgather 的区别

| 特性 | gemm_allgather | gemm_allreduce |
|------|----------------|----------------|
| 通信模式 | TPUT 到远端 | TPUT<AtomicAdd> 到所有 rank |
| 数据分布 | 每个 rank 计算不同行 | 每个 rank 计算相同位置 |
| 结果 | 收集所有 rank 的数据 | 归约所有 rank 的数据（求和） |
| Reduce 操作 | 无 | 通过 TPUT<AtomicAdd> 自动完成 |
| 信号量 | 队列信号 | 队列信号 |

## 技术要点

### 1. TPUT<AtomicAdd> Usage

```cpp
// 将本地数据原子加到所有 rank 的 reduced_output（包括自己）
// 每个 rank 将自己的 tile 数据原子加到所有 rank 的 reduced_output
for (int r = 0; r < nranks; ++r) {
    __gm__ float *dst_ptr = ShmemPtr(reduced_output, r) + tile_offset;
    Global dstG(dst_ptr, tileShape, tileStride);
    // TPUT<AtomicAdd> 直接写入 reduced_output，无需中间缓冲区
    pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pingTile, pongTile);
}
// 所有 rank 执行完成后，每个 rank 的 reduced_output 都包含相同的结果
// reduced_output = sum(C_part_0 + C_part_1 + ... + C_part_r-1)
```

**Key Features**:
- **Direct Write**: `TPUT<AtomicAdd>` writes directly to `reduced_output`, no `recv_buffers` needed
- **Atomic Operation**: Ensures correctness of concurrent writes from multiple ranks
- **Memory Optimization**: Avoids `nranks × outputSize` intermediate buffers, only needs 1MB debug buffer

### 2. Device-Side Synchronization

- Use `ShmemDeviceBarrierAll()` to ensure all ranks' atomic operations complete
- Use `pipe_barrier(PIPE_ALL)` to ensure all pipeline stages complete

### 3. Queue Polling Optimization

- Use `TTEST` for non-blocking checks
- Use `TWAIT` for blocking waits
- Poll multiple queues to improve parallelism

## TWAIT 阻塞等待优化详解

### 优化原理

通信核在等待计算核完成 tile 时，使用 PTO 硬件指令 `TWAIT` 进行阻塞等待。该优化通过直接使用 `TWAIT` 而不是 `TTEST + TWAIT` 模式，减少了指令开销并简化了代码逻辑。

### 优化前 vs 优化后

**优化前**（TTEST + TWAIT 模式）：
```cpp
// 先使用 TTEST 检查数据是否已就绪
if (!pto::comm::TTEST(sig, heads[wait_queue] + 1, pto::comm::WaitCmp::GE)) {
    // 数据未就绪，再使用 TWAIT 阻塞等待
    pto::comm::TWAIT(sig, heads[wait_queue] + 1, pto::comm::WaitCmp::GE);
}
```

**优化后**（直接 TWAIT）：
```cpp
// 直接使用 TWAIT，硬件级别自动处理数据已就绪的情况
pto::comm::TWAIT(sig, heads[wait_queue] + 1, pto::comm::WaitCmp::GE);
```

### 工作原理

1. **TWAIT 硬件行为**：
   - `TWAIT` 是 PTO 硬件指令，在硬件级别检查信号条件
   - 如果条件已满足（数据已就绪），`TWAIT` 会立即返回，不会阻塞
   - 如果条件未满足（数据未就绪），`TWAIT` 会硬件级别阻塞，直到条件满足

2. **信号机制**：
   - 使用 `queue->count` 作为信号变量
   - 计算核完成 tile 后，通过 `MultiBlockEnqueue` 增加 `count`
   - 通信核等待 `count >= heads[wait_queue] + 1` 的条件

3. **自动唤醒**：
   - 当计算核更新 `count` 时，硬件自动唤醒等待的通信核
   - 无需软件轮询，减少 CPU 开销

### 优化流程

```
通信核轮询流程：
┌─────────────────────────────────────────────────────────┐
│ 1. 轮询所有分配的队列（使用 PerBlockQueueTryDequeue）  │
│    - 使用 TTEST 非阻塞检查                             │
│    - 如果找到就绪的 tile，立即处理                      │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ 是否找到就绪的 tile? │
              └───────┬───────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
        是│                       │否
          │                       │
          ▼                       ▼
    ┌──────────┐    ┌──────────────────────────────┐
    │ 处理 tile │    │ 找到有 pending tile 的队列    │
    │ TPUT操作  │    │ 直接使用 TWAIT 阻塞等待      │
    └──────────┘    │ - 数据已就绪：立即返回         │
                    │ - 数据未就绪：硬件阻塞等待     │
                    └──────────────────────────────┘
```

### 性能收益

- **减少指令开销**：移除了冗余的 `TTEST` 调用
- **简化代码逻辑**：直接使用 `TWAIT`，代码更简洁
- **硬件级别优化**：利用硬件自动处理，无需软件判断
- **保持功能正确性**：`TWAIT` 在数据已就绪时会立即返回，不影响性能

### 关键代码实现

```cpp
// 在 comm_kernel.cpp 中
if (wait_queue >= 0) {
    // TWAIT optimization: directly use TWAIT for blocking wait
    // TWAIT will return immediately if data is already available (hardware-level check)
    // This reduces the overhead of TTEST + TWAIT pattern
    volatile __gm__ PerBlockQueue* pq = GetMyBlockQueue(qset, wait_queue);
    pto::comm::Signal sig(const_cast<__gm__ int32_t*>(&pq->count));
    pto::comm::TWAIT(sig, heads[wait_queue] + 1, pto::comm::WaitCmp::GE);
}
```

### 与 MultiBlockQueueSet 的协作

- **单生产者保证**：每个队列只有一个计算 block 写入，确保 `count` 更新的原子性
- **单消费者保证**：每个队列只有一个通信 block 读取，避免竞争
- **硬件同步**：`TWAIT` 在硬件级别检查 `count`，确保缓存一致性

## 两级双缓冲机制详解

### 架构概述

GEMM 计算核实现了优化的**两级双缓冲**机制，以最大化流水线并行度并隐藏 DMA 延迟。该优化通过实现三个流水线阶段的完全重叠，显著提升了性能：GM→L1 加载、L1→L0 提取和 Cube 计算。

Two-level double buffering includes:

1. **L1-level Double Buffering** (`mte2DBFlag`): Controls which L1 buffer slot TLOAD operations use
   - Two L1 buffers: `aMatTile[0]` and `aMatTile[1]` for matrix A
   - Two L1 buffers: `bMatTile[0]` and `bMatTile[1]` for matrix B
   - Each buffer caches `stepK` (default: 4) K-slices to reduce GM→L1 DMA frequency

2. **L0-level Double Buffering** (`mte1DBFlag`): Controls which L0 buffer slot TEXTRACT and TMATMUL operations use
   - Two L0 buffers: `aTile[0]` and `aTile[1]` for matrix A
   - Two L0 buffers: `bTile[0]` and `bTile[1]` for matrix B

### K-slice and Tile Concepts and Relationships

To better understand the two-level double buffering mechanism, it's important to clarify the concepts of **K-slice** and **Tile**:

#### Concept Definitions

- **Tile (Output Block)**:
  - Meaning: A complete output block of the output matrix C
  - Size: `baseM × baseN` (e.g., 128 × 256 = 32,768 elements)
  - Position: The output matrix C is divided into multiple tiles, each corresponding to an output region
  - Function: The complete output block that is finally written to `shmem_output`

- **K-slice (K-dimension Slice)**:
  - Meaning: A slice along the K dimension, used to compute part of a tile
  - Size:
    - Matrix A's K-slice: `baseM × baseK` (e.g., 128 × 64)
    - Matrix B's K-slice: `baseK × baseN` (e.g., 64 × 256)
  - Position: In K-loop iterations, each iteration processes one K-slice
  - Function: Accumulated into the output tile during the K-loop

#### Complete Matrix Multiplication Example

Taking a complete matrix multiplication on one rank as an example (assuming `M=16384, K=4096, N=4096`, 8 ranks, each rank's `k_per_rank=512`):

**1. Tile Partitioning of Output Matrix C**

```
Output matrix C[16384 × 4096] is divided into Tiles:

        N=4096 (16 tiles, 256 columns each)
      ┌─────────────────────────────────────┐
      │ Tile(0,0) │ Tile(0,1) │ ... │ Tile(0,15) │
M=    │ 128×256   │ 128×256   │     │ 128×256   │
16384 ├───────────┼───────────┼─────┼───────────┤
(128  │ Tile(1,0) │ Tile(1,1) │ ... │ Tile(1,15) │
tiles)│ 128×256   │ 128×256   │     │ 128×256   │
      │           │           │     │           │
      │    ...    │    ...    │ ... │    ...    │
      │           │           │     │           │
      │Tile(127,0)│Tile(127,1)│ ... │Tile(127,15)│
      │ 128×256   │ 128×256   │     │ 128×256   │
      └─────────────────────────────────────┘

Total: 128 × 16 = 2048 Tiles
Each Tile size: 128 × 256 = 32,768 elements
```

**2. Process of Computing One Tile (Example: Tile(0,0))**

```
Computing Tile(0,0) = A[0:128, :] × B[:, 0:256]

Input matrix A[16384 × 512]         Input matrix B[512 × 4096]
┌─────────────┐                     ┌─────────────────────────┐
│ A[0:128, :] │ ← Row block 0      │ B[:, 0:256]             │ ← Column block 0
│  128×512    │                     │  512×256                │
└─────────────┘                     └─────────────────────────┘
     ↓                                       ↓
     └─────────── K-loop (8 iterations) ────┘
```

**3. K-slice Processing in K-loop**

```
K dimension (512) is divided into 8 K-slices, each of size 64:

K dimension: 0    64   128  192  256  320  384  448  512
             │    │    │    │    │    │    │    │    │
             ├────┼────┼────┼────┼────┼────┼────┼────┤
             │ K0 │ K1 │ K2 │ K3 │ K4 │ K5 │ K6 │ K7 │
             └────┴────┴────┴────┴────┴────┴────┴────┘
             Each K-slice = 64

K-loop for computing Tile(0,0):
┌─────────────────────────────────────────────────────┐
│ cTile[128×256] = 0  // Initialize output tile       │
│                                                      │
│ for kIter = 0 to 7:  // 8 iterations, each processes │
│                      // one K-slice                  │
│   ┌──────────────────────────────────────────────┐  │
│   │ kIter=0:                                      │  │
│   │   aSlice = A[0:128, 0:64]     // 128×64      │  │
│   │   bSlice = B[0:64, 0:256]    // 64×256      │  │
│   │   cTile += aSlice × bSlice                   │  │
│   └──────────────────────────────────────────────┘  │
│   ┌──────────────────────────────────────────────┐  │
│   │ kIter=1:                                      │  │
│   │   aSlice = A[0:128, 64:128]   // 128×64      │  │
│   │   bSlice = B[64:128, 0:256]   // 64×256      │  │
│   │   cTile += aSlice × bSlice                   │  │
│   └──────────────────────────────────────────────┘  │
│   ...                                                 │
│   ┌──────────────────────────────────────────────┐  │
│   │ kIter=7:                                      │  │
│   │   aSlice = A[0:128, 448:512]  // 128×64      │  │
│   │   bSlice = B[448:512, 0:256]  // 64×256      │  │
│   │   cTile += aSlice × bSlice                   │  │
│   └──────────────────────────────────────────────┘  │
│                                                      │
│   TSTORE(cTile)  // Write complete tile to shmem_output │
└─────────────────────────────────────────────────────┘
```

**4. K-slice Caching in Two-level Double Buffering**

```
K-loop iteration process (L1 caches 4 K-slices):

Iterations 0-3:  TLOAD A[0:128, 0:256] → L1 (cache 4 K-slices)
                 ├─ kIter=0: TEXTRACT A[0:128, 0:64]   → L0 → Cube compute
                 ├─ kIter=1: TEXTRACT A[0:128, 64:128] → L0 → Cube compute
                 ├─ kIter=2: TEXTRACT A[0:128, 128:192] → L0 → Cube compute
                 └─ kIter=3: TEXTRACT A[0:128, 192:256] → L0 → Cube compute

Iterations 4-7:  TLOAD A[0:128, 256:512] → L1 (cache 4 K-slices)
                 ├─ kIter=4: TEXTRACT A[0:128, 256:320] → L0 → Cube compute
                 ├─ kIter=5: TEXTRACT A[0:128, 320:384] → L0 → Cube compute
                 ├─ kIter=6: TEXTRACT A[0:128, 384:448] → L0 → Cube compute
                 └─ kIter=7: TEXTRACT A[0:128, 448:512] → L0 → Cube compute

Advantage: GM→L1 DMA frequency reduced from 8 times to 2 times (4x optimization)
```

**5. Complete Computation Flow**

```
Complete computation flow for one Rank:

┌─────────────────────────────────────────────────────────────┐
│ Outer loop: Iterate over all Tiles (2048 tiles)            │
│ for tile_idx = 0 to 2047:                                  │
│   mi = tile_idx / 16  // Tile index in M dimension (0-127) │
│   ni = tile_idx % 16  // Tile index in N dimension (0-15)   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐ │
│   │ Inner loop: K-loop (8 iterations)                    │ │
│   │ for kIter = 0 to 7:                                  │ │
│   │   ┌───────────────────────────────────────────────┐ │ │
│   │   │ 1. L1 caching phase (every 4 iterations)        │ │ │
│   │   │    if (kIter % 4 == 0):                        │ │ │
│   │   │      TLOAD A[mi*128:(mi+1)*128, kIter*64:...] │ │ │
│   │   │      TLOAD B[kIter*64:..., ni*256:(ni+1)*256] │ │ │
│   │   │      → Cache to L1 (4 K-slices)                │ │ │
│   │   └───────────────────────────────────────────────┘ │ │
│   │   ┌───────────────────────────────────────────────┐ │ │
│   │   │ 2. L0 extraction phase (each iteration)        │ │ │
│   │   │    TEXTRACT aSlice from L1                    │ │ │
│   │   │    TEXTRACT bSlice from L1                    │ │ │
│   │   │    → Extract single K-slice to L0             │ │ │
│   │   └───────────────────────────────────────────────┘ │ │
│   │   ┌───────────────────────────────────────────────┐ │ │
│   │   │ 3. Cube computation phase (each iteration)     │ │ │
│   │   │    if (kIter == 0):                            │ │ │
│   │   │      cTile = aSlice × bSlice                  │ │ │
│   │   │    else:                                       │ │ │
│   │   │      cTile += aSlice × bSlice  // Accumulate   │ │ │
│   │   └───────────────────────────────────────────────┘ │ │
│   │ end  // K-loop ends                                 │ │
│   └─────────────────────────────────────────────────────┘ │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐ │
│   │ 4. Write phase                                       │ │
│   │    TSTORE(cTile) → shmem_output[mi*128, ni*256]    │ │
│   │    → Write complete Tile to output matrix            │ │
│   └─────────────────────────────────────────────────────┘ │
│ end  // Tile loop ends                                    │
└─────────────────────────────────────────────────────────────┘
```

#### Relationship Summary

- **One Tile = Accumulation of multiple K-slices**: Computing one Tile requires traversing all K dimensions (e.g., 8 K-loop iterations)
- **K-slice is an intermediate data block in computation**: Each K-loop iteration processes one K-slice and accumulates it into the tile
- **L1 caching optimization**: Load 4 K-slices together into L1, reducing GM→L1 DMA frequency (from 8 times to 2 times)
- **Parallelism**: Multiple tiles can be computed in parallel, improving overall throughput

This design achieves:
1. **Parallelism**: Multiple tiles can be computed in parallel
2. **Cache optimization**: L1 caches multiple K-slices, reducing memory access
3. **Pipeline**: L1→L0→Cube three-stage pipeline overlap

### Data Flow Pipeline

```
时间 →
L1 (MTE2):  [TLOAD A0,B0] [TLOAD A1,B1] [TLOAD A2,B2] ...
L0 (MTE1):       [EXT k0] [EXT k1] [EXT k2] [EXT k3] [EXT k0'] ...
Cube (M):             [MUL k0] [ACC k1] [ACC k2] [ACC k3] [MUL k0'] ...
                      ↑ 三级流水线完全并行 ↑
```

### 实现细节

**步骤 1：L1 缓存优化（stepK=4）**

每 `stepK` 次迭代（默认：4），将包含 `stepK` 个 K-slice 的更大面板加载到 L1 缓存：

```cpp
constexpr uint32_t G_STEP_KA = 4;    // A: 缓存 4 个 K-slice → TLOAD [128, 256] = 64KB
constexpr uint32_t G_STEP_KB = 4;    // B: 缓存 4 个 K-slice → TLOAD [256, 256] = 128KB

// L1 tile 类型：更大以容纳 stepK 个 slice
using TileMatAData = Tile<TileType::Mat, half, G_BASE_M, G_BASE_K * G_STEP_KA, ...>;
using TileMatBData = Tile<TileType::Mat, half, G_BASE_K * G_STEP_KB, G_BASE_N, ...>;
```

**步骤 2：使用 TEXTRACT 进行 L1→L0 传输**

使用 `TEXTRACT` 从缓存的 L1 面板中提取单个 K-slice，而不是使用 `TMOV`（需要匹配的 tile 形状）：

```cpp
// 从缓存的 L1 面板中提取当前 K-slice
// 对于 A 矩阵（ColMajor）：从行 0、列 kModStepKa * baseK 开始提取
TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], 0, kModStepKa * baseK);

// 对于 B 矩阵（RowMajor）：从行 (kIter % stepKb) * baseK、列 0 开始提取
TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], (kIter % stepKb) * baseK, 0);
```

**步骤 3：独立的缓冲区管理**

两个缓冲区级别独立运行：

```cpp
uint8_t mte2DBFlag = 0;  // L1 级双缓冲标志
uint8_t mte1DBFlag = 0;  // L0 级双缓冲标志

// 在 ProcessKIteration 中：
if (kModStepKa == 0) {
    // TLOAD 到 mte2DBFlag 指示的 L1 缓冲区
    wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)mte2DBFlag);
    TLOAD(aMatTile[mte2DBFlag], gmA);
    TLOAD(bMatTile[mte2DBFlag], gmB);
    mte2DBFlag = (mte2DBFlag == 0) ? 1 : 0;  // 切换 L1 缓冲区
}

const uint32_t currMte2Idx = (mte2DBFlag == 0) ? 1 : 0;  // 从另一个 L1 缓冲区读取

// 从 L1 提取到 mte1DBFlag 指示的 L0 缓冲区
wait_flag(PIPE_M, PIPE_MTE1, (event_t)mte1DBFlag);
TEXTRACT(aTile[mte1DBFlag], aMatTile[currMte2Idx], ...);
TEXTRACT(bTile[mte1DBFlag], bMatTile[currMte2Idx], ...);

// 使用 L0 缓冲区进行 TMATMUL
TMATMUL_ACC(cTile, cTile, aTile[mte1DBFlag], bTile[mte1DBFlag]);
mte1DBFlag = (mte1DBFlag == 0) ? 1 : 0;  // 切换 L0 缓冲区
```

### 同步逻辑

同步确保正确的执行顺序：

1. **TLOAD 等待 TEXTRACT**：在重用 L1 缓冲区之前，等待 TEXTRACT 完成读取
   ```cpp
   wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)mte2DBFlag);
   ```

2. **TEXTRACT 等待 TMATMUL**：在覆盖 L0 缓冲区之前，等待 TMATMUL 完成使用
   ```cpp
   wait_flag(PIPE_M, PIPE_MTE1, (event_t)mte1DBFlag);
   ```

3. **完成信号**：完成 stepK 组中的所有 slice 后，发出信号表示 L1 缓冲区可以重用
   ```cpp
   if ((kIter + 1) % stepKa == 0) {
       set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)currMte2Idx);
   }
   ```

### 性能收益

- **4x 减少 GM→L1 DMA 频率**：每 4 次迭代才加载一次，而不是每次迭代都加载
- **完全流水线重叠**：MTE2（GM→L1）、MTE1（L1→L0）和 M（Cube）可以并行运行
- **隐藏 DMA 延迟**：当 Cube 计算当前 K-slice 时，L1 可以加载下一个 stepK 块
- **提高带宽利用率**：更大的 TLOAD 操作（A 为 64KB，B 为 128KB）更好地利用内存带宽

### 内存布局

```
L1 内存布局：
[pingA(64KB) | pongA(64KB) | pingB(128KB) | pongB(128KB)] = 384KB 总计

L0 内存布局：
[aTile[0](16KB) | aTile[1](16KB) | bTile[0](32KB) | bTile[1](32KB)] = 96KB 总计
```

## 参考

- [PTO 通信指令集](../../../include/pto/comm/)
- [TPUT 指令文档](../../../include/pto/comm/TPut.hpp)
- [gemm_allgather Demo](../gemm_allgather/)
- [CANN 文档](../../../../docs/cann_8.0/)
