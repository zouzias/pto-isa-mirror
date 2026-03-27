# GEMM AllReduce Demo - Device 侧计算与通信

本 Demo 实现了 GEMM + AllReduce 算子，**所有计算和通信都在 Device 侧（NPU）执行**，无需 Host 参与。通过 `TPUT<AtomicAdd>` 实现远程原子加，直接在 Device 侧完成 AllReduce 操作。

## 数据并行模式

本实现支持**数据并行模式**（Data Parallel Mode）：

### 矩阵分割策略

假设总输入矩阵为：
- **A矩阵**：M × K（默认：16384 × 4096）
- **B矩阵**：K × N（默认：4096 × 4096）

对于 r 个 rank，矩阵按 **K 维度**分割：
- **每个 rank 的 A 矩阵**：M × (K/r)（例如：16384 × 512，当 r=8 时）
  - A 矩阵按**列**分割（K 维度是列维度）
- **每个 rank 的 B 矩阵**：(K/r) × N（例如：512 × 4096，当 r=8 时）
  - B 矩阵按**行**分割（K 维度是行维度）

**约束**：K 必须能被 rank 数量整除（`G_K % n_ranks == 0`）

### 计算流程

每个 rank 计算：
```
C_part = A_part[M × (K/r)] × B_part[(K/r) × N] = C_part[M × N]
```

### AllReduce 操作

所有 rank 的输出通过 AllReduce 求和：
```
C_final = sum(C_part_0 + C_part_1 + ... + C_part_r-1) = C_final[M × N]
```

### 数学正确性

由于矩阵乘法满足：
```
A × B = [A_0 | A_1 | ... | A_r-1] × [B_0; B_1; ...; B_r-1]
       = A_0 × B_0 + A_1 × B_1 + ... + A_r-1 × B_r-1
```

因此数据并行模式的结果与完整矩阵乘法的结果一致。

## 架构设计

### 整体架构

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
│   关键特性:                                                                   │
│   - 计算和通信完全在 Device 侧执行（AICORE 标识）                                  │
│   - 使用 TPUT<AtomicAdd> 实现远程原子加，无需额外的 reduce 阶段                      │
│   - 计算核完成 tile 后通过队列信号量启动通信核                                       │
│   - 通信核轮询队列，一旦 tile 就绪立即进行 TPUT<AtomicAdd>                          │
│   - 所有 rank 的数据通过原子加直接累加到 reduced_output                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心创新：TPUT<AtomicAdd> 实现 Device-Side AllReduce

传统 AllReduce 实现需要：
1. 每个 rank 发送数据到远端
2. 接收远端数据
3. 在本地执行 reduce 操作

本实现使用 **TPUT<AtomicAdd>** 直接实现：
- 每个 rank 将自己的 GEMM 结果通过 `TPUT<AtomicAdd>` 原子加到**所有 rank** 的 `reduced_output`（包括自己）
- 所有操作在 Device 侧完成，无需 Host 参与
- 避免了显式的 reduce 阶段，简化了实现并提高了性能

## 核心组件

### 1. 计算内核 (`gemm_compute_kernel.cpp`)

- **架构**: Cube-only (dav-c220-cube)
- **AICORE 标识**: 所有计算在 Device 侧执行
- **功能**: 执行 GEMM 计算 (C = A × B)
- **信号机制**: 计算完每个 tile 后，将 tile index 入队到 `MultiBlockQueueSet`
- **多 block 支持**: 多个计算 block 并行执行，每个 block 有独立队列（无锁竞争）
- **Tile 处理**: 使用**两级双缓冲** K-loop 优化，实现 L1 缓存复用和完全流水线重叠
  - **L1 级双缓冲** (`mte2DBFlag`): 每 4 次 K 迭代加载一次，缓存 4 个 K-slice
  - **L0 级双缓冲** (`mte1DBFlag`): 使用 TEXTRACT 从 L1 缓存提取单个 K-slice
  - **性能优化**: 4x 减少 GM→L1 DMA 频率，三级流水线（MTE2/MTE1/Cube）完全并行

### 2. 通信内核 (`comm_kernel.cpp`)

- **架构**: Vector-only (dav-c220-vec)
- **AICORE 标识**: 所有通信在 Device 侧执行
- **功能**: 
  1. **轮询阶段**: 使用 `TTEST`/`TWAIT` 轮询计算核队列，检测就绪的 tile
  2. **原子加阶段**: 使用 `TPUT<AtomicAdd>` 将 tile 数据原子加到所有 rank 的 `reduced_output`
     - 每个 rank 将自己的数据加到所有 rank（包括自己）
     - 最终每个 rank 的 `reduced_output` 包含所有 rank 数据的和
  3. **同步阶段**: 使用 `ShmemDeviceBarrierAll()` 确保所有原子操作完成
- **多 block 支持**: 通信 block 分担队列轮询任务，提高并行度

### 3. 队列机制 (`ready_queue.hpp`)

- **设计**: 每个计算 block 一个独立队列（单生产者）
- **优势**: 无原子操作开销，O(B) 轮询代替 O(N) flag 轮询
- **同步**: 使用 PTO `TTEST`/`TWAIT` 指令进行硬件友好的同步
- **数据结构**: `MultiBlockQueueSet` 管理多个 `PerBlockQueue`

## 数据流

### AllReduce 数据流（使用 TPUT<AtomicAdd>）

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

### 关键数据结构

- **`shmem_output`**: 每个 rank 的本地 GEMM 结果（对称堆内存）
  - 大小：`M × N × sizeof(float)`
  - 存储每个 rank 计算的部分结果 `C_part = A_part × B_part`
- **`reduced_output`**: AllReduce 的最终结果（对称堆内存）
  - 大小：`M × N × sizeof(float)`
  - 初始化为 0
  - 通过 `TPUT<AtomicAdd>` 直接累加所有 rank 的数据
  - 最终每个 rank 的 `reduced_output` 包含相同的结果（所有 rank 数据的和）
  - **注意**：`TPUT<AtomicAdd>` 直接写入 `reduced_output`，无需中间缓冲区
- **`recv_buffers`**: 最小调试缓冲区（1MB，仅用于调试）
  - **不用于实际通信**：实际通信通过 `TPUT<AtomicAdd>` 直接写入 `reduced_output`
  - 内存优化：从 `nranks × outputSize` 减少到 1MB，支持 8+ rank 测试

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

#### 5. 性能指标计算

```cpp
// GFLOPS 计算（数据并行模式）
// 每个 rank 的 FLOPS：2 * M * (K/r) * N，其中 r 是 rank 数量
uint32_t k_per_rank = G_K / n_ranks;
double gemm_flops_per_rank = 2.0 * M * k_per_rank * N;  // 每个 rank 的浮点运算
double gemm_flops_total = 2.0 * M * K * N;  // 总系统的浮点运算

// Per-rank GFLOPS（每个 rank 实际计算的性能）
double compute_gflops = gemm_flops_per_rank / (compute_time_us * 1e-6) / 1e9;

// Total system GFLOPS（所有 rank 组合的总吞吐量）
double seq_gflops = gemm_flops_total / (seq_avg * 1e-6) / 1e9;
double pipe_gflops = gemm_flops_total / (pipe_avg * 1e-6) / 1e9;

// 加速比计算
double speedup = seq_avg / pipe_avg;
double time_saved = seq_avg - pipe_avg;
double time_saved_percent = (time_saved / seq_avg) * 100.0;
```

**注意**：在数据并行模式下，每个 rank 只计算部分 K 维度（K/r），因此：
- **Per-rank FLOPS** = `2 * M * (K/r) * N`：用于评估单个 rank 的计算性能
- **Total system FLOPS** = `2 * M * K * N`：用于评估整个系统的总吞吐量

### 性能报告输出

运行后会输出以下性能数据：

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

**性能指标说明**：
- **Compute GFLOPS (per rank)**：基于 `2 * M * (K/r) * N`，评估单个 rank 的计算性能
- **Throughput (total)**：基于 `2 * M * K * N`，评估整个系统的总吞吐量
- **Comm Data**：每个 rank 发送的数据量 = `M × N × sizeof(float) × n_ranks`（发送到所有 rank，包括自己）

### 性能分析指标说明

| 指标 | 说明 |
|------|------|
| **Compute Kernel** | 纯 GEMM 计算时间，作为性能上限参考 |
| **Sequential Execution** | 串行执行时间（计算→通信），无重叠 |
| **Pipelined Execution** | 流水线执行时间（计算||通信），有重叠 |
| **Speedup** | 流水线相对于串行的加速比 |
| **Time Saved** | 流水线节省的时间（绝对值和百分比） |
| **Throughput (GFLOPS)** | 基于总执行时间的吞吐量 |
| **Std Dev** | 标准差，反映性能稳定性 |

### 性能优化要点

1. **计算-通信重叠**: 通过双流并行执行，隐藏通信延迟
2. **细粒度同步**: 使用队列机制实现 tile 级别的细粒度同步
3. **Device-Side 执行**: 所有操作在 Device 侧完成，避免 Host-Device 数据传输开销
4. **原子操作优化**: 使用 `TPUT<AtomicAdd>` 直接实现 AllReduce，避免额外的 reduce 阶段

## 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| G_M | 16384 | 矩阵 M 维度（全局） |
| G_K | 4096 | 矩阵 K 维度（全局，会被分割到各 rank） |
| G_N | 4096 | 矩阵 N 维度（全局） |
| G_BASE_M | 128 | Tile M 维度 |
| G_BASE_K | 64 | Tile K 维度 |
| G_BASE_N | 256 | Tile N 维度 |
| COMPUTE_BLOCK_NUM | 32 | 计算 block 数量 |
| COMM_BLOCK_NUM | 40 | 通信 block 数量 |
| WARMUP_ITERS | 5 | Warmup 迭代次数 |
| MEASURE_ITERS | 100 | 性能测量迭代次数 |

**数据并行模式参数**（运行时计算）：
- `k_per_rank = G_K / n_ranks`：每个 rank 的 K 维度大小
- 每个 rank 的输入：`A_part[M, k_per_rank]` 和 `B_part[k_per_rank, N]`
- 每个 rank 的输出：`C_part[M, N]`（完整大小，但只包含部分计算结果）

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

## 性能测试执行

本项目的性能测试包含：**主程序内置性能统计**（Warmup / 纯计算 / 顺序执行 / 流水线执行）与**可选核配置基准测试**。以下为可直接执行的命令。

### 1. 主程序性能测试（推荐）

一条命令完成构建并运行 GEMM AllReduce，输出各阶段耗时、GFLOPS、带宽与流水线加速比：

```bash
# 默认 8 rank，从 device 0 开始
./run_performance_test.sh

# 指定 rank 数与首设备
./run_performance_test.sh --nranks 8 --first-device 0

# 仅运行不重新编译（需已执行过 run.sh 或 run_performance_test.sh）
./run_performance_test.sh --nranks 8 --skip-build
```

或使用原有 `run.sh`（同样会跑主程序内置性能测试）：

```bash
./run.sh --nranks 8
# 等价于：构建后执行 ./gemm_allreduce --nranks 8 --first-device 0
```

### 2. 单核基准与最优核数配置

在已编译的前提下（先执行一次 `./run.sh` 或 `./run_performance_test.sh`）：

```bash
# 单通信核 TPUT 带宽
./run_benchmark_comm.sh --nranks 8 --first-device 0 --num-tiles 100

# 单计算核 GEMM GFLOPS
./run_benchmark_compute.sh --device 0

# 自动测单核并写回最优核数配置（会改 comm_kernel.cpp / gemm_compute_kernel.cpp，需重新编译）
./run_optimal_config.sh --nranks 8 --first-device 0 --total-cores 24
```

### 3. 直接调用可执行文件（在 build 目录下）

```bash
cd build
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /home/ntlab/qifeng/pypto/third_party_path/shmem/install/set_env.sh

# 主程序性能测试
./gemm_allreduce --nranks 8 --first-device 0

# 单核基准
./benchmark_single_comm_core --nranks 8 --first-device 0 --num-tiles 100
./benchmark_single_compute_core --device 0
./benchmark_optimal_config --nranks 8 --first-device 0 --total-cores 24
```

更多说明见 [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)。

## 文件结构

```
gemm_allreduce/
├── CMakeLists.txt           # 构建配置
├── README.md                # 本文档（英文版）
├── README_zh.md             # 本文档（中文版）
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

### 1. TPUT<AtomicAdd> 的使用

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

**关键特性**：
- **直接写入**：`TPUT<AtomicAdd>` 直接写入 `reduced_output`，无需 `recv_buffers`
- **原子操作**：确保多 rank 并发写入的正确性
- **内存优化**：避免了 `nranks × outputSize` 的中间缓冲区，只需 1MB 调试缓冲区

### 2. Device-Side 同步

- 使用 `ShmemDeviceBarrierAll()` 确保所有 rank 的原子操作完成
- 使用 `pipe_barrier(PIPE_ALL)` 确保所有 pipeline 阶段完成

### 3. 队列轮询优化

- 使用 `TTEST` 进行非阻塞检查
- 使用 `TWAIT` 进行阻塞等待
- 轮询多个队列，提高并行度

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

两级双缓冲包括：

1. **L1 级双缓冲**（`mte2DBFlag`）：控制 TLOAD 操作使用哪个 L1 缓冲区槽位
   - 两个 L1 缓冲区：`aMatTile[0]` 和 `aMatTile[1]` 用于矩阵 A
   - 两个 L1 缓冲区：`bMatTile[0]` 和 `bMatTile[1]` 用于矩阵 B
   - 每个缓冲区缓存 `stepK`（默认：4）个 K-slice，以减少 GM→L1 DMA 频率

2. **L0 级双缓冲**（`mte1DBFlag`）：控制 TEXTRACT 和 TMATMUL 操作使用哪个 L0 缓冲区槽位
   - 两个 L0 缓冲区：`aTile[0]` 和 `aTile[1]` 用于矩阵 A
   - 两个 L0 缓冲区：`bTile[0]` 和 `bTile[1]` 用于矩阵 B

### K-slice 与 Tile 的概念与关系

为了更好地理解两级双缓冲机制，需要明确 **K-slice** 和 **Tile** 的概念：

#### 概念定义

- **Tile（输出块）**：
  - 含义：输出矩阵 C 的一个完整输出块
  - 大小：`baseM × baseN`（例如：128 × 256 = 32,768 个元素）
  - 位置：输出矩阵 C 被划分为多个 tiles，每个 tile 对应一个输出区域
  - 作用：最终写入 `shmem_output` 的完整输出块

- **K-slice（K 维度切片）**：
  - 含义：K 维度上的一个切片，用于计算一个 tile 的一部分
  - 大小：
    - 矩阵 A 的 K-slice：`baseM × baseK`（例如：128 × 64）
    - 矩阵 B 的 K-slice：`baseK × baseN`（例如：64 × 256）
  - 位置：在 K-loop 迭代中，每次迭代处理一个 K-slice
  - 作用：在 K-loop 中累加到输出 tile

#### 完整矩阵乘示例

以一个 rank 上的完整矩阵乘为例（假设 `M=16384, K=4096, N=4096`，8 个 ranks，每个 rank 的 `k_per_rank=512`）：

**1. 输出矩阵 C 的 Tile 划分**

```
输出矩阵 C[16384 × 4096] 被划分为 Tiles：

        N=4096 (16个tiles，每个256列)
      ┌─────────────────────────────────────┐
      │ Tile(0,0) │ Tile(0,1) │ ... │ Tile(0,15) │
M=    │ 128×256   │ 128×256   │     │ 128×256   │
16384 ├───────────┼───────────┼─────┼───────────┤
(128  │ Tile(1,0) │ Tile(1,1) │ ... │ Tile(1,15) │
个    │ 128×256   │ 128×256   │     │ 128×256   │
tiles)│           │           │     │           │
      │    ...    │    ...    │ ... │    ...    │
      │           │           │     │           │
      │Tile(127,0)│Tile(127,1)│ ... │Tile(127,15)│
      │ 128×256   │ 128×256   │     │ 128×256   │
      └─────────────────────────────────────┘

总共：128 × 16 = 2048 个 Tiles
每个 Tile 大小：128 × 256 = 32,768 个元素
```

**2. 计算一个 Tile 的过程（以 Tile(0,0) 为例）**

```
计算 Tile(0,0) = A[0:128, :] × B[:, 0:256]

输入矩阵 A[16384 × 512]         输入矩阵 B[512 × 4096]
┌─────────────┐                ┌─────────────────────────┐
│ A[0:128, :] │ ← 第0行块      │ B[:, 0:256]             │ ← 第0列块
│  128×512    │                │  512×256                │
└─────────────┘                └─────────────────────────┘
     ↓                                    ↓
     └─────────── K-loop (8次迭代) ───────┘
```

**3. K-loop 中的 K-slice 处理**

```
K维度 (512) 被分为 8 个 K-slices，每个 64：

K维度: 0    64   128  192  256  320  384  448  512
       │    │    │    │    │    │    │    │    │
       ├────┼────┼────┼────┼────┼────┼────┼────┤
       │ K0 │ K1 │ K2 │ K3 │ K4 │ K5 │ K6 │ K7 │
       └────┴────┴────┴────┴────┴────┴────┴────┘
       每个 K-slice = 64

计算 Tile(0,0) 的 K-loop：
┌─────────────────────────────────────────────────────┐
│ cTile[128×256] = 0  // 初始化输出 tile              │
│                                                      │
│ for kIter = 0 to 7:  // 8次迭代，每次处理一个K-slice │
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
│   TSTORE(cTile)  // 写入完整的 tile 到 shmem_output │
└─────────────────────────────────────────────────────┘
```

**4. 两级双缓冲中的 K-slice 缓存**

```
K-loop 迭代过程（L1 缓存 4 个 K-slice）：

迭代 0-3:  TLOAD A[0:128, 0:256] → L1 (缓存4个K-slice)
           ├─ kIter=0: TEXTRACT A[0:128, 0:64]   → L0 → Cube计算
           ├─ kIter=1: TEXTRACT A[0:128, 64:128] → L0 → Cube计算
           ├─ kIter=2: TEXTRACT A[0:128, 128:192] → L0 → Cube计算
           └─ kIter=3: TEXTRACT A[0:128, 192:256] → L0 → Cube计算

迭代 4-7:  TLOAD A[0:128, 256:512] → L1 (缓存4个K-slice)
           ├─ kIter=4: TEXTRACT A[0:128, 256:320] → L0 → Cube计算
           ├─ kIter=5: TEXTRACT A[0:128, 320:384] → L0 → Cube计算
           ├─ kIter=6: TEXTRACT A[0:128, 384:448] → L0 → Cube计算
           └─ kIter=7: TEXTRACT A[0:128, 448:512] → L0 → Cube计算

优势：GM→L1 DMA 频率从 8次 减少到 2次（4x优化）
```

**5. 完整计算流程**

```
一个 Rank 的完整计算流程：

┌─────────────────────────────────────────────────────────────┐
│ 外层循环：遍历所有 Tiles (2048个)                            │
│ for tile_idx = 0 to 2047:                                   │
│   mi = tile_idx / 16  // M方向的tile索引 (0-127)            │
│   ni = tile_idx % 16  // N方向的tile索引 (0-15)              │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ 内层循环：K-loop (8次迭代)                             │  │
│   │ for kIter = 0 to 7:                                   │  │
│   │   ┌──────────────────────────────────────────────┐  │  │
│   │   │ 1. L1缓存阶段（每4次迭代）                      │  │  │
│   │   │    if (kIter % 4 == 0):                        │  │  │
│   │   │      TLOAD A[mi*128:(mi+1)*128, kIter*64:...] │  │  │
│   │   │      TLOAD B[kIter*64:..., ni*256:(ni+1)*256] │  │  │
│   │   │      → 缓存到 L1 (4个K-slice)                   │  │  │
│   │   └──────────────────────────────────────────────┘  │  │
│   │   ┌──────────────────────────────────────────────┐  │  │
│   │   │ 2. L0提取阶段（每次迭代）                      │  │  │
│   │   │    TEXTRACT aSlice from L1                   │  │  │
│   │   │    TEXTRACT bSlice from L1                   │  │  │
│   │   │    → 提取单个K-slice到L0                      │  │  │
│   │   └──────────────────────────────────────────────┘  │  │
│   │   ┌──────────────────────────────────────────────┐  │  │
│   │   │ 3. Cube计算阶段（每次迭代）                    │  │  │
│   │   │    if (kIter == 0):                           │  │  │
│   │   │      cTile = aSlice × bSlice                  │  │  │
│   │   │    else:                                      │  │  │
│   │   │      cTile += aSlice × bSlice  // 累加        │  │  │
│   │   └──────────────────────────────────────────────┘  │  │
│   │ end  // K-loop结束                                  │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │ 4. 写入阶段                                          │  │
│   │    TSTORE(cTile) → shmem_output[mi*128, ni*256]    │  │
│   │    → 写入完整的Tile到输出矩阵                        │  │
│   └──────────────────────────────────────────────────────┘  │
│ end  // Tile循环结束                                        │
└─────────────────────────────────────────────────────────────┘
```

#### 关系总结

- **一个 Tile = 多个 K-slice 的累加结果**：计算一个 Tile 需要遍历所有 K 维度（例如 8 次 K-loop 迭代）
- **K-slice 是计算过程中的中间数据块**：每次 K-loop 迭代处理一个 K-slice，累加到 tile
- **L1 缓存优化**：每 4 个 K-slice 一起加载到 L1，减少 GM→L1 DMA 频率（从 8 次减少到 2 次）
- **并行性**：多个 tiles 可以并行计算，提高整体吞吐量

这种设计实现了：
1. **并行性**：多个 tiles 可以并行计算
2. **缓存优化**：L1 缓存多个 K-slice，减少内存访问
3. **流水线**：L1→L0→Cube 三级流水线重叠

### 数据流管道

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

## 版本对比

| 版本 | 同步方式 | 重叠粒度 | 描述 |
|------|----------|----------|------|
| `gemm_allreduce`（本版本）| TPUT<AtomicAdd> | 逐 Tile | Device 侧 AllReduce，使用原子加 |
| `gemm_allreduce_host_barrierMode` | Host barrier + Event | 逐 Kernel | 双流 + Host 端协调 |
| `gemm_allreduce_single_stream` | 顺序执行 | 无 | 单流，先 GEMM 后 AllReduce |
| `gemm_allreduce_without_isa` | Host shmem | 无 | 测试版本，不使用 PTO-ISA 通信 |

## 参考

- [PTO 通信指令集](../../../include/pto/comm/)
- [TPUT 指令文档](../../../include/pto/comm/TPut.hpp)
- [gemm_allgather Demo](../gemm_allgather/)
- [CANN 文档](../../../../docs/cann_8.0/)
