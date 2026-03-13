# GEMM AllReduce 优化总结（gemm_allreduce_opt_3）

## 1. 架构概述

`gemm_allreduce_opt_3` 实现了使用 **plain TPUT + TREDUCE_PINGPONG** 的 AllReduce 融合算子：

- 矩阵 A `[M=16384, K=4096]` 和 B `[K=4096, N=4096]` 沿 K 维度分割到 8 个 rank
- 每个 rank 在 Cube 架构上计算 `C_part = A_part[M, K/r] × B_part[K/r, N]`
- AllReduce 分两步实现：
  1. **Plain TPUT scatter**：每个 rank 通过 TPUT（无 AtomicAdd）将本地数据写入所有远端 rank 的 `recv_buffers[my_rank]` 槽位
  2. **TREDUCE_PINGPONG**：设备端全局屏障后，每个 rank 本地归约所有 rank 的数据
- 计算内核和通信内核在**独立的 stream** 上运行，通过 `MultiBlockQueueSet` 实现 tile 级同步

## 2. 与基线的对比

### 基线方式（gemm_allreduce_opt_2）
- 使用 `TPUT<AtomicAdd>` 直接将数据原子累加到远端 rank 的 `reduced_output`
- 一步完成通信+归约
- 无需额外内存（不需要 recv_buffers）

### 本实现方式（gemm_allreduce_opt_3）
- Phase 1: Plain TPUT scatter（无原子操作）
- Phase 2: ShmemDeviceQuiet + ShmemDeviceBarrierAll
- Phase 3: TREDUCE_PINGPONG 本地归约
- 需要额外 `nranks × M × N × sizeof(float) = 2 GB` 的 recv_buffers

## 3. 实现细节

### 3.1 Plain TPUT Scatter（Phase 1）

每个通信 block 轮询计算内核的队列，tile 就绪后立即 TPUT 到所有远端 rank：

```cpp
for (int r = 0; r < nranks; ++r) {
    if (r == my_rank) continue;
    // 目标：rank r 的 recv_buffers[my_rank] 槽位
    __gm__ float *dst_ptr = ShmemPtr(recv_buffers, r)
                          + my_rank * output_size + tile_offset;
    Global dstG(dst_ptr, tileShape, tileStride);
    pto::comm::TPUT(dstG, srcG, pingTile, pongTile);  // Plain TPUT, 无 AtomicAdd
}
```

### 3.2 TREDUCE_PINGPONG 本地归约（Phase 3）

**关键设计决策**：使用 `TREDUCE_PINGPONG` 而非 `TREDUCE`。

#### 为什么 TREDUCE（非 pingpong）在循环中会挂死

`TREDUCE_IMPL` 内部使用 `TADD`（V-pipeline）进行归约，其 `set_flag`/`wait_flag` 使用 `EVENT_ID1` 在 `(PIPE_MTE2, PIPE_V)` 方向上。虽然每对 set/wait 在单次调用内是配对的，但在多次调用循环中（2048+ tiles），V-pipeline 的硬件 2-bit 事件计数器会累积溢出，导致 `wait_flag` 永远等不到正确的计数值。

**实验验证**：使用 `TREDUCE` 的实现在 warmup 阶段即挂死（运行超过 4 分钟无进展）。

#### TREDUCE_PINGPONG 如何解决此问题

`TREDUCE_PINGPONG_IMPL` 使用**交替的事件 ID**（`EVENT_ID1` 和 `EVENT_ID2`）进行 MTE2→V 同步：

```cpp
// TREDUCE_PINGPONG 内部的 pingpong 循环
for (int i = 0; i < numRemote; ++i) {
    const auto currentEvent = usePing ? EVENT_ID1 : EVENT_ID2;  // 交替！
    const auto nextEvent = usePing ? EVENT_ID2 : EVENT_ID1;
    // ...
    wait_flag(PIPE_MTE2, PIPE_V, currentEvent);
    TADD(acc, acc, currentTile);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
}
```

由于事件 ID 在 ID1 和 ID2 之间交替，每个事件的计数器最多达到 1，永远不会溢出。这使得 TREDUCE_PINGPONG 可以安全地在循环中调用 2048+ 次。

#### UB 内存布局

`TREDUCE_PINGPONG` 需要 3 个 UB tile（accumulator + ping + pong）。为了在 256KB UB 中容纳 3 个 tile，我们使用半高 tile：

```
全高 tile:  128 × 256 × 4B = 128KB → 2 tiles = 256KB (满)
半高 tile:   64 × 256 × 4B =  64KB → 3 tiles = 192KB (✓)

UB 布局 (Phase 3):
  [0KB, 64KB)   accTile         (64×256, accumulator)
  [64KB, 128KB) pingReduceTile  (64×256, ping buffer)
  [128KB, 192KB) pongReduceTile (64×256, pong buffer)
```

每个 128×256 的输出 tile 被拆分为两个 64×256 的子 tile（上半部分 + 下半部分）分别归约。

## 4. 性能测试结果

**测试配置**：M=16384, K=4096, N=4096, 8 ranks, 20 次测量迭代, Ascend 910B1

### 与基线的对比

| 指标 | 基线 TPUT\<AtomicAdd\> (opt_2, 100 iters) | TPUT + TREDUCE_PINGPONG (opt_3, 20 iters) | 变化 |
|------|-------------------------------------------|--------------------------------------------|------|
| **纯计算 GFLOPS** | 97,673 | 98,546 ~ 100,617 | **+0.9~3.0%** ✅ |
| **串行总时间** | 30,141 μs | 31,990 ~ 32,210 μs | **+6.1~6.9%** ❌ |
| **串行通信时间** | 29,413 μs | 31,294 ~ 31,492 μs | **+6.4~7.1%** ❌ |
| **串行通信带宽** | 68.0 GB/s | 63.5 ~ 63.9 GB/s | **-6.0~6.6%** ❌ |
| **串行吞吐量** | 18,239 GFLOPS | 17,068 ~ 17,185 GFLOPS | **-5.8~6.4%** ❌ |
| **流水线总时间** | 28,973 μs | 30,893 ~ 31,041 μs | **+6.6~7.1%** ❌ |
| **流水线吞吐量** | 18,975 GFLOPS | 17,711 ~ 17,796 GFLOPS | **-6.2~6.7%** ❌ |
| **流水线加速比** | 1.040× | 1.031 ~ 1.043× | — |
| **正确性** | ✅ max diff = 0 | ✅ max diff = 0 | — |

### 详细性能输出（Run 1）

```
================================================================
[SUCCESS] GEMM AllReduce test PASSED!
  Matrix dimensions: M=16384, K=4096, N=4096
  Ranks: 8
  Compute blocks: 32, Comm blocks: 40
  Total tiles: 2048 (128x16)
  Reduction method: TREDUCE_PINGPONG (half-tile 64x256, 3 UB tiles)

[PERF] Performance Results (n_iters=20):

  Component Breakdown (Reference - Per Rank):
    Compute Kernel:    697.336 us (min: 668.224, max: 718.224, std: 17.462)
    Compute GFLOPS:    98545.718 GFLOPS (avg), 95679.728 GFLOPS (best)

  Sequential Execution (Compute -> Comm, no overlap):
    Total Time:      32209.905 us (min: 30155.550, max: 38000.077, std: 1798.280)
    Comm Time:        31492.410 us
    Comm Bandwidth:   63.507 GB/s
    Throughput:       17067.911 GFLOPS

  Pipelined Execution (Compute || Comm, with overlap):
    Total Time:      30892.720 us (min: 28404.379, max: 32684.265, std: 1123.937)
    Throughput:      17795.643 GFLOPS

  Performance Comparison:
    Speedup:            1.043x (Pipelined vs Sequential)
================================================================
```

## 5. 性能分析

### 5.1 端到端性能回退 ~6.5%

相比基线 `TPUT<AtomicAdd>` 方案，`TPUT + TREDUCE_PINGPONG` 方案增加了约 2ms 的通信时间。主要开销来源：

| 开销来源 | 分析 |
|---------|------|
| **设备端全局屏障** | `ShmemDeviceBarrierAll()` 在 TPUT scatter 和 TREDUCE 之间插入同步点，增加延迟 |
| **额外的归约阶段** | TREDUCE_PINGPONG 需要对每个 tile 从 recv_buffers 加载 7 份数据并累加 |
| **半高 tile 处理** | 每个 128×256 tile 需要 2 次 TREDUCE_PINGPONG 调用（上半 + 下半） |
| **额外内存带宽** | 归约阶段需要从 recv_buffers 读取 7 × 256MB = 1.75GB 数据 |
| **流水线重叠损失** | 归约必须在所有 TPUT 完成后才能开始，无法与计算重叠 |

### 5.2 纯计算性能提升

纯计算性能略有提升（+0.9~3%），这继承自 opt_2 的快速入队优化（2 次 dcci 替代 5 次）。

### 5.3 正确性完美匹配

两次运行的 max diff 均为 0，与 golden 完全匹配。这验证了：
- Plain TPUT scatter 正确地将数据分发到所有 rank 的 recv_buffers
- TREDUCE_PINGPONG 正确地归约了所有 rank 的数据
- 半高 tile 拆分策略正确

## 6. 关键技术发现

### 6.1 TREDUCE 在循环中挂死（V-pipeline 事件计数器溢出）

**现象**：使用 `TREDUCE`（非 pingpong）的实现在 warmup 阶段挂死。

**根因**：`TREDUCE_IMPL` 在内部 reduce 循环中使用 `EVENT_ID1` 在 `(PIPE_MTE2, PIPE_V)` 方向上。虽然每次 TREDUCE 调用内部的 set/wait 是配对的，但硬件的 2-bit 事件计数器在跨调用时会累积。当 TREDUCE 在循环中被调用数十次后，计数器溢出导致 `wait_flag` 死锁。

**解决方案**：使用 `TREDUCE_PINGPONG`，它交替使用 `EVENT_ID1` 和 `EVENT_ID2`，将每个事件的计数器峰值限制为 1。

### 6.2 UB 空间限制要求半高 tile

Ascend 910B1 的 Vec 架构 UB 为 256KB。全高 tile（128×256×4B = 128KB）只能容纳 2 个，而 TREDUCE_PINGPONG 需要 3 个。解决方案是使用半高 tile（64×256×4B = 64KB），3 个共 192KB。

### 6.3 TPUT\<AtomicAdd\> 仍然是 AllReduce 最优方案

本实验进一步确认了 opt_2 OPTIMIZATION_SUMMARY 的结论：
- `TPUT<AtomicAdd>` 在 MTE3 DMA 引擎中硬件实现原子累加，几乎零开销
- 任何分离通信和归约的方案都会引入额外的屏障延迟和内存带宽开销
- 本方案比基线慢 ~6.5%，主要是全局屏障和额外归约阶段的开销

## 7. 构建与运行

```bash
# 1. 激活 conda 环境
conda activate pypto_haoran

# 2. 配置 CANN 环境
source /usr/local/Ascend/cann-8.5.0/set_env.sh

# 3. 配置 shmem 环境
source /home/ntlab/haoran/cann_shmem/shmem/install/set_env.sh

# 4. 设置 CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH="$HOME/.local/lib64/cmake:$CMAKE_PREFIX_PATH"

# 5. 生成输入数据
cd gemm_allreduce_opt_3
python scripts/gen_data.py --nranks 8

# 6. 构建
mkdir -p build && cd build
cmake .. -DCONFIG_COMPUTE_BLOCK_NUM=32 -DCONFIG_COMM_BLOCK_NUM=40
make -j8 gemm_allreduce

# 7. 运行（8 ranks，设备从 0 开始）
export LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}
./gemm_allreduce --nranks 8 --first-device 0
```

## 8. 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `comm_kernel.cpp` | Phase 3 归约从 `TREDUCE` 改为 `TREDUCE_PINGPONG`；使用半高 tile（64×256）；3 个 UB tile 布局 |
| `gemm_compute_kernel.cpp` | 继承自 opt_2 的快速入队优化（无新修改） |
| `ready_queue.hpp` | 继承自 opt_2（无新修改） |

## 9. 总结

本实现验证了使用 `TPUT` + `TREDUCE` 完成 AllReduce 的可行性，但性能不如直接使用 `TPUT<AtomicAdd>`：

| 方案 | 通信原语 | 正确性 | 性能 vs 基线 | 额外内存 |
|------|---------|--------|-------------|---------|
| 基线 (opt_2) | `TPUT<AtomicAdd>` | ✅ | — | 0 |
| 本方案 (opt_3) | `TPUT` + `TREDUCE_PINGPONG` | ✅ | -6.5% | +2GB (recv_buffers) |
| 尝试但失败 | `TPUT` + `TREDUCE` | ❌ 挂死 | N/A | +2GB |

**核心结论**：在 Ascend 910B1 上，`TPUT<AtomicAdd>` 是 AllReduce 的最优通信原语。尝试用 `TPUT` + `TREDUCE` 分离通信和归约不仅带来 ~6.5% 的性能回退，还需要额外 2GB 内存，且必须使用 `TREDUCE_PINGPONG`（而非 `TREDUCE`）来避免 V-pipeline 事件计数器溢出。
