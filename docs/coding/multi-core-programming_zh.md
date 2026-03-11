# 多核并行编程

本文档介绍 PTO 的多核并行编程技术，帮助开发者充分利用 Ascend 多核架构实现高性能算子。

## 目录

- [1. 多核架构概述](#1-多核架构概述)
- [2. SPMD 编程模式](#2-spmd-编程模式)
- [3. MPMD 编程模式](#3-mpmd-编程模式)
- [4. 负载均衡](#4-负载均衡)
- [5. 核间通信](#5-核间通信)
- [6. 性能优化](#6-性能优化)

---

## 1. 多核架构概述

### 1.1 Ascend 多核架构

**硬件配置**：
- A2/A3：24 个 AI Core
- A5：更多核心（具体数量依型号）

**架构特点**：
```
┌─────────────────────────────────┐
│         Host CPU                │
└────────────┬────────────────────┘
             │
    ┌────────┴────────┐
    │   NPU Device    │
    │  ┌───┬───┬───┐  │
    │  │C0 │C1 │...│  │  AI Cores
    │  └───┴───┴───┘  │
    │  ┌───────────┐  │
    │  │    GM     │  │  Global Memory
    │  └───────────┘  │
    └─────────────────┘
```

**核心特性**：
- 每个核心独立执行
- 共享全局内存（GM）
- 独立的 L1 缓存
- 通过 GM 进行核间通信

### 1.2 并行编程模型

**两种主要模式**：

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| **SPMD** | 所有核运行相同代码 | 规则的数据并行 |
| **MPMD** | 不同核运行不同代码 | 流水线、生产者-消费者 |

---

## 2. SPMD 编程模式

### 2.1 基本概念

**SPMD (Single Program, Multiple Data)**：
- 所有核心执行相同的程序
- 通过 `block_idx` 区分不同的数据块
- 最常用的并行模式

### 2.2 基础示例

**向量加法**：
```cpp
__global__ __aicore__ void VecAddKernel(__gm__ float* out,
                                        __gm__ const float* in0,
                                        __gm__ const float* in1,
                                        uint32_t totalLength) {
  // 获取当前核心 ID
  int block_idx = get_block_idx();
  int block_num = get_block_num();
  
  // 计算当前核心负责的数据范围
  int elements_per_block = (totalLength + block_num - 1) / block_num;
  int start = block_idx * elements_per_block;
  int end = min(start + elements_per_block, totalLength);
  
  // 处理当前块
  for (int i = start; i < end; i += TILE_SIZE) {
    int size = min(TILE_SIZE, end - i);
    
    using TileT = Tile<TileType::Vec, float, 8, 256>;
    TileT a, b, c;
    
    TLOAD(a, GlobalTensor(in0 + i));
    TLOAD(b, GlobalTensor(in1 + i));
    TADD(c, a, b);
    TSTORE(GlobalTensor(out + i), c);
  }
}
```

### 2.3 2D 数据划分

**矩阵乘法示例**：
```cpp
__global__ __aicore__ void MatMulKernel(__gm__ float* C,
                                        __gm__ const float* A,
                                        __gm__ const float* B,
                                        int M, int K, int N) {
  // 获取核心 ID
  int block_idx = get_block_idx();
  
  // 2D 划分：M 和 N 维度
  int blocks_m = (M + TILE_M - 1) / TILE_M;
  int blocks_n = (N + TILE_N - 1) / TILE_N;
  
  int block_m = block_idx / blocks_n;
  int block_n = block_idx % blocks_n;
  
  // 计算当前核心负责的矩阵块
  int m_start = block_m * TILE_M;
  int n_start = block_n * TILE_N;
  
  // 确保不越界
  if (m_start >= M || n_start >= N) return;
  
  int m_size = min(TILE_M, M - m_start);
  int n_size = min(TILE_N, N - n_start);
  
  // 执行矩阵乘法
  TileAcc acc;
  TFILL(acc, 0);
  
  for (int k = 0; k < K; k += TILE_K) {
    int k_size = min(TILE_K, K - k);
    
    TLOAD(tileA, A[m_start:m_start+m_size, k:k+k_size]);
    TLOAD(tileB, B[k:k+k_size, n_start:n_start+n_size]);
    TMATMUL_ACC(acc, tileA, tileB);
  }
  
  TSTORE(C[m_start:m_start+m_size, n_start:n_start+n_size], acc);
}
```

### 2.4 3D 数据划分

**卷积示例**：
```cpp
__global__ __aicore__ void ConvKernel(...) {
  int block_idx = get_block_idx();
  
  // 3D 划分：Batch, Height, Width
  int blocks_h = (H + TILE_H - 1) / TILE_H;
  int blocks_w = (W + TILE_W - 1) / TILE_W;
  
  int block_b = block_idx / (blocks_h * blocks_w);
  int block_h = (block_idx / blocks_w) % blocks_h;
  int block_w = block_idx % blocks_w;
  
  // 处理当前 3D 块
  process_conv_block(block_b, block_h, block_w);
}
```

---

## 3. MPMD 编程模式

### 3.1 基本概念

**MPMD (Multiple Program, Multiple Data)**：
- 不同核心执行不同的程序
- 适合流水线和生产者-消费者模式
- 需要核间同步

### 3.2 任务分派模式

**方法1：单入口 + switch**
```cpp
__global__ __aicore__ void MPMDKernel(__gm__ float* out,
                                      __gm__ const float* in,
                                      uint32_t task_id) {
  switch (task_id) {
    case 0:
      ProducerTask(out, in);
      break;
    case 1:
      ConsumerTask(out, in);
      break;
    case 2:
      ProcessorTask(out, in);
      break;
    default:
      break;
  }
}
```

**方法2：多入口**
```cpp
// 生产者 kernel
__global__ __aicore__ void ProducerKernel(...) {
  // 生产数据
  for (int i = 0; i < N; i++) {
    produce_data(buffer[i]);
    signal_consumer();  // 通知消费者
  }
}

// 消费者 kernel
__global__ __aicore__ void ConsumerKernel(...) {
  // 消费数据
  for (int i = 0; i < N; i++) {
    wait_producer();  // 等待生产者
    consume_data(buffer[i]);
  }
}
```

### 3.3 流水线模式

**三阶段流水线**：
```cpp
__global__ __aicore__ void PipelineKernel(__gm__ float* out,
                                          __gm__ const float* in,
                                          uint32_t stage_id) {
  switch (stage_id) {
    case 0:  // Stage 1: Load
      for (int i = 0; i < N; i++) {
        TLOAD(buffer1[i], in[i]);
        signal_stage2();
      }
      break;
      
    case 1:  // Stage 2: Compute
      for (int i = 0; i < N; i++) {
        wait_stage1();
        TCOMPUTE(buffer2[i], buffer1[i]);
        signal_stage3();
      }
      break;
      
    case 2:  // Stage 3: Store
      for (int i = 0; i < N; i++) {
        wait_stage2();
        TSTORE(out[i], buffer2[i]);
      }
      break;
  }
}
```

---

## 4. 负载均衡

### 4.1 静态负载均衡

**均匀划分**：
```cpp
// 方法1：简单均分
int elements_per_block = totalLength / block_num;
int start = block_idx * elements_per_block;
int end = (block_idx == block_num - 1) ? 
          totalLength : start + elements_per_block;

// 方法2：向上取整均分
int elements_per_block = (totalLength + block_num - 1) / block_num;
int start = block_idx * elements_per_block;
int end = min(start + elements_per_block, totalLength);
```

**2D 均匀划分**：
```cpp
// 计算最优的 2D 划分
int blocks_m = (int)sqrt(block_num * M / N);
int blocks_n = block_num / blocks_m;

// 调整以充分利用所有核心
while (blocks_m * blocks_n < block_num && blocks_n < N / TILE_N) {
  blocks_n++;
  blocks_m = block_num / blocks_n;
}
```

### 4.2 动态负载均衡

**工作窃取模式**：
```cpp
// 使用原子操作实现动态任务分配
__gm__ atomic<int> next_task = 0;

__global__ __aicore__ void DynamicKernel(...) {
  while (true) {
    // 原子获取下一个任务
    int task_id = next_task.fetch_add(1);
    
    if (task_id >= total_tasks) break;
    
    // 处理任务
    process_task(task_id);
  }
}
```

### 4.3 负载不均衡检测

**检测方法**：
```cpp
// 记录每个核心的执行时间
#ifdef PROFILE
  auto start = GetTime();
  
  // 执行任务
  process_block(block_idx);
  
  auto end = GetTime();
  execution_times[block_idx] = end - start;
#endif

// 分析负载均衡性
float max_time = *max_element(execution_times.begin(), 
                              execution_times.end());
float min_time = *min_element(execution_times.begin(), 
                              execution_times.end());
float imbalance = (max_time - min_time) / max_time;

if (imbalance > 0.2) {
  printf("Warning: Load imbalance detected: %.2f%%\n", 
         imbalance * 100);
}
```

---

## 5. 核间通信

### 5.1 通过全局内存通信

**基本模式**：
```cpp
// 核心 0：写入数据
__global__ __aicore__ void Writer(__gm__ float* shared_buffer) {
  if (get_block_idx() == 0) {
    TLOAD(tile, local_data);
    TSTORE(shared_buffer, tile);
    // 设置标志表示数据已就绪
    shared_buffer[FLAG_OFFSET] = 1;
  }
}

// 核心 1：读取数据
__global__ __aicore__ void Reader(__gm__ float* shared_buffer) {
  if (get_block_idx() == 1) {
    // 等待数据就绪
    while (shared_buffer[FLAG_OFFSET] != 1) {
      // 自旋等待
    }
    TLOAD(tile, shared_buffer);
    process(tile);
  }
}
```

### 5.2 使用原子操作同步

**计数器同步**：
```cpp
__gm__ atomic<int> counter = 0;

__global__ __aicore__ void SyncKernel(...) {
  // 每个核心完成工作后增加计数器
  process_local_work();
  
  counter.fetch_add(1);
  
  // 等待所有核心完成
  while (counter.load() < block_num) {
    // 自旋等待
  }
  
  // 继续下一阶段
  next_stage_work();
}
```

### 5.3 屏障同步

**软件屏障**：
```cpp
class Barrier {
  __gm__ atomic<int> counter;
  __gm__ atomic<int> generation;
  int num_threads;
  
public:
  void wait() {
    int gen = generation.load();
    
    if (counter.fetch_add(1) == num_threads - 1) {
      // 最后一个到达的线程
      counter.store(0);
      generation.fetch_add(1);
    } else {
      // 等待所有线程到达
      while (generation.load() == gen) {
        // 自旋等待
      }
    }
  }
};

__global__ __aicore__ void BarrierKernel(...) {
  Barrier barrier(block_num);
  
  // 阶段 1
  phase1_work();
  barrier.wait();
  
  // 阶段 2
  phase2_work();
  barrier.wait();
  
  // 阶段 3
  phase3_work();
}
```

---

## 6. 性能优化

### 6.1 减少核间通信

**策略1：增大数据块**
```cpp
// 不好：频繁通信
for (int i = 0; i < N; i++) {
  process_small_block(i);
  sync_with_other_cores();  // 每次都同步
}

// 好：批量处理
for (int i = 0; i < N; i += BATCH_SIZE) {
  process_large_block(i, BATCH_SIZE);
  sync_with_other_cores();  // 批量同步
}
```

**策略2：本地化计算**
```cpp
// 尽量让每个核心独立完成工作
__global__ __aicore__ void LocalizedKernel(...) {
  int block_idx = get_block_idx();
  
  // 每个核心处理完整的子问题
  // 无需与其他核心通信
  process_independent_subproblem(block_idx);
}
```

### 6.2 优化数据划分

**考虑数据局部性**：
```cpp
// 2D 矩阵：按块划分而非按行/列
// 好：每个核心访问连续的内存块
for (int bm = 0; bm < blocks_m; bm++) {
  for (int bn = 0; bn < blocks_n; bn++) {
    int block_id = bm * blocks_n + bn;
    if (block_id == get_block_idx()) {
      process_block(bm, bn);
    }
  }
}
```

### 6.3 避免伪共享

**问题**：
```cpp
// 不好：多个核心写入相邻位置
__gm__ float results[NUM_CORES];

__global__ __aicore__ void BadKernel(...) {
  int idx = get_block_idx();
  results[idx] = compute();  // 可能导致缓存行冲突
}
```

**解决方案**：
```cpp
// 好：使用 padding 避免伪共享
constexpr int CACHE_LINE_SIZE = 64;
constexpr int PADDING = CACHE_LINE_SIZE / sizeof(float);

__gm__ float results[NUM_CORES * PADDING];

__global__ __aicore__ void GoodKernel(...) {
  int idx = get_block_idx();
  results[idx * PADDING] = compute();  // 避免缓存行冲突
}
```

### 6.4 性能测量

**测量核心利用率**：
```cpp
#ifdef PROFILE
  __gm__ uint64_t start_times[NUM_CORES];
  __gm__ uint64_t end_times[NUM_CORES];
  
  __global__ __aicore__ void ProfileKernel(...) {
    int idx = get_block_idx();
    
    start_times[idx] = GetCycles();
    
    // 执行工作
    do_work();
    
    end_times[idx] = GetCycles();
  }
  
  // 分析结果
  uint64_t max_time = 0;
  uint64_t total_time = 0;
  
  for (int i = 0; i < NUM_CORES; i++) {
    uint64_t time = end_times[i] - start_times[i];
    max_time = max(max_time, time);
    total_time += time;
  }
  
  float efficiency = (float)total_time / (NUM_CORES * max_time);
  printf("Core efficiency: %.2f%%\n", efficiency * 100);
#endif
```

---

## 7. 最佳实践

### 7.1 选择合适的并行模式

**决策树**：
```
数据可以均匀划分？
├─ 是 → 使用 SPMD
│   └─ 数据是规则的矩阵/张量？
│       ├─ 是 → 2D/3D 划分
│       └─ 否 → 1D 划分
└─ 否 → 考虑 MPMD 或动态负载均衡
    └─ 有明显的流水线阶段？
        ├─ 是 → 使用 MPMD 流水线
        └─ 否 → 使用动态任务分配
```

### 7.2 优化检查清单

**并行设计**：
- [ ] 选择了合适的并行模式（SPMD/MPMD）
- [ ] 数据划分均匀
- [ ] 考虑了数据局部性
- [ ] 最小化核间通信

**负载均衡**：
- [ ] 每个核心的工作量相近
- [ ] 处理了边界情况
- [ ] 测量了核心利用率

**同步优化**：
- [ ] 只在必要时同步
- [ ] 使用细粒度同步
- [ ] 避免死锁

**性能验证**：
- [ ] 测量了并行加速比
- [ ] 分析了核心利用率
- [ ] 识别了性能瓶颈

---

## 8. 实战案例

### 案例1：GEMM 多核优化

**2D 划分策略**：
```cpp
// 24 核：4×6 划分
constexpr int BLOCKS_M = 4;
constexpr int BLOCKS_N = 6;

__global__ __aicore__ void GEMMKernel(...) {
  int block_idx = get_block_idx();
  int block_m = block_idx / BLOCKS_N;
  int block_n = block_idx % BLOCKS_N;
  
  // 每个核心处理 M/4 × N/6 的块
  int m_start = block_m * (M / BLOCKS_M);
  int n_start = block_n * (N / BLOCKS_N);
  
  // 执行局部 GEMM
  local_gemm(m_start, n_start, ...);
}
```

**性能结果**：
- 单核：50 TFLOPS
- 24 核：1100 TFLOPS
- 加速比：22× (效率 92%)

### 案例2：Flash Attention 多核

**序列维度划分**：
```cpp
__global__ __aicore__ void FlashAttnKernel(...) {
  int block_idx = get_block_idx();
  int seq_per_block = (SEQ_LEN + block_num - 1) / block_num;
  
  int seq_start = block_idx * seq_per_block;
  int seq_end = min(seq_start + seq_per_block, SEQ_LEN);
  
  // 每个核心处理一段序列
  // 无需核间通信
  for (int i = seq_start; i < seq_end; i += TILE_SIZE) {
    process_attention_block(i);
  }
}
```

---

## 参考资源

- [编程模型](ProgrammingModel_zh.md)
- [流水线与并行执行](pipeline-parallel_zh.md)
- [性能调优最佳实践](performance-best-practices_zh.md)
- [GEMM 优化案例](../../kernels/manual/a2a3/gemm_performance/README_zh.md)





