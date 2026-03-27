# GEMM AllReduce — 多卡 GEMM + AllReduce 融合算子

## 概述

本项目在昇腾 910B (A2/A3) NPU 上实现了一个 **GEMM + AllReduce 融合算子**，采用双流（Compute Stream + Comm Stream）计算通信重叠设计，通过 PTO 通信指令集在 HCCL RDMA 窗口上完成 AllReduce。

核心思路：将全局矩阵乘法 C\[M×N\] = A\[M×K\] × B\[K×N\] 沿 K 维度拆分到多张卡上，每张卡独立计算局部 GEMM 后，通过 **ReduceScatter + AllGather** 三阶段通信完成 AllReduce 归约，每张卡最终得到完整的 C\_final。

**平台要求**：Ascend 910B (A2/A3)、CANN 8.5、bisheng 编译器、MPICH

## 快速开始

```bash
# 1. 环境准备
conda activate <your-conda-env>          # 需含 Python + NumPy
source /usr/local/Ascend/cann-*/set_env.sh  # 或设置 ASCEND_CANN_PATH

# 2. 构建并运行（8 卡）
cd pto-comm-isa/kernels/manual/a2a3/gemm_ar
./run.sh --nranks 8 --soc-version Ascend910B1

# 3. 指定起始设备编号
FIRST_DEVICE=0 ./run.sh --nranks 8 --soc-version Ascend910B1

# 4. 自定义 block 分配
./run.sh --nranks 8 --compute-blocks 20 --comm-blocks 4
```

`run.sh` 会自动完成：清理构建 → cmake → make → 计算 HCCL_BUFFSIZE → `mpirun -n $NRANKS ./gemm_allreduce`。

## 数据并行策略（K 维切分）

```
全局矩阵乘法：C[M×N] = A[M×K] × B[K×N]

默认参数: M=16384, K=16384, N=4096, 8 ranks

K 维按 rank 均分，每个 rank 持有:
  A_part [M × (K/r)] = [16384 × 2048]
  B_part [(K/r) × N] = [2048 × 4096]

每个 rank 独立计算局部 GEMM:
  C_rank [M × N] = A_part × B_part

AllReduce 归约:
  C_final = C_rank_0 + C_rank_1 + ... + C_rank_7

数学原理:
  A × B = [A_0 | A_1 | ... | A_{r-1}] × [B_0; B_1; ...; B_{r-1}]
        = Σ A_i × B_i
```

## 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  Compute Stream (Cube)                  Comm Stream (Vector)        │
│                                                                     │
│  ┌──────────────────────┐                                           │
│  │ Compute Tile → TSTORE│──┐                                        │
│  │ pipe_barrier(PIPE_ALL)│  │                                       │
│  │ Enqueue tile_idx      │  │ Ready Queue                           │
│  └──────────────────────┘  │                                        │
│  ┌──────────────────────┐  │     ┌─────────────────────────────┐   │
│  │ Compute Tile → TSTORE│──┼────→│ Phase 1: ReduceScatter TPUT │   │
│  └──────────────────────┘  │     │   每 tile 只发给 owner rank  │   │
│  ┌──────────────────────┐  │     │     ↓ HcclHostBarrier        │   │
│  │ Compute Tile → TSTORE│──┘     │ Phase 2: Local TREDUCE      │   │
│  └──────────────────────┘        │   仅对 owned tiles 归约      │   │
│         ...                      │     ↓ HcclHostBarrier        │   │
│                                  │ Phase 3: AllGather TPUT      │   │
│                                  │   归约结果广播到所有 rank     │   │
│                                  └─────────────────────────────┘   │
│                                                                     │
│  关键特性:                                                           │
│  - 双流并行：Cube 计算流 + Vector 通信流，计算与通信重叠执行          │
│  - 逐 tile 信号：计算完一个 tile 即可被通信侧消费                    │
│  - 通信量降低 4×：ReduceScatter 只发 owner，AllGather 只由 owner 广播 │
└──────────────────────────────────────────────────────────────────────┘
```

## 三阶段通信流水线

AllReduce 被拆分为 3 次 kernel launch，中间通过 host 端 HcclHostBarrier 同步：

```
Phase 1 (ReduceScatter):  GemmCommRSKernel
  ↓ 轮询 Ready Queue，取到就绪 tile 后 TPUT 到 owner rank 的 recv_buffers
  ↓ HcclHostBarrier + aclrtSynchronizeStream

Phase 2 (Reduce):         GemmCommReduceKernel
  ↓ 每个 rank 对自己 owned 的 tile 做 nranks-way TREDUCE 求和
  ↓ HcclHostBarrier + aclrtSynchronizeStream

Phase 3 (AllGather):      GemmCommAGKernel
  ↓ 每个 rank 将归约结果 TPUT 到所有其他 rank 的 reduced_output
  ↓ HcclHostBarrier + aclrtSynchronizeStream
```

为何需要 3 次 launch：HCCL 不提供 device 端跨 rank barrier，`HcclHostBarrier`（= `HcclBarrier` + `aclrtSynchronizeStream`）是唯一的同步原语，而它要求 kernel 已经返回。

### Tile 分配与归约

```
输出矩阵 C[M×N] 被划分为 tile:
  tile 大小: 128 × 256 (128 KB/tile)
  M 方向: 16384/128 = 128 块
  N 方向: 4096/256  = 16 块
  总 tile 数: 128 × 16 = 2048

Owner 分配 (round-robin):
  owner(tile_idx) = tile_idx % nranks
  → Rank 0 拥有: tile 0, 8, 16, ..., 2040 (256 个)
  → Rank 1 拥有: tile 1, 9, 17, ..., 2041 (256 个)
  → ...

通信量 (每 rank):
  ReduceScatter: 1792 tiles × 128KB = 224 MB (只发给 owner)
  AllGather:     256 tiles × 7 ranks × 128KB = 224 MB (owner 广播)
  合计: 448 MB/rank (相比广播方案的 1.75 GB 降低 ~4×)
```

## 内存布局与 HCCL 窗口

只有被远端 TPUT 写入的 buffer 需要放在 HCCL RDMA 窗口中，本地读写的 buffer 使用普通 `aclrtMalloc`。

| 缓冲区 | 大小 | 位置 | 原因 |
|--------|------|------|------|
| `recv_buffers` | nranks × M × N × 4B | **HCCL 窗口** | Phase 1 远端写入 |
| `reduced_output` | M × N × 4B | **HCCL 窗口** | Phase 3 远端写入 |
| `gemm_output` | M × N × 4B | **aclrtMalloc** | 仅本地读写 |
| `src0_dev`, `src1_dev` | 输入矩阵 | **aclrtMalloc** | 仅本地读写 |

窗口大小由 `HCCL_BUFFSIZE` 环境变量控制，`run.sh` 自动计算：`(nranks + 1) × M × N × 4 / 1MB + 64MB`。

所有窗口内 buffer 必须在每个 rank 上分配在相同偏移处（通过 `WindowAlloc` 顺序递增分配器），以保证 `HcclRemotePtr` 地址转换的正确性。

## 计算内核

### 核心参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| G\_M | 16384 | 矩阵 M 维度 |
| G\_K | 16384 | 矩阵 K 维度（全局，须整除 nranks） |
| G\_N | 4096 | 矩阵 N 维度 |
| G\_BASE\_M | 128 | Tile M 维度 |
| G\_BASE\_K | 64 | Tile K 维度 |
| G\_BASE\_N | 256 | Tile N 维度 |
| G\_STEP\_KA/KB | 4 | L1 缓存 K-slice 数（4× 减少 DMA 次数） |
| COMPUTE\_BLOCK\_NUM | 24 | 计算 block 数（可通过 `--compute-blocks` 配置） |
| COMM\_BLOCK\_NUM | 24 | 通信 block 数（可通过 `--comm-blocks` 配置） |

### 两级双缓冲流水线

```
时间 →
L1 (MTE2):  [TLOAD A0,B0]                [TLOAD A1,B1]              ...
L0 (MTE1):       [TEXTRACT k0] [k1] [k2] [k3] [TEXTRACT k0'] ...
Cube (M):             [TMATMUL k0] [ACC k1] [ACC k2] [ACC k3] [TMATMUL k0'] ...
                      ↑ 三级流水线完全并行 ↑
```

每个计算 block 负责一组 tile，对每个 tile：
1. **K-loop**：每 4 次迭代做一次 TLOAD（L1 缓存优化），每次迭代 TEXTRACT + TMATMUL\_ACC
2. **TSTORE**：写入 `gemm_output`
3. **pipe\_barrier(PIPE\_ALL)**：确保 GM 写入完成
4. **MultiBlockEnqueueFast**：入队 `tile_idx`，通知通信 kernel

### Ready Queue 无锁信号机制

- 每个计算 block 一个独立队列（单生产者单消费者，无需原子操作）
- 生产端：计算 kernel 通过 `PerBlockQueueEnqueueFast`（2 次 `dcci`）入队
- 消费端：通信 kernel 通过 `TTEST` 硬件指令轮询（非内存 busy-wait）
- 支持 `TWAIT` 阻塞等待，避免空轮询浪费

## 执行流程

```
1. MPI 初始化 → Rank 0 生成随机矩阵 (seed=42) + CPU golden reference
2. MPI Broadcast 输入数据到所有 rank
3. HCCL 通信器初始化（MPI 广播 root info，自动检测 MESH/RING 拓扑）
4. Warmup (5 iter)
5. Compute-only 测量 (5 iter) — 纯 GEMM 性能基准
6. Sequential 测量 (10 iter)  — 计算→通信串行执行
7. Pipelined 测量 (10 iter)   — 计算‖通信双流重叠
8. 验证运行 + golden 对比
9. 输出性能报告
```

## 性能报告说明

| 指标 | 含义 |
|------|------|
| Compute-only | 纯 GEMM 时间（无通信） |
| Sequential | 计算 → 通信串行，无重叠 |
| Pipelined | 计算 ‖ 通信，双流并行 |
| Speedup | Sequential / Pipelined |
| Time saved | 重叠带来的时间节省 |
| Overlap eff | 重叠效率（节省时间 / 较短阶段时间） |

## 文件结构

```
gemm_ar/
├── CMakeLists.txt              # 构建配置（3 个 target：cube kernel, vec kernel, host exe）
├── run.sh                      # 一键构建+运行脚本（自动计算 HCCL_BUFFSIZE、发现 MPI 路径）
├── main.cpp                    # 入口：MPI 初始化、数据生成、CPU golden 计算、数据广播
├── gemm_compute_kernel.cpp     # GEMM 计算内核（Cube 架构，L1 缓存 + ready queue 信号）
├── comm_kernel.cpp             # 通信内核（Vector 架构，3 阶段 AllReduce）
│                               #   + host 端编排、HCCL 初始化、性能测量、验证
├── common.hpp                  # HcclRemotePtr 设备端包装（RDMA 窗口地址转换）
├── hccl_context.h              # HcclDeviceContext 结构体（每 rank 的 RDMA 窗口地址）
├── ready_queue.hpp             # 多 block 无锁 tile 队列（compute→comm 信号机制）
└── comm_mpi.h                  # MPI 动态加载包装（dlopen/dlsym，免硬链接依赖）
```

## 修改矩阵维度

1. 在 `comm_kernel.cpp`、`gemm_compute_kernel.cpp`、`main.cpp` 三个文件中同步修改 `CONFIG_G_M`、`CONFIG_G_K`、`CONFIG_G_N`
2. K 必须能被 nranks 整除
3. `HCCL_BUFFSIZE` 由 `run.sh` 自动计算，无需手动调整

也可以通过 CMake 参数传入，无需修改源码：

```bash
cmake -DCONFIG_G_M=8192 -DCONFIG_G_K=8192 -DCONFIG_G_N=2048 ..
```

## 常见问题

| 问题 | 原因与解决 |
|------|-----------|
| `HCCL window too small` | 窗口不够大。检查 `HCCL_BUFFSIZE`，公式：`(nranks+1) × M × N × 4 bytes + margin` |
| `HcclGetRootInfo failed: 7` | 上次运行残留脏状态。执行 `rm -rf /dev/shm/sem.hccl*; ipcrm -a` 或等待 ~30s 重试 |
| HCCL 初始化后挂死 | rank 同步问题，检查所有 rank 是否到达 `CommMpiBarrier` |
| 通信 kernel 段错误 | 通常是窗口地址无效，验证 `windowsIn[]` 值非零 |
| `aclInit repeat init` (100002) | 无害，代码已做保护，同一进程只调用一次 `aclInit` |
| `--allow-run-as-root` 失败 | 本项目使用 MPICH，此选项是 OpenMPI 专用 |

## 关键设计决策

### 为何选择 ReduceScatter + AllGather

相比直接广播（TPUT\<AtomicAdd\>到所有 rank），ReduceScatter + AllGather 的通信量降低 4×：
- 每个 tile 在 ReduceScatter 阶段只发送给唯一 owner（1 次 TPUT vs 7 次）
- AllGather 阶段只有 owner 广播归约结果
- 不同 rank 写入 owner 的不同 slot，无写冲突，因此可用 plain TPUT 代替 TPUT\<AtomicAdd\>

### TREDUCE\_PINGPONG 防止事件计数器溢出

V 流水线事件计数器只有 2-bit 容量。普通 TREDUCE 在循环中持续使用 EVENT\_ID1 会导致溢出，TREDUCE\_PINGPONG 交替使用 EVENT\_ID1/EVENT\_ID2，每个计数器最大值始终为 1。

### 头文件包含顺序

`common.hpp`（通过 `kernel_operator.h`）必须在 `<iostream>` 之前包含，否则 `std::dec` 与 AscendC 宏冲突。

## 构建系统

- **编译器**：bisheng（CANN 内置 clang 15.0.5）
- **Cube kernel**：`--cce-aicore-arch=dav-c220-cube -DMEMORY_BASE`
- **Vec kernel**：`--cce-aicore-arch=dav-c220-vec -DMEMORY_BASE`
- **Host 可执行文件**：`-xc++` 标准编译
- **链接库**：`runtime`、`ascendcl`、`hcomm`、`tiling_api`
- pto-comm-isa 的 include 路径**必须放在首位**，以覆盖 CANN 自带的 `pto_tile.hpp`
