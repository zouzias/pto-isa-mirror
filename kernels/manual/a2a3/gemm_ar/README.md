# GEMM AllReduce — 多卡 GEMM + AllReduce 融合算子

## 概述

本项目在昇腾 910B (A2/A3) NPU 上实现了一个 **GEMM + AllReduce 融合算子**，采用双流（Compute Stream + Comm Stream）计算通信重叠设计，通过 PTO 通信指令集在 HCCL RDMA 窗口上完成 AllReduce。

核心思路：每张卡持有独立的输入矩阵 A\_i\[M×K\] 和共享的权重矩阵 B\[K×N\]，各自计算完整 GEMM C\_i = A\_i × B，然后通过 **ReduceScatter + Reduce + AllGather** 三阶段通信完成 AllReduce 归约，每张卡最终得到 C\_final = Σ C\_i。典型应用场景为分布式训练中的梯度聚合。

**关键优化**：
- **全链路 FP16**：MatMul L0C FP32 累加器输出经 FixPipe 硬件自动 cast 为 FP16 写入 GM，通信数据和最终输出均为 FP16，通信量比 FP32 方案减半
- **零 Host Barrier**：三阶段通信合并为单次 kernel launch，阶段间同步通过 device-side `TNOTIFY`/`TWAIT` 信号完成，完全消除 `HcclHostBarrier` 的 host-device 往返开销
- **两级设备端同步**：跨 rank 同步（block 0 执行 RDMA 窗口原子操作）+ rank 内跨 block 同步（block 0 通过 GM flag 广播给其他 block）

**平台要求**：Ascend 910B (A2/A3)、CANN 8.5、bisheng 编译器、MPICH

## 快速开始

```bash
# 1. 环境准备
conda activate <your-conda-env>                    # 需含 Python + NumPy
source /usr/local/Ascend/cann-*/set_env.sh         # run.sh 也会自动检测

# 2. 构建并运行（8 卡）
cd pto-comm-isa/kernels/manual/a2a3/gemm_ar
./run.sh --nranks 8 --soc-version Ascend910B1

# 3. 指定起始设备编号
FIRST_DEVICE=0 ./run.sh --nranks 8 --soc-version Ascend910B1

# 4. 自定义 block 分配
./run.sh --nranks 8 --compute-blocks 20 --comm-blocks 4
```

`run.sh` 会自动完成：清理构建 → cmake → make → 计算 HCCL_BUFFSIZE → `mpirun -n $NRANKS ./gemm_allreduce`。

### 环境变量说明

`run.sh` 通过环境变量适配不同服务器环境，均有自动检测逻辑，大多数情况无需手动设置：

| 环境变量 | 用途 | 默认行为 |
|---------|------|---------|
| `ASCEND_CANN_PATH` | CANN `set_env.sh` 的完整路径 | 自动 glob `/usr/local/Ascend/cann-*/set_env.sh` 取最新版 |
| `MPI_SEARCH_DIRS` | MPI `bin/` 目录搜索路径（空格分隔） | 搜索 `/usr/local/mpich/bin`、`/home/mpich/bin`、`/home/*/mpich/bin`、`/home/*/*/mpich/bin` |
| `ASCEND_DRIVER_PATH` | Ascend driver 路径（CMake 使用） | 默认 `/usr/local/Ascend/driver` |
| `MPI_LIB_PATH` | `libmpi.so` 绝对路径（运行时动态加载） | 由 `run.sh` 根据找到的 MPI 自动设置 |
| `CONDA_PREFIX` | Conda 环境路径（自动由 `conda activate` 设置） | 激活 conda 环境后自动生效 |

## 数据并行策略（独立 A，共享 B）

```
语义: C_final = Σ (A_i × B)    for i in [0, nranks)

默认参数: M=5416, K=6144, N=1408, 8 ranks

每个 rank i 持有:
  A_i [M × K] = [5416 × 6144]    (FP16, 每 rank 独立)
  B   [K × N] = [6144 × 1408]    (FP16, 所有 rank 共享)

每个 rank 独立计算完整 GEMM:
  C_i [M × N] = A_i × B          (L0C FP32 累加，GM 输出 FP16)

AllReduce 归约 (全 FP16):
  C_final = C_0 + C_1 + ... + C_7

典型应用场景:
  分布式训练中，每个 rank 持有不同 batch 的梯度 A_i，
  乘以共享权重 B，然后 AllReduce 聚合结果。
```

## 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  Compute Stream (Cube)                  Comm Stream (Vector)        │
│                                                                     │
│  ┌──────────────────────┐                                           │
│  │ Compute Tile → TSTORE│──┐     ┌─────────────────────────────┐   │
│  │ (L0C FP32→GM FP16)   │  │     │ Single Kernel:              │   │
│  │ pipe_barrier(PIPE_ALL)│  │     │   GemmCommAllKernel         │   │
│  │ Enqueue tile_idx      │  │     │                             │   │
│  └──────────────────────┘  │     │ Phase 1: ReduceScatter      │   │
│  ┌──────────────────────┐  │     │   TPUT FP16 tile→owner rank │   │
│  │ Compute Tile → TSTORE│──┼────→│     ↓ DeviceBarrier (phase0)│   │
│  └──────────────────────┘  │     │ Phase 2: Local TREDUCE      │   │
│  ┌──────────────────────┐  │     │   owned tiles FP16 归约     │   │
│  │ Compute Tile → TSTORE│──┘     │     ↓ DeviceBarrier (phase1)│   │
│  └──────────────────────┘        │ Phase 3: AllGather TPUT     │   │
│         ...                      │   FP16 归约结果广播全 rank   │   │
│                                  └─────────────────────────────┘   │
│                                                                     │
│  关键特性:                                                           │
│  - 独立 A + 共享 B：每 rank 全量 K 计算，Cube 利用率高               │
│  - 全链路 FP16：GEMM→通信→输出全程 FP16，通信量比 FP32 减半          │
│  - 单次 kernel launch：三阶段合并，阶段间用 DeviceBarrier 同步        │
│  - 零 Host Barrier：跨 rank 同步由 TNOTIFY/TWAIT 在设备端完成        │
│  - 逐 tile 信号：计算完一个 tile 即可被通信侧消费                    │
│  - 通信量降低 4×：ReduceScatter 只发 owner，AllGather 只由 owner 广播 │
└──────────────────────────────────────────────────────────────────────┘
```

## 三阶段通信流水线

AllReduce 的三个阶段合并在单次 `GemmCommAllKernel` launch 中，阶段间通过 device-side `DeviceBarrier` 同步：

```
GemmCommAllKernel (单次 launch):
  Phase 1 (ReduceScatter):
    轮询 Ready Queue，取到就绪 tile 后 TPUT(FP16) 到 owner rank 的 recv_buffers
    ↓ DeviceBarrier(phase=0)  — 跨 rank + rank 内同步

  Phase 2 (Reduce):
    每个 rank 对自己 owned 的 tile 做 nranks-way TREDUCE(FP16) 求和
    ↓ DeviceBarrier(phase=1)  — 跨 rank + rank 内同步

  Phase 3 (AllGather):
    每个 rank 将归约结果 TPUT(FP16) 到所有其他 rank 的 reduced_output
```

### DeviceBarrier：两级设备端同步

```
DeviceBarrier(phase):
  pipe_barrier(PIPE_ALL)                    // 确保本 block 流水线刷完

  if block_idx == 0:                        // 只有 block 0 做跨 rank 信号
    for each remote rank r:
      TNOTIFY(remote signal_matrix[phase][my_rank], 1, AtomicAdd)   // 写远端
    for each remote rank r:
      TWAIT(local signal_matrix[phase][r], 1, GE)                   // 等远端
    TNOTIFY(local_broadcast_flag[phase], 1, Set)                    // 通知本 rank 其他 block
  else:
    TWAIT(local_broadcast_flag[phase], 1, GE)                       // 等 block 0 广播

  pipe_barrier(PIPE_ALL)
```

**为何需要两级同步**：`pipe_barrier(PIPE_ALL)` 只同步单个 AI Core 内部的流水线，无法同步同一 kernel 中不同 block（AI Core）之间的执行。block 0 完成跨 rank 信号交换后，通过 GM flag + `TNOTIFY/TWAIT` 广播给同 rank 的其他 block。

**Signal Matrix 布局**（位于 HCCL RDMA 窗口内）：
```
[0 .. MAX_RANKS-1]           Phase 0 跨 rank 计数器（RS 完成）
[MAX_RANKS .. 2*MAX_RANKS-1] Phase 1 跨 rank 计数器（Reduce 完成）
[2*MAX_RANKS]                Phase 0 rank 内广播 flag
[2*MAX_RANKS+1]              Phase 1 rank 内广播 flag
```
总计 `(2 * MAX_RANKS + 2) * sizeof(int32_t)` = 72 字节（对齐到 128 字节）。

### Tile 分配与归约

```
输出矩阵 C[M×N] 被划分为 tile:
  tile 大小: 128 × 256 × FP16 = 64 KB/tile
  M 方向: M_padded / 128 块
  N 方向: N_padded / 256 块

示例 (M=5416, N=1408, padded 5504x1536, 8 ranks):
  tile 数: 43 × 6 = 258

Owner 分配 (round-robin):
  owner(tile_idx) = tile_idx % nranks

通信量 (每 rank, FP16):
  ReduceScatter: (total - owned) tiles × 64KB
  AllGather:     owned tiles × (nranks-1) × 64KB
  通信量比 FP32 方案减半
```

## 内存布局与 HCCL 窗口

只有被远端 TPUT/TNOTIFY 写入的 buffer 需要放在 HCCL RDMA 窗口中，本地读写的 buffer 使用普通 `aclrtMalloc`。

| 缓冲区 | 大小 | 位置 | 原因 |
|--------|------|------|------|
| `recv_buffers` | nranks × M × N × 2B | **HCCL 窗口** | Phase 1 远端 TPUT 写入（FP16） |
| `reduced_output` | M × N × 2B | **HCCL 窗口** | Phase 3 远端 TPUT 写入（FP16） |
| `signal_matrix` | (2\*MAX\_RANKS+2) × 4B, 对齐 64B | **HCCL 窗口** | DeviceBarrier 跨 rank TNOTIFY 写入 |
| `gemm_output` | M × N × 2B | **aclrtMalloc** | 仅本地读写（FP16） |
| `src0_dev`, `src1_dev` | 输入矩阵（FP16） | **aclrtMalloc** | 仅本地读写 |

窗口大小由 `HCCL_BUFFSIZE` 环境变量控制，`run.sh` 自动计算：`(nranks + 1) × M × N × 2 / 1MB + 64MB`。

所有窗口内 buffer 必须在每个 rank 上分配在相同偏移处（通过 `WindowAlloc` 顺序递增分配器），以保证 `HcclRemotePtr` 地址转换的正确性。`signal_matrix` 在每轮迭代开始前通过 `aclrtMemset` 清零。

## 计算内核

### 核心参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| G\_M | 5416 | 矩阵 M 维度（自动 pad 到 128 对齐） |
| G\_K | 6144 | 矩阵 K 维度（每 rank 使用全量 K，须整除 G\_BASE\_K × G\_STEP\_KA） |
| G\_N | 1408 | 矩阵 N 维度（自动 pad 到 256 对齐） |
| G\_BASE\_M | 128 | Tile M 维度 |
| G\_BASE\_K | 64 | Tile K 维度 |
| G\_BASE\_N | 256 | Tile N 维度 |
| G\_STEP\_KA/KB | 4 | L1 缓存 K-slice 数（4× 减少 DMA 次数） |
| COMPUTE\_BLOCK\_NUM | 24 | 计算 block 数（可通过 `--compute-blocks` 配置） |
| COMM\_BLOCK\_NUM | 24 | 通信 block 数（可通过 `--comm-blocks` 配置） |

### FP16 输出

MatMul 计算在 L0C 使用 FP32 累加器，最终通过 `TSTORE` 写入 GM 时，FixPipe 硬件自动执行 FP32→FP16 cast（`copy_matrix_cc_to_gm` 指令的 `quantizationMode` 字段控制）。输入 A/B 仍为 FP16，无需手动精度转换。

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
1. MPI 初始化 → Rank 0 生成随机矩阵 (seed=42):
   - 每 rank 一个独立的 A_i[M×K]
   - 所有 rank 共享 B[K×N]
   - CPU golden reference = Σ(A_i × B) (FP32)
2. MPI Broadcast 输入数据到所有 rank
3. HCCL 通信器初始化（MPI 广播 root info，自动检测 MESH/RING 拓扑）
4. 分配 HCCL 窗口内存 (recv_buffers + reduced_output + signal_matrix)
5. Warmup (5 iter)
6. Compute-only 测量 (5 iter) — 纯 GEMM 性能基准
7. Sequential 测量 (10 iter)  — 计算→通信串行执行
8. Pipelined 测量 (10 iter)   — 计算‖通信双流重叠
9. 验证运行：FP16 输出转 FP32 后与 golden 对比 (eps=0.01)
10. 输出性能报告
```

每轮迭代前通过 `resetState` 清零所有缓冲区（含 `signal_matrix`），确保 DeviceBarrier 计数器归零。

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
├── gemm_ar_config.h            # 全局参数配置（矩阵维度、tile 大小、block 数量）
├── main.cpp                    # 入口：MPI 初始化、数据生成（独立 A + 共享 B）、HCCL 初始化、
│                               #   窗口分配（含 signal_matrix）、性能测量、FP16 验证
├── gemm_compute_kernel.cpp     # GEMM 计算内核（Cube 架构，L0C FP32→GM FP16 自动 cast）
├── comm_kernel.cpp             # 通信内核（Vector 架构，单 kernel 三阶段 AllReduce）
│                               #   包含 DeviceBarrier（两级设备端同步）
├── common.hpp                  # HcclRemotePtr 设备端包装（RDMA 窗口地址转换）
├── hccl_context.h              # HcclDeviceContext 结构体（每 rank 的 RDMA 窗口地址）
├── ready_queue.hpp             # 多 block 无锁 tile 队列（compute→comm 信号机制）
└── comm_mpi.h                  # MPI 动态加载包装（dlopen/dlsym，免硬链接依赖）
```

## 修改矩阵维度

修改 `gemm_ar_config.h` 中的 `CONFIG_G_M`、`CONFIG_G_K`、`CONFIG_G_N` 即可，所有源文件通过 include 共享配置。也可通过 CMake 参数传入：

```bash
cmake -DCONFIG_G_M=8192 -DCONFIG_G_K=8192 -DCONFIG_G_N=2048 ..
```

约束：K 必须能被 `G_BASE_K × G_STEP_KA`（默认 64×4=256）整除。`HCCL_BUFFSIZE` 由 `run.sh` 自动计算。

## 常见问题

| 问题 | 原因与解决 |
|------|-----------|
| `HCCL window too small` | 窗口不够大。检查 `HCCL_BUFFSIZE`，公式：`(nranks+1) × M × N × 2 bytes + margin` |
| `HcclGetRootInfo failed: 7` | 上次运行残留脏状态。执行 `rm -rf /dev/shm/sem.hccl*; ipcrm -a` 或等待 ~30s 重试 |
| HCCL 初始化后挂死 | rank 同步问题，检查所有 rank 是否到达 `CommMpiBarrier` |
| 通信 kernel 段错误 | 通常是窗口地址无效，验证 `windowsIn[]` 值非零 |
| DeviceBarrier 死锁 | signal\_matrix 未在迭代间清零，检查 `resetState` 是否 memset 了 signal\_matrix |
| 验证失败 max\_diff 较大 | FP16 精度有限，验证容差为 eps=0.01；若 diff 异常大，检查 DeviceBarrier 同步逻辑 |
| `aclInit repeat init` (100002) | 无害，代码已做保护，同一进程只调用一次 `aclInit` |
| `--allow-run-as-root` 失败 | 本项目使用 MPICH，此选项是 OpenMPI 专用 |

## 关键设计决策

### 为何选择 ReduceScatter + AllGather

每个 rank 计算的是完整的 C\_i = A\_i × B，AllReduce 对所有 C\_i 求和。相比直接广播（TPUT\<AtomicAdd\>到所有 rank），ReduceScatter + AllGather 的通信量降低 4×：
- 每个 tile 在 ReduceScatter 阶段只发送给唯一 owner（1 次 TPUT vs 7 次）
- AllGather 阶段只有 owner 广播归约结果
- 不同 rank 写入 owner 的不同 slot，无写冲突，因此可用 plain TPUT 代替 TPUT\<AtomicAdd\>

### 为何使用 FP16 而非 FP32 通信

MatMul 累加器在 L0C 使用 FP32，但最终输出经 TSTORE 写入 GM 时由 FixPipe 硬件自动 cast 为 FP16。全链路 FP16 使通信数据量减半，直接降低 RS 和 AG 阶段的 DMA 传输时间。FP16 精度对于大多数深度学习推理/训练场景已足够。

### 为何消除 Host Barrier

原设计中每个 `HcclHostBarrier` = `HcclBarrier` + `aclrtSynchronizeStream`，涉及 host-device 往返 + 跨 rank host 端同步，单次约 100-150 us。三次 barrier + 三次 kernel launch 共计约 300-600 us 纯开销。通过 `TNOTIFY`/`TWAIT` 在设备端完成跨 rank 同步，并合并为单次 kernel launch，消除了这些开销。

### 两级 DeviceBarrier 设计

`pipe_barrier(PIPE_ALL)` 只同步单个 AI Core 的流水线，无法同步同一 kernel 中不同 AI Core（block）。因此 DeviceBarrier 采用两级设计：
1. block 0 通过 HCCL RDMA 窗口的 `TNOTIFY`/`TWAIT` 完成跨 rank 同步
2. block 0 通过 GM flag 的 `TNOTIFY(Set)` 广播给同 rank 其他 block，其他 block 通过 `TWAIT` 等待

### 头文件包含顺序

`common.hpp`（通过 `kernel_operator.h`）必须在 `<iostream>` 之前包含，否则 `std::dec` 与 AscendC 宏冲突。

## 性能实测

8 卡 Ascend 910B，M=5416, K=6144, N=1408（padded 5504x1536），258 tiles。每 rank 计算完整 GEMM C\_i = A\_i × B，AllReduce 对 8 个 C\_i 求和。

性能数据待实测更新。

## 构建系统

- **编译器**：bisheng（CANN 内置 clang 15.0.5）
- **Cube kernel**：`--cce-aicore-arch=dav-c220-cube -DMEMORY_BASE`
- **Vec kernel**：`--cce-aicore-arch=dav-c220-vec -DMEMORY_BASE`
- **Host 可执行文件**：`-xc++` 标准编译
- **链接库**：`runtime`、`ascendcl`、`hcomm`、`tiling_api`
- pto-comm-isa 的 include 路径**必须放在首位**，以覆盖 CANN 自带的 `pto_tile.hpp`
