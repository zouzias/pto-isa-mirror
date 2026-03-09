# 核配置性能测试指南

本指南说明如何使用性能测试工具来测试单个通信核和计算核的效率，并自动配置最优的核数量。

## 概述

本目录包含三个测试程序：

1. **benchmark_single_comm_core**: 测试单个通信核的TPUT传输效率
2. **benchmark_single_compute_core**: 测试单个计算核的GEMM计算效率
3. **benchmark_optimal_config**: 主测试程序，自动测试并配置最优核数（总数为24）

## 编译

在编译主程序时，这些测试程序会自动编译：

```bash
cd build
cmake ..
make -j16
```

编译完成后，会在 `build/` 目录下生成：
- `benchmark_single_comm_core`
- `benchmark_single_compute_core`
- `benchmark_optimal_config`

## 使用方法

### 方法1: 自动测试和配置（推荐）

运行主测试程序，它会自动：
1. 测试单个通信核性能
2. 测试单个计算核性能
3. 计算最优配置
4. 更新源代码文件

```bash
cd build
./benchmark_optimal_config --nranks 8 --first-device 0 --total-cores 24
```

参数说明：
- `--nranks N`: 使用的rank数量（默认：8）
- `--first-device ID`: 第一个设备ID（默认：0）
- `--total-cores N`: 总核数（默认：24）

### 方法2: 使用已知值快速配置

如果已经知道单个核的性能值，可以使用 `--skip-benchmark` 选项：

```bash
./benchmark_optimal_config --skip-benchmark --total-cores 24
```

### 方法3: 单独测试

#### 测试单个通信核

**方法1：使用运行脚本（推荐）**
```bash
./run_benchmark_comm.sh --nranks 8 --first-device 0 --num-tiles 100
```

**方法2：直接运行（需要先设置环境变量）**
```bash
cd build
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /home/ntlab/qifeng/pypto/third_party_path/shmem/install/set_env.sh
./benchmark_single_comm_core --nranks 8 --first-device 0 --num-tiles 100
```

输出示例：
```
[BENCHMARK] Single Comm Core Performance:
  Tiles: 100
  Ranks: 8
  Avg Time: 1234.56 us
  Total Data: 256.00 MB
  Bandwidth: 1.36 GB/s
```

#### 测试单个计算核

**方法1：使用运行脚本（推荐）**
```bash
./run_benchmark_compute.sh --device 0
```

**方法2：直接运行（需要先设置环境变量）**
```bash
cd build
source /usr/local/Ascend/cann-8.5.0/set_env.sh
./benchmark_single_compute_core --device 0
```

输出示例：
```
[BENCHMARK] Single Compute Core Performance:
  Tiles: 100
  Avg Time: 567.89 us
  Total Ops: 134.22 GFLOP
  Performance: 11600.00 GFLOPS
```

## 配置原理

程序会根据以下信息计算最优配置：

1. **单个通信核带宽** (GB/s): 从 `benchmark_single_comm_core` 获取
2. **单个计算核性能** (GFLOPS): 从 `benchmark_single_compute_core` 获取
3. **总计算量**: M × K × N × 2 操作
4. **总通信量**: M × N × sizeof(float) × nranks 字节

最优配置的目标是让计算时间和通信时间尽可能匹配，以实现最佳的重叠效率。

## 配置更新

运行 `benchmark_optimal_config` 后，程序会自动更新以下文件：

- `comm_kernel.cpp`: 更新 `CONFIG_COMM_BLOCK_NUM` 和 `CONFIG_COMPUTE_BLOCK_NUM`
- `gemm_compute_kernel.cpp`: 更新 `CONFIG_COMPUTE_BLOCK_NUM`

更新后，需要重新编译：

```bash
cd build
cmake ..
make -j16
./gemm_allreduce --nranks 8
```

## 示例输出

```
=== Core Performance Benchmark ===
Testing individual core performance to determine optimal configuration...

Step 1: Benchmarking single comm core...
[BENCHMARK] Single Comm Core Performance:
  Bandwidth: 1.36 GB/s

Step 2: Benchmarking single compute core...
[BENCHMARK] Single Compute Core Performance:
  Performance: 11600.00 GFLOPS

=== Performance Analysis ===
Single Comm Core Bandwidth: 1.36 GB/s
Single Compute Core Performance: 11600.00 GFLOPS
Total Computation: 549.76 GFLOP
Total Communication: 2.00 GB

=== Optimal Configuration ===
Comm Cores: 4
Compute Cores: 20
Total Cores: 24
Expected Overlap Efficiency: 95.23%

[INFO] Updated source files with optimal configuration.
[INFO] Please rebuild and run the main program to test the new configuration.
```

## 注意事项

1. 测试需要多rank环境，确保有足够的设备可用
2. 测试时间可能较长，请耐心等待
3. 配置更新会修改源代码文件，建议先备份
4. 更新配置后必须重新编译才能生效

## 故障排除

### 问题：测试程序无法找到设备

确保已正确设置环境变量：
```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /home/ntlab/qifeng/pypto/third_party_path/shmem/install/set_env.sh
```

### 问题：Shmem初始化失败

检查：
- 是否有足够的设备
- 设备ID是否正确
- Shmem环境是否正确配置

### 问题：配置更新失败

手动检查并更新：
- `comm_kernel.cpp` 第102-107行
- `gemm_compute_kernel.cpp` 第156-159行
