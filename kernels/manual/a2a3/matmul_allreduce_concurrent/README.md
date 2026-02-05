# MATMUL + AllReduce Concurrent Demo

本示例展示了如何在昇腾 AI Core 上实现 **Cube 单元与 Vector 单元的混合编译和并发执行**。

## 概述

在昇腾 AI Core 架构中，每个 AI Core 包含：
- **1 个 Cube 单元**：用于矩阵计算（GEMM）
- **2 个 Vector 单元**：用于向量计算和通信操作

本示例利用这一硬件特性，让 **Cube 单元执行矩阵乘法 (GEMM)**，同时让 **Vector 单元执行 AllReduce 通信**，两者在同一个 kernel 中并发运行，实现计算与通信的重叠（Overlap）。

## 架构原理

```
┌─────────────────────────────────────────────────────┐
│                    AI Core                          │
├─────────────────────┬───────────────────────────────┤
│     Cube 单元       │       Vector 单元 (x2)        │
│   ┌─────────────┐   │   ┌─────────────────────────┐ │
│   │             │   │   │  SubBlock 0  SubBlock 1 │ │
│   │    GEMM     │   │   │  ┌───────┐  ┌───────┐   │ │
│   │  (矩阵乘法)  │   │   │  │AllRed │  │AllRed │   │ │
│   │             │   │   │  │uce    │  │uce    │   │ │
│   └─────────────┘   │   │  └───────┘  └───────┘   │ │
│                     │   └─────────────────────────┘ │
└─────────────────────┴───────────────────────────────┘
         ↑                          ↑
         │                          │
    __DAV_CUBE__               __DAV_VEC__
    编译时宏                    编译时宏
```

## 关键技术点

### 1. 混合架构编译

使用 `--cce-aicore-arch=dav-c220` 编译选项，编译器会同时生成 Cube 和 Vector 代码：

```cmake
target_compile_options(${NAME}_kernel PRIVATE 
    ${CMAKE_CCE_COMPILE_OPTIONS} 
    --cce-aicore-arch=dav-c220   # 混合架构
    -DMEMORY_BASE -std=c++17)
```

### 2. 编译时条件分支

使用编译时宏来区分 Cube 和 Vector 代码路径：

```cpp
// 检测编译时宏
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

// 在 kernel 中使用 if constexpr 进行编译时分支
if constexpr (DAV_CUBE) {
    // GEMM 代码 - 只在 Cube 单元上执行
    RunGEMMOnCore<...>(...);
}

if constexpr (DAV_VEC) {
    // AllReduce 代码 - 只在 Vector 单元上执行
    RunAllReduceOnCore<...>(...);
}
```

### 3. 并发执行

当 kernel 启动时：
- Cube 单元自动执行 `DAV_CUBE` 分支中的代码
- Vector 单元自动执行 `DAV_VEC` 分支中的代码
- 两者**同时运行**，无需额外同步

## 文件结构

```
matmul_allreduce_concurrent/
├── CMakeLists.txt                          # 构建配置
├── run.sh                                  # 运行脚本
├── main.cpp                                # Host 端程序入口
├── matmul_allreduce_concurrent_kernel.cpp  # Device 端 kernel 实现
├── matmul_allreduce_concurrent.h           # 配置和统计结构定义
├── common.hpp                              # 通用工具函数
└── README.md                               # 本文档
```

## 使用方法

### 环境要求

> **⚠️ 重要**：本示例必须使用 **CANN 8.5.0.alpha002** 或更高版本编译。

### 环境准备

确保已设置以下环境变量：

```bash
# CANN 版本要求: 8.5.0.alpha002+
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export SHMEM_HOME_PATH=/path/to/shmem
```

### 编译和运行

```bash
# 普通模式运行
./run.sh

# 启用 Debug 模式（开启 cce::printf 输出）
./run.sh -d

# 指定参数运行
./run.sh -r npu -v Ascend910B4 -d
```

### 命令行选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `-r, --run-mode` | 运行模式：npu 或 sim | npu |
| `-v, --soc-version` | SoC 版本 | Ascend910B4 |
| `-d, --debug` | 启用 Debug 模式 | 关闭 |
| `-h, --help` | 显示帮助信息 | - |

## 配置参数

在 `matmul_allreduce_concurrent.h` 中可以调整以下参数：

```cpp
// AI Core 数量
#define TOTAL_BLOCK_NUM 4

// GEMM 使用的 Cube 数量（每个 block 1 个 Cube）
#define GEMM_BLOCK_NUM TOTAL_BLOCK_NUM

// AllReduce 使用的 Vector 数量（每个 block 2 个 Vector）
#define ALLREDUCE_BLOCK_NUM TOTAL_BLOCK_NUM * 2
```

## 性能分析

运行后会输出性能统计信息：

```
================================================================
  MATMUL + ALLREDUCE Concurrent Demo Results
================================================================
  GEMM Performance:
    - Matrix size:      128x64 * 64x128
    - Avg latency:      XX.XX us

  AllReduce Performance:
    - Data size:        XX KB
    - Avg latency:      XX.XX us

  Concurrent Analysis:
    - Effective time:   XX.XX us  (实际执行时间)
    - Sequential time:  XX.XX us  (串行执行时间)
    - Overlap savings:  XX.XX%    (节省的时间百分比)
================================================================
```

## 注意事项

1. **同步点**：Cube 和 Vector 单元有各自独立的同步机制，跨单元同步需要使用 `ShmemDeviceBarrierAll()` 等全局同步原语。

2. **内存访问**：Cube 和 Vector 单元共享 Global Memory，但有独立的 L1/UB 缓存，需注意数据一致性。

3. **调试输出**：使用 `-d` 选项启用 `cce::printf` 调试输出。由于多个单元并发执行，printf 输出可能交错。

## 相关文档

- [PTO-ISA 机器模型](../../../../docs/mkdocs/src/manual/02-machine-model.md)
- [TPRINT 调试输出](../../../../docs/isa/TPRINT.md)
- [通信原语 API](../../../../docs/isa/comm/README.md)