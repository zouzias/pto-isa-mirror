# 第一章：PTO 入门指南

> **PTO (Portable Tile Operations)** — 昇腾 NPU 的 Tile 编程模型。

本章介绍 PTO 的核心概念，并带你编写第一个内核。

---

## 1.1 什么是 PTO？

PTO 是昇腾 NPU 的**指令级 Tile 编程模型**。可以把它理解为 Tile 操作的"汇编语言"——你需要显式控制：

- **数据位置**（全局内存、UB、L1、L0）
- **数据移动**（TLOAD、TSTORE、TMOV）
- **执行时序**（流水线事件同步）

### PTO vs AscendC

| 方面 | PTO | AscendC |
|------|-----|---------|
| 抽象层级 | 底层 Tile | 高层 API |
| 内存管理 | 显式 TASSIGN | 自动分配 |
| 同步机制 | 手动 Event | 隐式处理 |
| 使用场景 | 编译器后端、微内核 | 应用层内核 |

**何时使用 PTO：**
- 构建编译器后端（PyPTO → PTO → 二进制）
- 实现需要精确控制的微内核
- 学习 NPU 架构内部原理

---

## 1.2 NPU 架构概览

在写代码之前，先了解目标硬件：

```
┌─────────────────────────────────────────────────────────────┐
│                      全局内存 (GM)                           │
│                       (HBM / DDR)                            │
└───────────────────────────────┬─────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
    ┌──────────┐         ┌──────────┐         ┌──────────┐
    │  AICore  │         │  AICore  │   ...   │  AICore  │
    │    0     │         │    1     │         │    N     │
    └──────────┘         └──────────┘         └──────────┘
```

### AICore 内部结构

```
┌─────────────────────────────────────────────────────────────┐
│                         AICore                               │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  L1 缓冲区 (512KB)                   │    │
│  │               Mat Tile（矩阵暂存）                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│        ┌──────────────────┼──────────────────┐              │
│        ▼                  ▼                  ▼              │
│  ┌──────────┐      ┌──────────┐       ┌───────────┐        │
│  │   L0A    │      │   L0B    │       │    L0C    │        │
│  │  (Left)  │      │ (Right)  │       │   (Acc)   │        │
│  │   64KB   │      │   64KB   │       │   128KB   │        │
│  └──────────┘      └──────────┘       └───────────┘        │
│        │                │                   ▲              │
│        └────────────────┴───────────────────┘              │
│                         │                                   │
│                    ┌────┴────┐                              │
│                    │  Cube   │  （矩阵乘法单元）             │
│                    └────┬────┘                              │
│                         │                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              UB - 统一缓冲区 (192KB)                  │   │
│  │               Vec Tile（向量运算）                    │   │
│  │                      ┌────────┐                      │   │
│  │                      │ Vector │                      │   │
│  │                      │  Unit  │                      │   │
│  │                      └────────┘                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 内存区域总结

| 区域 | 大小 (A2A3) | Tile 类型 | 用途 |
|------|-------------|-----------|------|
| UB | 192KB | `Vec` | 向量运算、临时数据 |
| L1 | 512KB | `Mat` | 矩阵暂存、大型 Tile |
| L0A | 64KB | `Left` | TMATMUL 左操作数 |
| L0B | 64KB | `Right` | TMATMUL 右操作数 |
| L0C | 128KB | `Acc` | TMATMUL 累加器 |

---

## 1.3 核心概念

### Tile（数据块）

**Tile** 是存储在片上内存中的固定大小二维数据块：

```cpp
// 定义 Tile 类型
using MyTile = Tile<
    TileType::Vec,    // 位置：UB
    float,            // 数据类型
    64,               // 行数（编译时固定）
    64,               // 列数（编译时固定）
    BLayout::RowMajor // 缓冲区布局
>;

// 创建 Tile 实例，指定"有效"维度
MyTile tile(32, 32);  // 只有 32x32 是有效数据，其余是填充区域
```

**关键点：** Tile 尺寸在编译时固定，但"有效区域"可以在运行时更小。

### GlobalTensor（全局张量）

**GlobalTensor** 是指向全局内存 (GM) 数据的句柄：

```cpp
// Shape: [1, 1, 1, 64, 64]  (5维形状)
// Stride: [64, 64, 64, 64, 1]  (行优先步长)
using Shape5D = Shape<1, 1, 1, 64, 64>;
using Stride5D = Stride<64, 64, 64, 64, 1>;
using GMTensor = GlobalTensor<float, Shape5D, Stride5D>;

GMTensor src(ptr);  // 包装 GM 指针
```

### 指令集

PTO 指令负责数据移动和计算：

| 指令 | 操作 |
|------|------|
| `TLOAD` | GM → Tile（加载到片上） |
| `TSTORE` | Tile → GM（存回全局内存） |
| `TASSIGN` | 将 Tile 映射到内存地址 |
| `TADD` | 逐元素加法 |
| `TMUL` | 逐元素乘法 |
| `TMATMUL` | 矩阵乘法 |

### Event（流水线同步）

NPU 有多条可以并行运行的流水线：

```
PIPE_MTE2 ─────┐ (GM → L1 加载)
PIPE_MTE1 ────┐│ (L1 → L0 移动)
PIPE_M ──────┐││ (Cube 计算)
PIPE_V ─────┐│││ (Vector 计算)
            ││││
            ▼▼▼▼
        ┌────────┐
        │ AICore │
        └────────┘
```

**Event** 确保操作在依赖的前序操作完成后再开始：

```cpp
Event<Op::TLOAD, Op::TADD> evt;  // TLOAD → TADD 依赖关系

evt = TLOAD(tile, src);          // 加载完成时返回事件
TADD(dst, tile, tile2, evt);     // 等待 evt 后再计算
```

---

## 1.4 第一个 PTO 内核：向量加法

我们来实现两个 64×64 浮点数组的 `C = A + B`。

### 完整内核代码

```cpp
// tadd_kernel.cpp
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int Rows, int Cols>
__global__ AICORE void runVectorAdd(
    __gm__ T __out__ *out,
    __gm__ T __in__ *src0,
    __gm__ T __in__ *src1)
{
    // 1. 定义张量/Tile 类型
    using Shape5D = Shape<1, 1, 1, Rows, Cols>;
    using Stride5D = Stride<Cols, Cols, Cols, Cols, 1>;  // 行优先
    using GMTensor = GlobalTensor<T, Shape5D, Stride5D>;
    using TileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor>;

    // 2. 创建 Tile
    TileData src0Tile(Rows, Cols);
    TileData src1Tile(Rows, Cols);
    TileData dstTile(Rows, Cols);

    // 3. 分配 UB 内存（手动偏移）
    TASSIGN(src0Tile, 0x00000);  // UB[0]
    TASSIGN(src1Tile, 0x10000);  // UB[64KB]
    TASSIGN(dstTile,  0x20000);  // UB[128KB]

    // 4. 包装 GM 指针
    GMTensor gmSrc0(src0);
    GMTensor gmSrc1(src1);
    GMTensor gmDst(out);

    // 5. 声明流水线同步事件
    Event<Op::TLOAD, Op::TADD> loadEvent;
    Event<Op::TADD, Op::TSTORE_VEC> storeEvent;

    // 6. 加载 → 计算 → 存储
    TLOAD(src0Tile, gmSrc0);
    loadEvent = TLOAD(src1Tile, gmSrc1);

    storeEvent = TADD(dstTile, src0Tile, src1Tile, loadEvent);

    TSTORE(gmDst, dstTile, storeEvent);
    out = gmDst.data();
}

// 主机端调用的启动函数
template <typename T, int Rows, int Cols>
void LaunchVectorAdd(T *out, T *src0, T *src1, void *stream) {
    runVectorAdd<T, Rows, Cols><<<1, nullptr, stream>>>(out, src0, src1);
}

// 显式实例化
template void LaunchVectorAdd<float, 64, 64>(float*, float*, float*, void*);
```

### 代码解析

**步骤 1-2：定义类型并创建 Tile**

```cpp
using TileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor>;
TileData src0Tile(Rows, Cols);  // 整个 Tile 都是有效数据
```

**步骤 3：分配 UB 内存**

```cpp
TASSIGN(src0Tile, 0x00000);  // 64×64×4 = 16KB，从偏移 0 开始
TASSIGN(src1Tile, 0x10000);  // 64KB 偏移（留有余量）
TASSIGN(dstTile,  0x20000);  // 128KB 偏移
```

**为什么要手动偏移？** PTO 给你完全的控制权。你需要负责确保 Tile 不会重叠（除非你有意让它们共享内存）。

**步骤 4-5：包装指针并声明事件**

```cpp
GMTensor gmSrc0(src0);        // GM 句柄
Event<Op::TLOAD, Op::TADD> loadEvent;  // 同步点
```

**步骤 6：带正确同步的执行流程**

```cpp
TLOAD(src0Tile, gmSrc0);              // 触发第一个加载（异步）
loadEvent = TLOAD(src1Tile, gmSrc1);  // 触发第二个加载，获取事件

storeEvent = TADD(dstTile, src0Tile, src1Tile, loadEvent);  // 等待加载完成

TSTORE(gmDst, dstTile, storeEvent);   // 等待 TADD 完成
```

---

## 1.5 数据流向

```
全局内存                      UB                       全局内存
┌───────────┐              ┌────────────┐           ┌───────────┐
│   src0    │───TLOAD───▶  │ src0Tile   │           │           │
│   src1    │───TLOAD───▶  │ src1Tile   │           │           │
│           │              │     │      │           │           │
│           │              │   TADD     │           │           │
│           │              │ dstTile=   │──TSTORE─▶ │    out    │
│           │              │  A + B     │           │           │
└───────────┘              └────────────┘           └───────────┘

流水线：
  MTE2 ──────────────▶ V ──────────────▶ MTE3
  (加载)               (计算)             (存储)
         loadEvent              storeEvent
```

---

## 1.6 运行测试

### 前置条件

```bash
# 加载 CANN 环境
source /usr/local/Ascend/cann/set_env.sh

# 克隆 pto-isa（如需要）
git clone https://gitcode.com/cann/pto-isa.git
cd pto-isa
```

### 使用 run_st.py

运行 PTO 测试最简单的方式：

```bash
# 在 A3 模拟器上运行
python3 tests/script/run_st.py -r sim -v a3 -t tadd_tutorial

# 在 A5 模拟器上运行
python3 tests/script/run_st.py -r sim -v a5 -t tadd_tutorial
```

该脚本会自动：
1. 编译内核和测试框架
2. 生成黄金数据
3. 在 NPU 模拟器上运行
4. 比较结果

---

## 1.7 常见陷阱

### 缺少 Event

❌ **错误写法：**
```cpp
TLOAD(tile, src);
TADD(dst, tile, tile2);  // 可能在加载完成前就执行了！
```

✅ **正确写法：**
```cpp
Event<Op::TLOAD, Op::TADD> evt;
evt = TLOAD(tile, src);
TADD(dst, tile, tile2, evt);  // 等待加载完成
```

### 内存重叠

❌ **错误写法：**
```cpp
TASSIGN(tile1, 0x0);
TASSIGN(tile2, 0x100);  // 如果 tile1 大于 256 字节就会重叠！
```

✅ **正确写法：** 根据 Tile 大小计算偏移：
```cpp
size_t tileBytes = Rows * Cols * sizeof(T);
TASSIGN(tile1, 0x0);
TASSIGN(tile2, tileBytes);  // 无重叠
```

### 错误的头文件

❌ `#include <pto/pto.h>` — 旧路径，可能不可用
✅ `#include <pto/pto-inst.hpp>` — 当前正确的头文件

---

## 1.8 本章小结

本章你学到了：

- **PTO** 是昇腾 NPU 的底层 Tile 编程模型
- **内存层次**：GM → L1 → L0 → 计算 → L0 → L1 → GM
- **核心类型**：`Tile`、`GlobalTensor` 和基于模板的形状定义
- **指令集**：`TLOAD`、`TSTORE`、`TASSIGN`、`TADD`
- **Event** 用于跨流水线同步操作

### 下一步

- **第二章**：矩阵乘法 — 使用 Cube 单元和 TMATMUL
- **第三章**：动态形状 — 处理运行时维度
- **第四章**：多核并行 — 在多个 AICore 间分配工作

---

## 快速参考

### Tile 类型 → 内存映射

| TileType | 内存 | 用途 |
|----------|------|------|
| `Vec` | UB | 向量运算 |
| `Mat` | L1 | 矩阵暂存 |
| `Left` | L0A | TMATMUL 左操作数 |
| `Right` | L0B | TMATMUL 右操作数 |
| `Acc` | L0C | TMATMUL 累加器 |

### 常用指令

| 指令 | 来源 | 目标 | 描述 |
|------|------|------|------|
| `TLOAD` | GM | Tile | 从全局内存加载 |
| `TSTORE` | Tile | GM | 存储到全局内存 |
| `TASSIGN` | — | — | 将 Tile 映射到地址 |
| `TADD` | Tiles | Tile | 逐元素加法 |
| `TMUL` | Tiles | Tile | 逐元素乘法 |

### Event 模板

```cpp
Event<Op::Producer, Op::Consumer> event;
event = PRODUCER_OP(...);
CONSUMER_OP(..., event);
```

---

> 📖 继续阅读 [第二章：矩阵乘法](chapter2-matmul.md)
