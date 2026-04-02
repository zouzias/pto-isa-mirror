# 融合算子开发指南

本指南为在 PTO Tile Library 中开发融合算子提供全面指导。

## 目录
1. [概述](#概述)
2. [文件结构](#文件结构)
3. [核心概念](#核心概念)
4. [融合模式](#融合模式)
5. [同步策略](#同步策略)
6. [内存管理](#内存管理)
7. [类型处理](#类型处理)
8. [最佳实践](#最佳实践)
9. [示例](#示例)

## 概述

融合算子将多个 PTO 指令组合成单个操作，以实现：
- 减少内存传输
- 提高流水线效率
- 最小化延迟
- 优化硬件资源利用率

常见的融合模式包括：
- **TAXPY**: TLOAD + TADD + TSTORE
- **TROWEXPANDADD**: TLOAD + TROWEXPANDADD + TSTORE
- **TCOLEXPANDADD**: TLOAD + TCOLEXPANDADD + TSTORE
- **TPARTADD**: TLOAD + TPARTADD + TSTORE
- **TMATMUL**: TLOAD + TMATMUL + TSTORE

## 文件结构

```cpp
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to License for details. You may not use this file except in compliance with License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;

namespace FusedOperatorName {

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runFusedOp(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
}

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchFusedOp(T *out, T *src0, T *src1, void *stream)
{
}

template void launchFusedOp<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);

} // namespace FusedOperatorName
```

## 核心概念

### GlobalTensor 和 Tile 声明
缺少DN排布
```cpp
using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
using DynStridDim5 = Stride<vRows * vCols, vRows * vCols, vRows * vCols, vCols, 1>;
using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
```

### 使用 TASSIGN 分配缓冲区

```cpp
TileData src0Tile(vRows, vCols);
TileData src1Tile(vRows, vCols);
TileData dstTile(vRows, vCols);

TASSIGN(src0Tile, 0x0);
TASSIGN(src1Tile, sizeof(T) * TileData::Numel);
TASSIGN(dstTile, 2 * sizeof(T) * TileData::Numel);
```

**关键点**：
- 每个 tile 必须有唯一的缓冲区地址
- 使用递增偏移量sizeof(T) * TileData::Numel
- 缓冲区大小由 tile 维度和数据类型决定

### GlobalTensor 初始化
补充动态shape/stride场景
```cpp
GlobalData src0Global(src0);
GlobalData src1Global(src1);
GlobalData dstGlobal(out);
```

## 融合模式

### 模式 1：基于事件的融合（推荐）

**使用场景**：具有清晰依赖关系的简单融合

```cpp
Event<Op::TLOAD, Op::TADD> event0;
Event<Op::TADD, Op::TSTORE_VEC> event1;

TLOAD(src0Tile, src0Global);
event0 = TLOAD(src1Tile, src1Global);
event1 = TADD(dstTile, src0Tile, src1Tile, event0);
TSTORE(dstGlobal, dstTile, event1);
```

**优势**：
- 自动依赖跟踪
- 编译器优化
- 代码更简洁
- 同时支持手动和自动模式

### 模式 2：手动标志同步

**使用场景**：具有自定义流水线控制的复杂融合

```cpp
TLOAD(src0Tile, src0Global);
TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
TADD(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
TSTORE(dstGlobal, dstTile);
```

**何时使用**：
- 需要对流水线阶段进行细粒度控制
- 复杂的多阶段融合
- 需要手动优化的性能关键代码

### 模式 3：标量操作融合

**示例：TAXPY (Y = a*X + Y)**

```cpp
Event<Op::TLOAD, Op::TAXPY> event0;
Event<Op::TAXPY, Op::TSTORE_VEC> event1;

TLOAD(src0Tile, src0Global);
event0 = TLOAD(dstTile, dstGlobal);
event1 = TAXPY(dstTile, src0Tile, (T)scalar, event0);
TSTORE(dstGlobal, dstTile, event1);
```

**关键点**：
- 标量参数必须转换为 tile 数据类型
- 输出 tile 可以重用为输入（原地操作）

### 模式 4：广播操作融合

**示例：TROWEXPANDADD**

```cpp
using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1Row, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
using TileDataSrc1 = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;

TileDataSrc1 src1Tile(validRow, 1);

TLOAD(src0Tile, src0Global);
TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
TROWEXPANDADD(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
TSTORE(dstGlobal, dstTile);
```

**关键点**：
- 广播数据使用不同的形状/布局
- 沿行广播使用 ColumnMajor 布局
- 全局张量使用 DN（Dense）布局

### 模式 5：矩阵操作融合

**示例：TMATMUL**

```cpp
using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

using LeftTile = TileLeft<U, M, K, validM, validK>;
using RightTile = TileRight<S, K, N, validK, validN>;
using AccTile = TileAcc<T, M, N, validM, validN>;

TileMatAData aMatTile;
TileMatBData bMatTile;
LeftTile aTile;
RightTile bTile;
AccTile cTile;

TLOAD(aMatTile, src0Global);
TLOAD(bMatTile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
TMOV(aTile, aMatTile);
TMOV(bTile, bMatTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
TSTORE(dstGlobal, cTile);
```

**关键点**：
- 矩阵操作使用 TileType::Mat
- 需要 TileLeft、TileRight、TileAcc 特化
- 多个流水线阶段（MTE2 -> MTE1 -> M -> FIX）
- TMOV 将 Mat tile 转换为操作 tile

## 同步策略

### 基于事件的同步

```cpp
Event<Op::Producer, Op::Consumer> event;

event = PRODUCER(outputTile, inputTile);
CONSUMER(nextTile, outputTile, event);
```

**流水线阶段**：
- `Op::TLOAD`: 从全局内存加载
- `Op::TSTORE_VEC`: 存储到全局内存
- `Op::TADD`、`Op::TMUL` 等：向量操作
- `Op::TMATMUL`: 矩阵乘法

### 手动标志同步

```cpp
set_flag(srcPipe, dstPipe, eventId);
wait_flag(srcPipe, dstPipe, eventId);
```

**流水线线类型**：
- `PIPE_MTE1`、`PIPE_MTE2`、`PIPE_MTE3`: 内存传输引擎
- `PIPE_V`: 向量单元
- `PIPE_M`: 矩阵单元
- `PIPE_FIX`: 格式转换单元
- `PIPE_S`: 标量单元
- `PIPE_ALL`: 所有流水线

**事件 ID**：
- `EVENT_ID0` 到 `EVENT_ID7`: 可用的事件标识符

### 屏障同步

```cpp
pipe_barrier(PIPE_ALL);
```

**使用场景**：所有流水线阶段的全局同步

## 内存管理

### 缓冲区分配策略

```cpp
size_t size = Row * Col * sizeof(T);
TASSIGN(src0Tile, 0x0);
TASSIGN(src1Tile, size);
TASSIGN(dstTile, size * 2);
```

**最佳实践**：
- 计算实际大小（字节）
- 对多个 tile 使用累积偏移量
- 确保缓冲区不重叠

### 对齐考虑

```cpp
constexpr uint16_t alignedRows = ((validRows * sizeof(T) + 31) / 32) * (32 / sizeof(T));
```

**为什么需要对齐**：
- 硬件要求 32 字节对齐
- 防止性能下降
- 避免硬件错误

### 临时缓冲区

```cpp
TileDataTmp tmpTile(validRow, validCol);
TASSIGN(tmpTile, size + size1);
```

**使用场景**：
- 中间结果
- 复杂操作的临时空间
- 对齐填充

## 类型处理

### aclFloat16 到 half 转换

```cpp
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchFusedOp(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runFusedOp<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runFusedOp<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
    }
}
```

**为什么使用此模式**：
- aclFloat16 是 API 类型
- half 是硬件类型
- NPU 执行需要转换

### 混合类型操作

```cpp
template <typename T, typename U, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runMixedOp(__gm__ T __out__ *out, __gm__ U __in__ *src0, float scalar)
{
    using SrcGlobalData = GlobalTensor<U, DynShapeDim5, DynStridDim5>;
    using SrcTileData = Tile<TileType::Vec, U, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    SrcTileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);

    event1 = TAXPY(dstTile, src0Tile, (U)scalar, event0);
}
```

**使用场景**：
- 操作期间类型转换
- 输入和输出使用不同精度
- 标量广播

## 最佳实践

### 1. 始终使用命名空间

```cpp
namespace FusedOperatorName {
}
```

**优势**：
- 避免符号冲突
- 更好的代码组织
- 更清晰的意图

### 2. 优先使用基于事件的融合

```cpp
Event<Op::TLOAD, Op::TADD> event0;
event0 = TLOAD(src1Tile, src1Global);
TADD(dstTile, src0Tile, src1Tile, event0);
```

**优势**：
- 自动依赖管理
- 更好的编译器优化
- 代码更简洁

### 3. 为手动模式使用条件编译

```cpp
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
```

**优势**：
- 两种模式使用单一代码库
- 编译器为每种模式优化
- 更易于维护

### 4. 验证 Tile 维度

```cpp
PTO_STATIC_ASSERT(kTRows_ <= 256, "Tile rows must not exceed 256");
PTO_STATIC_ASSERT(kTCols_ <= 256, "Tile cols must not exceed 256");
```

### 5. 记录缓冲区布局

```cpp
size_t size = Row * Col * sizeof(T);
TASSIGN(src0Tile, 0x0);           // Buffer 0: 0 to size
TASSIGN(src1Tile, size);           // Buffer 1: size to 2*size
TASSIGN(dstTile, size * 2);         // Buffer 2: 2*size to 3*size
```

### 6. 处理原地操作

```cpp
event0 = TLOAD(dstTile, dstGlobal);
event1 = TAXPY(dstTile, src0Tile, (T)scalar, event0);
TSTORE(dstGlobal, dstTile, event1);
```

**关键点**：
- 首先将输出作为输入加载
- 对同一 tile 使用相同角色
- 最小化内存传输

### 7. 为常见情况优化

```cpp
template void launchFusedOp<float, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);
template void launchFusedOp<aclFloat16, 16, 256, 16, 256>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
```

**常见尺寸**：
- float: 64x64, 32x32, 16x16
- aclFloat16: 16x256, 8x768, 4x1024
- int32_t: 64x64, 32x32
- int16_t: 64x64, 32x128

### 8. 使用 constexpr 进行编译时计算

```cpp
constexpr uint16_t alignedRows = ((validRows * sizeof(T) + 31) / 32) * (32 / sizeof(T));
```

**优势**：
- 在编译时计算
- 无运行时开销
- 更好的优化

### 9. 分离启动函数

```cpp
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchFusedOp(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runFusedOp<half, kTRows_, kTCols_, vRows, vCols>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        run fusedOp<T, kTRows_, kTCols_, vRows, vCols><<<1, nullptr, stream>>>(out, src0, src1);
    }
}
```

**优势**：
- 处理类型转换
- 提供清晰的 API
- 分离关注点

### 10. 优先在 CPU 模拟器上测试

始终在 NPU 硬件之前在 CPU 模拟器上测试

```bash
python3 tests/run_cpu.py --testcase <testcase> --gtest_filter '<test>'
```

**优势**：
- 更快的迭代
- 更好的调试
- 尽早发现错误

## 示例

### 示例 1：简单逐元素融合

```cpp
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TADD> event0;
    Event<Op::TADD, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TADD(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}
```

### 示例 2：标量融合 (TAXPY)

```cpp
template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
__global__ AICORE void runTAxpy(__gm__ T __out__ *out, __gm__ T __in__ *src0, float scalar)
{
    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, vCols, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x10000);

    GlobalData src0Global(src0);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TAXPY> event0;
    Event<Op::TAXPY, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(dstTile, dstGlobal);
    event1 = TAXPY(dstTile, src0Tile, (T)scalar, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}
```

### 示例 3：广播融合 (TROWEXPANDADD)

```cpp
template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
__global__ AICORE void runTRowExpandAdd(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    constexpr uint16_t src1Row = ((validRow * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, Row, Col>, Stride<1, 1, 1, Col, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, Row, Col, BLayout::RowMajor, -1, -1>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1Row, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;

    TileDataDst src0Tile(validRow, validCol);
    TileDataSrc1 src1Tile(validRow, 1);
    TileDataDst dstTile(validRow, validCol);
    size_t size = Row * Col * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(src1Tile, size);

    GlobalDataDst src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (src0eqdst) {
        TROWEXPANDADD(dstTile, src0Tile, src1Tile);
    } else {
        TROWEXPANDADD(dstTile, src1Tile, src0Tile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}
```

### 示例 4：矩阵融合 (TMATMUL)

```cpp
template <typename T, typename U, typename S, typename B, int validM, int validK, int validN, bool isBias>
__global__ AICORE void RunTMATMUL(__gm__ T *out, __gm__ U *src0, __gm__ S *src1, __gm__ B *src2)
{
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(U);
    constexpr int M = CeilAlign<int>(validM, 16);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using GlobalDataSrc0 = GlobalTensor<U, pto::Shape<1, 1, 1, validM, validK>,
                                        pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<S, pto::Shape<1, 1, 1, validK, validN>,
                                        pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<T, pto::Shape<1, 1, 1, validM, validN>,
                                       pto::Stride<1 * validM * validN, 1 * validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, U, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, S, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<U, M, K, validM, validK>;
    using RightTile = TileRight<S, K, N, validK, validN>;
    using AccTile = TileAcc<T, M, N, validM, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;

    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cTile, aTile, bTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TSTORE(dstGlobal, cTile);
    out = dstGlobal.data();
}
```

## 测试和验证

### 单元测试

```cpp
TEST(FusedOpTest, BasicTest) {
    const int rows = 64;
    const int cols = 64;
    const int size = rows * cols;

    std::vector<float> src0(size, 1.0f);
    std::vector<float> src1(size, 2.0f);
    std::vector<float> dst(size);

    launchFusedOp<float, 64, 64, 64, 64>(dst.data(), src0.data(), src1.data(), stream);

    for (int i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(dst[i], src0[i] + src1[i]);
    }
}
```

### 性能测试

```bash
# CPU 模拟器
python3 tests/run_cpu.py --testcase fusedop --gtest_filter 'FusedOpTest.PerfTest'

# NPU 硬件
python3 tests/script/run_st.py -r npu -v a3 -t fusedop -g FusedOpTest.PerfTest
```

## 常见陷阱

### 1. 缓冲区偏移错误

**问题**：
```cpp
TASSIGN(src0Tile, 0x0);
TASSIGN(src1Tile, 0x0);  // Same offset!
```

**解决方案**：
```cpp
TASSIGN(src0Tile, 0x0);
TASSIGN(src1Tile, 0x10000);
```

### 2. 缺少类型转换

**问题**：
```cpp
TAXPY(dstTile, src0Tile, scalar, event0);  // scalar is float, tile is half
```

**解决方案**：
```cpp
TAXPY(dstTile, src0Tile, (T)scalar, event0);
```

### 3. 事件依赖错误

**问题**：
```cpp
TLOAD(src0Tile, src0Global);
TLOAD(src1Tile, src1Global);
TADD(dstTile, src0Tile, src1Tile);  // Missing event dependency
```

**解决方案**：
```cpp
Event<Op::TLOAD, Op::TADD> event0;
TLOAD(src0Tile, src0Global);
event0 = TLOAD(src1Tile, src1Global);
TADD(dstTile, src0Tile, src1Tile, event0);
```

### 4. 对齐问题

**问题**：
```cpp
using TileData = Tile<TileType::Vec, T, 63, 63, BLayout::RowMajor, -1, -1>;
```

**解决方案**：
```cpp
constexpr uint16_t alignedRows = ((63 * sizeof(T) + 31) / 32) * (32 / sizeof(T));
using TileData = Tile<TileType::Vec, T, alignedRows, alignedCols, BLayout::RowMajor, -1, -1>;
```

### 5. 忘记 aclFloat16 转换

**问题**：
```cpp
template void launchFusedOp<aclFloat16, 64, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
// No conversion in launch function
```

**解决方案**：
```cpp
if constexpr (std::is_same_v<T, aclFloat16>) {
    runFusedOp<half, kTRows_, kTCols_, vRows, vCols>
        <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
}
```

## 参考

- PTO 指令参考：`include/pto/pto-inst.hpp`
- 常量和类型：`include/pto/common/constants.hpp`
- 测试示例：`tests/npu/a2a3/src/st/testcase/`
- 构建系统：`tests/script/`
