# TExtract 指令实现对比分析：A5 vs Kirin9030 vs KirinX90

## 背景

根据 readme.md 的描述：
> PTO（Parallel Tile Operation）是昇腾 CANN 定义的一套面向 tile 编程的虚拟 ISA。include/pto/npu目录下有不同硬件架构对应同一个指令的不同实现，其中 a2a3 与 a5 的硬件架构差异较大，kirin9030 以及 kirinX90 和 a5 是相似但不完全相同的硬件架构。
>
> 现在需要做的是将 kirin9030 的指令对齐到 a5，kirinX90 的指令对齐到 kirin9030。

本文档对 kirin9030、kirinX90 和 a5 的 TExtract 指令实现进行详细对比分析，包括功能、支持的数据类型、分型布局、处理场景差异等。

---

## TExtract 指令概述

### 指令定义

**TExtract** (Tile Extract) 是 PTO ISA 中用于从源 Tile 提取子块到目标 Tile 的数据搬运指令。它是实现矩阵分块计算和流水线处理的关键基础设施。

### 核心功能

TExtract 指令的核心功能是从一个大 Tile 中提取指定位置的子块，搬运到另一个 Tile。提取位置通过 `indexRow` 和 `indexCol` 参数指定，提取大小由目标 Tile 的 `ValidRow` 和 `ValidCol` 决定。

```
源 Tile (SrcTile)                目标 Tile (DstTile)
┌────────────────────┐           ┌──────────────┐
│                    │           │              │
│    ┌──────────────┐│  Extract  │  DstTile     │
│    │ indexRow,    ││  ======>  │  (子块)      │
│    │ indexCol     ││           │              │
│    │   子块       ││           │              │
│    └──────────────┘│           └──────────────┘
│                    │
└────────────────────┘
```

### 支持的数据流路径

TExtract 支持以下主要数据流路径：

| 源 Tile | 目标 Tile | 数据流 | 用途 |
|---------|-----------|--------|------|
| **Mat (L1/CBUF)** | **Left (L0A)** | L1 → L0A | Cube 左矩阵输入准备 |
| **Mat (L1/CBUF)** | **Right (L0B)** | L1 → L0B | Cube 右矩阵输入准备 |
| **Vec (UB)** | **Mat (L1)** | UB → L1 | 向量数据写入 L1 |
| **Acc (L0C)** | **Mat (L1)** | L0C → L1 | 累加器结果写回 |
| **Acc (L0C)** | **Vec (UB)** | L0C → UB | 累加器结果写入向量缓冲区（量化） |
| **Vec (UB)** | **Vec (UB)** | UB → UB | 向量内部子块提取 |
| **ConvTile** | **Right (L0B)** | L1 → L0B | 卷积专用矩阵输入 |
| **Mat (L1)** | **ScaleLeft/Right** | L1 → MX Scale Buffer | MX 格式缩放因子加载 |

### 关键应用场景

#### 1. 矩阵乘法流水线

在矩阵乘法计算中，需要将 L1 中的矩阵分块加载到 L0A/L0B 进行 Cube 计算：

```
┌─────────────────────────────────────────────────────────────────┐
│                     矩阵乘法流水线                                │
├─────────────────────────────────────────────────────────────────┤
│  L1 (MatTile)                                                   │
│  ┌───────┬───────┬───────┬───────┐                              │
│  │ Block │ Block │ Block │ Block │                              │
│  │  0    │  1    │  2    │  3    │                              │
│  └───┬───┴───┬───┴───────┴───────┘                              │
│      │       │                                                   │
│      │       │                                                   │
│  TExtract(indexRow=0, indexCol=0)  TExtract(indexRow=0, indexCol=K)│
│      │       │                                                   │
│      ▼       ▼                                                   │
│  ┌───────┐ ┌───────┐                                             │
│  │ L0A   │ │ L0B   │  ──Cube计算──>  L0C (Acc)                   │
│  └───────┘ └───────┘                                             │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. 累加器结果处理

矩阵乘法完成后，需要将 L0C 中的累加结果搬运出去：

```
┌───────────────────────────────────────────────────┐
│              累加器结果处理                        │
├───────────────────────────────────────────────────┤
│                                                   │
│  L0C (AccTile)                                    │
│  ┌────────────────────┐                          │
│  │  累加结果 (M×N)     │                          │
│  └─────────┬──────────┘                          │
│            │                                      │
│            │                                      │
│  ┌─────────┴──────────┬─────────────────────┐    │
│  │                    │                     │    │
│  │  TExtractAccToMat  │  TExtractAccToVec   │    │
│  │  (NZ → L1)         │  (NZ → UB + 量化)   │    │
│  │                    │                     │    │
│  ▼                    ▼                     │    │
│  ┌───────┐          ┌───────┐               │    │
│  │ Mat   │          │ Vec   │               │    │
│  │ (L1)  │          │ (UB)  │               │    │
│  └───────┘          └───────┘               │    │
└───────────────────────────────────────────────────┘
```

#### 3. 量化/反量化流程

在量化计算中，TExtract 结合 Fixpipe Buffer 实现向量量化：

```
┌───────────────────────────────────────────────────┐
│            量化流程示例                            │
├───────────────────────────────────────────────────┤
│                                                   │
│  L0C (int32_t 累加结果)                           │
│           │                                       │
│           │ TExtractAccToVec + FpTile             │
│           │ (SetFPC 设置量化表地址)               │
│           │                                       │
│           ▼                                       │
│  ┌───────────────┐                               │
│  │ Vec (half)    │  ← 量化后结果                  │
│  │ (UB)          │                                │
│  └───────────────┘                               │
│                                                   │
│  Fixpipe Buffer (量化缩放表)                      │
│  ┌───────────────────┐                           │
│  │ scale[0],scale[1] │                           │
│  │ scale[2],...      │                           │
│  └───────────────────┘                           │
└───────────────────────────────────────────────────┘
```

#### 4. ConvTile 处理

卷积计算使用特殊的 ConvTile 格式（FRACTAL_Z），需要通过 TExtract 提取到 L0B：

```
┌───────────────────────────────────────────────────┐
│          ConvTile 提取流程                        │
├───────────────────────────────────────────────────┤
│                                                   │
│  ConvTile [C1HW, N, 16, C0]                       │
│  ┌────────────────────────────┐                  │
│  │  特殊卷积矩阵布局           │                  │
│  │  (FRACTAL_Z 或 FRACTAL_Z_3D)│                  │
│  └───────────────┬────────────┘                  │
│                  │                                │
│                  │ TEXTRACT_CONVTILE_IMPL         │
│                  │                                │
│                  ▼                                │
│  ┌───────────────────┐                           │
│  │ L0B (Right)       │  ← 卷积输入矩阵           │
│  │ ColMajor + RowMajor│                          │
│  └───────────────────┘                           │
└───────────────────────────────────────────────────┘
```

### 指令参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| **DstTile** | Tile 模板参数 | 目标 Tile，决定提取大小 |
| **SrcTile** | Tile 模板参数 | 源 Tile，提供数据 |
| **indexRow** | uint16_t | 提取起始行位置 |
| **indexCol** | uint16_t | 提取起始列位置 |
| **reluMode** | ReluPreMode | Relu 模式 (NoRelu/NormalRelu) |
| **preQuantScalar** | uint64_t | 标量量化参数 |
| **FpTile** | Scaling Tile | 向量量化表 |

### 分型布局约束

TExtract 对源和目标 Tile 的布局有严格要求：

| 目标 Tile | 源 Tile 允许布局 | 目标 Tile 要求布局 |
|-----------|----------------|------------------|
| **Left (L0A)** | ColMajor+RowMajor, RowMajor+ColMajor, 或 RowMajor(单行) | **RowMajor + ColMajor** (NZ) |
| **Right (L0B)** | ColMajor+RowMajor, RowMajor+ColMajor | **ColMajor + RowMajor** (ZN) |
| **Vec (ND)** | RowMajor | RowMajor |
| **Vec (NZ)** | !RowMajor + RowMajor | !RowMajor + RowMajor |
| **Mat (NZ)** | Acc: !RowMajor + RowMajor | !RowMajor + RowMajor 或 RowMajor+NoneBox |

---

## 一、总体结构对比

| 特性 | A5 | Kirin9030 | KirinX90 | A2A3 |
|------|:--:|:---------:|:--------:|:----:|
| **文件大小** | 1039 行 | 842 行 | 452 行 | 914 行 |
| **核心函数数** | ~25 | ~18 | ~12 | ~20 |
| **数据类型支持** | 10+ 类型 | 6 类型 | 6 类型 | 4 类型 |
| **FP4 支持** | ✅ | ❌ | ❌ | ❌ |
| **FP8 支持** | ✅ | ❌ | ❌ | ❌ |
| **MX Scale** | ✅ | ❌ | ❌ | ❌ |
| **Vec→Vec ND** | ✅ | ✅ | ❌ | ✅ |
| **Vec→Vec NZ** | ✅ | ✅ | ❌ | ✅ |
| **Acc→Vec** | ✅ | ✅ | ❌ | ❌ |

---

## 二、数据类型支持详细对比

### 2.1 is_textract_supported_type 定义

| 架构 | 支持的数据类型 |
|------|--------------|
| **A5** | `int8_t`, `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t`, `half`, `bfloat16_t`, `float`, `float4_e2m1x2_t`, `float4_e1m2x2_t`, `float8_e8m0_t` |
| **Kirin9030** | `int8_t`, `uint8_t`, `half`, `int16_t`, `uint16_t`, `int32_t` |
| **KirinX90** | `int8_t`, `uint8_t`, `half`, `int16_t`, `uint16_t`, `int32_t` |
| **A2A3** | `int8_t`, `half`, `bfloat16_t`, `float` |

### 2.2 数据类型对比表

| 数据类型 | A5 | Kirin9030 | KirinX90 | A2A3 | 用途说明 |
|----------|:--:|:---------:|:--------:|:----:|---------|
| **int8_t** | ✅ | ✅ | ✅ | ✅ | INT8 量化计算 |
| **uint8_t** | ✅ | ✅ | ✅ | ❌ | UINT8 量化计算 |
| **int16_t** | ✅ | ✅ | ✅ | ❌ | INT16 中间计算 |
| **uint16_t** | ✅ | ✅ | ✅ | ❌ | UINT16 中间计算 |
| **int32_t** | ✅ | ✅ | ✅ | ❌ | Acc 累加器类型 |
| **half** | ✅ | ✅ | ✅ | ✅ | FP16 矩阵计算 |
| **bfloat16_t** | ✅ | ❌ | ❌ | ✅ | BF16 矩阵计算 |
| **float** | ✅ | ✅ | ✅ | ✅ | FP32 矩阵计算 |
| **hifloat8_t** | ✅ | ❌ | ❌ | ❌ | 华为自定义 8-bit FP |
| **float8_e4m3_t** | ✅ | ❌ | ❌ | ❌ | MXFP8 E4M3 |
| **float8_e5m2_t** | ✅ | ❌ | ❌ | ❌ | MXFP8 E5M2 |
| **float8_e8m0_t** | ✅ | ❌ | ❌ | ❌ | MX 缩放因子类型 |
| **float4_e2m1x2_t** | ✅ | ❌ | ❌ | ❌ | MXFP4 E2M1 (每字节 2 元素) |
| **float4_e1m2x2_t** | ✅ | ❌ | ❌ | ❌ | MXFP4 E1M2 (每字节 2 元素) |

### 2.3 CheckTExtractToL0 函数对比

| 架构 | 检查内容 |
|------|---------|
| **A5** | 无单独函数，通过 `CommonCheck()` 检查：`is_textract_supported_type` + `DstType == SrcType` + `SrcTile Fractal 有效性` |
| **Kirin9030** | `CheckTExtractToL0<DstType, SrcType>()`: `DstType == SrcType` + 类型限制为 `half/(u)int8/(u)int16` |
| **KirinX90** | `CheckTExtractToL0<DstType, SrcType>()`: 同 Kirin9030 |
| **A2A3** | `CheckTExtract<DstTile, SrcTile, DstType, SrcType>()`: `(SrcTile::Loc == Acc) || DstType == SrcType` + 类型限制为 `int8/half/bf16/float` |

---

## 三、功能路径支持对比

### 3.1 TEXTRACT_IMPL 主入口对比

| 源→目标路径 | A5 | Kirin9030 | KirinX90 | A2A3 |
|------------|:--:|:---------:|:--------:|:----:|
| **Vec→Vec (ND)** | ✅ TExtractVecToVecNDDispatch | ✅ TExtractVecToVecNDDispatch | ❌ 不支持 | ✅ TExtractVecToVecNDDispatch |
| **Vec→Vec (NZ)** | ✅ TExtractVecToVecNZImpl | ✅ TExtractVecToVecNZImpl | ❌ 不支持 | ✅ TExtractVecToVecNZDispatch |
| **Mat→Left (L0A)** | ✅ TExtractToLeft | ✅ TExtractToLeft | ✅ TExtractToLeft | ✅ TExtractToLeft |
| **Mat→Right (L0B)** | ✅ TExtractToRight | ✅ TExtractToRight | ✅ TExtractToRight | ✅ TExtractToRight |
| **Vec→Mat** | ✅ TExtractVecToMat | ✅ TExtractVecToMat | ✅ TExtractVecToMat | ❌ 不支持 |
| **Acc→Mat** | ✅ TExtractAccToMat | ✅ TExtractAccToMat | ✅ TExtractAccToMat | ✅ TExtractAccToMat |
| **Acc→Vec** | ✅ TExtractAccToVec | ✅ TExtractAccToVec | ❌ 不支持 | ❌ 不支持 |
| **ConvTile→L0B** | ✅ TEXTRACT_CONVTILE_IMPL | ✅ TEXTRACT_CONVTILE_IMPL | ✅ TEXTRACT_CONVTILE_IMPL | ✅ TEXTRACT_CONVTILE_IMPL |
| **Mat→ScaleLeft (MX)** | ✅ TExtractToAmx | ❌ static_assert 失败 | ❌ static_assert 失败 | ❌ 不支持 |
| **Mat→ScaleRight (MX)** | ✅ TExtractToBmx | ❌ static_assert 失败 | ❌ static_assert 失败 | ❌ 不支持 |

### 3.2 路径详细说明

#### 3.2.1 Vec→Vec ND (RowMajor + NoneBox)

**功能**: 从源 VecTile (ND 格式) 提取子块到目标 VecTile (ND 格式)

| 架构 | 实现路径 | 硬件指令 |
|------|---------|---------|
| **A5** | 4 路径：Impl(DMA)、AlignedImpl(vlds/vsts)、VectorImpl(vldas/vldus/vsts)、ScalarImpl(标量) | DMA + 向量指令 |
| **Kirin9030** | 4 路径：同 A5 | DMA + 向量指令 |
| **KirinX90** | ❌ 无 | — |
| **A2A3** | 3 路径：Aligned(DMA)、Unaligned(DMA+vcopy)、Scalar(标量) | pto_copy_ubuf_to_ubuf + vcopy |

**关键差异**:
- A5/Kirin9030 有 4 种实现路径，根据对齐情况选择最优路径
- A5 支持 FP4 类型，使用 `isFp4Type` 模板参数特殊处理
- A2A3 使用纯 DMA 路径，无向量指令

#### 3.2.2 Vec→Vec NZ (!RowMajor + RowMajor SFractal)

**功能**: 从源 VecTile (NZ 格式) 提取子块到目标 VecTile (NZ 格式)

| 架构 | 实现路径 | 硬件指令 |
|------|---------|---------|
| **A5** | 3 路径：Impl(DMA)、ScalarImpl(标量) | pto_copy_ubuf_to_ubuf |
| **Kirin9030** | 3 路径：同 A5 | pto_copy_ubuf_to_ubuf |
| **KirinX90** | ❌ 无 | — |
| **A2A3** | 3 路径：Aligned(DMA)、Unaligned(DMA+vcopy)、Scalar(标量) | pto_copy_ubuf_to_ubuf + vcopy |

**关键差异**:
- A5 有 FP4 特殊处理：`byteValidCol = validCol / 2`，`byteIndexCol = indexCol / 2`
- Kirin9030 无 FP4 特殊处理
- A2A3 有 Aligned/Unaligned 两种 DMA 路径

#### 3.2.3 Mat→Left (L0A)

**功能**: 从源 MatTile (L1/CBUF) 提取子块到目标 LeftTile (L0A)

| 架构 | 实现方式 | 硬件指令 |
|------|---------|---------|
| **A5** | TExtractToA (带 `isFp4Type` 参数) + load_cbuf_to_ca_s4/ca | load_cbuf_to_ca_s4 (FP4)、load_cbuf_to_ca (B16/B32) |
| **Kirin9030** | TExtractToA (无 FP4 参数) | load_cbuf_to_ca |
| **KirinX90** | TExtractToA (无 FP4 参数) | load_cbuf_to_ca |
| **A2A3** | TExtractToANonTranspose (img2colv2) + TExtractToATranspose (load_cbuf_to_ca_transpose/img2colv2) | img2colv2_cbuf_to_ca、load_cbuf_to_ca_transpose |

**关键差异**:
- **A5 独有 FP4 支持**: 使用 `load_cbuf_to_ca_s4` 硬件指令
- **A5 FP8 特殊处理**: 对于 typeSize==1 (B8 类型)，需要分块循环 (`M_STEP_MIN_VAL_B8=2`)
- **A2A3 独有 Transpose 实现**: 使用 `img2colv2_cbuf_to_ca` + `load_cbuf_to_ca_transpose`

#### 3.2.4 Mat→Right (L0B)

**功能**: 从源 MatTile (L1/CBUF) 提取子块到目标 RightTile (L0B)

| 架构 | 实现方式 | 硬件指令 |
|------|---------|---------|
| **A5** | TExtractToB (带 `isFp4Type` 参数) + pto_load_cbuf_to_cb | pto_load_cbuf_to_cb、load_cbuf_to_cb_s4 |
| **Kirin9030** | TExtractToB (无 FP4 参数) | pto_load_cbuf_to_cb |
| **KirinX90** | TExtractToB (无 FP4 参数) | pto_load_cbuf_to_cb |
| **A2A3** | TExtractToBNonTranspose + TExtractToBTranspose | pto_load_cbuf_to_cb、img2colv2_cbuf_to_cb、load_cbuf_to_cb_transpose |

**关键差异**:
- A5 有 FP4 专用指令 `load_cbuf_to_cb_s4`
- A2A3 有专门的 Transpose 实现函数

#### 3.2.5 Acc→Mat

**功能**: 从源 AccTile (L0C) 提取子块到目标 MatTile (L1/CBUF)

| 架构 | 实现方式 | 硬件指令 | 参数数量 |
|------|---------|---------|:--------:|
| **A5** | TExtractAccToMat | copy_matrix_cc_to_cbuf | 26 |
| **Kirin9030** | TExtractAccToMat | copy_matrix_cc_to_cbuf | 22 |
| **KirinX90** | TExtractAccToMat | copy_matrix_cc_to_cbuf | 12 |
| **A2A3** | TExtractAccToMat | copy_matrix_cc_to_cbuf | 8 |

**关键差异**:
- A5 支持 `channelSplitEnable` (float + SFractalSize==512/1024)
- A5/Kirin9030 支持更多参数（如 `enableNz2Nd`、`enableNz2Dn`）
- KirinX90/A2A3 参数简化，不支持复杂转换

#### 3.2.6 Acc→Vec

**功能**: 从源 AccTile (L0C) 提取子块到目标 VecTile (UB)

| 架构 | 是否支持 | 硬件指令 | 特性 |
|------|:--------:|---------|------|
| **A5** | ✅ | copy_matrix_cc_to_ub (26 参数) | AccToVecMode、STPhase、dualDstCtl、subBlockId |
| **Kirin9030** | ✅ | copy_matrix_cc_to_ub (22 参数) | AccToVecMode、dualDstCtl、subBlockId |
| **KirinX90** | ❌ | — | — |
| **A2A3** | ❌ | — | — |

**关键差异**:
- **KirinX90/A2A3 不支持 Acc→Vec**，这是重大功能缺失
- A5/Kirin9030 支持 `AccToVecMode` (SingleModeVec0/Vec1, DualModeSplitM/SplitN)
- A5 支持 `STPhase` 参数

---

## 四、分型布局支持对比

### 4.1 分型布局定义

| 布局类型 | isRowMajor | SFractal | BFractal | 说明 |
|----------|:----------:|:--------:|:--------:|------|
| **ND** | true | NoneBox | RowMajor | 标准行主格式 |
| **DN** | false | NoneBox | ColMajor | 列主格式 |
| **NZ** | false | RowMajor | ColMajor | Cube NZ 格式 (L0A/L0B/Acc) |
| **ZN** | true | ColMajor | RowMajor | Cube ZN 格式 |
| **ZZ** | true | RowMajor | RowMajor | ZZ 格式 (A5 only) |
| **FRACTAL_Z** | — | — | — | 卷积专用格式 [C1HW,N,16,C0] |

### 4.2 TExtractToLeft 布局检查

| 架构 | SrcTile 允许的布局 | DstTile 要求的布局 |
|------|-------------------|-------------------|
| **A5** | `(ColMajor && isRowMajor)` 或 `(RowMajor && !isRowMajor)` 或 `(Rows==1 && isRowMajor)` | `RowMajor && !isRowMajor` |
| **Kirin9030** | 同 A5 | 同 A5 |
| **KirinX90** | 同 A5 | 同 A5 |
| **A2A3** | 同 A5 | `RowMajor && isRowMajor` ⚠️ **不同** |

**关键差异**: A2A3 的 DstTile 要求 `RowMajor && isRowMajor`，与其他架构不同

### 4.3 TExtractToRight 布局检查

| 架构 | SrcTile 允许的布局 | DstTile 要求的布局 |
|------|-------------------|-------------------|
| **A5** | `(ColMajor && isRowMajor)` 或 `(RowMajor && !isRowMajor)` | `ColMajor && isRowMajor` |
| **Kirin9030** | 同 A5 | 同 A5 |
| **KirinX90** | 同 A5 | 同 A5 |
| **A2A3** | 同 A5 | 同 A5 |

---

## 五、Compact 模式支持对比

### 5.1 Compact 模式定义

| Compact 模式 | 说明 |
|-------------|------|
| **Null** | 默认模式，无特殊压缩 |
| **Normal** | 正常压缩模式，支持动态 validRow/validCol |
| **RowPlusOne** | 行数加一优化，减少 bank conflict |
| **RowAlignedPadding** | 行对齐填充 |

### 5.2 TExtractToACompact/TExtractToBCompact 对比

| 特性 | A5 | Kirin9030 | KirinX90 | A2A3 |
|------|:--:|:---------:|:--------:|:----:|
| **支持 Compact 模式** | ✅ | ✅ | ✅ | ✅ |
| **FP4 参数** | ✅ `isFp4Type` | ❌ | ❌ | ❌ |
| **动态 validRow/validCol** | ✅ | ✅ | ✅ | ✅ |
| **Transpose 版本** | ✅ TExtractToATransCompact | ✅ TExtractToATransCompact | ✅ TExtractToATransCompact | ✅ TExtractToATransposeCompact |
| **KAligned 参数** | ❌ | ❌ | ❌ | ✅ (A2A3 独有) |

**关键差异**:
- A2A3 独有 `isKAligned` 参数，用于 float 类型的 K 维对齐优化
- A5 独有 `isFp4Type` 参数支持 FP4 类型

---

## 六、量化支持对比

### 6.1 QuantPre 模式

| 功能 | A5 | Kirin9030 | KirinX90 | A2A3 |
|------|:--:|:---------:|:--------:|:----:|
| **ReluPreMode** | ✅ NoRelu/NormalRelu | ✅ NoRelu/NormalRelu | ✅ NoRelu/NormalRelu | ✅ NoRelu/NormalRelu |
| **标量量化 (preQuantScalar)** | ✅ | ✅ | ✅ | ✅ |
| **向量量化 (FpTile)** | ✅ | ✅ | ✅ | ✅ |
| **AccToVecMode** | ✅ | ✅ | ❌ | ❌ |
| **STPhase** | ✅ | ✅ | ❌ | ❌ |

### 6.2 TEXTRACT_IMPL 重载对比

| 重载签名 | A5 | Kirin9030 | KirinX90 | A2A3 |
|---------|:--:|:---------:|:--------:|:----:|
| `TEXTRACT_IMPL(dst, src, indexRow, indexCol)` | ✅ | ✅ | ✅ | ✅ |
| `TEXTRACT_IMPL<ReluPreMode>(dst, src)` | ✅ Acc→Mat/Vec | ✅ Acc→Mat/Vec | ✅ 仅 Acc→Mat | ✅ 仅 Acc→Mat |
| `TEXTRACT_IMPL<mode, ReluPreMode>(dst, src)` | ✅ Acc→Vec | ✅ Acc→Vec | ❌ | ❌ |
| `TEXTRACT_IMPL(dst, src, preQuantScalar)` | ✅ Acc→Mat/Vec | ✅ Acc→Mat/Vec | ✅ 仅 Acc→Mat | ✅ 仅 Acc→Mat |
| `TEXTRACT_IMPL<mode>(dst, src, preQuantScalar)` | ✅ Acc→Vec | ✅ Acc→Vec | ❌ | ❌ |
| `TEXTRACT_IMPL(dst, src, fp)` | ✅ Acc→Mat/Vec | ✅ Acc→Mat/Vec | ✅ 仅 Acc→Mat | ✅ 仅 Acc→Mat |
| `TEXTRACT_IMPL<mode>(dst, src, fp)` | ✅ Acc→Vec | ✅ Acc→Vec | ❌ | ❌ |

---

## 七、硬件指令详细对比

### 7.1 Mat→L0A 硬件指令

| 指令 | A5 | Kirin9030 | KirinX90 | A2A3 | 说明 |
|------|:--:|:---------:|:--------:|:----:|------|
| **load_cbuf_to_ca** | ✅ | ✅ | ✅ | ❌ | 标准加载 |
| **load_cbuf_to_ca_s4** | ✅ (FP4) | ❌ | ❌ | ❌ | FP4 专用 |
| **img2colv2_cbuf_to_ca** | ❌ | ❌ | ❌ | ✅ | img2col 加载 |
| **load_cbuf_to_ca_transpose** | ❌ | ❌ | ❌ | ✅ | B8 转置 |

### 7.2 Mat→L0B 硬件指令

| 指令 | A5 | Kirin9030 | KirinX90 | A2A3 | 说明 |
|------|:--:|:---------:|:--------:|:----:|------|
| **pto_load_cbuf_to_cb** | ✅ | ✅ | ✅ | ✅ | 标准加载 |
| **load_cbuf_to_cb_s4** | ✅ (FP4) | ❌ | ❌ | ❌ | FP4 专用 |
| **img2colv2_cbuf_to_cb** | ❌ | ❌ | ❌ | ✅ | img2col 加载 |
| **load_cbuf_to_cb_transpose** | ❌ | ❌ | ❌ | ✅ | B8 转置 |

### 7.3 Acc→Mat 硬件指令参数对比

```cpp
// A5 copy_matrix_cc_to_cbuf (26 参数)
copy_matrix_cc_to_cbuf(
    dstAddr, srcData,
    sid, validCol, validRow, dstStride, srcStride,
    0, 0, 0, QuantPre, reluMode,
    channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0,
    false, false, false, false, false, enableNz2Dn
);

// Kirin9030 copy_matrix_cc_to_cbuf (22 参数)
copy_matrix_cc_to_cbuf(
    dstAddr, srcData,
    sid, validCol, validRow, dstStride, srcStride,
    0, 0, QuantPre, reluMode,
    channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0,
    false, false, false, false, false, enableNz2Dn
);

// KirinX90 copy_matrix_cc_to_cbuf (12 参数)
copy_matrix_cc_to_cbuf(
    dstAddr, srcData,
    sid, validCol, validRow, dstStride, srcStride,
    0, 0, QuantPre, reluMode,
    channelSplitEnable, false, 0, 0, false, false, 0,
    false, false, false, false, false, false
);

// A2A3 copy_matrix_cc_to_cbuf (8 参数)
copy_matrix_cc_to_cbuf(
    dstAddr, srcAddr,
    sid, nSize, validRow, dstStrideD, srcStride,
    0, QuantPre, reluMode, false, false
);
```

### 7.4 Acc→Vec 硬件指令参数对比

```cpp
// A5 copy_matrix_cc_to_ub (26 参数)
copy_matrix_cc_to_ub(
    dstAddr, srcData,
    sid, validCol, validRow, dstStride, srcStride,
    dualDstCtl, subBlockId, 0,
    unitFlagCtrl, quantPre, reluMode,
    channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0,
    false, false, false, false, false, enableNz2Dn
);

// Kirin9030 copy_matrix_cc_to_ub (22 参数)
copy_matrix_cc_to_ub(
    dstAddr, srcData,
    sid, validCol, validRow, dstStride, srcStride,
    dualDstCtl, subBlockId, 0, 0,
    quantPre, reluMode,
    channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0,
    false, false, false, false, false, enableNz2Dn
);

// KirinX90/A2A3: 不支持此指令
```

---

## 八、MX Scale 支持对比

### 8.1 TExtractToAmx/TExtractToBmx

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|:--:|:---------:|:--------:|
| **支持 MX ScaleLeft** | ✅ TExtractToAmx | ❌ `static_assert(sizeof(DstTile::DType)==0)` | ❌ `static_assert(sizeof(DstTile::DType)==0)` |
| **支持 MX ScaleRight** | ✅ TExtractToBmx | ❌ `static_assert(sizeof(DstTile::DType)==0)` | ❌ `static_assert(sizeof(DstTile::DType)==0)` |
| **硬件指令** | load_cbuf_to_ca_mx/cb_mx | — | — |
| **数据类型** | float8_e8m0_t | — | — |

### 8.2 A5 MX Scale 实现详情

```cpp
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TExtractToAmx(...) {
    static_assert((DstTileData::SFractal == RowMajor && DstTileData::isRowMajor), "TMov_mx: DstTile Invalid Fractal.");
    using DataType = typename DstTileData::DType;
    uint16_t rowStartPosition = indexRow >> SHIFT_MX_ROW;  // SHIFT_MX_ROW = 4
    uint16_t colStartPosition = (indexCol * sizeof(DataType)) >> SHIFT_MX_COL;  // SHIFT_MX_COL = 1
    
    if constexpr (DstTileData::Rows == 1) {
        uint8_t shiftCol = CeilDivision(validCol * sizeof(DataType), SCALE_CUBE_BLOCK_SIZE) * CO_SIZE_SCALE;
        uint8_t colStep = (shiftCol * sizeof(DataType)) >> SHIFT_MX_COL;
        load_cbuf_to_ca_mx(dstAddr, srcAddr, rowStartPosition, colStartPosition, 1, colStep, srcStride, dstStride);
    } else if constexpr (DstTileData::Compact == CompactMode::Normal) {
        uint16_t validRowAlign = CeilDivision(validRow, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        uint8_t rowStep = validRowAlign >> SHIFT_MX_ROW;
        uint8_t colStep = (validCol * sizeof(DataType)) >> SHIFT_MX_COL;
        load_cbuf_to_ca_mx(dstAddr, srcAddr, rowStartPosition, colStartPosition, rowStep, colStep, srcStride, dstStride);
    } else {
        constexpr uint8_t rowStep = DstTileData::Rows >> SHIFT_MX_ROW;
        constexpr uint8_t colStep = (DstTileData::Cols * sizeof(DataType)) >> SHIFT_MX_COL;
        load_cbuf_to_ca_mx(dstAddr, srcAddr, rowStartPosition, colStartPosition, rowStep, colStep, srcStride, dstStride);
    }
}
```

---

## 九、ConvTile 支持对比

### 9.1 TExtractToBConv 对比

| 特性 | A5 | Kirin9030 | KirinX90 | A2A3 |
|------|:--:|:---------:|:--------:|:----:|
| **支持 ConvTile→L0B** | ✅ | ✅ | ✅ | ✅ |
| **源布局** | FRACTAL_Z 或 FRACTAL_Z_3D | FRACTAL_Z 或 FRACTAL_Z_3D | FRACTAL_Z 或 FRACTAL_Z_3D | FRACTAL_Z 或 FRACTAL_Z_3D |
| **目标布局** | ColMajor + RowMajor | ColMajor + RowMajor | ColMajor + RowMajor | ColMajor + RowMajor |
| **支持类型** | int8/u8/i16/u16/i32/u32/half/bf16/float | int8/u8/i16/u16/i32/u32/half/float | int8/u8/i16/u16/i32/u32/half/float | int8/half/bf16/float |

### 9.2 ConvTile 实现差异

| 架构 | 实现方式 |
|------|---------|
| **A5** | `pto_load_cbuf_to_cb<false>` |
| **Kirin9030** | `pto_load_cbuf_to_cb<false>` |
| **KirinX90** | `pto_load_cbuf_to_cb<false>` |
| **A2A3** | 循环调用 `pto_load_cbuf_to_cb`，按行遍历 |

---

## 十、Vec→Vec 实现详细对比

### 10.1 Vec→Vec ND 实现

| 实现路径 | A5 | Kirin9030 | A2A3 | 说明 |
|---------|:--:|:---------:|:----:|------|
| **Impl (DMA)** | TExtractVecToVecNDImpl | TExtractVecToVecNDImpl | TExtractVecToVecNDAligned | 32B 对齐时使用 DMA |
| **AlignedImpl (向量)** | TExtractVecToVecNDAlignedImpl | TExtractVecToVecNDAlignedImpl | — | 使用 vlds/vsts |
| **VectorImpl (非对齐)** | TExtractVecToVecNDVectorImpl | TExtractVecToVecNDVectorImpl | TExtractVecToVecNDUnaligned | 使用 vldas/vldus/vsts |
| **ScalarImpl (单元素)** | TExtractVecToVecNDScalarImpl | TExtractVecToVecNDScalarImpl | TExtractVecToVecNDScalar | 标量操作 |
| **FP4 特殊处理** | ✅ | ❌ | ❌ | BLOCK_BYTE_SIZE 对齐检查 |

### 10.2 Vec→Vec NZ 实现

| 实现路径 | A5 | Kirin9030 | A2A3 | 说明 |
|---------|:--:|:---------:|:----:|------|
| **Impl (DMA)** | TExtractVecToVecNZImpl | TExtractVecToVecNZImpl | TExtractVecToVecNZAligned | c0Size 对齐时使用 DMA |
| **ScalarImpl (单元素)** | TExtractVecToVecNZScalarImpl | TExtractVecToVecNZScalarImpl | TExtractVecToVecNZScalar | 标量操作 |
| **FP4 特殊处理** | ✅ `byteValidCol/2` | ❌ | ❌ | FP4 每字节 2 元素 |
| **Unaligned 路径** | ❌ | ❌ | ✅ TExtractVecToVecNZUnaligned | 使用 DMA + vcopy |

---

## 十一、关键缺失功能分析

### 11.1 KirinX90 vs Kirin9030 缺失功能

| 缺失功能 | 影响场景 | 对齐建议 |
|----------|---------|---------|
| **Vec→Vec ND** | 无法从 Vec 提取 ND 子块，影响 UB 数据处理 | 实现 TExtractVecToVecNDDispatch |
| **Vec→Vec NZ** | 无法从 Vec 提取 NZ 子块，影响 UB 数据处理 | 实现 TExtractVecToVecNZImpl |
| **Acc→Vec** | 无法从 Acc 提取到 Vec，影响量化后数据流 | 实现 TExtractAccToVec + copy_matrix_cc_to_ub |
| **copy_matrix_cc_to_ub** | 无 Acc→Vec 硬件指令 | 补齐 22 参数版本 |
| **STPhase** | 无存储阶段控制 | 补齐 unitFlagCtrl 参数 |
| **AccToVecMode** | 无多模式 Vec 输出 | 补齐 dualDstCtl/subBlockId 参数 |

### 11.2 Kirin9030/KirinX90 vs A5 缺失功能

| 缺失功能 | 影响场景 | 对齐建议 |
|----------|---------|---------|
| **FP4 类型** | 无法使用 MXFP4 E2M1/E1M2 | 补齐 load_cbuf_to_ca_s4/cb_s4 + isFp4Type 参数 |
| **FP8 类型** | 无法使用 MXFP8 E4M3/E5M2 | 补齐类型支持 |
| **bfloat16_t** | 无法使用 BF16 类型 | 补齐类型支持 |
| **hifloat8_t** | 无法使用华为自定义 FP8 | 补齐类型支持 |
| **float8_e8m0_t** | 无法使用 MX 缩放因子 | 补齐类型支持 |
| **MX ScaleLeft/Right** | 无法使用 MX 格式矩阵乘 | 实现 TExtractToAmx/Bmx + load_cbuf_to_ca_mx/cb_mx |
| **channelSplitEnable** | 无 float channel split 模式 | 补齐 SFractalSize==512/1024 分支 |

---

## 十二、代码结构对比总结

### 12.1 函数层级对比

```
A5 (1039 行)                    Kirin9030 (842 行)             KirinX90 (452 行)
├── TEXTRACT_IMPL (主入口)       ├── TEXTRACT_IMPL              ├── TEXTRACT_IMPL
│   ├── Vec→Vec ND/NZ           │   ├── Vec→Vec ND/NZ          │   ❌ 无 Vec→Vec
│   ├── ConvTile→L0B            │   ├── ConvTile→L0B           │   ├── ConvTile→L0B
│   └── TEXTRACT_TILE_IMPL      │   └── TEXTRACT_TILE_IMPL     │   └── TEXTRACT_TILE_IMPL
│       ├── Mat→Left            │       ├── Mat→Left           │       ├── Mat→Left
│       ├── Mat→Right           │       ├── Mat→Right          │       ├── Mat→Right
│       ├── Vec→Mat             │       ├── Vec→Mat            │       ├── Vec→Mat
│       ├── Acc→Mat             │       ├── Acc→Mat            │       ├── Acc→Mat
│       ├── Acc→Vec             │       ├── Acc→Vec            │       ❌ 无 Acc→Vec
│       ├── ScaleLeft (MX)      │       ❌ static_assert        │       ❌ static_assert
│       └── ScaleRight (MX)     │       ❌ static_assert        │       ❌ static_assert
│
├── TExtractToA/B (带 isFp4Type) ├── TExtractToA/B              ├── TExtractToA/B
│   ├── TExtractToAVector       │   ├── TExtractToAVector       │   ├── TExtractToAVector
│   ├── TExtractToACompact      │   ├── TExtractToACompact      │   ├── TExtractToACompact
│   ├── TExtractToATransCompact │   ├── TExtractToATransCompact │   ├── TExtractToATransCompact
│   ├── TExtractToBCompact      │   ├── TExtractToBCompact      │   ├── TExtractToBCompact
│   └── TExtractToBTransCompact │   └ TExtractToBTransCompact   │   └ TExtractToBTransCompact
│
├── TExtractToAmx/Bmx (MX)      ├── ❌ 无                       ├── ❌ 无
│
├── TExtractAccToMat (26 参数)   ├── TExtractAccToMat (22 参数) ├── TExtractAccToMat (12 参数)
├── TExtractAccToVec (26 参数)   ├── TExtractAccToVec (22 参数) ├── ❌ 无
│
├── Vec→Vec (FP4 特殊处理)      ├── Vec→Vec (无 FP4)           ├── ❌ 无
│   ├── NDImpl/Aligned/Vector   │   ├── NDImpl/Aligned/Vector
│   └ NZImpl/Scalar            │   └ NZImpl/Scalar
│
└── SetFPC (indexCol 参数)       └── SetFPC (indexCol 参数)     └── SetFPC (indexCol 参数)
```

### 12.2 总结

| 对比项 | A5 | Kirin9030 | KirinX90 |
|--------|:--:|:---------:|:--------:|
| **代码行数** | 1039 | 842 | 452 |
| **数据类型** | 10+ | 6 | 6 |
| **功能完整度** | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| **Vec→Vec** | ✅ ND/NZ | ✅ ND/NZ | ❌ |
| **Acc→Vec** | ✅ | ✅ | ❌ |
| **MX Scale** | ✅ | ❌ | ❌ |
| **FP4/FP8** | ✅ | ❌ | ❌ |
| **copy_matrix 参数** | 26 | 22 | 12 |

---

## 十三、对齐建议

### 13.1 KirinX90 对齐 Kirin9030

1. **补齐 Vec→Vec ND**: 实现 TExtractVecToVecNDDispatch 及 4 个实现路径
2. **补齐 Vec→Vec NZ**: 实现 TExtractVecToVecNZImpl + ScalarImpl
3. **补齐 Acc→Vec**: 实现 TExtractAccToVec + copy_matrix_cc_to_ub (22 参数)
4. **补齐 AccToVecMode**: 支持 SingleModeVec0/Vec1, DualModeSplitM/SplitN
5. **补齐 STPhase**: 支持 unitFlagCtrl 参数

### 13.2 Kirin9030 对齐 A5

1. **补齐 FP4 类型**: 实现 load_cbuf_to_ca_s4/cb_s4 + isFp4Type 模板参数
2. **补齐 FP8 类型**: 补齐 float8_e4m3/e5m2/hifloat8 类型支持
3. **补齐 bfloat16_t**: 补齐类型支持
4. **补齐 MX Scale**: 实现 TExtractToAmx/Bmx + load_cbuf_to_ca_mx/cb_mx
5. **补齐 channelSplitEnable**: 支持 float + SFractalSize==512 分支
6. **补齐 copy_matrix_cc_to_cbuf 参数**: 从 22 参数扩展到 26 参数