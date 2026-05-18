# Kirin9030 与 A5 同一指令支持方式差异分析

## 分析范围

基于 `include/pto/npu/a5/` 与 `include/pto/npu/kirin9030/` 的源码对比，分析两架构对**同一指令**在数据类型、数据排布、场景功能等方面的支持差异。

---

## 一、两架构共同支持的指令

### 1.1 完全相同的指令（直接复用 a5 实现，81 条）

以下指令 k9030 无任何重写，直接 `#include "pto/npu/a5/..."`，实现代码**完全一致**：

| 类别 | 指令 | 数量 |
|------|------|:----:|
| **二元运算** | TAdd, TSub, TMul, TDiv, TMax, TMin, TAxpy, TPrelu, TLRelu | 9 |
| **标量运算** | TAddS, TSubS, TMulS, TDivS, TMaxS, TMins, TMaxs | 7 |
| **位运算** | TAnd, TOr, TXor, TShl, TShr, TAndS, TOrS, TXorS, TShlS, TShrS | 10 |
| **比较** | TCmp, TCmps | 2 |
| **选择** | TSel, TSels | 2 |
| **一元运算** | TAbs(UnaryOp), TExp(UnaryOp), TLog(UnaryOp), TSqrt(UnaryOp), TRsqrt, TRelu(UnaryOp), TNeg(UnaryOp) | 7 |
| **数据变换** | TAssign, TSubView, TReshape, TConcat, TTrans, TFillPad, TExpandS, TTri | 8 |
| **行广播** | TRowExpand, TRowExpandAdd, TRowExpandSub, TRowExpandMul, TRowExpandDiv, TRowExpandMax, TRowExpandMin | 7 |
| **列广播** | TColExpand, TColExpandAdd, TColExpandSub, TColExpandMul, TColExpandDiv, TColExpandMax, TColExpandMin | 7 |
| **行规约** | TRowSum, TRowMax, TRowMin, TRowProd, TRowArgMax, TRowArgMin | 6 |
| **列规约** | TColSum, TColProd, TColMax, TColMin, TColReduceIdx | 5 |
| **部分运算** | TPartAdd, TPartMul, TPartMax, TPartMin, TPartArgMax, TPartArgMin | 6 |
| **其他** | TDeQuant, TGetScaleAddr, TGatherB, TScatter, TImg2col, SetFmatrix, SetImg2colRpt, SetImg2colPadding, TSort32, TMrgSort, TPow, Tci, THistogram, TBinSOp | 14 |

**小计：81 条** — 无差异。

### 1.2 k9030 有专属实现的指令（12 条）

| # | 指令 | 文件 | 实现方式 |
|---|------|------|---------|
| 1 | TLOAD | `kirin9030/TLoad.hpp` | 原生，kX90 还需 include a2a3 |
| 2 | TSTORE | `kirin9030/TStore.hpp` | 原生 |
| 3 | TMATMUL | `kirin9030/TMatmul.hpp` | 原生（含 TGEMV / TSETTF32MODE） |
| 4 | TGEMV | `kirin9030/TMatmul.hpp` | 原生（矩阵×向量乘） |
| 5 | TSETTF32MODE | `kirin9030/TMatmul.hpp` | 原生（a5 无此指令） |
| 6 | TCVT | `kirin9030/TCvt.hpp` | 原生 |
| 7 | TMOV | `kirin9030/TMov.hpp` | 原生 |
| 8 | TEXTRACT | `kirin9030/TExtract.hpp` | 原生 |
| 9 | TGATHER | `kirin9030/TGather.hpp` | 原生 |
| 10 | TQUANT | `kirin9030/TQuant.hpp` | 原生 |
| 11 | TSYNC | `kirin9030/TSync.hpp` | 原生 |
| 12 | TINSERT | `kirin9030/TInsert.hpp` | 薄包装（19 行，宏重写后委托 a5） |

**小计：12 条** — 需要逐条分析差异。

---

## 二、实现不同指令的逐条对比

---

### 2.1 TLOAD

**指令含义**：将数据从全局内存 (Global Memory) 搬运到本地缓冲区（Vec 路径→UB，Cube 路径→L1）。

#### 支持的数据格式

| 数据类型 | a5 | kirin9030 |
|----------|:--:|:---------:|
| int8 / uint8 | ✓ | ✓ |
| int16 / uint16 | ✓ | ✓ |
| half | ✓ | ✓ |
| int32 / uint32 | ✓ | ✓ |
| float | ✓ | ✓ |
| int64 / uint64 | ✓ | ✓ |
| **bfloat16** | ✓ | ✓（sizeof==2 隐式覆盖） |
| **bfloat16 conv tile** | ✓ | **✗**（CheckConvTileData 仅列 half/float） |
| **float8_e4m3 / float8_e5m2** | ✓（通过 sizeof 检查 + if constexpr） | 由 header.hpp 类型别名 `= int8_t` 覆盖 |
| **float4_e1m2x2 / float4_e2m1x2** | **✓** stride 减半、动检 `BLOCK_BYTE_SIZE*2`、padCount 调整 | **✗** |
| **hifloat8** | ✓ | 由别名覆盖 |

#### 支持的数据排布

| 排布/布局 | a5 | kirin9030 |
|----------|:--:|:---------:|
| ND (RowMajor) | ✓ | ✓ |
| DN (ColMajor, 非分形) | ✓ | ✓ |
| NZ (ColMajor + RowMajor 分形) | ✓ | ✓ |
| ZN (RowMajor + ColMajor 分形) | — | **✓**（`TLoadCubeDN2ZN`） |
| **DN→NZ (Cube)** | **✓** `copy_gm_to_cbuf_multi_dn2nz` | **✗** `static_assert(false)` |

#### 场景/功能差异

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **L2 cache 控制参数** | ✓ `copy_gm_to_ubuf_align_v2` 有 l2 cache 参数 | ✗ 无此参数 |
| **Loop 模式复位** | 条件设置 + **使用后复位** | 无条件设置 + **不复位** |
| **Pad 值管理** | `set_mov_pad_val()` (vec) + 守卫 + **清零复位** | `set_pad_val_outtoub()` + **不复位** |
| **类型分发机制** | `if constexpr` on `sizeof(DType)` | `LoadTypeBySize_t` trait |
| **FP4 支持** | ✓ 全面 | ✗ |
| **DN→ZN 转置立方体加载** | ✗ | ✓ |

---

### 2.2 TSTORE

**指令含义**：将本地缓冲区（Vec→UB，Cube→L1/L0C）的数据写回全局内存。

#### 支持的数据格式

| 数据类型 | a5 | kirin9030 |
|----------|:--:|:---------:|
| 基础类型系列 | ✓ | ✓ |
| **bfloat16** | ✓（含 `set_atomic_bf16`） | ✗ |
| **FP4** | ✓ stride 调整 | ✗ |

#### 支持的数据排布

| 排布 | a5 | kirin9030 |
|------|:--:|:---------:|
| ND | ✓ | ✓ |
| DN | ✓ | ✓ |
| NZ | ✓ | ✓ |
| **MatTile 存储** | **✓** | **✗** |
| **Acc NC1HWC0 布局** | **✓** | **✗** |

#### 场景/功能差异

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **bfloat16 atomic 累加** | ✓ | ✗ |
| **MatTile 写回** | ✓ | ✗ |
| **NC1HWC0 卷积输出** | ✓ | ✗ |
| **32B 对齐要求 (ND Vec)** | ✓ `Cols*sizeof %32 ==0` | ✓ 同 |
| **Vec 循环方式** | 硬件 loop 寄存器 | 硬件 loop 寄存器 |

---

### 2.3 TMATMUL

**指令含义**：`C = A * B`，在 Cube 单元上执行矩阵乘累加。

#### 支持的数据格式

| Accumulator (C) | A (Left) | B (Right) | a5 | kirin9030 |
|:---------------:|:--------:|:---------:|:--:|:---------:|
| **half** | half | half | — | ✓ |
| **float** | half | half | ✓ | — |
| | bfloat16 | bfloat16 | ✓ | — |
| | float | float | ✓ | — |
| | float8_e4m3 | float8_e4m3 | ✓ | — |
| | float8_e4m3 | float8_e5m2 | ✓ | — |
| | float8_e5m2 | float8_e4m3 | ✓ | — |
| | float8_e5m2 | float8_e5m2 | ✓ | — |
| | hifloat8 | hifloat8 | ✓ | — |
| **int32_t** | int8_t | int8_t | ✓ | ✓ |
| **MXFP8 (C=float)** | fp8e4m3/fp8e5m2 混合 4 组合 | → `mad_mx()` | ✓ | ✗ |
| **MXFP4 (C=float)** | fp4e1m2x2/fp4e2m1x2 混合 4 组合 | → `mad_mx()` | ✓ | ✗ |

#### 支持的数据排布（两者一致）

| Tile | isRowMajor | SFractal | 含义 |
|------|:----------:|:--------:|------|
| Left (A) | false | RowMajor | BFractal ColMajor + SFractal RowMajor |
| Right (B) | true | ColMajor | BFractal RowMajor + SFractal ColMajor |
| Acc (C) | false | RowMajor | BFractal ColMajor + SFractal RowMajor |
| Bias | true, Rows=1 | — | 单行 RowMajor |

#### 场景/功能差异

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **MXFP8 微缩浮点矩阵乘** | ✓ `mad_mx()` 完整硬件支持 | ✗ `static_assert("no support")` |
| **MXFP4 微缩浮点矩阵乘** | ✓ `mad_mx()` 完整硬件支持 | ✗ stub |
| **TF32 模式切换** | ✗ | ✓ `TSETTF32MODE`（硬件不允开启） |
| **GEMV 控制** | ✓ 显式 `gemvCtrl` 模板参数 | ✗ 硬编码 false |
| **混合精度 (e4m3×e5m2)** | ✓ | 不适用 |
| **Bias 类型限制** | MX 下必须 float | 同 A/B 类型 |

---

### 2.4 TGEMV

**指令含义**：`C = A * b`，矩阵×向量乘法，TMatmul 在 m=1 时的特化。

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **实现** | 调用 TMatmul，`gemvCtrl=false` | 调用 TMatmul，硬编码 `false`（实质同） |
| **类型支持** | 同 TMatmul（含 MX 变体） | 同 TMatmul（仅 half/int32 累加器） |
| **MX 变体** | ✓ TGEMV_MX_IMPL (3 overloads) | ✗ |

**结论**：实质行为一致，但 a5 多出 MX 变体。

---

### 2.5 TSETTF32MODE

**指令含义**：设置 TF32 (TensorFloat-32) 精度模式开关。

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **是否存在** | ✗ 无此指令 | ✓ |
| **功能** | — | `static_assert(!isEnable)` — 硬件不支持开启，仅能保证关闭 |
| **控制位** | — | bit 46 (TF32_MODE_BIT) + bit 47 (TF32_TRANS_MODE_BIT 定义未用) |

---

### 2.6 TCVT

**指令含义**：在不同数据类型之间转换 tile 数据的数值类型。

#### 支持的数据格式

| 数据类型 | a5 | kirin9030 |
|----------|:--:|:---------:|
| int8, uint8 | ✓ | ✓ |
| int16, uint16 | ✓ | ✓ |
| half | ✓ | ✓ |
| int32, uint32 | ✓ | ✓ |
| float | ✓ | ✓ |
| int64, uint64 | ✓ | ✓ |
| **bfloat16** | **✓** | **✗** |
| **float8_e4m3** | **✓** | **✗** |
| **float8_e5m2** | **✓** | **✗** |
| **hifloat8** | **✓** | **✗** |
| **float8_e8m0** | **✓** | **✗** |
| **float4_e1m2x2** | **✓** | **✗** |
| **float4_e2m1x2** | **✓** | **✗** |

#### 排布/实现差异

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **转换引擎** | 基于**寄存器**（注释 "无需 UB 临时缓冲区"） | 基于 **UB 临时缓冲区** |
| **饱和位** | `SAT_MODE_BIT_60/59/48` | ✓ 同 |

**本质差异**：a5 支持 17 种数据类型的互相转换，k9030 仅支持 10 种基础类型，缺少 bf16/fp8/fp4 全系。

---

### 2.7 TMOV

**指令含义**：在不同 Tile 类型之间搬移数据，包括 Vec→Vec、Acc→Mat、Acc→Vec、Mat→Left/Right/Bias/Scaling 等。

#### 各操作支持的数据类型

| TMov 操作 | a5 | kirin9030 |
|-----------|:--:|:---------:|
| **ToBt (Bias)** — 从 Acc/Mat 搬移到 Bias Buffer | int32→int32, float→float, **half→float, bf16→float** | int32→int32, half→half（仅同类型） |
| **ToFb (Fixpipe/Scaling)** — 搬移到 Fixpipe Buffer | 无类型限制 | 无类型限制 |
| **ToLeft** — 搬移到 Cube L0A (Left 输入) | **11 种**：int8, hifp8, fp8e5m2, fp8e4m3, half, bf16, float, int16, uint16, fp4e2m1, fp4e1m2 | **2 种**：half, int8 |
| **ToRight** — 搬移到 Cube L0B (Right 输入) | 同上 11 种 | 同上 2 种 |
| **CcToCb (Acc→Mat)** — 累加器→Mat Tile | float/int32 源 → 目标含 hifp8/bf16/fp8 | half/int32 源 → 目标含 int16 |
| **CcToUb (Acc→Vec)** — 累加器→Vec Tile | 同上 + DualMode | 同上 + DualMode |
| **VecToVec** — Vec→Vec 拷贝 | Vector intrinsic (`vlds/vsts`) | Vector intrinsic (`vlds/vsts`) |
| **ToVecNd2Nz** — ND→NZ 搬移 | **✓** VF scatter 流水线 | **✓** VF scatter 流水线 |
| **ScaleLeft/ScaleRight** — MX 缩放因子搬移 | **✓** `TExtractToAmx/Bmx` | **✗** `static_assert("not supported")` |

#### 场景/功能差异

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **Bias 类型转换** (half→float) | ✓ | ✗ |
| **Left/Right 高级类型支持** (fp8/fp4/bf16) | ✓ | ✗ |
| **ScaleLeft/ScaleRight** | ✓ | ✗ stub |
| **ConvTile 搬移** | ✓ `TMOV_CONVTILE_IMPL` | ✓（在 TExtract 中） |
| **ND→ZZ 布局搬移** | ✓ | ✗ |
| **Dual-destination Acc→Vec** | ✓ | ✓ |
| **STPhase 流水线阶段控制** | ✓ | ✓ |
| **Fb 缓冲区** | 4KB | 7KB |

---

### 2.8 TEXTRACT

**指令含义**：从 Cube 累加器 (Acc) 或向量 (Vec) Tile 中抽取数据，搬运到目标 Tile（Vec/Mat/Left/Right/Bias/Scaling 等）。

#### 支持的数据类型（Acc→Mat/Acc→Vec 路径）

| Src | Quant | Dst | a5 | kirin9030 |
|:---:|:-----:|:---:|:--:|:---------:|
| float | ✓ | int8/uint8 | ✓ | —（Src 不为 float） |
| | | hifloat8 | ✓ | — |
| | | half | ✓ | — |
| | | bf16 | ✓ | — |
| | | fp8e4m3 | ✓ | — |
| | | float | ✓ | — |
| half | ✓ | half | — | ✓ |
| | | int8/uint8 | — | ✓ |
| | | int16 | — | **✓**（k9030 独有） |
| int32 | ✓ | half | ✓ | ✓ |
| | | int8/uint8 | ✓ | ✓ |
| | | int16 | — | **✓**（k9030 独有） |
| | | bf16 | ✓ | — |
| float | ✗ | half/bf16/float | ✓ | — |
| half | ✗ | half | — | ✓ |
| int32 | ✗ | int32 | ✓ | ✓ |

#### 场景/功能差异

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **FP4 立方体数据抽取** | ✓ `isFp4Type` + `load_cbuf_to_ca_s4` | ✗ |
| **Scale Tile 抽取** (ScaleLeft/Right) | ✓ `TExtractToAmx/Bmx` | ✗ `static_assert("not supported")` |
| **B4 循环** (4-bit 分形) | ✓ `SHIFT_M_STEP_B4` / `M_STEP_MIN_VAL_B4` | ✗ |
| **ConvTile 抽取** (含 FRACTAL_Z_3D) | ✓ | ✓ |
| **Acc→Vec 双目标** (DualModeSplitM/N) | ✓ | ✓ |
| **int16 输出类型** | ✗ | ✓ |

---

### 2.9 TGATHER

**指令含义**：根据索引数组从源 Tile 中收集数据到目标 Tile。

#### 支持的数据类型

| 数据类型 | a5 | kirin9030 |
|----------|:--:|:---------:|
| int8 / uint8 | ✓ | ✓ |
| int16 / uint16 | ✓ | ✓ |
| int32 / uint32 | ✓ | ✓ |
| half | ✓ | ✓ |
| float | ✓ | ✓ |
| **bfloat16** | **✓** | **✗** |
| **float8_e4m3**（专用内核） | **✓** | **✗** |
| **float8_e5m2**（专用内核） | **✓** | **✗** |
| **hifloat8** | **✓** | **✗** |

#### 场景/功能差异

| 功能 | a5 | kirin9030 |
|------|:--:|:---------:|
| **基础索引收集** | ✓ | ✓ |
| **FP8 收集** (e4m3/e5m2 专用内核) | ✓ | ✗ |
| **条件收集** (GT/EQ 比较后筛选收集，8 变体) | ✓ | ✗ |
| **BF16 掩码收集** | ✓ | ✗ |
| **自定义 CEIL 宏** | 无（用共用 `CeilDivision`） | ✓（局部宏） |

---

### 2.10 TQUANT

**指令含义**：将浮点数据量化为低精度整数或 MX (Micro-Scaling) 格式。

#### 支持的量化模式

| 量化模式 | a5 | kirin9030 |
|---------|:--:|:---------:|
| **INT8_SYM** (float→int8) | ✓ | ✓（代码完全一致） |
| **INT8_ASYM** (float→uint8) | ✓ | ✓（代码完全一致） |
| **MXFP8** (float/bf16/half → float8_e4m3) | ✓ 完整流水线 | ✗ `static_assert("does not support")` |
| **MXFP4 E2M1** (bf16/half → float4_e2m1x2) | ✓ 完整流水线 | ✗ stub |

#### 场景/功能差异

| 场景 | a5 | kirin9030 |
|------|:--:|:---------:|
| **INT8 对称/非对称量化** | ✓ | ✓（一致） |
| **MXFP8 量化 (含 AbsReduceMax)** | ✓ 3 变体 + 指数提取 + 量化值计算 | ✗ |
| **MXFP4 量化 (含 E2M1 编解码)** | ✓ 完整流水线 | ✗ |
| **bfloat16/half 输入→MX** | ✓ | ✗ |
| **2D 行步进非连续场景** | ✓ | ✗ |
| **零填充工具 ZeroPadSourceTile** | ✓ | ✗ |

---

### 2.11 TSYNC

**指令含义**：在不同硬件流水线阶段（PIPE_M/MTE1/MTE2/MTE3/V/FIX）之间执行同步屏障，确保数据依赖关系。

#### 场景/功能差异

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **跨核同步** (cross-core event) | ✓ `wait_intra_block(srcPipe, CrossCoreId)` / `set_intra_block()` | **✗** |
| **允许的 Barrier 管道** | 仅 `PIPE_MTE2` / `PIPE_MTE3` / `PIPE_ALL` | 额外允许 `PIPE_M` / `PIPE_MTE1` / `PIPE_FIX` |
| **Event::RecordEvent** | 断言 `!IsCrossCore` 再 Init | 无条件 Init |
| **额外断言** | `IsCrossCore \|\| srcPipe != PIPE_ALL` / `!CrossCore \|\| !AutoToken` | 无 |

---

### 2.12 TINSERT

**指令含义**：将数据插入到目标 Tile 的指定位置，支持 DMA 和 Vector 两种路径。

| 维度 | a5 | kirin9030 |
|------|:--:|:---------:|
| **实现方式** | 637 行完整算法（Vec→Vec ND 对齐/向量实现 + DMA 路径） | **19 行薄包装**：定义 `COPY_CC_TO_CUBF` 宏（将 `copy_matrix_cc_to_cbuf` 第 7 参数改为常量 0）后 `#include "a5/TInsert.hpp"` |
| **功能差异** | 完整 | 宏重写参数后完全委托 a5，**行为一致** |

---

## 三、差异总览表

| 指令 | 数据类型差异 | 排布/格式差异 | 场景功能差异 | 差异根因 |
|:----:|:-----------:|:------------:|:-----------:|---------|
| **TLOAD** | FP4 ✗, bf16 conv ✗ | DN→NZ ✗, DN→ZN ✓ | L2 cache ✗, Loop/Pad 管理不同 | 硬件 micro-architecture 不同 |
| **TSTORE** | FP4 ✗, bf16 ✗ | MatTile ✗, NC1HWC0 ✗ | 32B 对齐方式同 | 硬件存储通路不同 |
| **TMATMUL** | **累加器 float→half** | 分形布局同 | MX ✗, TF32 ✓, gemvCtrl 显式化 | 核心算术单元不同 |
| **TGEMV** | 同 TMATMUL | 同 | MX 变体 ✗ | 同 TMATMUL |
| **TSETTF32MODE** | — | — | a5 无此指令 | k9030 有 TF32 控制位但硬件不支持 |
| **TCVT** | **bf16/fp8/fp4/hifp8/e8m0 ✗** | 转换引擎不同（寄存器 vs UB） | 7 种高级类型不支持 | k9030 硬件无高级类型寄存器 |
| **TMOV** | Left/Right 11种→2种, Bias转 ✗ | ScaleTile ✗, ND→ZZ ✗ | ConvTile 位置不同, Fb 7KB | 硬件寄存器类型有限 |
| **TEXTRACT** | FP4 ✗, int16 ✓ | ScaleTile ✗, B4 循环 ✗ | ConvTile 同, DualMode 同 | k9030 无 FP4 立方体通路 |
| **TGATHER** | bf16/fp8/hifp8 ✗ | — | 条件收集 ✗, FP8 内核 ✗ | k9030 无高级类型收集硬件 |
| **TQUANT** | MXFP8/FP4 ✗ | — | INT8 一致, MX 流水线 ✗ | k9030 无 MX 硬件 |
| **TSYNC** | — | — | 跨核 ✗, 管道约束不同 | k9030 无跨核同步机制 |
| **TINSERT** | — | — | **实质一致**（包装委托 a5） | 仅宏参数重写 |

### 核心结论

1. **累加器类型切换**（a5=`float`, k9030=`half`）是所有差异的最大源头，影响 TMATMUL + TMOV + TEXTRACT 三个指令的量化和类型检查链。
2. **高级数据类型**（bf16/fp8/fp4/hifp8）k9030 在 5 个指令（TLOAD/TCVT/TMOV/TGATHER/TQUANT）中完全不支持，这是 k9030 硬件能力上限。
3. **MX 矩阵乘**（`mad_mx`）a5 有完整硬件支持，k9030 完全无此能力。
4. **跨核同步** a5 有完整机制，k9030 不支持。
5. **排布差异**相对较小：除 DN→NZ/MatTile/NC1HWC0/ScaleTile 外，大部分分形布局要求一致。
6. **29 条 a5 独有/暂缓指令**（TFMod/TRem/Expdif/Random/Prefetch/FIFO/MGather/MScatter）两架构都不支持，不在分析范围内。
