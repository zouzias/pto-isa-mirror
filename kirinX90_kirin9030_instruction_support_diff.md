# KirinX90 与 Kirin9030 同一指令支持方式差异分析

## 分析范围

基于两架构的 `header.hpp` 及实现文件，对比两架构**共同实现**的指令在数据类型、数据排布、场景功能等方面的差异。

---

## 一、两架构共同支持的指令

### 1.1 完全相同的指令（57 条）

以下指令两架构使用**完全相同的实现代码**，无差异：

#### 复用 kirin9030 原生（3 条）

| 指令 | 来源 | 说明 |
|------|------|------|
| **TMatmul** (+TGEMV/TSETTF32MODE) | `kirin9030/TMatmul.hpp` | kX90 直接 include k9030，代码一致 |
| **TGather** | `kirin9030/TGather.hpp` | kX90 直接 include k9030，代码一致 |
| **TSync** | `kirin9030/TSync.hpp` | kX90 直接 include k9030，代码一致 |

#### 复用 a5 实现（27 条）

两架构都 `#include` 相同的 `a5/*.hpp`：

| 类别 | 指令 | 数量 |
|------|------|:----:|
| **二元运算** | TAdd, TSub, TMul, TDiv, TMax, TMin | 6 |
| **标量运算** | TAddS, TDivS, TMulS | 3 |
| **一元运算** | TUnaryOp (TAbs/TExp/TLog/TSqrt/TNot/TRelu/TNeg), TRsqrt | 8 |
| **比较** | TCmps | 1 |
| **选择** | TSel | 1 |
| **变换** | TReshape, TFillPad, TTrans | 3 |
| **规约** | TColSum, TRowReduce (TROWSUM/TROWMAX/TROWMIN) | 4 |
| **排序** | TSort32, TMrgSort | 2 |
| **广播** | TRowExpand | 1 |
| **部分运算** | TPartAdd, TPartMax, TPartMin | 3 |
| **其他** | Tci, TBinSOp | 2 |

#### 复用 a2a3 实现（2 条，经过 a5 中转或直连）

| 指令 | k9030 路径 | kX90 路径 | 最终来源 |
|------|-----------|----------|---------|
| **TSubView** | `a5/TSubView.hpp` → `a2a3/TSubView.hpp` | `a2a3/TSubView.hpp` 直接 | 同为 a2a3 代码 |
| **TAssign** | `a5/TAssign.hpp` → `a2a3/TAssign.hpp` | `a2a3/TAssign.hpp` 直接 | 同为 a2a3 代码 |

**小计：57 条** — 无差异。

### 1.2 实现不同的指令（5 条）

| # | 指令 | k9030 | kX90 | 差异性质 |
|---|------|-------|------|---------|
| 1 | TLOAD | 原生 | 原生 | 底层 intrinsic + 循环机制完全不同 |
| 2 | TSTORE | 原生 | 原生 | 底层 intrinsic + 功能集不同 |
| 3 | TMOV | 原生 | 原生 | 功能集大幅删减 + intrinsic 参数不同 |
| 4 | TExtract | 原生（含公开 API） | 内部 helper（公开 API 来自 a2a3） | 角色定位完全不同 |
| 5 | TCVT | 原生 | 原生（条件 `__DAV_VEC__`） | 转换引擎 + 硬件 workaround 不同 |

**小计：5 条** — 以下逐条详细对比。

---

## 二、实现不同指令的逐条对比

---

### 2.1 TLOAD

**指令含义**：将数据从全局内存 (GM) 搬运到本地缓冲区（Vec→UB，Cube→L1）。

#### 支持的数据格式

| 数据类型 | kirin9030 | kirinX90 |
|----------|:---------:|:---------:|
| int8 / uint8 | ✓ | ✓ |
| int16 / uint16 | ✓ | ✓ |
| half | ✓ | ✓ |
| int32 / uint32 | ✓ | ✓ |
| float | ✓ | ✓ |
| int64 / uint64 | ✓ | ✓ |
| bfloat16 | ✓（sizeof==2 隐式） | ✓（显式列在 CheckNormalTileData） |
| **bfloat16 conv tile** | **✗** | **✓** |
| float8 / float4 / hifloat8 | 类型别名覆盖 | 类型别名覆盖 |

#### 支持的数据排布

| 排布 | kirin9030 | kirinX90 |
|------|:---------:|:---------:|
| ND (RowMajor) | ✓ | ✓ |
| DN (ColMajor, 无分形) | ✓ | ✓ |
| NZ (ColMajor + RowMajor) | ✓ | ✓ |
| **DN→ZN** (RowMajor + ColMajor) | **✓** | **✗** |

#### 场景/功能差异

| 维度 | kirin9030 | kirinX90 | 差异性质 |
|------|-----------|----------|---------|
| **GM→UB intrinsic** | `copy_gm_to_ubuf_align_v2` (10 参, 按字节) | `copy_gm_to_ubuf_align` (9 参, 按 32B block) | **不同 intrinsic 函数名和参数模板** |
| **GM→L1 intrinsic** | `copy_gm_to_cbuf_align_v2` (stride 模式) | `copy_gm_to_cbuf` (gap 模式) | 不同 intrinsic |
| **ND→NZ intrinsic** | 通用 `copy_gm_to_cbuf_multi_nd2nz` + `set_mte2_nz_para()` | 类型后缀 `_b8`/`_b16` 显式变体 | 不同 dispatch 方式 |
| **循环方式** | **硬件 loop 寄存器** (`set_loop2/1_stride_outtoub`, `set_loop_size_outtoub`)。仅最外层留软件循环 | **手动 C++ for 循环**（3 层嵌套，每层一个 DMA） | **架构级差异** |
| **Pad 值设置 (Vec)** | `set_pad_val_outtoub(GetPadValue<...>())` | `set_mov_pad_val(GetPadValue<...>())` | 不同 intrinsic |
| **Pad 值设置 (Cube)** | `set_pad_val_outtol1(padCount)` — 无条件 | 有条件（`PadVal != Null/Zero`）+ 清零复位 | 管理方式不同 |
| **大小单位** | 全程**字节** (`GetByteSize<T>(count)`) | Element + **32B block** (value >> SHIFT_BLOCK_BYTE) | 不同约定 |
| **nBurst 类型** | `uint32_t` | `uint16_t` | 类型不同 |
| **布局分发** | `isRowMajor` + `SFractal` 枚举组合 | `GetTileLayoutCustom<TileData>()` | 不同分发方式 |
| **bfloat16 conv tile** | ✗ | ✓ | kX90 功能更多 |
| **loop 管理** | 设置后**不复位** | 不涉及（无硬件 loop） | 不适用 |
| **pad 清理** | 设置后**不复位** | 使用后 `set_pad_val_outtol1(0)` | kX90 更规范 |

---

### 2.2 TSTORE

**指令含义**：将本地缓冲区数据写回全局内存。

#### 支持的数据格式

| 数据类型 | kirin9030 | kirinX90 |
|----------|:---------:|:---------:|
| 基础类型系列 | ✓ | ✓ |
| **bfloat16** | **✗**（无显式处理） | **✓**（`set_atomic_bf16`） |
| **int64_t atomic** | **✗**（static_assert 排除） | **✓**（未排除） |

#### 支持的数据排布

| 排布 | kirin9030 | kirinX90 |
|------|:---------:|:---------:|
| ND | ✓ | ✓ |
| DN | ✓ | ✓ |
| NZ | ✓ | ✓ |
| **MatTile 存储** (TStoreMat) | **✗** | **✓**（含 Nd2Nd / Dn2Dn / Nz2Nz） |
| **Acc NC1HWC0** (卷积布局) | **✗** | **✓**（TStoreAccNz2NC1HWC0） |

#### 场景/功能差异

| 维度 | kirin9030 | kirinX90 | 差异性质 |
|------|-----------|----------|---------|
| **UB→GM intrinsic** | `copy_ubuf_to_gm_align_v2` | `copy_ubuf_to_gm_align` | 不同 intrinsic |
| **Vec 循环** | 硬件 loop 寄存器 (`set_loop_size_ubtoout` 等) | 手动 for 循环 | **架构级差异** |
| **MatTile 存储** | ✗ 不支持 | ✓ 新增（完整 Nd/Dn/Nz） | **kX90 功能新增** |
| **NC1HWC0 布局** | ✗ | ✓ | kX90 功能新增 |
| **32B 对齐检查 (ND Vec)** | ✓ `Cols*sizeof %32 ==0`（严格） | ✗ 无检查（宽松） | kX90 约束放宽 |
| **bfloat16 atomic 累加** | ✗ | ✓ `set_atomic_bf16()` | kX90 功能新增 |
| **量化 bit 编码** | bit[29] + bit[38:34] | 仅 bit[38:34] | 编码位不同 |
| **PIPE_FIX barrier** | 无 | `TStoreAccFp` 中显式添加 | kX90 新增 |
| **ND 参数设置** | `set_nd_para(ndParaSPR)` | `set_loop3_para(config)` | 不同硬件 intrinsic |

---

### 2.3 TMOV

**指令含义**：在不同 Tile 类型之间搬移数据（Vec↔Vec, Acc→Mat, Acc→Vec, Mat→Left/Right/Bias/Scaling 等）。

#### 支持的数据格式

| TMov 操作 | kirin9030 | kirinX90 |
|-----------|:---------:|:---------:|
| **ToBt (Bias)** — 搬移到 Bias Buffer | int32→int32, half→half | int32→int32, half→half（同） |
| **ToFb (Scaling/Fixpipe)** — 搬移到 Fixpipe Buffer | **无类型限制** | **仅 `uint64_t`**（`static_assert`） |
| **ToLeft / ToRight** — 搬移到 Cube L0A/L0B | **2 种**：half, int8 | **2 种**：half, int8（经 TExtract 检查） |
| **CcToCb (Acc→Mat)** — 累加器→Mat | half/int32 源，目标含 int16 | half/int32 源，目标含 int16（同） |
| **CcToUb (Acc→Vec)** — 累加器→Vec | 同上 + DualMode | **不支持** |
| **Vec→Vec** | Vector intrinsic (`vlds/vsts` + `MaskReg`) | **DMA 方式** (`pto_copy_ubuf_to_ubuf`) |
| **Vec→Mat** | ✓ 支持（通过 TExtractVecToMat） | ✗ 不支持 |
| **Vec Nd2Nz** | ✓ VF scatter 流水线 | ✗ 不支持 |
| **ScaleLeft/ScaleRight** — MX 缩放因子 | `static_assert("not supported")` | 未引用（不涉及） |

#### 场景/功能差异

| 维度 | kirin9030 | kirinX90 | 差异性质 |
|------|-----------|----------|---------|
| **Acc→Cb intrinsic** | `copy_matrix_cc_to_cbuf(..., **23 参数**：含 channelSplitEnable, enableNz2Nd, enableNz2Dn, 多个 padding)` | `copy_matrix_cc_to_cbuf(..., **12 参数**：简化版，无以上参数)` | **不同的参数模板** |
| **Acc→Ub 双目标** | ✓ 完整支持（AccToVecMode + STPhase + GetDualDstCtl） | **✗ 完全缺失** | kX90 功能缺失 |
| **Vec→Vec 实现** | **Vector intrinsic** (`vlds/vsts` + `RegTensor` + `MaskReg`) | **DMA 块拷贝** (`pto_copy_ubuf_to_ubuf`) | **原理级差异** |
| **Vec→Mat** | ✓ | ✗ | kX90 功能缺失 |
| **ND→NZ 搬移** | ✓ (`TMovToVecNd2Nz` VF scatter) | ✗ | kX90 功能缺失 |
| **ConvTile** | 已迁至 TExtract | ✓ (`TMOV_CONVTILE_IMPL`，仅 FRACTAL_Z) | 所在位置不同 |
| **Acc→Cb 的 srcStride** | `CeilAlignment(validRow, BLOCK_LEN)` (对齐到 32) | `SrcTileData::Rows` (原始值) | 对齐策略不同 |
| **Bias 最大容量** | 4.0 KB (`PTO_BIAS_SIZE_BYTES`) | **1.0 KB** (`PTO_BIAS_SIZE_BYTES`) | **kX90 更小** |
| **Fb 存储类型限制** | 无限制 | **限制为 `uint64_t` 仅** | kX90 更严格 |
| **Fb 缓冲区大小** | 7.0 KB (`PTO_FBUF_SIZE_BYTES`) | **4 KB** (硬编码 4096) | **kX90 更小** |
| **STPhase 控制** | ✓ 模板参数传递到 TMovCcToUb | ✗ 不支持 | kX90 功能缺失 |
| **TMOV_IMPL overload 数量** | 7 个（含 mode/STPhase/scalar/fp 变体） | 简化版（Vec/Acc→Mat/ConvTile） | kX90 更少 |
| **Fb 存储类型限制** | 无限制 | **限制 `uint64_t` 仅** | |

---

### 2.4 TExtract

**指令含义**：从 Cube 累加器 (Acc) 或向量 (Vec) Tile 中抽取数据，搬运到目标 Tile（Vec/Mat/Left/Right 等）。

#### 本质定位差异

| 维度 | kirin9030 (842 行) | kirinX90 (452 行) |
|------|-------------------|-------------------|
| **角色** | 提供**全部功能**：内部 helper + 公开 `TEXTRACT_IMPL` | **仅内部 helper**：供 TMov 使用（TExtractToA/ToB/AccToMat） |
| **公开 API 来源** | 自身提供 `TEXTRACT_IMPL` | **`a2a3/TExtract.hpp`**（header.hpp line 33） |

#### 功能集差异

| 功能 | kirin9030 | kirinX90 | 差异性质 |
|------|:---------:|:---------:|---------|
| **Acc→Vec 抽取**（含 DualMode SplitM/N） | ✓ | **✗** | kX90 全部删除 |
| **ConvTile 抽取**（TExtractToBConv） | ✓ | **✗** | kX90 全部删除（由 TMov 处理） |
| **Vec→Vec ND 抽取**（5 个变体） | ✓ | **✗** | kX90 全部删除 |
| **Vec→Vec NZ 抽取**（2 个变体） | ✓ | **✗** | kX90 全部删除 |
| **Acc→Mat 抽取** | ✓ | ✓ | 两架构都有 |
| **FP4/B4 支持**（KHALF/SHIFT_M_STEP_B4） | **✗** | **✓** | **kX90 新增** |
| **TExtractToLeft 行检查** | `Rows == 1 && isRowMajor`（严格） | `isRowMajor`（宽松） | kX90 约束放宽 |
| **TExtractVecToMat 参数** | 静态 DstTile 维度 | 运行时 dstValidRow/Col 参数 | 参数化方式不同 |
| **ScaleLeft/Right** | `static_assert(false)` | `static_assert(false)` | 一致（均 stub） |

---

### 2.5 TCVT

**指令含义**：在不同数据类型之间转换 tile 数据的数值类型。

#### 支持的数据格式（一致）

| 数据类型 | kirin9030 | kirinX90 |
|----------|:---------:|:---------:|
| int8 / uint8 | ✓ | ✓ |
| int16 / uint16 | ✓ | ✓ |
| half | ✓ | ✓ |
| int32 / uint32 | ✓ | ✓ |
| float | ✓ | ✓ |
| int64 / uint64 | ✓ | ✓ |

两架构都只支持 10 种基础类型，无 bf16/fp8/fp4 等高级类型。

#### 实现差异

| 维度 | kirin9030 (2121 行) | kirinX90 (2379 行) | 差异性质 |
|------|-------------------|-------------------|---------|
| **转换引擎** | 基于 **UB 临时缓冲区** | 基于**寄存器**（注释 "无需 UB 临时缓冲区"） | **原理级差异** |
| **CTRL 寄存器饱和位** | 正常使用 `SAT_MODE_BIT_60/59/48` | **硬件有 bug** → 改用 `RS_ENABLE` 参数 fallback | **硬件 workaround** |
| **CastMode 枚举** | 6 种：EXPAND/ROUND/ROUND_SAT/ROUND_PART/ROUND_SAT_PART/SAT_PART | **7 种**：上述 6 种 + **`SAT_ROUND`** | kX90 新增一种 |
| **PyTorch 对齐** | 无 | `EDGE_CASE_ALIGN_ENABLE 1` | kX90 新增边缘对齐模式 |
| **条件编译** | 无条件始终生效 | **`__DAV_VEC__` 保护** | kX90 有条件编译 |

---

## 三、差异汇总

### 3.1 两架构实现相同的指令（57 条）

| 来源 | 指令 | 数量 |
|------|------|:----:|
| 复用 k9030 原生 | TMatmul(+TGEMV/TSETTF32MODE), TGather, TSync | 3 |
| 复用 a5 | TAdd, TSub, TMul, TDiv, TMax, TMin, TAddS, TDivS, TMulS, TUnaryOp(7), TRsqrt, TCmps, TSel, TReshape, TFillPad, TTrans, TColSum, TRowReduce(3), TSort32, TMrgSort, TRowExpand, TPartAdd, TPartMax, TPartMin, Tci, TBinSOp | 27 |
| 复用 a2a3(同代码) | TSubView, TAssign | 2 |
| 共计 | — | **57** |

### 3.2 两架构实现不同的指令（5 条）

| 指令 | 主要差异 | 差异分类 |
|:----:|---------|---------|
| **TLOAD** | intrinsic 函数名+参数模板完全不同；k9030 用硬件 loop 寄存器，kX90 用手动 for 循环；pad 管理方式不同；bfloat16 conv tile 支持差异 | **架构级**（循环机制 + intrin API） |
| **TSTORE** | intrinsic 名不同；kX90 新增 MatTile/NC1HWC0/bfloat16 atomic 支持；32B 对齐放宽；量化编码位不同 | **架构级**（intrin API + 功能扩展） |
| **TMOV** | Acc→Cb 参数数 23→12；Acc→Ub 双目标/STPhase 缺失；Vec→Vec DMA 代替 vector intrin；Bias/Fb 容量更小；Fb 限 uint64 | **功能删减 + API 差异** |
| **TExtract** | 角色不同（kX90 仅内部 helper）；Acc→Vec/ConvTile/Vec→Vec 全部删除；FP4 新增；行检查放宽 | **功能删减 + 角色变更** |
| **TCVT** | 基于寄存器 vs UB 缓冲；CTRL bug→RS_ENABLE 回退；CastMode 多 SAT_ROUND；条件编译 | **原理级 + 硬件 workaround** |

### 3.3 差异根因总结

| 根因 | 影响指令 | 说明 |
|------|---------|------|
| **硬件循环机制不同** | TLOAD, TSTORE | k9030 有硬件 loop 寄存器，kX90 用手动 for 循环 |
| **底层 intrin 函数不同** | TLOAD, TSTORE, TMOV | 数据搬运 / 矩阵搬移的硬件 API 完全不同 |
| **功能集取舍** | TSTORE, TMOV, TExtract | kX90 删除双目标/Vec→Mat/ND→NZ 等，新增 MatTile/NC1HWC0/bfloat16 |
| **硬件 bug workaround** | TCVT | CTRL 寄存器在 kX90 上不可用 |
| **角色重组** | TExtract | kX90 将公开 API 交给 a2a3，自身仅保留内部 helper |
| **缓冲区容量差异** | TMOV | Bias 1KB vs 4KB, Fb 4KB vs 7KB |
