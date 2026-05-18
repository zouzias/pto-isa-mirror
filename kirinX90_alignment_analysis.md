# KirinX90 与 Kirin9030 对齐分析

## 总览

KirinX90 与 Kirin9030 的 align 工作，通过 `include/pto/npu/kirinX90/header.hpp` 体现。整体策略为：**差异化指令原生重写 + 基础设施扩展 + 复用 kirin9030/a5/a2a3**。

header.hpp 共 **37 条有效 include**，来自四个来源：

| 来源 | 数量 | 涵盖内容 |
|------|:----:|---------|
| **kirinX90 原生 (5)** | TLoad / TStore / TMov / TExtract(内部) / TCvt |
| **基础设施扩展 (2)** | common.hpp / datatype.hpp |
| **复用 kirin9030 (4)** | utils.hpp / TSync / TMatmul / TGather |
| **复用 a2a3 (3)** | TSubView / TAssign / TExtract(公开API) |
| **复用 a5 (27)** | 基础算术/比较/排序/一元运算等 |

---

## 一、已完成的对齐工作

### 1.1 原生重写 — 针对硬件差异

KirinX90 对其与 kirin9030 硬件行为不同的指令做了完整的原生重写：

#### TLoad (646 行)

| 维度 | kirin9030 | kirinX90 |
|------|-----------|----------|
| GM→UB intrinsic | `copy_gm_to_ubuf_align_v2` (10 参, stride 按字节) | `copy_gm_to_ubuf_align` (9 参, gap 按 32B block) |
| GM→L1 intrinsic | `copy_gm_to_cbuf_align_v2` (stride 模式) | `copy_gm_to_cbuf` (gap 模式) |
| ND→NZ intrinsic | 通用 `copy_gm_to_cbuf_multi_nd2nz` + `set_mte2_nz_para()` | 类型后缀 `_b8`/`_b16` 显式变体 |
| 循环方式 | 硬件 loop 模式 (`set_loop*_stride` / `set_loop_size`) | 手动 C++ for 循环 (3 层嵌套) |
| Pad 值设置 | `set_pad_val_outtoub()` / `set_pad_val_outtol1()` | `set_mov_pad_val()` |
| 大小单位 | 全程按字节 (`GetByteSize<T>(count)`) | Element + 32B block |
| 布局分发 | `isRowMajor` + `SFractal` 枚举 | `GetTileLayoutCustom<TileData>()` |
| bfloat16 conv tile | **不支持** | **支持** |

#### TStore (677 行)

| 维度 | kirin9030 | kirinX90 |
|------|-----------|----------|
| UB→GM intrinsic | `copy_ubuf_to_gm_align_v2` | `copy_ubuf_to_gm_align` |
| Vec 循环 | 硬件 loop 寄存器 | 手动 for 循环 |
| MatTile 支持 | **不支持** | **新增**：TStoreMat / TStoreMat2GmNd2Nd / Dn2Dn / Nz2Nz |
| Acc 布局 | 仅 ND / NZ | **新增 NC1HWC0**：`TStoreAccNz2NC1HWC0` |
| Align 要求 | 要求 32B 对齐 | **放宽**：无 32B 对齐检查 |
| bfloat16 | 不支持 | **新增**：`set_atomic_bf16()` |
| 量化位编码 | 使用 bit[29] + bit[38:34] | 仅使用 bit[38:34] |
| 显式 barrier | 无 | **新增**：`pipe_barrier(PIPE_FIX)` in TStoreAccFp |

#### TMov (264 行)

| 维度 | kirin9030 | kirinX90 |
|------|-----------|----------|
| Acc→Cb signature | 23 参数（含 channelSplit / Nz2Nd / Nz2Dn 等） | 12 参数（简化版） |
| Acc→Ub (双目标) | **完整支持**：AccToVecMode + STPhase + GetDualDstCtl | **不支持** |
| Vec→Vec 实现 | Vector intrinsic (`vlds`/`vsts` + `MaskReg` 谓词) | DMA 方式 (`pto_copy_ubuf_to_ubuf`) |
| Vec→Mat | 支持 (通过 TExtractVecToMat) | 不支持 |
| ND→NZ 搬移 | **支持**：`TMovToVecNd2Nz` VF scatter | **不支持** |
| ConvTile | 已迁至 TExtract | **已支持**：`TMOV_CONVTILE_IMPL` |
| ScaleLeft/Right | **stub 禁止** | **不存在**（未引用） |
| Fb 类型限制 | 无限制 | **限制为 `uint64_t` 仅** |
| Bias 最大容量 | 4.0 KB | 1.0 KB |
| `TMOV_IMPL` overload 数量 | 7 个（含 mode/STPhase/scalar/fp 变体） | 简化（仅 Vec/Vec→Mat/Acc→Mat/ConvTile） |

#### TExtract (452 行) — 仅内部帮助函数

| 维度 | kirin9030 | kirinX90 |
|------|-----------|----------|
| Acc→Vec 抽取 | **有**（含 AccToVecMode 多种模式） | **已删除** |
| ConvTile 抽取 | **有**（TExtractToBConv + TEXTRACT_CONVTILE_IMPL） | **已删除** |
| Vec→Vec ND 抽取 | **有**（5 个变体） | **已删除** |
| Vec→Vec NZ 抽取 | **有**（2 个变体） | **已删除** |
| FP4/B4 支持 | **无** | **新增**：`KHALF=2`, `SHIFT_M_STEP_B4=2`, `M_STEP_MIN_VAL_B4=4` |
| 公开 TEXTRACT_IMPL | 由自己提供 | 来自 a2a3/TExtract.hpp 包装 |
| 文件大小 | 842 行 | 452 行 |

#### TCvt (2379 行)

- 完整原生重写，与 kirin9030 不同的转换引擎
- 对应 CTRL 寄存器 issue，使用 `RS_ENABLE` fallback（见注释 "KirinX90 set CTRL unsuccess"）
- 7 种 CastMode，支持 PyTorch edge-case alignment
- 条件编译：仅在 `__DAV_VEC__` 时生效

### 1.2 基础设施扩展

#### common.hpp (54 行)

- **包装 kirin9030/common.hpp**（`#include` 后追加 X90 特有内容）
- **新增** `CheckTMovAccToMat()`：比 kirin9030 的 `CheckTMovAccValid` 更严格
  - 增加了 `SFractalSize == TileConfig::fractalABSize` 检查
  - 增加了 `Cols * sizeof(DstType) % C0_SIZE_BYTE == 0` 对齐检查

#### datatype.hpp (25 行)

- 继承 kirin9030/datatype.hpp 的全部 TypeGet 特化
- **新增** `TypeGet<vector_bf16>` 恒等映射（`__DAV_VEC__` 保护）

### 1.3 复用 kirin9030（4 条）

| 文件 | 来源 | 说明 |
|------|------|------|
| `utils.hpp` | `kirin9030/utils.hpp` | 间接复用 a5 |
| `TSync.hpp` | `kirin9030/TSync.hpp` | 直接复用 |
| `TMatmul.hpp` | `kirin9030/TMatmul.hpp` | 直接复用 |
| `TGather.hpp` | `kirin9030/TGather.hpp` | 直接复用 |

### 1.4 复用 a2a3（3 条）

| 指令 | 来源 | 与 kirin9030 差异 |
|------|------|-----------------|
| TSubView | `a2a3/TSubView.hpp` | kirin9030 复用 a5，kirinX90 复用 a2a3 |
| TAssign | `a2a3/TAssign.hpp` | kirin9030 复用 a5，kirinX90 复用 a2a3 |
| TExtract | `a2a3/TExtract.hpp` | kirin9030 有原生实现，kirinX90 包装 a2a3 |

### 1.5 复用 a5（27 条）

TAdd / TAddS / TSub / TMul / TDiv / TDivS / TMulS / TMax / TMin / TMrgSort / TCmps / TColSum / TReshape / TRowReduce (TRowSum/TRowMax/TRowMin) / TFillPad / TTrans / Tci / TSel / TSort32 / TRowExpand / TPartAdd / TPartMax / TPartMin / TRsqrt / TUnaryOp (TAbs/TExp/TLog/TSqrt/TNot/TRelu/TNeg) / TBinSOp

---

## 二、仍需做的对齐工作

### 2.1 可直接复用 a5 的缺失指令（47 条）

这些指令在 kirin9030 中通过 `#include "pto/npu/a5/..."` 直接复用，kirinX90 理论上也可以同样方式接入，只需在 `header.hpp` 中添加对应 include：

| 类别 | 指令 | 数量 |
|------|------|:----:|
| 位运算 | TAnd / TAndS / TOr / TOrS / TXor / TXorS / TShl / TShlS / TShr / TShrS | **10** |
| 标量运算 | TSubS / TMaxS / TMins | **3** |
| 线性运算 | TAxpy | **1** |
| 激活函数 | TPrelu / TLRelu | **2** |
| 比较 | TCmp（已有 TCmps） | **1** |
| 选择 | TSels | **1** |
| 列展开基版 | TColExpand | **1** |
| 列展开变体 | TColExpandAdd / TColExpandSub / TColExpandMul / TColExpandDiv / TColExpandMax / TColExpandMin | **6** |
| 行展开变体 | TRowExpandAdd / TRowExpandSub / TRowExpandMul / TRowExpandDiv / TRowExpandMax / TRowExpandMin | **6** |
| 行规约扩展 | TRowProd / TRowReduceIdx (TRowArgMax/TRowArgMin) | **2** |
| 列规约扩展 | TColProd / TColMax / TColMin / TColReduceIdx | **4** |
| 部分运算扩展 | TPartMul / TPartArgMax / TPartArgMin | **3** |
| 数据变换 | TConcat / TExpandS / TTri | **3** |
| 量化 | TDeQuant / TGetScaleAddr | **2** |
| Gather/Scatter | TGatherB / TScatter | **2** |
| 图像处理 | TImg2col / SetFmatrix / SetImg2colRpt / SetImg2colPadding | **4** |
| 其它 | TPow / THistogram | **2** |

**小计：47 条** — 这些只需在 header.hpp 添加 include，无需额外实现。

### 2.2 需 kirinX90 原生实现的缺失指令（2 条）

| 指令 | kirin9030 来源 | 说明 |
|------|---------------|------|
| **TQuant** | kirin9030 原生 | 包含 INT8_SYM/INT8_ASYM 量化。kirinX90 需要评估是否支持 MXFP8/MXFP4（kirin9030 也 stub 了） |
| **TInsert** | kirin9030 包装 a5 | 已在 X90 header 中注释（line 57）。可能需要适配 COPY_CC_TO_CUBF 宏 |

### 2.3 与 a5→kirin9030 共同的暂缓指令（15 条）

这些在 kirin9030 header 中也已注释或标记 "to be evaluated"：

| 指令 | kirin9030 状态 |
|------|---------------|
| TFMod / TFModS | 已注释 |
| TRem / TRemS | 已注释 |
| TRowExpandExpdif | 已注释 |
| TColExpandExpdif | 已注释 |
| TRandom | to be evaluated |
| TPrefetch | to be evaluated |
| TPush / TPop / TAlloc / TFree | to be evaluated |
| MGather / MScatter | to be evaluated |
| TMATMUL_MX | stub |

**小计：15 条** — 暂时无需处理，等 kirin9030 侧先确定后同步。

### 2.4 原生实现的潜在差异（需验证）

以下指令虽可复用 kirin9030 或 a5，但需验证在 kirinX90 硬件上是否行为正确：

| 指令 | 风险点 |
|------|--------|
| **TMatmul** (复用了 k9030) | k9030 TMatmul 使用 mad() 硬编码 gemvCtrl=false，X90 硬件可能支持不同 |
| **TSync** (复用了 k9030) | k9030 同步允许 PIPE_M/PIPE_MTE1/PIPE_FIX，X90 可能不同 |
| **TGather** (复用了 k9030) | k9030 的 CEIL() / batchSize 计算可能与 X90 硬件不匹配 |

---

## 三、工作量总结

### 已完成的 align 工作

| 工作 | 内容 | 涉及 |
|------|------|------|
| 原生重写 | TLoad / TStore / TMov (不同 intrinsic / 循环模式 / 功能集) | 3 个核心指令 |
| 简化重写 + FP4 新增 | TExtract (仅内部 helpers) | 1 个内部文件 |
| 完整重写 | TCvt (不同硬件行为) | 1 个核心指令 |
| 基础设施扩展 | common.hpp + datatype.hpp | 2 个扩展文件 |
| 复用 kirin9030 | TSync / TMatmul / TGather / utils | 4 条 |
| 复用 a2a3 | TSubView / TAssign / TExtract (公开 API) | 3 条 |
| 复用 a5 | 27 条基础指令 | 27 条 |

### 仍需完成的 align 工作

| 工作 | 内容 | 数量 | 难度 |
|------|------|:----:|:----:|
| **直接加 include** | 复用 a5 的 47 条指令 | 47 | ★☆☆ 低（仅改 header.hpp） |
| **评估 + 原生实现** | TQuant（如需要 MX） | 1 | ★★★ 高 |
| **评估 + 适配** | TInsert（COPY_CC_TO_CUBF 宏检查） | 1 | ★★☆ 中 |
| **暂缓处理** | TFMod/R等下 15 条 | 15 | ★☆☆ 等待决策 |
| **验证兼容性** | TMatmul/TSync/TGather 在 X90 硬件上 | 3 | ★★☆ 需测试 |

### 优先级建议

| 优先级 | 内容 | 原因 |
|:-----:|------|------|
| P0 | header.hpp 添加 47 条 a5 include | 零成本，快速缩小差距 |
| P1 | 验证 TMatmul/TSync/TGather 兼容性 | 已在复用，必须保证正确 |
| P2 | 评估 TInsert 并取消注释 | 对齐基础操作 |
| P3 | 评估 TQuant 是否真正需要 | 如不需要 MX，可直接复用 |
| P4 | 跟进 15 条暂缓指令决策 | 等 kirin9030 侧先确定 |
