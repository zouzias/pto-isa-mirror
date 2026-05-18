# Kirin9030 与 A5 架构适配分析

## 分析背景

基于 `include/pto/npu/a5/` 与 `include/pto/npu/kirin9030/` 的源码对比，分析 kirin9030 为了与 a5 对齐所做的适配工作。

整体策略：**差异化指令原生实现 + 相同指令直接复用 a5**。

---

## 1. 架构适配策略总览

```
kirin9030/header.hpp  (137 行, 96 条有效 include)
├── 原生实现 (10 个文件, 12 条指令)
│   ├── TLoad.hpp       ── TLOAD
│   ├── TStore.hpp      ── TSTORE
│   ├── TMatmul.hpp     ── TMATMUL / TGEMV / TSETTF32MODE
│   ├── TMov.hpp        ── TMOV
│   ├── TExtract.hpp    ── TEXTRACT
│   ├── TCvt.hpp        ── TCVT
│   ├── TGather.hpp     ── TGATHER
│   ├── TQuant.hpp      ── TQUANT
│   ├── TSync.hpp       ── TSYNC
│   └── TInsert.hpp     ── TINSERT (19 行薄包装, 委托给 a5)
├── 基础设施 (3 个文件)
│   ├── common.hpp      ── 寄存器类型 / 量化模式 / 有效性检查
│   ├── datatype.hpp    ── TypeGet 类型映射 (59 行精简版)
│   └── utils.hpp       ── 仅 10 行 `#include <pto/npu/a5/utils.hpp>`
└── 复用 a5 (~80 条 include)
    ├── 二元运算: TAdd / TSub / TMul / TDiv / TMax / TMin / TAxpy /
    │             TPrelu / TLRelu
    ├── 标量运算: TAddS / TSubS / TMulS / TDivS / TMaxS / TMins
    ├── 位运算:   TAnd / TOr / TXor / TShl / TShr / TAndS / TOrS /
    │             TXorS / TShlS / TShrS
    ├── 比较:     TCmp / TCmps
    ├── 选择:     TSel / TSels
    ├── 一元:     TUnaryOp (TAbs/TExp/TLog/TSqrt/TNot/TRelu/TNeg) / TRsqrt
    ├── 行操作:   TRowExpand / TRowExpandAdd/Sub/Mul/Div/Max/Min /
    │             TRowReduce / TRowReduceIdx / TRowProd
    ├── 列操作:   TColExpand / TColExpandAdd/Sub/Mul/Div/Max/Min /
    │             TColSum / TColProd / TColMax / TColMin / TColReduceIdx
    ├── 部分:     TPartAdd / TPartMul / TPartMax / TPartMin /
    │             TPartArgMax / TPartArgMin
    ├── 变换:     TAssign / TSubView / TReshape / TConcat / TTrans /
    │             TFillPad / TExpandS / TTri
    ├── 其他:     TSort32 / TMrgSort / TPow / Tci / THistogram /
    │             TImg2col / SetFmatrix / SetImg2colRpt / SetImg2colPadding /
    │             TGatherB / TScatter / TDeQuant / TGetScaleAddr / TBinSOp
    └── 注释暂缓: TFMod / TFModS / TRem / TRemS /
                   TRowExpandExpdif / TColExpandExpdif /
                   TRandom / TPrefetch / TPush / TPop / TAlloc / TFree /
                   MGather / MScatter (共 14 条)
```

---

## 2. 原生实现详细差异

### 2.1 TLoad — 数据搬运

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| L2 cache 控制 | `copy_gm_to_ubuf_align_v2` 有 **10 参数**，含 `l2 cache ctl` | **9 参数**，去掉 l2 cache 控制位 |
| DN→NZ cube 加载 | 支持 | `static_assert(false)` **编译期拒绝** |
| FP4 类型 | `float4_e1m2x2_t` / `float4_e2m1x2_t` 全面支持，stride 减半 | **不支持** |
| Loop 模式管理 | 条件设置 + 使用后重置 | 始终设置，**不重置** |
| Pad 值设置 | `set_mov_pad_val()` (vec) + 守卫条件 (cube) | `set_pad_val_outtoub()` (vec) + 无条件设置 (cube) |
| Pad 清理 | 使用后显式清零 | **不复位** |
| 类型分发 | `if constexpr` on `sizeof(DType)` | `LoadTypeBySize_t` compile-time trait |
| 文件大小 | 1306 行 | 615 行 |

### 2.2 TStore — 数据存储

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| Nd2Nd Acc 存储 | `copy_cbuf_to_gm(..., 0, 0, 0, ..., srcInitial)` | `copy_cbuf_to_gm(..., 0, 0, ..., srcInitial)` (少一个 0) |
| NZ 分支 Acc 存储 | `copy_cbuf_to_gm(..., 0, 0, 0)` | `copy_cbuf_to_gm(..., 0, 0)` |
| Loop 模式管理 | 条件设置 + 后复位 | 始终设置 + **不复位** |

### 2.3 TMatmul — 矩阵乘法

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| 累加类型 | `float` 或 `int32_t` | `half` 或 `int32_t` |
| A/B 输入类型 | half / bf16 / float / fp8e4m3 / fp8e5m2 / hifloat8 / int8_t | 仅 `half` 和 `int8_t` |
| MXFP8/FP4 | 完整 `mad_mx()` 支持 | **stub 禁止** `static_assert` |
| TF32 模式 | 不涉及 | **有 `TSETTF32MODE`**（硬件不支持，始终关闭） |
| `mad()` 签名 | 显式 `gemvCtrl` 模板参数 | 硬编码 `gemvCtrl=false` |
| GEMV 控制 | 作为模板参数传递 | 隐式 |
| 分形布局检查 | 统一严格 | 分 PTO_NPU_ARCH_KIRIN9030 (严格) / KIRINX90 (宽松) |
| 文件大小 | 355 行 | 230 行 |

### 2.4 TMov — 数据搬移

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| Acc→Ub 双目标 | 支持 `dualDstCtl` + `subBlockId` | 不支持，硬编码 `0` |
| Acc→Cb 签名 | `copy_matrix_cc_to_cbuf(..., 0, 0, 0, QuantPre, ...)` | `copy_matrix_cc_to_cbuf(..., 0, 0, QuantPre, ...)` (少一个 0) |
| 左/右 Tile 类型 | 11 种 (含 fp8/fp4/bf16) | 仅 `half` 和 `int8_t` |
| Fb 缓冲区容量 | 4KB 硬编码 `4096` | 7KB `PTO_FBUF_SIZE_BYTES` |
| Bias 类型转换 | `half→float`, `bf16→float` | 仅同类型拷贝 |
| ScaleLeft/Right | 完整实现 `TExtractToAmx`/`TExtractToBmx` | **stub 禁止** |
| ND→ZZ 布局搬移 | **新增** `TMovNdTo2Zz` | **无此能力** |
| ConvTile 搬移 | **新增** `TMOV_CONVTILE_IMPL` | **无此能力** |
| STPhase 控制 | 传递到所有 overload | 不支持 |
| 文件大小 | 722 行 | 491 行 |

### 2.5 TExtract — 数据抽取

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| FP4 通路 | `isFp4Type` 模板参数 + `load_cbuf_to_ca_s4` / `load_cbuf_to_cb_s4` | **无** |
| ScaleLeft/Right | `TExtractToAmx` / `TExtractToBmx` (完整 107 行) | **stub 禁止** |
| B4 循环 | 有 (`SHIFT_M_STEP_B4` / `M_STEP_MIN_VAL_B4`) | **无** |
| 新增常量 | `KHALF` / `SHIFT_MX_COL` / `SHIFT_MX_ROW` / `CO_SIZE_SCALE` / `SCALE_CUBE_BLOCK_SIZE` | 未定义 |
| `c0Size` 计算 | 含 fp4 分支 | 不含 |
| 文件大小 | 1039 行 | 842 行 |

### 2.6 TGather — 收集

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| FP8 收集 | `TGather_fp8_e4m3` / `TGather_fp8_e5m2` 专用内核 | **无** |
| 条件收集 | 8 变体 (float/half/b32/b16 × GT/EQ) 含 SPR 散列 | **无** |
| BF16 支持 | 2 参数 overload 支持 | **无** |
| `CEIL()` 宏 | 无 (使用共用 `CeilDivision`) | 定义了局部宏 |
| 批大小常数 | 硬编码 `256` | `CCE_VL` |
| 文件大小 | 810 行 | 261 行 |

### 2.7 TQuant — 量化

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| MXFP8 量化 | 完整流水线: AbsReduceMax (3 变体) + 指数提取 + 量化值计算 + 2D 行步进 | **stub 禁止** |
| MXFP4 E2M1 量化 | 完整流水线: 含 E2M1 编解码 + PackE2M1SignedCodeBytes 等 | **stub 禁止** |
| 零填充工具 | `ZeroPadSourceTile` (VL 对齐 / 不对齐) | **无** |
| `FlatTile1D` | 有 (通过 TReshape 展平) | 有 (include a5 的) |
| INT8_SYM/ASYM | 有 | 有 (**代码完全一致**) |
| 文件大小 | 1424 行 | 134 行 |

### 2.8 TCvt — 类型转换

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| 文件大小 | 3105 行 | 2121 行 |
| 转换引擎 | 基于寄存器 ("无需 UB 临时缓冲区") | 基于 UB 临时缓冲区 |
| 基础类型 | int8/16/32/64, uint8/16/32/64, half, float | 完全相同 |
| 高级类型 | bf16 / fp8e4m3 / fp8e5m2 / hifloat8 / fp8e8m0 / fp4e1m2x2 / fp4e2m1x2 | **全部不支持** |
| 饱和位 | `SAT_MODE_BIT_60/59/48` | 完全相同 |
| 架构守卫 | 无 | 无 |

### 2.9 TSync — 同步

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| 跨核同步 | `wait_intra_block` / `set_intra_block` 支持 | **不支持** |
| 允许同步管道 | 仅 `PIPE_MTE2` / `PIPE_MTE3` / `PIPE_ALL` | 额外允许 `PIPE_M` / `PIPE_MTE1` / `PIPE_FIX` |
| 额外断言 | `IsCrossCore` 检查 + `AutoToken` 检查 | 无 |
| `Event::RecordEvent` | 断言 `!IsCrossCore` 后调用 Init | 无条件调用 Init |

---

## 3. 基础设施差异

### 3.1 common.hpp

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| `GetByteSize` FP4 处理 | `(value+1)>>1` | **无** (`sizeof(T)*value` 仅) |
| `RegTensor::Print()` | 声明 AICORE Print 方法 | **已删除** |
| `CreatePredicate` | `CreatePredicateImpl` + 封装 | 直接实现 (无封装层) |
| `GetCastPreQuantMode` | `F322F16` / `F322BF16` 支持 float→half/bf16 | `static_assert(SrcType == DstType)` **禁止转型** |
| `GetScalarPreQuantMode` | 源 `float` → dst 含 hifloat8 / bf16 / fp8e4m3 | 源 `half` → dst 新增 `int16_t` (DEQS16) |
| `GetVectorPreQuantMode` | 源 `float` 含 VQF322HIF8_PRE 等 | 源 `half` → dst 含 VQF162S16_PRE |
| `CheckTMovAccValid` | 累加器源类型 `float` / `int32_t` | 累加器源类型 `half` / `int32_t` |
| 版权年份 | 2025 | 2026 |

### 3.2 datatype.hpp

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| 文件大小 | 148 行 | 59 行 |
| `TypeGet` 特化 | 10 基础 + 7 exotic (`__DAV_VEC__` 保护) | 仅 10 基础 |
| `TypeGet<vector_X>` 恒等 | 有 (10 个 + 3 guarded) | **全部删除** |
| 守卫 | `__DAV_VEC__` 保护高级类型 | 无 |

### 3.3 utils.hpp

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| 实现方式 | 91 行: VECTOR_REG_WIDTH, DistVST 枚举, GetDistVst, PSetWithType, GetScaleAddr | **10 行**: 仅 `#include <pto/npu/a5/utils.hpp>` |

---

## 4. 包装层适配

### TInsert.hpp

| 适配项 | a5 | kirin9030 |
|--------|----|-----------|
| 实现方式 | 637 行完整算法 | **19 行包装器**: 定义 `COPY_CC_TO_CUBF` 宏 (将第7参改为常量 0) 后 include a5 版 |
| 包含守卫 | `TINSERT_HPP` | `TINSERT_HPP_KIRIN9030` |

---

## 5. header.hpp 类型别名适配

Kirin9030 通过宏将高级类型别名为基础整数类型，避免编译错误：

```cpp
#define bfloat16_t      half
#define float8_e4m3_t   int8_t
#define float8_e5m2_t   int8_t
#define hifloat8_t      int8_t
#define float8_e8m0_t   int8_t
#define float4_e2m1x2_t int64_t
#define float4_e1m2x2_t int64_t
// ... 文件末尾 #undef 全部
```

---

## 6. 暂缓接入的 a5 指令 (14 条)

| 指令 | 原因 | header.hpp 状态 |
|------|------|-----------------|
| TFMod / TFModS | 已注释 | `// #include "pto/npu/a5/TFModS.hpp"` |
| TRem / TRemS | 已注释 | `// #include "pto/npu/a5/TRem.hpp"` |
| TRowExpandExpdif | 已注释 | `// #include "pto/npu/a5/TRowExpandExpdif.hpp"` |
| TColExpandExpdif | 已注释 | `// #include "pto/npu/a5/TColExpandExpdif.hpp"` |
| TRandom | 待评估 | `// TRandom to be evaluated` |
| TPrefetch | 待评估 | `// TPrefetch to be evaluated` |
| TPush / TPop / TAlloc / TFree | 待评估 | `// TPush/TPop/TAlloc/TFree to be evaluated` |
| MGather / MScatter | 待评估 | `// MGather/MScatter to be evaluated` |

---

## 7. 适配工作量统计

| 工作类别 | 涉及文件 | 行数 (k9030) | 行数 (a5) | 变更性质 |
|---------|----------|:-----------:|:--------:|---------|
| **TLoad** 硬件差异适配 | `kirin9030/TLoad.hpp` | 615 | 1306 | L2 cache、DN2NZ、FP4 裁剪、Loop/Pad 管理 |
| **TStore** 硬件差异适配 | `kirin9030/TStore.hpp` | 530 | ~1200 | 同上模式 |
| **TMatmul** 类型切换 + MX stub | `kirin9030/TMatmul.hpp` | 230 | 355 | 累加器 `float→half`、FP8 裁剪、TF32 添加 |
| **TMov** 多维度适配 | `kirin9030/TMov.hpp` | 491 | 722 | 双目标、ScaleTile、ConvTile、STPhase 裁剪 |
| **TExtract** FP4 + Scale 裁剪 | `kirin9030/TExtract.hpp` | 842 | 1039 | FP4/ScaleTile/B4 循环 stub |
| **TGather** FP8 + 条件收集裁剪 | `kirin9030/TGather.hpp` | 261 | 810 | FP8/条件收集/bf16 裁剪 |
| **TQuant** MX 量化 stub | `kirin9030/TQuant.hpp` | 134 | 1424 | MXFP8/MXFP4 stub, INT8 不变 |
| **TCvt** 高级类型裁剪 | `kirin9030/TCvt.hpp` | 2121 | 3105 | bf16/fp8/fp4 等高级类型删除 |
| **TSync** 跨核同步裁剪 | `kirin9030/TSync.hpp` | 129 | 155 | 跨核通路删除, 管道约束放宽 |
| **TInsert** 薄包装 | `kirin9030/TInsert.hpp` | 19 | 637 | 宏定义后委托 a5 |
| **common.hpp** 类型 + 量化调整 | `kirin9030/common.hpp` | 160 | 193 | 累加器/量化源 `float→half` |
| **datatype.hpp** 精简 | `kirin9030/datatype.hpp` | 59 | 148 | 高级类型 + 恒等映射删除 |
| **utils.hpp** 委托 | `kirin9030/utils.hpp` | 10 | 91 | 直接 `#include <pto/npu/a5/utils.hpp>` |
| **复用 a5** (~80 指令) | `header.hpp` | 0 | 0 | 零成本 include |

### 总结

| 维度 | 数据 |
|------|:----:|
| **原生实现文件** | 10 个 (差异化指令) |
| **基础设施文件** | 3 个 (common/datatype/utils) |
| **复用 a5 指令** | ~80 条 |
| **暂缓指令** | 14 条 |
| **核心适配模式** | 累加器 `float→half`、高级类型裁剪、Intrinsic 签名差异、跨核同步裁剪、薄包装委托 |
| **代码量对比** | kirin9030 原生实现合计 ~5372 行 vs a5 对应合计 ~10039 行 (约 53% 精简) |
