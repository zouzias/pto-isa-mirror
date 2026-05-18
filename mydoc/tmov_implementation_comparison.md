# TMov 指令实现对比分析：A5 vs Kirin9030 vs KirinX90

## 背景

根据 readme.md 的描述，需要对 kirin9030、kirinX90 和 a5 的 TMov 指令实现进行对比分析，以了解三者的异同，为指令对齐工作提供参考。

---

## 一、总体结构对比

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|:--:|:---------:|:--------:|
| **文件大小** | 722 行 | 491 行 | 264 行 |
| **依赖模块** | TExtract.hpp, TPartAdd.hpp | TExtract.hpp, a5/TPartAdd.hpp | TExtract.hpp (自身实现) |
| **函数总数** | ~20 | ~15 | ~10 |
| **数据类型支持** | 10+ 类型 | 6 类型 | 6 类型 |
| **布局转换支持** | ND/NZ/DN/ZZ/MX | ND/NZ/DN | ND/NZ |

---

## 二、核心函数对比

### 2.1 TMovToBt (Bias Table 搬运)

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **数据类型** | int32_t→int32_t, float→float, half→float, bf16→float | int32_t→int32_t, half→half | int32_t→int32_t, half→half |
| **类型转换** | ✅ half→float (convControl=true) | ❌ 无转换 | ❌ 无转换 |
| **对齐要求** | col*sizeof(DstType) % 64 == 0 | col*sizeof(DstType) % 64 == 0 | col*sizeof(SrcType) % 64 == 0 |
| **Bias Table 大小** | 4 KiB | 4 KiB (PTO_BIAS_SIZE_BYTES) | 1 KiB (PTO_BIAS_SIZE_BYTES) |
| **burstLen 计算** | srcRow*srcCol*sizeof(SrcType) >> 5 (32B单位) | SrcTile::Numel*sizeof(SrcType) >> 5 | srcRow*srcCol*sizeof(SrcType) / 64 |
| **硬件指令** | copy_cbuf_to_bt | copy_cbuf_to_bt | copy_cbuf_to_bt |

**关键差异**：
1. **A5 支持类型转换**：half/bf16→float，kirin9030/kirinX90 不支持
2. **Bias Table 容量差异**：kirinX90 仅 1 KiB，a5/kirin9030 为 4 KiB

### 2.2 TMovToFb (Fixpipe Buffer 搬运)

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **数据类型** | 任意 (无类型检查) | 任意 (无类型检查) | **仅 uint64_t** |
| **对齐要求** | col*sizeof(DstType) % 128 == 0 | col*sizeof(DstType) % 128 == 0 | col*sizeof(SrcType) % 128 == 0 |
| **Fixpipe Buffer 大小** | 4 KiB | 7 KiB (PTO_FBUF_SIZE_BYTES) | 未显式限制 |
| **burstLen 计算** | srcRow*srcCol*sizeof(SrcType) >> 6 (64B单位) | SrcTile::Numel*sizeof(SrcType) / 128 | srcRow*srcCol*sizeof(SrcType) / 128 |
| **硬件指令** | copy_cbuf_to_fbuf | copy_cbuf_to_fbuf | copy_cbuf_to_fbuf |

**关键差异**：
1. **KirinX90 类型限制**：仅支持 uint64_t，其他架构无限制
2. **Fixpipe Buffer 容量**：kirin9030 最大 (7 KiB)

### 2.3 TMovCcToCb (Acc→Mat/L1 搬运)

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **支持路径** | NZ→NZ, NZ→ND, NZ→DN | NZ→NZ, NZ→ND, NZ→DN | 仅 NZ→NZ |
| **ChannelSplit** | ✅ (float + SFractalSize==512) | ✅ (float + SFractalSize==512) | ❌ 无 |
| **Loop3Para** | ✅ (NZ→ND/NZ→DN) | ✅ (NZ→ND/NZ→DN) | ❌ 无 |
| **ChannelPara** | ✅ (NZ→DN) | ✅ (NZ→DN) | ❌ 无 |
| **dstStride 计算** | GetTmovAccDstStride() (复杂逻辑) | GetTmovAccDstStride() (相同逻辑) | DstTile::Rows (简化) |
| **硬件指令参数** | 26 参数 | 22 参数 | 8 参数 |

**关键差异**：
1. **路径支持**：A5/kirin9030 支持 NZ→ND/DN 转换，kirinX90 仅 NZ→NZ
2. **硬件指令参数数量**：A5 最复杂 (26 参数)，kirinX90 最简化 (8 参数)

### 2.4 TMovCcToUb (Acc→Vec/UB 搬运)

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **是否支持** | ✅ | ✅ | ❌ **不支持** |
| **支持路径** | NZ→NZ, NZ→ND, NZ→DN | NZ→NZ, NZ→ND, NZ→DN | — |
| **AccToVecMode** | ✅ SingleModeVec0/Vec1, DualModeSplitM/SplitN | ✅ SingleModeVec0/Vec1, DualModeSplitM/SplitN | — |
| **STPhase** | ✅ 支持 | ✅ 支持 | — |
| **硬件指令** | copy_matrix_cc_to_ub | copy_matrix_cc_to_ub | — |

**关键差异**：
- **KirinX90 不支持 Acc→Vec 路径**，这是重大功能缺失

### 2.5 TMovToLeft/TMovToRight (Mat→L0A/L0B 搬运)

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **数据类型检查** | CommonCheck() (10+ 类型) | CommonCheck() (half/int8) | 无显式检查 |
| **FP4 支持** | ✅ float4_e2m1x2/float4_e1m2x2 | ❌ 无 | ❌ 无 |
| **FP8 支持** | ✅ float8_e4m3/float8_e5m2/hifloat8 | ❌ 无 | ❌ 无 |
| **调用 TExtract** | TExtractToA/B (带 isFp4Type 参数) | TExtractToA/B (无 FP4 参数) | TExtractToA/B (无 FP4 参数) |
| **Compact 模式** | ✅ Normal/Null/RowPlusOne/RowAlignedPadding | ✅ Normal/Null | ✅ Normal/Null |

**关键差异**：
1. **A5 独有 FP4/FP8 支持**：通过 isFp4Type 模板参数实现
2. **A5 使用 load_cbuf_to_ca_s4**：FP4 专用硬件指令

### 2.6 TMovToVec (Vec→Vec 搬运)

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **ND→Vec** | TMovVecToVec (vlds/vsts) | TMovVecToVec (vlds/vsts) | TMovToVecImpl (pto_copy_ubuf_to_ubuf) |
| **ND→NZ 转换** | TMovToVecNd2Nz (vsstb) | TMovToVecNd2Nz (vsstb) | ❌ **不支持** |
| **实现方式** | 向量指令 (vlds/vsts/vsstb) | 向量指令 (vlds/vsts/vsstb) | DMA (pto_copy_ubuf_to_ubuf) |

**关键差异**：
1. **KirinX90 使用 DMA而非向量指令**：性能可能较低
2. **KirinX90 不支持 ND→NZ 转换**：关键功能缺失

---

## 三、数据类型支持对比

### 3.1 TMov 支持的数据类型

| 数据类型 | A5 | Kirin9030 | KirinX90 | 说明 |
|----------|:--:|:---------:|:--------:|------|
| **int8_t** | ✅ | ✅ | ✅ | 所有架构支持 |
| **uint8_t** | ✅ | ✅ | ✅ | 所有架构支持 |
| **int16_t** | ✅ | ✅ | ✅ | 所有架构支持 |
| **uint16_t** | ✅ | ✅ | ✅ | 所有架构支持 |
| **int32_t** | ✅ | ✅ | ✅ | 所有架构支持 |
| **half** | ✅ | ✅ | ✅ | 所有架构支持 |
| **bfloat16_t** | ✅ | ❌ | ❌ | 仅 A5 支持 |
| **float** | ✅ | ✅ | ✅ | 所有架构支持 |
| **int64_t** | ❌ | ❌ | ❌ | 均不支持 |
| **float8_e4m3_t** | ✅ | ❌ | ❌ | 仅 A5 支持 |
| **float8_e5m2_t** | ✅ | ❌ | ❌ | 仅 A5 支持 |
| **hifloat8_t** | ✅ | ❌ | ❌ | 仅 A5 支持 |
| **float8_e8m0_t** | ✅ | ❌ | ❌ | 仅 A5 支持 (MX) |
| **float4_e2m1x2_t** | ✅ | ❌ | ❌ | 仅 A5 支持 |
| **float4_e1m2x2_t** | ✅ | ❌ | ❌ | 仅 A5 支持 |

### 3.2 TExtract 支持的数据类型

| 数据类型 | A5 | Kirin9030 | KirinX90 |
|----------|:--:|:---------:|:--------:|
| **is_textract_supported_type** | int8, fp8系列, half, bf16, float, fp4系列 | int8, uint8, half, int16, uint16, int32 | int8, uint8, half, int16, uint16, int32 |

---

## 四、布局转换支持对比

### 4.1 支持的布局转换路径

| 转换路径 | A5 TMov | Kirin9030 TMov | KirinX90 TMov | 说明 |
|----------|:-------:|:--------------:|:-------------:|------|
| **Mat→Bias** | ✅ | ✅ | ✅ | L1→BT |
| **Mat→Scaling** | ✅ | ✅ | ✅ | L1→FB |
| **Mat→Left** | ✅ | ✅ | ✅ | L1→L0A |
| **Mat→Right** | ✅ | ✅ | ✅ | L1→L0B |
| **Mat→ScaleLeft** | ✅ | ❌ | ❌ | MX 左缩放 |
| **Mat→ScaleRight** | ✅ | ❌ | ❌ | MX 右缩放 |
| **Acc→Vec** | ✅ | ✅ | ❌ | L0C→UB |
| **Acc→Mat** | ✅ | ✅ | ✅ | L0C→L1 (仅 NZ→NZ) |
| **Vec→Vec (ND)** | ✅ | ✅ | ✅ | UB→UB |
| **Vec→Vec (NZ)** | ✅ | ✅ | ❌ | UB→UB NZ |
| **Vec→Vec (ND→NZ)** | ✅ | ✅ | ❌ | UB ND→NZ 转换 |
| **Vec→Mat** | ✅ | ✅ | ✅ | UB→L1 |
| **ND→ZZ** | ✅ | ❌ | ❌ | A5 独有 |
| **ConvTile→Right** | ✅ | ✅ | ✅ | FRACTAL_Z→L0B |

### 4.2 NZ→ND/DN 转换参数

| 参数 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **SetLoop3Para** | ✅ | ✅ | ❌ |
| **SetChannelPara** | ✅ (NZ→DN) | ✅ (NZ→DN) | ❌ |
| **channelSplitEnable** | ✅ (float+SFractalSize==512) | ✅ (float+SFractalSize==512) | ❌ |

---

## 五、量化支持对比

### 5.1 QuantPre 模式支持

| 功能 | A5 | Kirin9030 | KirinX90 |
|------|:--:|:---------:|:--------:|
| **标量量化 (preQuantScalar)** | ✅ TMovCcToUb/Cb | ✅ TMovCcToUb/Cb | ✅ 仅 TMovCcToCb |
| **向量量化 (FpTile)** | ✅ TMovCcToUb/Cb + SetFPC | ✅ TMovCcToUb/Cb + SetFPC | ✅ 仅 TMovCcToCb + SetFPC |
| **ReluPreMode** | ✅ NoRelu/NormalRelu | ✅ NoRelu/NormalRelu | ✅ NoRelu/NormalRelu |
| **AccToVecMode** | ✅ SingleModeVec0/Vec1, DualModeSplitM/SplitN | ✅ SingleModeVec0/Vec1, DualModeSplitM/SplitN | ❌ 无 (不支持 Acc→Vec) |
| **STPhase** | ✅ Unspecified 等 | ✅ Unspecified 等 | ❌ 无 |

### 5.2 SetFPC 函数对比

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **参数** | fp tile | fp tile + indexCol | fp tile + indexCol |
| **地址计算** | (addr >> 7) << 8 | (addr >> 7) << 8 | (addr >> 7) << 8 |
| **用途** | TMovCcToUb/Cb | TMovCcToUb/Cb | 仅 TMovCcToCb |

---

## 六、TExtract 实现对比

### 6.1 TExtractToA/B 核心函数

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **FP4 支持** | ✅ load_cbuf_to_ca_s4/cb_s4 | ❌ 无 | ❌ 无 |
| **FP8 特殊处理** | ✅ M_STEP_MIN_VAL_B8=2, 分块循环 | ✅ M_STEP_MIN_VAL_B8=2, 分块循环 | ✅ M_STEP_MIN_VAL_B8=2, 分块循环 |
| **FP4 特殊处理** | ✅ M_STEP_MIN_VAL_B4=4, 分块循环 | ❌ 无 | ❌ 无 |
| **Transpose 支持** | ✅ 模板参数 | ✅ 模板参数 | ✅ 模板参数 |
| **Compact 模式** | ✅ | ✅ | ✅ |
| **MX 支持** | ✅ TExtractToAmx/Bmx | ❌ static_assert 失败 | ❌ static_assert 失败 |

### 6.2 TExtractAccToVec (Acc→Vec)

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **是否支持** | ✅ | ✅ | ❌ **不支持** |
| **硬件指令** | copy_matrix_cc_to_ub | copy_matrix_cc_to_ub | — |
| **参数** | 26 参数 | 22 参数 | — |
| **AccToVecMode** | ✅ | ✅ | — |
| **dualDstCtl** | ✅ GetDualDstCtl() | ✅ GetDualDstCtl() | — |

### 6.3 TExtractVecToVecND (Vec→Vec ND)

| 特性 | A5 | Kirin9030 | KirinX90 |
|------|----|-----------|----------|
| **实现路径数** | 4 (Impl/AlignedImpl/VectorImpl/ScalarImpl) | 4 (同上) | ❌ **不支持** |
| **FP4 特殊处理** | ✅ BLOCK_BYTE_SIZE 对齐 | ❌ 无 | — |
| **Dispatch 逻辑** | ✅ kStridesAligned/kValidColAligned | ✅ kStridesAligned/kValidColAligned | — |

---

## 七、硬件缓冲区容量对比

| 缓冲区 | A5 | Kirin9030 | KirinX90 |
|--------|:--:|:---------:|:--------:|
| **Bias Table (BT)** | 4 KiB | 4 KiB | **1 KiB** |
| **Fixpipe Buffer (FB)** | 4 KiB | **7 KiB** | 未显式限制 |
| **ScaleLeft/Right Buffer** | 4 KiB (各) | **0 (不支持)** | **0 (不支持)** |

---

## 八、关键缺失功能分析

### 8.1 KirinX90 vs Kirin9030 缺失功能

| 缺失功能 | 影响 |
|----------|------|
| **Acc→Vec (TMovCcToUb)** | 无法将累加器结果写入向量缓冲区，限制了量化后的数据流向 |
| **Vec→Vec ND→NZ 转换** | 无法在 UB 内将 ND 格式转换为 NZ 格式，影响 Cube 输入准备 |
| **NZ→ND/DN 转换参数** | 缺少 SetLoop3Para/SetChannelPara，无法进行 NZ 格式转换 |
| **Vec→Vec NZ 操作** | 无法处理 NZ 格式的向量数据 |
| **TExtractAccToVec** | 无法从 Acc 提取子块到 Vec |
| **TExtractVecToVecND/NZ** | 无法在 Vec 间进行子块提取 |
| **ChannelSplit** | 无法使用 float channel split 模式 |

### 8.2 Kirin9030/KirinX90 vs A5 缺失功能

| 缺失功能 | 影响 |
|----------|------|
| **bfloat16_t** | 无法使用 bf16 数据类型 |
| **FP8 系列 (e4m3/e5m2/hifloat8/e8m0)** | 无法使用低精度浮点类型，影响 AI 推理性能 |
| **FP4 系列 (e2m1x2/e1m2x2)** | 无法使用最低精度浮点类型 |
| **MX Scale (ScaleLeft/ScaleRight)** | 无法使用 MX 格式的缩放因子 |
| **ND→ZZ 转换** | 无法将 ND 格式转换为 ZZ 格式 |
| **Acc→Vec 类型转换 (half/bf16→float)** | Bias Table 无法进行类型转换 |

---

## 九、copy_matrix_cc_to_cbuf/ub 参数对比

### 9.1 A5 copy_matrix_cc_to_cbuf (26 参数)

```
copy_matrix_cc_to_cbuf(
    dstAddr, srcData,
    sid, validCol, validRow, dstStride, srcStride,
    0, 0, 0, QuantPre, reluMode,
    channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0,
    false, false, false, false, false, enableNz2Dn
)
```

### 9.2 Kirin9030 copy_matrix_cc_to_cbuf (22 参数)

```
copy_matrix_cc_to_cbuf(
    dstAddr, srcData,
    sid, validCol, validRow, dstStride, srcStride,
    0, 0, QuantPre, reluMode,
    channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0,
    false, false, false, false, false, enableNz2Dn
)
```

### 9.3 KirinX90 copy_matrix_cc_to_cbuf (8 参数)

```
copy_matrix_cc_to_cbuf(
    dstAddr, srcAddr,
    sid, validCol, SrcTile::Rows, dstStride, srcStride,
    0, QuantPre, reluMode, false, false
)
```

**参数差异总结**：
- KirinX90 硬件指令参数大幅简化，缺失 channelSplit、enableNz2Nd、enableNz2Dn 等关键控制参数
- A5 有额外的 sid 后参数 (3 个 0)，支持更复杂的 DMA 控制

---

## 十、总结

### 10.1 实现成熟度对比

| 架构 | 代码行数 | 功能完整度 | 数据类型 | 布局转换 | 量化支持 |
|------|:--------:|:---------:|:--------:|:--------:|:--------:|
| **A5** | 722 | ★★★★★ | 15+ | 全路径 | 全模式 |
| **Kirin9030** | 491 | ★★★★☆ | 6 | NZ→ND/DN | 标量+向量 |
| **KirinX90** | 264 | ★★☆☆☆ | 6 | 仅 NZ→NZ | 仅 Acc→Mat |

### 10.2 KirinX90 对齐 Kirin9030 的关键任务

1. **补齐 Acc→Vec 路径**：实现 TMovCcToUb 和 TExtractAccToVec
2. **补齐 Vec→Vec ND→NZ 转换**：实现 TMovToVecNd2Nz
3. **补齐 NZ→ND/DN 参数**：添加 SetLoop3Para/SetChannelPara 逻辑
4. **补齐 TExtractVecToVecND/NZ**：实现 Vec→Vec 子块提取
5. **扩展 copy_matrix_cc_to_cbuf 参数**：从 8 参数扩展到 22 参数
6. **补齐 ChannelSplit 支持**：添加 float+SFractalSize==512 分支

### 10.3 Kirin9030 对齐 A5 的关键任务

1. **补齐低精度类型**：FP8/FP4/hifloat8/bfloat16/e8m0
2. **补齐 MX Scale 支持**：ScaleLeft/ScaleRight + load_cbuf_to_ca_mx/cb_mx
3. **补齐 ND→ZZ 转换**：TMovNdTo2Zz
4. **补齐 Bias Table 类型转换**：half/bf16→float
5. **补齐 FP4 硬件指令**：load_cbuf_to_ca_s4/cb_s4
6. **扩展 Bias Table/Fixpipe Buffer**：从 4 KiB/7 KiB 到统一 4 KiB

### 10.4 架构依赖关系

```
A5 (最完整)
  │
  ├── 复用路径 ──→ Kirin9030 (复用 a5/TPartAdd.hpp, 自实现 TExtract/TMov)
  │                    │
  │                    ├── 复用路径 ──→ KirinX90 (复用 kirin9030 部分，但大量功能缺失)
  │
  └── 不复用 ──────→ KirinX90 (TExtract.hpp 为 a2a3 版本，非 kirin9030 版本)
```

**注意**：KirinX90 的 TExtract.hpp 来自 a2a3 而非 kirin9030，这可能导致功能不一致。