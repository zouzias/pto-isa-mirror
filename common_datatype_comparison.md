# common.hpp 与 datatype.hpp 三架构对比分析

---

## 一、common.hpp 对比

### 1.0 文件基本信息

| 维度 | a5 | kirin9030 | kirinX90 |
|------|:--:|:---------:|:---------:|
| **行数** | 193 | 160 | 54 |
| **版权年份** | 2025 | 2026 | 2026 |
| **关系** | — | 独立实现 | **包装 k9030**（`#include "pto/npu/kirin9030/common.hpp"` 后追加内容） |
| **命名空间** | `pto` | `pto` | `pto` |

### 1.1 GetByteSize

| 版本 | 实现 | 差异 |
|------|------|------|
| **a5** | 含 FP4 特殊分支：`float4_e1m2x2_t` / `float4_e2m1x2_t` 返回 `(value+1)>>1`，其余 `sizeof(T)*value` | **完整版** |
| **k9030** | 仅 `sizeof(T) * value` | **无 FP4 处理** |
| **kX90** | 继承 k9030 | 同 k9030 |

### 1.2 SupportBytes

```cpp
template <typename T, int U, int... Args>
AICORE constexpr bool SupportBytes() { ... }
```

**三者完全一致**。无差异。

### 1.3 寄存器类型别名

```cpp
using MaskReg = vector_bool;
using UnalignReg = vector_align;
using AddrReg = vector_address;
```

**三者完全一致**。

### 1.4 CreatePredicate

| 版本 | 实现 | 差异 |
|------|------|------|
| **a5** | `CreatePredicateImpl<T>()`（实现体）→ `CreatePredicate<T>()`（封装）**两层** | `plt_b8/b16/b32` + `POST_UPDATE` |
| **k9030** | 直接 `CreatePredicate<T>()` **单层** | 逻辑相同，去掉了包装层 |
| **kX90** | 继承 k9030 | 同 k9030 |

### 1.5 RegTensor

| 成员顺序 | a5 | k9030 | kX90 |
|---------|:--:|:-----:|:----:|
| 构造函数 | 第1 | 第3 | 继承 |
| RegType typedef | 第2 | **第1** | 继承 |
| reg 成员变量 | 第3 | **第2** | 继承 |
| `operator RegType&()` | 第4 | 第4 | 继承 |
| `Print()` 方法 | **有** | **无** | 继承（无） |

**关键差异**：
- `RegTensor` 的成员**声明顺序不同**：a5 是"构造→类型→变量→操作符"，k9030 是"类型→变量→构造→操作符"
- **a5 特有** `AICORE void Print() const;` 声明（k9030/kX90 均无）

### 1.6 GetCastPreQuantMode

| 版本 | 支持 src→dst 转换 | 返回值 |
|------|-------------------|--------|
| **a5** | 任意 src → `half` 返回 `F322F16`；任意 src → `bfloat16_t` 返回 `F322BF16` | 允许 float→half/bf16 的无参转型 |
| **k9030** | **`static_assert(SrcType == DstType)`** + `return NoQuant` | 禁止任何转型，必须同类型 |
| **kX90** | 继承 k9030 | 同 k9030 |

### 1.7 GetScalarPreQuantMode

| 源类型 | 目标类型 | a5 | k9030 | kX90 |
|--------|---------|:--:|:-----:|:----:|
| **float** | int8_t / uint8_t | QF322B8_PRE | — | — |
| | hifloat8_t | QF322HIF8_PRE | — | — |
| | half | QF322F16_PRE | — | — |
| | bfloat16_t | QF322BF16_PRE | — | — |
| | float8_e4m3_t | QF322B8_PRE | — | — |
| **int32_t** | half | DEQF16 | DEQF16 | 继承 |
| | int16_t | — | **DEQS16** | 继承 |
| | int8_t / uint8_t | REQ8 | REQ8 | 继承 |
| | bfloat16_t | QS322BF16_PRE | — | — |
| **half** | int16_t | — | **QF162S16_PRE** | 继承 |
| | int8_t / uint8_t | — | **QF162B8_PRE** | 继承 |

**关键差异**：
- a5 **以 `float` 为量化源类型**，支持 exotic 目标：hifloat8_t、bfloat16_t、float8_e4m3_t
- k9030 **以 `half` 替代 `float` 为量化源类型**，无 exotic 目标，但**新增了 `int16_t` 支持**
- kX90 继承 k9030 行为

### 1.8 GetVectorPreQuantMode

模式与 `GetScalarPreQuantMode` 完全相同，仅枚举名前缀 `V`：

| 差异 | a5 | k9030 | kX90 |
|------|:--:|:-----:|:----:|
| float→half | VQF322F16_PRE | — | — |
| float→bf16 | VQF322BF16_PRE | — | — |
| float→hifloat8 | VQF322HIF8_PRE | — | — |
| float→fp8e4m3 | VQF322B8_PRE | — | — |
| int32→half | VDEQF16 | VDEQF16 | 继承 |
| int32→int16 | — | **VDEQS16** | 继承 |
| half→int16 | — | **VQF162S16_PRE** | 继承 |
| half→int8/uint8 | — | **VQF162B8_PRE** | 继承 |

### 1.9 CheckTMovAccValid

| 维度 | a5 | k9030 | kX90 |
|------|:--:|:-----:|:----:|
| **SrcType 支持** | `float` 或 `int32_t` | `half` 或 `int32_t` | 继承 k9030 |
| **Quant 分支 src=float** | dst: int8/uint8/hifloat8/half/bf16/fp8e4m3/float | — | — |
| **Quant 分支 src=half** | — | dst: half/int8/uint8/**int16** | 继承 |
| **Quant 分支 src=int32** | dst: int8/uint8/half/bf16 | dst: half/int8/uint8/**int16** | 继承 |
| **非 Quant 分支 src=float** | dst: half/bf16/float | — | — |
| **非 Quant 分支 src=half** | — | dst == src (half/int32) | 继承 |
| **非 Quant 分支 src=int32** | dst == int32 | dst == src (half/int32) | 继承 |
| **Dst 布局限制** | nz2nz / nz2nd / nz2dn | nz2nz / nz2nd / nz2dn | 继承 |

**a5 额外还包含 DstTileData 的格式检查**（第 174-177 行），与 k9030 的 141-144 行完全一致。

### 1.10 GetDualDstCtl

**三者完全一致**。模板参数和逻辑相同。

### 1.11 kirinX90 新增：CheckTMovAccToMat

`kirinX90/common.hpp` 在继承 k9030 后新增了单独的校验函数：

```cpp
template <typename DstTileData, typename SrcTileData, typename DstType, typename SrcType, bool isCastQuant>
PTO_INTERNAL void CheckTMovAccToMat()
```

与 k9030 的 `CheckTMovAccValid` 相比的区别：

| 检查项 | CheckTMovAccValid (k9030) | CheckTMovAccToMat (kX90 新增) |
|--------|:-------------------------:|:----------------------------:|
| Src Loc | 仅 Acc | 仅 Acc |
| **Dst Loc** | Mat 或 Vec | **仅 Mat**（新增限制） |
| **SFractalSize** | 无限制 | **必须 == 512**（新增限制） |
| **Col 对齐** | 无 | `Cols * sizeof(DstType) % 32 == 0`（新增限制） |
| Dst 分形格式 | 宽松 | **必须 RowMajor**（新增限制） |

---

## 二、datatype.hpp 对比

### 2.0 文件基本信息

| 维度 | a5 | kirin9030 | kirinX90 |
|------|:--:|:---------:|:---------:|
| **行数** | 148 | 59 | 25 |
| **版权年份** | 2025 | 2026 | 2026 |
| **关系** | — | 独立实现 | **包装 k9030**（`#include` 后追加） |
| **包含守卫** | `PTO__DATATYPE_IMPL_H` | `PTO__DATATYPE_IMPL_H` | **`PTO_DATATYPE_HPP_KIRINX90`** |

### 2.2 TypeGet 特化 — 基础类型

三者都包含的 10 个 C++ 类型 → 向量类型映射：

| C++ 类型 | 向量类型 | a5 | k9030 | kX90 |
|----------|---------|:--:|:-----:|:----:|
| int8_t | vector_s8 | ✓ | ✓ | ✓ (继承) |
| uint8_t | vector_u8 | ✓ | ✓ | ✓ |
| int16_t | vector_s16 | ✓ | ✓ | ✓ |
| uint16_t | vector_u16 | ✓ | ✓ | ✓ |
| half | vector_f16 | ✓ | ✓ | ✓ |
| int32_t | vector_s32 | ✓ | ✓ | ✓ |
| uint32_t | vector_u32 | ✓ | ✓ | ✓ |
| float | vector_f32 | ✓ | ✓ | ✓ |
| int64_t | vector_s64 | ✓ | ✓ | ✓ |
| uint64_t | vector_u64 | ✓ | ✓ | ✓ |

### 2.3 TypeGet 特化 — 高级/Exotic 类型

| C++ 类型 | 向量类型 | a5 | k9030 | kX90 |
|----------|---------|:--:|:-----:|:----:|
| bfloat16_t | vector_bf16 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |
| float8_e5m2_t | vector_f8e5m2 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |
| float8_e4m3_t | vector_f8e4m3 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |
| hifloat8_t | vector_hif8 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |
| float8_e8m0_t | vector_f8e8m0 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |
| float4_e1m2x2_t | vector_f4e1m2x2 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |
| float4_e2m1x2_t | vector_f4e2m1x2 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |

### 2.4 TypeGet 恒等映射 — 向量类型 → 自身

| 向量类型 | a5 | k9030 | kX90 |
|----------|:--:|:-----:|:----:|
| vector_u64 / vector_s64 | ✓ | ✗ | ✗ |
| vector_u32 / vector_s32 | ✓ | ✗ | ✗ |
| vector_f32 | ✓ | ✗ | ✗ |
| vector_u16 / vector_s16 | ✓ | ✗ | ✗ |
| vector_f16 | ✓ | ✗ | ✗ |
| vector_u8 / vector_s8 | ✓ | ✗ | ✗ |
| vector_bf16 | ✓ (`__DAV_VEC__`) | ✗ | **✓** (`__DAV_VEC__`) |
| vector_hif8 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |
| vector_f8e8m0 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |
| vector_f4e1m2x2 | ✓ (`__DAV_VEC__`) | ✗ | ✗ |

**注意**：a5 也**没有** `vector_f8e5m2` 和 `vector_f8e4m3` 的恒等映射（只有标量→向量的映射）。

### 2.5 特有差异化

| 特性 | a5 | k9030 | kX90 |
|------|:--:|:-----:|:----:|
| 守卫类型 | `__DAV_VEC__` 保护 exotic 类型 | 无守卫 | `__DAV_VEC__` 保护 vector_bf16 恒等 |
| 标量→向量 exotic 映射 | **有**（7 个） | **无** | **无** |
| 向量恒等映射数量 | **14 个**（10 基础 + 4 exotic） | **0** | **1 个**（仅 vector_bf16） |

---

## 三、三层继承/包装关系

```
                    a5 (full implementation, 193 + 148 = 341 lines)
                     |
                     | 独立实现，差异较大
                     v
              kirin9030 (独立实现, 160 + 59 = 219 lines)
                     |
                     |  #include 继承 + 追加
                     v
              kirinX90  common.hpp: #include k9030 + 追加 CheckTMovAccToMat (54 lines)
                       datatype.hpp: #include k9030 + 追加 vector_bf16 恒等 (25 lines)
```

### 继承关系

| 函数/类型 | a5 → k9030 变化 | k9030 → kX90 变化 |
|-----------|:---------------:|:-----------------:|
| GetByteSize | 去掉 FP4 分支 | 不变（继承） |
| SupportBytes | 不变 | 不变 |
| CreatePredicate | 两层→单层（简化） | 不变 |
| RegTensor | 成员顺序 + Print() 删除 | 不变 |
| GetCastPreQuantMode | **转型→禁止转型** | 不变 |
| GetScalarPreQuantMode | float→half, 加 int16, 去 exotic | 不变 |
| GetVectorPreQuantMode | float→half, 加 int16, 去 exotic | 不变 |
| CheckTMovAccValid | float→half, 加 int16, 去 exotic | 不变（继承） |
| GetDualDstCtl | 不变 | 不变 |
| datatype exotic 类型 | 全部去掉 | 不变 |
| datatype 恒等映射 | 全部去掉 | **追加 vector_bf16** |
| **CheckTMovAccToMat** | — | **新增**（kX90 独有） |

---

## 四、核心差异总结

### 4.1 累加器类型链：`float ↔ half` 切换

这是驱动所有差异的根因：

```
a5:          Acc SrcType = float → GetScalarPreQuantMode src = float → GetCastPreQuantMode 允许 float→half/bf16
kirin9030:   Acc SrcType = half  → GetScalarPreQuantMode src = half  → GetCastPreQuantMode 禁止任何转型
kirinX90:    继承 k9030（Acc SrcType = half）
```

### 4.2 Advanced 数据类型支持

```
a5:          7 exotic 类型 (bf16/fp8e5m2/fp8e4m3/hifloat8/fp8e8m0/fp4) + 14 恒等映射
kirin9030:   0 exotic 类型, 0 恒等映射
kirinX90:    0 exotic 类型, 1 恒等映射 (vector_bf16)
```

### 4.3 量化目标类型差异

```
a5:          支持 hifloat8 / bfloat16 / float8_e4m3 → 禁用 int16
kirin9030:   不支持 hifloat8 / bfloat16 / float8_e4m3 → 启用 int16 (DEQS16/VDEQS16)
kirinX90:    同 k9030
```

### 4.4 文件组织策略

| 架构 | common.hpp | datatype.hpp |
|------|-----------|-------------|
| **a5** | 完整实现（量化和校验全量） | 完整实现（含 exotic 类型和全部恒等映射） |
| **k9030** | 独立精简（float→half 调整，去 exotic） | 独立精简（仅基础类型，无恒等映射） |
| **kX90** | 继承 k9030 + 追加 CheckTMovAccToMat | 继承 k9030 + 追加 vector_bf16 |
