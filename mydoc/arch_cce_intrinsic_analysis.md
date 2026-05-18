# arch_cce_intrinsic.hpp 文件分析

## 背景

根据 readme.md 的描述，需要对 PTO Tile Library 的核心组件进行分析理解。`arch_cce_intrinsic.hpp` 是实现跨架构兼容性的关键基础设施。

---

## 文件概述

### 文件位置

`include/pto/common/arch_cce_intrinsic.hpp`

### 文件大小

81 行代码

### 核心目的

根据文件头注释：
> Implementation of interface adaptation layer for device-side and cloud-side compatibility

**翻译**：实现接口适配层，用于设备端（NPU）和云端（CPU 模拟器）的兼容性。

---

## 核心作用

### 1. 硬件指令适配层

该文件是一个**硬件抽象层（HAL）**，用于统一不同 NPU 架构之间的底层硬件指令接口差异。它的主要作用是：

| 作用 | 说明 |
|------|------|
| **接口统一** | 提供统一的 `pto_*` 前缀函数接口 |
| **架构适配** | 通过宏定义分支处理不同架构的参数差异 |
| **可移植性** | 上层代码无需关心底层硬件差异 |
| **编译隔离** | `#ifndef __CPU_SIM` 确保仅在 NPU 上编译 |

### 2. 架构分组策略

文件将四种 NPU 架构分为两大组：

```
┌─────────────────────────────────────────────────────────────┐
│                    架构分组策略                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐       │
│  │ A2A3 || KirinX90    │    │ A5 || Kirin9030     │       │
│  │ (旧架构/相似架构)    │    │ (新架构/相似架构)    │       │
│  ├─────────────────────┤    ├─────────────────────┤       │
│  │ • copy_ubuf_to_ubuf │    │ • copy_ubuf_to_ubuf │       │
│  │   需要 sid=0 参数   │    │   无 sid 参数       │       │
│  │                     │    │                     │       │
│  │ • load_cbuf_to_cb   │    │ • load_cbuf_to_cb   │       │
│  │   6参数+addr_cal    │    │   8参数+Transpose   │       │
│  │                     │    │   模板参数          │       │
│  │                     │    │                     │       │
│  │ • vgatherb          │    │ • vgatherb          │       │
│  │   参数列表式        │    │   寄存器式+mask     │       │
│  └─────────────────────┘    └─────────────────────┘       │
│                                                             │
│  ┌─────────────────────┐                                   │
│  │ Kirin9030 特殊处理  │                                   │
│  ├─────────────────────┤                                   │
│  │ • vgatherb 无 mask  │                                   │
│  │ • create_cbuf_matrix│                                   │
│  │   使用 set_l0 +     │                                   │
│  │   set_l1_2d         │                                   │
│  └─────────────────────┘                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 封装函数详细分析

### 1. pto_copy_ubuf_to_ubuf

**功能**：UB（Unified Buffer）到 UB 的数据搬运

**源代码**：

```cpp
PTO_INTERNAL void pto_copy_ubuf_to_ubuf(__ubuf__ void *dst, __ubuf__ void *src, 
                                         uint16_t nBurst, uint16_t lenBurst,
                                         uint16_t srcGap, uint16_t dstGap)
{
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
    copy_ubuf_to_ubuf(dst, src, 0, nBurst, lenBurst, srcGap, dstGap);
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_A5)
    copy_ubuf_to_ubuf(dst, src, nBurst, lenBurst, srcGap, dstGap);
#endif
}
```

**差异对比**：

| 架构 | 参数数量 | 硬件指令调用 | 说明 |
|------|:--------:|-------------|------|
| **A2A3/KirinX90** | 7 参数 | `copy_ubuf_to_ubuf(dst, src, 0, nBurst, lenBurst, srcGap, dstGap)` | 第 3 参数 `sid=0` 硬编码 |
| **A5/Kirin9030** | 6 参数 | `copy_ubuf_to_ubuf(dst, src, nBurst, lenBurst, srcGap, dstGap)` | 无 `sid` 参数 |

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `dst` | `__ubuf__ void*` | 目标地址（UB） |
| `src` | `__ubuf__ void*` | 源地址（UB） |
| `nBurst` | `uint16_t` | Burst 次数（传输次数） |
| `lenBurst` | `uint16_t` | 每次 Burst 的长度（32B 为单位） |
| `srcGap` | `uint16_t` | 源地址 Burst 间间隔 |
| `dstGap` | `uint16_t` | 目标地址 Burst 间间隔 |
| `sid` | — | DMA Stream ID（仅旧架构需要） |

---

### 2. pto_load_cbuf_to_cb

**功能**：L1（CBUF）到 L0B（CB）的矩阵数据加载

**源代码**：

```cpp
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
using __cce_scalar::addr_cal_mode_t;
template <typename T>
PTO_INTERNAL void pto_load_cbuf_to_cb(__cb__ T *dst, __cbuf__ T *src, 
                                       uint16_t baseIdx, uint8_t repeat,
                                       uint16_t srcStride, uint16_t dstStride)
{
#if defined(PTO_NPU_ARCH_A2A3)
    load_cbuf_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, 0, false, addr_cal_mode_t(0));
#elif defined(PTO_NPU_ARCH_KIRINX90)
    load_cbuf_to_cb(dst, src, baseIdx, repeat, srcStride, dstStride, false, addr_cal_mode_t(0));
#endif
}
#elif defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030)
template <bool Transpose, typename T>
PTO_INTERNAL void pto_load_cbuf_to_cb(__cb__ T *dst, __cbuf__ T *src, 
                                       uint16_t mStartPosition, uint16_t kStartPosition,
                                       uint8_t mStep, uint8_t kStep, 
                                       uint16_t srcStride, uint16_t dstStride)
{
    load_cbuf_to_cb(dst, src, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, Transpose);
}
#endif
```

**差异对比**：

| 架构 | 参数数量 | 参数格式 | 特殊参数 |
|------|:--------:|---------|---------|
| **A2A3** | 9 参数 | `baseIdx, repeat, srcStride, dstStride` | `0, false, addr_cal_mode_t(0)` |
| **KirinX90** | 8 参数 | `baseIdx, repeat, srcStride, dstStride` | `false, addr_cal_mode_t(0)` |
| **A5/Kirin9030** | 9 参数 | `mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride` | `Transpose` 模板参数 |

**关键差异**：

1. **参数语义不同**：
   - A2A3/KirinX90：`baseIdx`（起始索引）、`repeat`（重复次数）
   - A5/Kirin9030：`mStartPosition`（M 维起始）、`kStartPosition`（K 维起始）、`mStep`、`kStep`

2. **Transpose 支持**：
   - A5/Kirin9030：通过模板参数 `Transpose` 支持矩阵转置加载
   - A2A3/KirinX90：无转置支持，需额外使用 `load_cbuf_to_cb_transpose`

3. **地址计算模式**：
   - A2A3/KirinX90：需要 `addr_cal_mode_t(0)` 参数
   - A5/Kirin9030：无此参数

**参数说明**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `dst` | `__cb__ T*` | 目标地址（L0B） |
| `src` | `__cbuf__ T*` | 源地址（L1） |
| `baseIdx` | `uint16_t` | 起始块索引（旧架构） |
| `repeat` | `uint8_t` | 重复次数（旧架构） |
| `mStartPosition` | `uint16_t` | M 维起始位置（新架构） |
| `kStartPosition` | `uint16_t` | K 维起始位置（新架构） |
| `mStep` | `uint8_t` | M 维步长（新架构） |
| `kStep` | `uint8_t` | K 维步长（新架构） |
| `srcStride` | `uint16_t` | 源矩阵步长 |
| `dstStride` | `uint16_t` | 目标矩阵步长 |
| `Transpose` | `bool` | 是否转置（模板参数） |

---

### 3. pto_vgatherb

**功能**：向量 Gather 操作，按索引从 UB 收集元素

**源代码**：

```cpp
#if defined(PTO_NPU_ARCH_A2A3)
template <typename T>
PTO_INTERNAL void pto_vgatherb(__ubuf__ T *dst, __ubuf__ uint32_t *src, 
                                uint32_t offsetAddr, uint16_t dstRepeatStride,
                                uint8_t dstBlockStride, uint8_t repeat)
{
    vgatherb(dst, src, offsetAddr, dstRepeatStride, dstBlockStride, repeat);
}
#elif defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
template <typename T, typename U, typename S>
PTO_INTERNAL void pto_vgatherb(T &dstReg, __ubuf__ U *base, S &idxReg, vector_bool &mask)
{
#if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
    vgatherb(dstReg, base, idxReg);
#else
    vgatherb(dstReg, base, idxReg, mask);
#endif
}
#endif
```

**差异对比**：

| 架构 | 接口风格 | 参数数量 | 特殊参数 |
|------|---------|:--------:|---------|
| **A2A3** | 参数列表式 | 6 参数 | `offsetAddr`, `dstRepeatStride`, `dstBlockStride`, `repeat` |
| **Kirin9030/KirinX90** | 寄存器式 | 3 参数 | `dstReg, base, idxReg`（无 mask） |
| **A5** | 寄存器式 | 4 参数 | `dstReg, base, idxReg, mask` |

**关键差异**：

1. **接口风格完全不同**：
   - A2A3：传统参数列表风格，适合循环调用
   - A5/Kirin9030/KirinX90：寄存器风格，适合向量编程

2. **Mask 参数**：
   - A5：需要 `vector_bool &mask` 参数，支持部分收集
   - Kirin9030/KirinX90：无 mask 参数，全量收集

---

### 4. pto_create_cbuf_matrix

**功能**：在 L1（CBUF）中创建矩阵并填充初始值

**源代码**：

```cpp
template <typename T, typename U>
PTO_INTERNAL void pto_create_cbuf_matrix(__cbuf__ T *dst, int64_t repeatConfig, U value)
{
#if defined(PTO_NPU_ARCH_KIRIN9030)
    set_l0_set_value_ui((uint32_t)value);
    set_l1_2d(dst, repeatConfig);
#else
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        create_cbuf_matrix_bf16(dst, repeatConfig, value);
    } else {
        create_cbuf_matrix(dst, repeatConfig, value);
    }
#endif
}
```

**差异对比**：

| 架构 | 实现方式 | 特殊处理 |
|------|---------|---------|
| **Kirin9030** | `set_l0_set_value_ui + set_l1_2d` | 两步操作，先设置值再填充 |
| **其他架构** | `create_cbuf_matrix` | 一步操作，BF16 有专用指令 |

**关键差异**：

1. **Kirin9030 使用两步操作**：
   - 第一步：`set_l0_set_value_ui((uint32_t)value)` 设置填充值
   - 第二步：`set_l1_2d(dst, repeatConfig)` 执行填充

2. **BF16 类型特殊处理**：
   - 其他架构：BF16 使用 `create_cbuf_matrix_bf16` 专用指令
   - Kirin9030：无 BF16 专用指令，使用通用方式

---

## 与 PTO 指令的关系

### 上层调用关系

```
┌─────────────────────────────────────────────────────────────┐
│                    调用层级关系                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PTO 指令层 (include/pto/npu/*/T*.hpp)                      │
│  ├── TExtract.hpp                                           │
│  │   └── TExtractVecToVecNDImpl()                           │
│  │       └── pto_copy_ubuf_to_ubuf()  ←── 调用适配层       │
│  │                                                           │
│  │   └── TExtractToB()                                      │
│  │       └── pto_load_cbuf_to_cb()    ←── 调用适配层       │
│  │                                                           │
│  ├── TMov.hpp                                                │
│  │   └── TMovToVec()                                        │
│  │       └── pto_copy_ubuf_to_ubuf()  ←── 调用适配层       │
│  │                                                           │
│  └── TFill.hpp                                               │
│      └── TFillMat()                                          │
│          └── pto_create_cbuf_matrix() ←── 调用适配层       │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  硬件适配层 (include/pto/common/arch_cce_intrinsic.hpp)     │
│  ├── pto_copy_ubuf_to_ubuf()                                │
│  ├── pto_load_cbuf_to_cb()                                  │
│  ├── pto_vgatherb()                                         │
│  └── pto_create_cbuf_matrix()                               │
│                                                             │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  硬件指令层 (CANN SDK 内部)                                  │
│  ├── copy_ubuf_to_ubuf()                                    │
│  ├── load_cbuf_to_cb()                                      │
│  ├── vgatherb()                                             │
│  └── create_cbuf_matrix()                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 具体调用示例

**TExtract.hpp 中的调用**：

```cpp
// TExtractVecToVecNDImpl (Vec→Vec ND)
template <typename T, typename DstTileData, typename SrcTileData>
void TExtractVecToVecNDImpl(...) {
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    
    // 调用适配层函数，无需关心架构差异
    pto_copy_ubuf_to_ubuf((__ubuf__ void *)dstAddr, (__ubuf__ void *)srcStart, 
                          validRow, rowBurstLen, srcGap, dstGap);
}

// TExtractToB (Mat→Right/L0B)
template <typename DstTileData, typename SrcTileData>
void TExtractToB(...) {
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    
    // A5/Kirin9030 使用模板参数 Transpose
    pto_load_cbuf_to_cb<false>(dstAddr, srcAddr, mStartPosition, kStartPosition, 
                               mStep, kStep, srcStride, dstStride);
}
```

---

## 编译隔离机制

### CPU 模拟器隔离

```cpp
#ifndef __CPU_SIM
// ... 所有硬件指令封装 ...
#endif // __CPU_SIM
```

**作用**：
- 当编译目标为 CPU 模拟器时（`__CPU_SIM` 宏定义），这些硬件指令封装不参与编译
- CPU 模拟器有自己的实现（在 `include/pto/cpu/` 目录）

---

## 设计模式分析

### 1. 模板参数适配

```cpp
template <bool Transpose, typename T>  // A5/Kirin9030: Transpose 模板参数
void pto_load_cbuf_to_cb(...);

template <typename T>                   // A2A3/KirinX90: 无 Transpose 参数
void pto_load_cbuf_to_cb(...);
```

**优势**：
- 编译期决定是否支持 Transpose，无运行时开销
- 类型安全，避免类型转换错误

### 2. 宏定义分支

```cpp
#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)
    // 旧架构实现
#elif defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_A5)
    // 新架构实现
#endif
```

**优势**：
- 编译期分支，零运行时开销
- 清晰的架构分组逻辑

### 3. 命名统一

所有封装函数使用 `pto_` 前缀：
- `pto_copy_ubuf_to_ubuf`
- `pto_load_cbuf_to_cb`
- `pto_vgatherb`
- `pto_create_cbuf_matrix`

**优势**：
- 统一的命名风格，易于识别
- 与硬件原生指令区分（原生无 `pto_` 前缀）

---

## 总结

### 文件定位

| 层级 | 文件 | 作用 |
|------|------|------|
| **应用层** | `kernels/*.cpp` | 用户算子代码 |
| **指令层** | `include/pto/npu/*/T*.hpp` | PTO 指令实现 |
| **适配层** | `include/pto/common/arch_cce_intrinsic.hpp` | **硬件指令适配** |
| **硬件层** | CANN SDK | 硬件原生指令 |

### 核心价值

1. **跨架构可移植性**：PTO 指令无需修改即可在不同 NPU 架构上运行
2. **接口统一性**：上层代码使用统一的 `pto_*` 接口
3. **编译期优化**：通过模板和宏实现零开销适配
4. **CPU/NPU 兼容**：通过 `__CPU_SIM` 宏实现模拟器隔离

### 对 KirinX90 对齐工作的启示

1. **KirinX90 与 A2A3 相似**：多数硬件指令参数格式相同
2. **关键差异点**：
   - `load_cbuf_to_cb` 参数数量（8 vs 9）
   - `vgatherb` 接口风格（寄存器式 vs 参数列表式）
3. **适配策略**：KirinX90 可复用 A2A3 的适配逻辑，无需单独分支