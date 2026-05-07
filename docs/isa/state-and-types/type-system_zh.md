# 类型系统

PTO 使用一套紧凑但架构可见的类型系统。合法性不会在原始类型名处结束。类型类别先说明当前操作数属于哪一类架构对象；layout、location、valid region 和目标 profile 等其他维度再决定该用法是否真的合法。

## 元素类型

PTO 支持浮点、整数和若干专用数值类型。

### 浮点类型

| 类型 | SSA 名称 | 位宽 | 说明 | A2/A3 | A5 |
| --- | --- | --- | --- | :---: | :---: |
| IEEE FP16 | `f16` / `half` | 16 | IEEE 754 半精度 | Yes | Yes |
| BF16 | `bf16` / `bfloat16_t` | 16 | Brain float 16（8 位指数） | Yes | Yes |
| IEEE FP32 | `f32` | 32 | IEEE 754 单精度 | Yes | Yes |
| FP8 E4M3 | `f8e4m3` / `float8_e4m3_t` | 8 | 4 位指数、3 位尾数 | No | Yes |
| FP8 E5M2 | `f8e5m2` / `float8_e5m2_t` | 8 | 5 位指数、2 位尾数 | No | Yes |
| HI Float8 | `hifloat8_t` | 8 | 高精度 float8 | No | Yes |
| Float4 E1M2x2 | `float4_e1m2x2_t` | 4 | 4-bit float4，2x2 打包 | No | Yes |
| Float4 E2M1x2 | `float4_e2m1x2_t` | 4 | 4-bit float4，2x2 打包 | No | Yes |

### 整数类型

| 类型 | SSA 名称 | 位宽 | 有符号性 | A2/A3 | A5 |
| --- | --- | --- | --- | :---: | :---: |
| int8 | `i8` | 8 | Signed | Yes | Yes |
| uint8 | `u8` | 8 | Unsigned | Yes | Yes |
| int16 | `i16` | 16 | Signed | Yes | Yes |
| uint16 | `u16` | 16 | Unsigned | Yes | Yes |
| int32 | `i32` | 32 | Signed | Yes | Yes |
| uint32 | `u32` | 32 | Unsigned | Yes | Yes |
| int64 | `i64` | 64 | Signed | Yes | Yes |
| uint64 | `u64` | 64 | Unsigned | Yes | Yes |

## 向量宽度

向量寄存器宽度 `N` 由元素类型和目标 profile 决定：

| 元素类型 | 向量宽度 N | 每寄存器字节数 | 说明 |
| --- | :---: | :---: | --- |
| f32 | 64 | 256 B | 64 × 32-bit |
| f16, bf16 | 128 | 256 B | 128 × 16-bit |
| i16, u16 | 128 | 256 B | 128 × 16-bit |
| i8, u8 | 256 | 256 B | 256 × 8-bit |
| f8e4m3, f8e5m2 | 256 | 256 B | 256 × 8-bit |

向量宽度本身在各 profile 间可移植。差别在于 A5 原生执行向量指令，而 CPU / A2 / A3 通过模拟或桥接实现。

## 向量寄存器类型

向量寄存器的 SSA 类型：

```text
!pto.vreg<NxDTYPE>
```

```text
!pto.vreg<64xf32>   -- 64 lanes of f32
!pto.vreg<128xf16>  -- 128 lanes of f16
!pto.vreg<256xi8>   -- 256 lanes of i8
```

## Tile Buffer 类型

Tile buffer 的 SSA 类型如下，完整参数见 [Tile 与有效区域](../programming-model/tiles-and-valid-regions_zh.md)：

```text
!pto.tile<loc=vec, f32, 16, 16, RowMajor, NoneBox, None, Zero>
!pto.tile_buf<loc=mat, bf16, 16, 16, RowMajor, NoneBox, None, Null>
!pto.tile_buf<loc=left, int8, 16, 16, RowMajor, RowMajor, NZ, Null>
```

## NaN 与 Inf 行为

对浮点类型，PTO 以 IEEE 754 语义为基础，并保留以下 implementation-defined 变化点：

| 行为 | 规则 |
| --- | --- |
| Quiet NaN 传播 | quiet NaN 输入得到 quiet NaN 输出 |
| Signaling NaN | signaling NaN 可能先被硬件 quiet 化 |
| Inf 算术 | 按 IEEE 754 要求产生和传播 |
| 非规格化数 | 硬件可选择 flush-to-zero |
| 舍入 | 由 `rnd` 属性控制：`rne`、`rz`、`rp`、`rm` |

FTZ 行为是 implementation-defined。`rnd` 属性控制那些会改变指数范围的操作的舍入方向，例如 `vcvt`。

## 类型转换规则

### 浮点到浮点

| 源 | 目标 | 行为 |
| --- | --- | --- |
| f16 → bf16 | Conversion | 按位重解释，不做数值转换 |
| bf16 → f16 | Conversion | 按位重解释，不做数值转换 |
| f16/bf16 → f32 | Promotion | 扩展到 f32；可精确表示的值保持不变 |
| f32 → f16/bf16 | Narrowing | 按 `rnd` 舍入；NaN/Inf 遵循 IEEE 754 |
| f8 → f16/f32 | Promotion | 扩展；可精确表示的值保持不变 |
| f16/f32 → f8 | Narrowing | 按 `rnd` 舍入；可能溢出为 Inf |

### 整数到整数

| 源 | 目标 | 行为 |
| --- | --- | --- |
| 扩宽（如 i8 → i16） | Zero/sign extend | 无符号零扩展；有符号符号扩展 |
| 缩窄（如 i16 → i8） | Truncation | 截断高位 |
| i32 → f32 | Conversion | 在 `[-2^24, 2^24]` 内精确；更大范围可能丢精度 |
| f32 → i32 | Conversion | 向零截断；可能溢出 |

### 浮点与整数之间

| 源 | 目标 | 行为 |
| --- | --- | --- |
| f32 → i8/u8/i16/u16 | Narrowing | 截断；可能溢出 |
| f32 → i32/u32 | Narrowing | 截断；可能溢出 |
| i8/u8 → f32 | Promotion | 小范围精确；更大值可能丢精度 |

### 转换操作

| 操作 | 指令集 | 说明 |
| --- | --- | --- |
| `pto.tcvt` | Tile | 在 tile 上逐元素转换类型 |
| `pto.vcvt` | Vector | 向量寄存器类型转换 |
| `pto.vtrc` | Vector | 向量截断/舍入 |
| `pto.vci` | Vector | 向整数压缩或索引式转换 |

## 约束

!!! warning "约束"
    - 指令集页面必须明确列出允许的操作数/结果类别。
    - 类型错误必须与更深层的合法性错误区分开来，例如 shape、layout、location intent 或 target profile。
    - 向量指令文档必须显式说明向量寄存器、mask、指针和对齐状态。
    - Tile 指令文档必须显式说明 tile role、shape 和 valid region 的交互。
    - 不存在隐式类型提升。`tadd(t, i8_tile, f32_immediate)` 之类形式在没有显式 `tcvt` 的情况下是非法的。

## 不允许的情形

!!! danger "不允许的情形"
    - 把类型类别检查误当成完整的 backend 合法性检查
    - 混淆标量状态与 tile / vector 有效载荷状态
    - 把 vector 和 tile 有效载荷类别写成可互换
    - 依赖隐式类型转换而非显式 `tcvt` / `vcvt`

## 相关页面

- [位置意图与合法性](./location-intent-and-legality_zh.md)
- [指令族总览](../instruction-families/README_zh.md)
- [规范来源](../reference/source-of-truth_zh.md)
- [Tile 与有效区域](../programming-model/tiles-and-valid-regions_zh.md)
