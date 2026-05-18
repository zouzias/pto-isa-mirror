# KirinX90 TLoad 测试用例分析

## 背景

KirinX90 的 TLoad 测试分布在两个测试套件中：

1. **`tload`** — 标准 ND VecTile 加载，使用 `Tile<TileType::Vec, ...>`，从 5D GlobalTensor 加载到 UB
2. **`tload_shape2d`** — MatTile 格式转换加载，使用 `Tile<TileType::Mat, ...>` 和 `TileShape2D`，测试不同布局间的转换

---

## 套件 1：`tload` — 标准 ND VecTile 加载（共 12 个用例）

| # | 用例名 | 类型 | Global 形状 | Tile(R×C) | Blocks | Pad | 目的 |
|---|--------|:----:|:-----------:|:----------:|:------:|:---:|------|
| 1 | `case_float_GT_128_128_VT_128_128_BLK1` | float | [1,1,1,128,128] | 128×128 | 1 | Null | **基线测试**：对齐 2D ND 加载，float |
| 2 | `case_float_GT_2_2_2_256_64_VT_256_64_BLK8` | float | [2,2,2,256,64] | 256×64 | 8 | Null | **5D 批量**：3 个 batch 维（2×2×2=8 块），测试批量 ND 加载 |
| 3 | `case_float_GT_128_127_VT_128_128_BLK1_PADMAX` | float | [1,1,1,128,127] | 128×128 | 1 | Max | **PadMax**：tile(128×128) > global(128×127)，多余列填充 +inf |
| 4 | `case_s16_GT_128_127_VT_128_128_BLK1_PADMAX` | int16 | [1,1,1,128,127] | 128×128 | 1 | Max | **PadMax + int16**：填充 INT16_MAX |
| 5 | `case_u8_GT_128_127_VT_128_128_BLK1_PADMIN` | uint8 | [1,1,1,128,127] | 128×128 | 1 | Min | **PadMin + uint8**：填充 UINT8_MIN（0） |
| 6 | `case_float_GT_32_64_128_VT_64_128_BLK32_DYN` | int16 | [1,1,32,64,128] | 64×128 | 32 | Null | **5D 动态 shape**：3 batch（1×1×32=32），动态 shape 分发 |
| 7 | `case_float_GT_32_64_128_VT_64_128_BLK32_STC` | int16 | [1,1,32,64,128] | 64×128 | 32 | Null | **5D 静态 shape**：与#6 相同的形状，但使用编译期静态 shape |
| 8 | `case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX` | float | [2,2,2,256,60] | 256×64 | 8 | Max | **PadMax + 批量**：tile 宽(64) > global(60)，填充 +inf |
| 9 | `case_int64_GT_128_128_VT_128_128_BLK1` | int64 | [1,1,1,128,128] | 128×128 | 1 | Null | **int64 基线**：64-bit 整数 ND 加载 |
| 10 | `case_uint64_GT_128_125_VT_128_128_BLK1_PADZERO` | uint64 | [1,1,1,128,125] | 128×128 | 1 | Zero | **PadZero + uint64**：填充 0 |
| 11 | `case_int64_GT_2_2_2_256_62_VT_256_64_BLK8_PADZERO` | int64 | [2,2,2,256,62] | 256×64 | 8 | Zero | **PadZero + 批量 + int64**：填充 0 |
| 12 | `case_uint64_GT_2_2_2_256_64_VT_256_64_BLK8` | uint64 | [2,2,2,256,64] | 256×64 | 8 | Null | **uint64 批量**：5D 对齐批量加载 |

### 覆盖维度

- **数据类型**：float、int16_t、uint8_t、int64_t、uint64_t
- **Pad 模式**：Null（不填充）、Max（+inf 或类型最大值）、Min（0）、Zero（0）
- **Shape 策略**：动态 shape 与静态 shape 对比
- **批量**：单块（1 block）到多块（32 blocks）
- **对齐**：对齐（128×128）与非对齐（128×127）

---

## 套件 2：`tload_shape2d` — MatTile 格式转换加载（共 16 个用例）

| # | 格式 | 类型 | Valid 形状 | Whole 形状 | L1 Tile | 目的 |
|:-:|:----:|:----:|:----------:|:----------:|:-------:|------|
| 1 | ND→NZ | half | 128×128 | 128×128 | 128×128 | **对齐 ND→NZ**，half，标准转换 |
| 2 | ND→NZ | int8 | 128×128 | 128×128 | 128×128 | **对齐 ND→NZ**，int8 |
| 3 | ND→NZ | float | 128×128 | 128×128 | 128×128 | **对齐 ND→NZ**，float |
| 4 | DN→NZ | half | 64×128 | 64×128 | 64×128 | **列主→NZ**（DN 是行主转置） |
| 5 | ND→NZ | half | 63×127 | 63×127 | 64×128 | **非对齐 ND→NZ**：valid(63×127) < L1(64×128) |
| 6 | ND→ND | float | 128×128 | 128×128 | 128×128 | **ND→ND 透传**：相同格式，float |
| 7 | ND→ND | int8 | 37×126 | 37×126 | 37×128 | **非对齐 ND→ND**：valid(37×126)，L1(37×128) |
| 8 | ND→NZ | half | 33×99 | 64×128 | 48×112 | **子矩阵 ND→NZ**：valid(33×99) 小于 whole(64×128) |
| 9 | ND→NZ | int8 | 59×119 | 64×128 | 64×128 | **子矩阵 ND→NZ**：int8 |
| 10 | DN→NZ | float | 51×123 | 64×128 | 64×128 | **子矩阵 DN→NZ**：float |
| 11 | DN→NZ | half | 63×127 | 63×127 | 64×128 | **非对齐 DN→NZ**：half |
| 12 | DN→DN | float | 128×128 | 128×128 | 128×128 | **DN→DN 透传**：列主到列主 |
| 13 | DN→DN | int8 | 37×126 | 37×126 | 64×126 | **非对齐 DN→DN**：int8 |
| 14 | NZ→NZ | half | [1,10,8,16,16] | [1,11,9,16,16] | 128×160 | **NZ→NZ 批量**：5D NZ，C0=16（half），含 batch dims |
| 15 | NZ→NZ | int8 | [1,8,4,16,32] | [1,9,4,16,32] | 80×256 | **NZ→NZ 批量**：int8，C0=32 |
| 16 | ND→ND | int64 | 59×119 | 59×124 | 59×120 | **非对齐 ND→ND int64** |

### 覆盖维度

- **格式转换**：ND→NZ、DN→NZ、ND→ND、DN→DN、NZ→NZ（共 5 种）
- **数据类型**：half、int8_t、float、int64_t
- **对齐性**：对齐（幂等边界）与非对齐（如 37×126、59×119、33×99）
- **子矩阵**：valid shape < whole shape 的子区域加载
- **NZ Fractal**：不同 C0 大小（half C0=16、int8 C0=32）
- **批量 NZ**：5D NZ 格式含 batch 维度的加载

---

## 总结

| 统计项 | tload | tload_shape2d |
|--------|:----:|:-------------:|
| 用例数量 | 12 | 16 |
| Tile 类型 | VecTile（UB） | MatTile（L1） |
| 数据类型 | float/int16/uint8/int64/uint64 | half/int8/float/int64 |
| 布局转换 | 无（ND only） | ND/NZ/DN 互转 |
| 测试重点 | ND 加载、Pad、批量、动/静态 shape | 格式转换、对齐/非对齐、子矩阵 |
