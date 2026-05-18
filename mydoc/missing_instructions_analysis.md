# KirinX90 缺失指令的硬件指令、数据类型、数据转换与分型分析

## 背景

KirinX90 相比 Kirin9030 缺少 8 个指令，这些指令由 kirin9030 原生实现或 a5 实现再被 kirin9030 复用。以下逐一分析每个指令所使用的硬件指令、支持的数据类型、支持的数据转换路径和分型（fractal）布局。

---

## 1. TInsert

**来源：** kirin9030/TInsert.hpp（薄封装，下层复用 a5/TInsert.hpp）

### 硬件指令

| 硬件指令/Intrinsic | 用途 |
|---|---|
| `copy_matrix_cc_to_cbuf` | Acc(L0C)→Mat(L1/CBUF) 搬运 |
| `copy_matrix_cc_to_ub` | Acc(L0C)→Vec(UBUF) 搬运 |
| `copy_ubuf_to_cbuf` | Vec(UBUF)→Mat(CBUF) 搬运 |
| `copy_ubuf_to_ubuf` | Vec(UBUF)→Vec(UBUF) 搬运 |
| `vlds` | 从 UBUF 向量加载到向量寄存器 |
| `vsts` / `vstus` / `vstas` | 向量寄存器→UBUF 存储（对齐/非对齐/流式） |
| `vsstb` | 向量 scatter store block（ND→NZ 转换） |
| `set_quant_pre` | 标量量化预缩放参数设置 |
| `set_fpc` | 向量量化/反量化 tensor 地址配置 |
| `set_loop3_para` / `set_channel_para` | NZ→ND/DN 转换参数配置 |
| `set_flag` / `wait_flag` | 管道间事件同步 |
| `plt_b8/b16/b32` | 谓词设置（向量掩码生成） |

### 数据类型

| 数据路径 | 支持类型 |
|---|---|
| Acc 源（L0C→Mat/Vec） | `half`, `int32_t` 仅 |
| Vec/Mat 源（非 Acc 路径） | `half`, `bfloat16`, `float`, `int32_t`, `float8_e4m3`, `float8_e5m2`, `hifloat8`, `int8_t`, `float8_e8m0`, `float4_e2m1x2`, `float4_e1m2x2` |
| Acc+量化路径（int32_t→） | `half`, `int16_t`, `int8_t`, `uint8_t` |
| Acc+量化路径（half→） | `half`, `int16_t`, `int8_t`, `uint8_t` |

### 数据转换

| 转换路径 | 说明 |
|---|---|
| Acc NZ→Mat/Vec NZ | `!isRowMajor && SFractal==RowMajor` 原生 NZ |
| Acc NZ→ND | `isRowMajor && SFractal==NoneBox` NZ 转行主 |
| Acc NZ→DN | `!isRowMajor && SFractal==NoneBox` NZ 转列主 |
| Vec ND→Vec ND | 直接行主块拷贝（对齐/VF路径/非对齐向量路径/标量路径） |
| Vec NZ→Vec NZ | NZ 块拷贝 |
| Vec ND→Vec NZ | ND→NZ 转换（via vsstb） |
| Vec ND→Mat ND | Vec→CBUF 行主 |
| Vec NZ→Mat NZ | Vec→CBUF NZ（支持 SPLIT2/SPLIT4 拆分模式） |

### 分型布局

| 布局 | isRowMajor | SFractal | 说明 |
|---|---|---|---|
| ND | true | NoneBox | 标准行主 |
| DN | false | NoneBox | 列主 |
| NZ | false | RowMajor | Cube NZ，Acc 输出格式 |
| ZN | true | ColMajor | Cube ZN |
| ZZ | true | RowMajor | ZZ（a5 only） |

### 量化模式

| 模式 | 含义 |
|---|---|
| DEQF16 | int32_t → half 标量反量化 |
| DEQS16 | int32_t → int16_t 标量反量化 |
| REQ8 | int32_t → int8_t/uint8_t 标量重量化 |
| QF162S16_PRE | half → int16_t 标量量化 |
| QF162B8_PRE | half → int8_t/uint8_t 标量量化 |
| VDEQF16 / VDEQS16 / VREQ8 | 对应向量版反量化/重量化 |
| VQF162S16_PRE / VQF162B8_PRE | 对应向量版量化 |

### 模式选项

- **TInsertMode:** `SPLIT2`, `SPLIT4`（Vec→Mat NZ 拆分 DMA 次数）
- **AccToVecMode:** `SingleModeVec0`, `SingleModeVec1`, `DualModeSplitM`, `DualModeSplitN`
- **ReluPreMode:** `NoRelu`, `NormalRelu`
- **CompactMode:** `Null`, `Normal`, `RowPlusOne`, `RowAlignedPadding`

---

## 2. TDeQuant

**来源：** a5/TDeQuant.hpp（kirin9030 直接复用）

### 硬件指令

| 硬件指令/Intrinsic | 用途 |
|---|---|
| `vlds(UNPK4_B8)` | 加载 4 个 int8_t 值，解包为 32-bit 寄存器 |
| `vlds(UNPK_B16)` | 加载 2 个 int16_t 值，解包为 32-bit 寄存器 |
| `vlds(BRC_B32)` | 加载并广播 scale/offset 标量 |
| `vcvt` | int32→float 类型转换（ROUND_Z 模式） |
| `vsub` | 减 offset（MODE_ZEROING） |
| `vmul` | 乘 scale（MODE_ZEROING） |
| `vsts(DIST_NORM)` | 存储结果回 UBUF |

### 数据类型

| 角色 | 类型 |
|---|---|
| 源（src） | `int8_t` 或 `int16_t` |
| 目标（dst） | `float` 仅 |
| scale / offset | `float` 仅 |

### 数据转换

运算公式：`dst[i][j] = (src[i][j] - offset[i]) * scale[i]`

- int8_t 路径：`UNPK4_B8`（4 int8→4 int32）→ `vcvt`（int32→float, ROUND_Z）
- int16_t 路径：`UNPK_B16`（2 int16→2 int32）→ `vcvt`（int16→float）

### 分型布局

- **仅支持 RowMajor**（ND 布局）
- scale/offset 为 ColMajor 列向量，shape `[rows, 1]`

---

## 3. TGetScaleAddr

**来源：** a5/TGetScaleAddr.hpp（kirin9030 直接复用）

### 硬件指令

| 硬件指令/Intrinsic | 用途 |
|---|---|
| `__cce_pto_get_mx_tile_scale()` | 专用硬件指令：将源 tile 地址右移 4 位得到 scale tile 地址 |

### 数据类型

- **类型无关**（操作地址而非数据值）
- Scale tiles 通常使用 `float8_e8m0_t`（MX 缩放因子专用 8-bit 指数-only 类型）

### 数据转换

`Address(dst) = Address(src) >> SHIFT_MX_ADDR`（`SHIFT_MX_ADDR = 4`）

将数据 tile 的片上地址转换为对应 scale tile 的地址。每 32 字节数据块对应 32 字节 scale 块。

### 分型布局

| Tile 类型 | 布局 | SFractal | 缓冲区 | Fractal 大小 |
|---|---|---|---|---|
| TileLeftScale | RowMajor | RowMajor | L0A | 32 bytes |
| TileRightScale | ColMajor | ColMajor | L0B | 32 bytes |

**注意：** Scale buffer 大小在 A5 上为各 4 KiB，在 kirin9030/kirinX90 上为 0（不支持）。

---

## 4. TImg2col

**来源：** a5/TImg2col.hpp（kirin9030 直接复用）

### 硬件指令

| 硬件指令/Intrinsic | 用途 |
|---|---|
| `img2colv2_cbuf_to_ca()` | **核心硬件指令**，执行 img2col 转换（cbuf→ca） |
| `set_fmatrix()` / `set_fmatrix_b()` | 设置特征矩阵配置寄存器（A/B 矩阵） |
| `set_l3d_rpt()` / `set_l3d_rpt_b()` | 设置 L3D repeat 配置寄存器 |
| `set_padding()` / `set_padding_b()` | 设置 padding 值寄存器 |

### 数据类型

`int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16`, `float`

### 数据转换

**NC1HWC0（5HD）→ Cube LHS 格式（L0A）**

- 输入 layout：`NC1HWC0` 或 `NDC1HWC0`
- 输出 layout：`!isRowMajor && SFractal == RowMajor`（Cube NZ 格式）
- 卷积参数：strideW/H, filterW/H, dilationW/H, transpose, channelSize

### 分型布局

| 角色 | 布局 |
|---|---|
| 源（ConvTile） | NC1HWC0 / NDC1HWC0 |
| 目标（TileData） | Cube L0A（ColMajor base + RowMajor fractal） |

---

## 5. SetFmatrix

**来源：** a5/SetFmatrix.hpp（kirin9030 直接复用）

### 硬件指令

| 硬件指令/Intrinsic | 用途 |
|---|---|
| `set_fmatrix()` | 设置 A 矩阵的特征矩阵配置寄存器（64-bit） |
| `set_fmatrix_b()` | 设置 B 矩阵的特征矩阵配置寄存器（64-bit） |

### 寄存器布局

| 比特位 | 字段 | 来源 |
|---|---|---|
| [15:0] | fmapW | 特征图宽度 |
| [31:16] | fmapH | 特征图高度 |
| [39:32] | padList[0] | top padding |
| [47:40] | padList[1] | bottom padding |
| [55:48] | padList[2] | left padding |
| [63:56] | padList[3] | right padding |

### 数据类型

类型无关（寄存器操作）

---

## 6. SetImg2colRpt

**来源：** a5/SetImg2colRpt.hpp（kirin9030 直接复用）

### 硬件指令

| 硬件指令/Intrinsic | 用途 |
|---|---|
| `set_l3d_rpt()` | 设置 A 矩阵 L3D repeat 配置寄存器 |
| `set_l3d_rpt_b()` | 设置 B 矩阵 L3D repeat 配置寄存器 |

### 寄存器布局

| 比特位 | 字段 | 来源 |
|---|---|---|
| [15:0] | repeatStride | src.GetRepeatStride() |
| [23:16] | repeatTime | src.GetRepeatTime() |
| [31:24] | repeatMode | src.GetRepeatMode() |
| [47:32] | dstStride | src.GetDstStride() |
| [63:48] | dstMposition | src.GetDstMposition() |

### 数据类型

类型无关（寄存器操作）

---

## 7. SetImg2colPadding

**来源：** a5/SetImg2colPadding.hpp（kirin9030 直接复用）

### 硬件指令

| 硬件指令/Intrinsic | 用途 |
|---|---|
| `set_padding()` | 设置 A 矩阵 padding 值寄存器 |
| `set_padding_b()` | 设置 B 矩阵 padding 值寄存器 |

### 数据类型

| 数据大小 | 类型 | 处理方式 |
|---|---|---|
| 1 byte | `int8_t`, `uint8_t` | byte-extend 到 uint16 |
| 2 bytes | `int16_t`, `uint16_t`, `half`, `bfloat16` | uint16 reinterpret |
| 4 bytes | `int32_t`, `uint32_t`, `float` | uint32 reinterpret |

### 分型布局

类型无关（寄存器操作）

---

## 8. THistogram

**来源：** a5/THistogram.hpp（kirin9030 直接复用）

### 硬件指令

| 硬件指令/Intrinsic | 用途 |
|---|---|
| `chistv2(Bin_N0/Bin_N1)` | **核心硬件指令**，计算字节级直方图 |
| `vcvt(PART_EVEN/PART_ODD)` | uint16→uint32 加宽 |
| `vadd(MODE_ZEROING)` | 累加桶计数 |
| `vlds(DINTLV_B8)` | 加载 uint16 数据并解交织为 MSB/LSB 字节 |
| `vlds(DINTLV_B16)` | 加载 uint32 数据并解交织为偶奇半字 |
| `vlds(BRC_B8)` | 广播加载索引值 |
| `vsts(INTLV_B32)` | 交织存储偶数/奇数桶计数 |
| `vcmp_eq` | 字节比较（级联过滤） |
| `vdintlv` | 向量解交织（bytes→bytes） |
| `vbr` | 向量广播（清零） |

### 数据类型

| 角色 | 类型 |
|---|---|
| 源（scores） | `uint16_t` 或 `uint32_t` |
| 目标（桶计数） | `uint32_t` 仅 |
| 索引（idx） | `uint8_t` 仅 |

### 数据转换

**uint16 路径：**
```
vlds(DINTLV_B8) → vector_u8 (MSB & LSB 分离)
                → chistv2 → vector_u16
                          → vcvt(PART_EVEN/ODD) → vector_u32
                                                → vadd → vector_u32（累加后存储）
```

**uint32 路径：**
```
vlds(DINTLV_B16) → vector_u16 (偶奇半字)
                 → vdintlv → 4 × vector_u8 (byte0/1/2/3)
                            → chistv2 → vector_u16
                                      → vcvt → vector_u32
                                              → vadd → vector_u32
```

**HistByte 枚举（4 基数排序趟次）：**

| 趟次 | HistByte | 说明 | uint16 | uint32 |
|---|---|---|---|---|
| 1st | BYTE_3 (MSB) | 最高字节，无需过滤 | 不支持 | 支持 |
| 2nd | BYTE_2 | 次高字节，1-row 过滤 | 不支持 | 支持 |
| 3rd | BYTE_1 | 第 2 字节，2-row 过滤 | 支持(isMSB=true) | 支持 |
| 4th | BYTE_0 (LSB) | 最低字节，3-row 过滤 | 支持(isMSB=false) | 支持 |

### 分型布局

| 角色 | 布局约束 |
|---|---|
| 源 tile | RowMajor 仅 |
| 目标 tile | RowMajor 仅 |
| 索引 tile（uint16 源） | DN 布局（ColMajor, NoneBox），恰好 1 列 |
| 索引 tile（uint32 源） | RowMajor，行数根据 HistByte 变化（0-3 行） |

---

## 总结

| 缺失指令 | 核心硬件指令 | 数据类型 | 转换路径 | 分型 |
|---|---|---|---|---|
| **TInsert** | copy_matrix_cc_to_cbuf/ub, copy_ubuf_to_cbuf/ubuf, vlds/vsts/vsstb | half, bf16, f32, i32, i8, fp8 系列 | NZ↔NZ, NZ→ND/DN, ND↔ND, ND→NZ, Vec→Mat | ND, DN, NZ, ZN, ZZ |
| **TDeQuant** | vlds(UNPK), vcvt, vsub, vmul, vsts | int8/int16→float | int8→float (UNPK4→cvt), int16→float (UNPK→cvt) | RowMajor only |
| **TGetScaleAddr** | __cce_pto_get_mx_tile_scale | 地址操作（f8_e8m0） | addr >> 4 | ScaleLeft/Right (32B fractal) |
| **TImg2col** | img2colv2_cbuf_to_ca, set_fmatrix/rpt/padding | i8/u8/i16/u16/i32/u32/half/bf16/f32 | NC1HWC0 → Cube L0A | NC1HWC0(5HD), Cube NZ |
| **SetFmatrix** | set_fmatrix | 寄存器操作 | fmapW/H + pad 写入寄存器 | — |
| **SetImg2colRpt** | set_l3d_rpt | 寄存器操作 | repeat 参数写入寄存器 | — |
| **SetImg2colPadding** | set_padding | 1B/2B/4B 类型 | padding 值写入寄存器 | — |
| **THistogram** | chistv2, vlds(DINTLV), vcvt, vadd, vcmp_eq | uint16/uint32→uint32 | DINTLV→chistv2→cvt→add | RowMajor, DN(idx) |
