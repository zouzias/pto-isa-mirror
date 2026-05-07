# 数据格式参考

**物理数据格式** 定义了 tile、vector 和 scalar 在内存及硬件寄存器中的表示方式。它涵盖 memory space、元素打包、地址对齐、VLane 架构，以及 PTO 逻辑视图与底层存储之间的关系。

## 内存空间

PTO 区分三类架构可见内存空间：

| 空间 | 位置 | 访问单位 | 带宽 | 访问模式 |
| --- | --- | --- | --- | --- |
| **GM** | 片外设备 DRAM | 字节级 | 低 | 通过 DMA 与 UB 交换 |
| **UB** | 片上 SRAM | 32-byte block | 高 | 批量 DMA 传输 |
| **Tile Register File (TRF)** | 片上 tile buffer | 元素级 | 最高 | 供 tile / vector 计算直接访问 |

GM 与 UB 之间通过 DMA 引擎搬运；UB 与 TRF 之间通过 `TLOAD` / `TSTORE` / `VLDS` / `VSTS` 等显式操作搬运。

## Tile Buffer 格式

Tile 在 UB 或 TRF 中占据一段连续区域。其逻辑形状 `(Rows, Cols)` 与物理存储格式分离。

### UB 中的格式

在 UB 中，tile 按 `BLayout` 存储：

对 `RowMajor`：

$$ \mathrm{addr}(r, c) = (r \times C + c) \times \mathrm{sizeof(DType)} $$

对 `ColMajor`：

$$ \mathrm{addr}(r, c) = (c \times R + r) \times \mathrm{sizeof(DType)} $$

### TRF 中的格式

TRF 保存 tile 的原生 `BLayout` 形式。TRF 不是字节可寻址空间，只能通过 `TLOAD` / `TSTORE` 与外部交互。

### 地址对齐

| 访问类型 | 对齐要求 |
| --- | --- |
| GM 读写 | 至少元素大小对齐 |
| UB DMA | 32-byte block 对齐 |
| TRF load/store | 元素大小对齐 |

## 元素类型编码

### 标准类型

| 类型 | C++ 类型 | SSA 名称 | 字节数 | 寄存器宽度 |
| --- | --- | --- | :---: | :---: |
| IEEE FP16 | `half` | `f16` | 2 | 128 lanes |
| BF16 | `bfloat16_t` | `bf16` | 2 | 128 lanes |
| IEEE FP32 | `float` | `f32` | 4 | 64 lanes |
| int8 | `int8_t` | `i8` | 1 | 256 lanes |
| uint8 | `uint8_t` | `u8` | 1 | 256 lanes |
| int16 | `int16_t` | `i16` | 2 | 128 lanes |
| uint16 | `uint16_t` | `u16` | 2 | 128 lanes |
| int32 | `int32_t` | `i32` | 4 | 64 lanes |
| uint32 | `uint32_t` | `u32` | 4 | 64 lanes |

### A5 专属类型

| 类型 | C++ 类型 | SSA 名称 | 字节数 | 说明 |
| --- | --- | --- | :---: | --- |
| FP8 E4M3 | `float8_e4m3_t` | `f8e4m3` | 1 | 256 lanes |
| FP8 E5M2 | `float8_e5m2_t` | `f8e5m2` | 1 | 256 lanes |
| HI Float8 | `hifloat8_t` | `hifloat8` | 1 | 256 lanes |
| Float4 E1M2x2 | `float4_e1m2x2_t` | `float4_e1m2x2` | 1 | 2x2 packed |
| Float4 E2M1x2 | `float4_e2m1x2_t` | `float4_e2m1x2` | 1 | 2x2 packed |

## 向量寄存器格式（VLane）

在 A5 上，向量寄存器由 **8 个 VLane** 组成，每个 VLane 为 32 字节。这一结构是架构可见的，尤其体现在 group reduction 操作中。

```text
vreg (256 bytes total):
| VLane0 | VLane1 | VLane2 | ... | VLane7 |
|  32B   |  32B   |  32B   |     |  32B   |
```

各类型在每个 VLane 中的元素数：

| 数据类型 | 每 VLane 元素数 | 每寄存器总元素数 |
| --- | :---: | :---: |
| i8 / u8 | 32 | 256 |
| i16 / u16 / f16 / bf16 | 16 | 128 |
| i32 / u32 / f32 | 8 | 64 |
| i64 / u64 | 4 | 32 |

Group reduction（如 `vcgadd`、`vcgmax`、`vcgmin`）会按 VLane 独立归约，每个 VLane 产生一个结果。

## Pad Value 编码

`Pad` 参数指定 valid region 外元素的填充值。

### 标准 Pad 值

| Pad | 含义 | `float` 编码 | `half`/`bf16` 编码 | `i8`/`u8` 编码 |
| --- | --- | --- | --- | --- |
| `Zero` | 置零 | `0x00000000` | `0x0000` | `0x00` |
| `Null` | 未定义，不应读取 | `0x00000000` | `0x0000` | `0x00` |
| `Min` | 填最小值 | `0xff800000` | `0xfc00` | `0xff` |
| `Max` | 填最大值 | `0x7f800000` | `0x7c00` | `0x7f` |

### 自定义 Pad（A5）

`PadValueCustom(value)` 允许在编译期指定 pad 的浮点值，例如 softmax mask 常用的 `-1.0f`。

## 分形布局编码

`TileLayoutCustom` 枚举描述运行时实际使用的布局：

| `TileLayoutCustom` | BLayout | SLayout | Fractal | Block Size | 用途 |
| --- | --- | --- | --- | :---: | --- |
| `ND` | RowMajor | NoneBox | — | — | 标准 tile |
| `DN` | ColMajor | NoneBox | — | — | 列优先 tile |
| `NZ` | ColMajor | RowMajor | NZ | 512 B | A5 matmul 左操作数 |
| `ZN` | RowMajor | ColMajor | ZN | 512 B | `NZ` 对称形式 |
| `ZZ` | RowMajor | RowMajor | ZZ | 512 B | CUBE 专用 |

## 常量参考

| 常量 | 值 | 单位 | 用途 |
| --- | --- | --- | --- |
| `BLOCK_BYTE_SIZE` | 32 | bytes | DMA block 传输单位 |
| `FIXP_BURST_UNIT_LEN` | 64 | half-words | DMA burst 长度 |
| `FRACTAL_NZ_ROW` | 16 | elements | NZ/ZN 分形行尺寸 |
| `CUBE_BLOCK_SIZE` | 512 | bytes | CUBE 分形块 |
| `C0_SIZE_BYTE` | 32 | bytes | Cube C0 尺寸 |
| `MX_COL_LEN` | 2 | elements | MX matmul 列块 |
| `MX_ROW_LEN` | 16 | elements | MX matmul 行块 |
| `MX_BLOCK_SIZE` | 32 | elements | MX matmul 块大小 |
| `TMP_UB_SIZE` | 8 × 1024 | bytes | 临时 UB 大小 |
| `TMP_UB_OFFSET` | 184 × 1024 | bytes | 临时 UB 偏移 |
| `MASK_LEN` | 64 | bits | 谓词 mask 宽度 |
| `BLOCK_LEN` | 16 | elements | 标准 block 长度 |
| `VLane_COUNT` | 8 | lanes | A5 每寄存器的 VLane 数 |

## 相关页面

- [类型系统](./type-system_zh.md)
- [布局参考](./layout_zh.md)
- [Tile 与有效区域](../programming-model/tiles-and-valid-regions_zh.md)
- [内存模型](../memory-model/consistency-baseline_zh.md)
