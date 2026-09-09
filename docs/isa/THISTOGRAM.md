# THISTOGRAM

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T04:11:21.026Z pushedAt=2026-08-29T09:05:18.435Z -->

> **Implementation status**: THISTOGRAM provides C++ built-in implementations on the Ascend 950PR/Ascend 950DT, Kirin9030 backends, and CPU simulation (`__CPU_SIM`), and has been registered in the virtual ISA index (`PTOISA`, `isa/README`, `manifest.yaml`, the instruction family matrix `appendix-d`, and the mkdocs navigation). It is available only on Ascend 950PR/Ascend 950DT/Kirin9030/CPU simulation (Atlas A2/A3 training products/Atlas A2/A3 inference products are not supported); no public bytecode encoding is available yet.

## Introduction

Computes a histogram over a specific **byte** of each source tile element (the occurrence count of each byte value 0–255), and supports cascade filtering based on "the already-processed higher-order bytes equal a given index". It is the **byte bucket counting primitive for radix sort**: the first pass counts the most significant byte (MSB); subsequent passes count lower-order bytes but only over elements whose higher-order bytes match the bucket index from the previous pass, thereby obtaining the distribution of the current radix digit within the prefix bucket.

A single call independently produces a set of 256 `uint32` counts (bins) for each valid row of the source tile.

## Mathematical Semantics

Assume that the source `src` has valid shape $R \times C$, with elements of type `uint16_t` or `uint32_t`. Denote $B_k(x)$ as the $k$-th byte of element $x$:

$$
B_0 = \text{bits } 7\text{–}0\ (\text{LSB}),\quad B_1 = \text{bits } 15\text{–}8,\quad B_2 = \text{bits } 23\text{–}16,\quad B_3 = \text{bits } 31\text{–}24\ (\text{MSB})
$$

The template parameter `byte` selects the byte $k\in\{0,1,2,3\}$ to be counted. For each source row $r\in[0,R)$ and each bucket value $b\in[0,256)$:

$$
\mathrm{dst}_{r,b} = \bigl|\{\,j\in[0,C)\ \big|\ B_k(\mathrm{src}_{r,j})=b\ \wedge\ F_{k}(r,j)\ \}\bigr|
$$

Where the cascade filtering $F_k$ is defined in high-order-first order (process $k=3$ first, then $k=2,1,0$):

| Source Dtype | `byte` $k$ | Filter $F_k(r,j)$ | Meaning of `idx` |
|----------|-----------|-----------------|-----------|
| `uint16` | `BYTE_1` (MSB) | Always true (first pass, no filtering) | Unused |
| `uint16` | `BYTE_0` (LSB) | $B_1(\mathrm{src}_{r,j})=\mathrm{idx}_{r}$ | 1 matching byte per row (higher-order) |
| `uint32` | `BYTE_3` (MSB) | Always true (first pass, no filtering) | Unused |
| `uint32` | `BYTE_2` | $B_3=\mathrm{idx}_{r,0}$ | 1 filter byte per row |
| `uint32` | `BYTE_1` | $B_3=\mathrm{idx}_{r,0}\ \wedge\ B_2=\mathrm{idx}_{r,1}$ | 2 filter bytes per row |
| `uint32` | `BYTE_0` (LSB) | $B_3=\mathrm{idx}_{r,0}\wedge B_2=\mathrm{idx}_{r,1}\wedge B_1=\mathrm{idx}_{r,2}$ | 3 filter bytes per row |

- `dst` always has 256 `uint32` buckets per row (corresponding to byte values 0–255).
- `uint16` sources support only `BYTE_0` / `BYTE_1` (only two bytes can be extracted).
- Unless otherwise stated, the semantics are defined within the valid region; the specific memory interleaving layout of the bucket counts (N0/N1 dual banks, even/odd split) is implementation-defined, and the logical result is 256 counts per row.

> `src` and `idx` are both ISA-visible tile operands (not compiler scratch).

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`, available on Ascend 950PR/Ascend 950DT/Kirin9030/CPU simulation (`PTO_NPU_ARCH_A5 || PTO_NPU_ARCH_KIRIN9030 || __CPU_SIM`). The `HistByte` enum is defined in `include/pto/common/type.hpp`.
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
enum class HistByte : uint8_t {
    BYTE_0 = 0, // LSB (bits 7-0)
    BYTE_1 = 1, // bits 15-8
    BYTE_2 = 2, // bits 23-16
    BYTE_3 = 3  // MSB (bits 31-24)
};

template <HistByte byte, typename TileDataDst, typename TileDataSrc, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst &dst, TileDataSrc &src, TileDataIdx &idx, WaitEvents &...events);
```

| Parameter | Direction | Meaning |
|------|------|------|
| `byte` | Template | Byte to be counted (`HistByte::BYTE_0`…`BYTE_3`) |
| `dst` | Output | Histogram result tile, `uint32_t`, row-major order, 256 buckets per row |
| `src` | Input | Source data tile, `uint16_t` or `uint32_t`, row-major order |
| `idx` | Input | Cascade filtering index tile, `uint8_t`, shape varies with `byte` and source dtype (see below) |
| `events...` | Input | Wait events (`WaitEvents`), implicit `TSYNC` before the instruction |

## Tile Size and Data Types

Assume that the source valid shape is $R \times C$:

| Tile | Dtype | Valid Shape | Layout | Description |
|------|-------|---------|------|------|
| `dst` | `uint32_t` | $R \times 256$ | RowMajor | 256 bucket counts per row |
| `src` | `uint16_t` or `uint32_t` | $R \times C$ | RowMajor | Data to be counted |
| `idx` (`uint16` source) | `uint8_t` | $R \times 1$ | ColMajor (DN) | 1 matching byte (high bits) per row |
| `idx` (`uint32` source) | `uint8_t` | $(3-k) \times C$ | RowMajor | 1 filter byte broadcast per row; 0 rows when $k=3$ (unused) |

> The physical row count of `idx` must be aligned to 32-byte blocks (`PTO_CEIL(rows · sizeof(uint8_t), 32)`); the `uint16` mode requires `idx` to use the DN layout (`BLayout::ColMajor` + `SLayout::NoneBox`) with exactly 1 column.

## Supported Input Dtypes

| Source Dtype | Destination Dtype | idx Dtype | Allowed `byte` | Description |
|----------|-----------|-----------|--------------|------|
| `U16` (`uint16_t`) | `U32` | `U8` | `BYTE_0`, `BYTE_1` | Only the low/high byte can be extracted. |
| `U32` (`uint32_t`) | `U32` | `U8` | `BYTE_0`…`BYTE_3` | All four bytes are available, paired with 0–3 rows of idx. |

> `dst` must be `uint32_t`, `idx` must be `uint8_t`, and `src` is restricted to `uint16_t`/`uint32_t`; other combinations are intercepted by `static_assert` in the implementation.

## Implementation Notes

THISTOGRAM executes on the vector pipe (`PIPE_V`):

1. **Byte extraction**: Deinterleaves the source elements into per-byte vectors — `uint16` uses `DINTLV_B8` (splitting out MSB/LSB), and `uint32` uses `DINTLV_B16` + `vdintlv` (splitting out 4 bytes).
2. **Cascade filtering**: Uses `vcmp_eq` to generate the predicate "processed high-order byte == idx", cascading AND per byte (the first pass MSB has no filtering).
3. **Byte histogram**: Uses the hardware `chistv2` to count the selected bytes under the filtering predicate, internally using N0/N1 dual banks and odd/even split accumulation; finally writes back 256 `uint32` buckets per row (`INTLV_B32` interleaved storage). The dual banks and odd/even split are implementation details; the logical result is 256 counts per row.

## Constraints

| Constraint | Scope | Reason |
|------|---------|------|
| `dst` is `uint32_t` and row-major order | All targets | 256-bucket count width and storage layout |
| `src` ∈ {`uint16_t`, `uint32_t`} and row-major order | All targets | Byte extraction path |
| `idx` is `uint8_t` | All targets | Filter byte word width |
| `uint16` source: `idx` is DN (ColMajor + NoneBox) with 1 column | Ascend 950PR/Ascend 950DT/Kirin9030/CPU | Single byte/row broadcast match |
| `uint32` source: `idx` row-major order, rows $=3-k$, columns $=$ source columns | Ascend 950PR/Ascend 950DT/Kirin9030/CPU | Index rows required for cascade filtering |
| `uint16` source allows only `BYTE_0` / `BYTE_1` | All targets | `uint16` has only 2 bytes |
| `dst` 256 buckets per row | All targets | Byte value space 0–255 |

## Examples

```cpp
// uint16 source: histogram the high-order byte (MSB) of each element (first pass of radix sort).
THISTOGRAM<HistByte::BYTE_1>(dstTile, srcTile, idxTile);

// uint16 source: histogram the low-order byte (LSB), counting only elements whose high-order byte == idx (second pass).
THISTOGRAM<HistByte::BYTE_0>(dstTile, srcTile, idxTile);
```

Typical tile declaration (`uint16` mode, source valid shape $R\times C$):

```cpp
using TileDataSrc = Tile<TileType::Vec, uint16_t, R, alignedC,        BLayout::RowMajor>;
using TileDataDst = Tile<TileType::Vec, uint32_t, R, 256,             BLayout::RowMajor>;
using TileDataIdx = Tile<TileType::Vec, uint8_t,  alignedIdxBytes, 1, BLayout::ColMajor>;
```

For complete ST examples, see `tests/npu/a5/src/st/testcase/thistogram/` (A5), `tests/npu/kirin9030/src/st/testcase/thistogram/` (Kirin9030), `tests/npu/kirinX90/src/st/testcase/thistogram/` (KirinX90), and `tests/cpu/st/testcase/thistogram/` (CPU reference implementation).
