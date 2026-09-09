# TPARTMAX

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-26T04:38:10.337Z pushedAt=2026-08-29T09:05:18.450Z -->

## Instruction Diagram

![TPARTMAX tile operation](../figures/isa/TPARTMAX.svg)

## Introduction

Performs element-wise maximum selection within the destination valid region. If both `src0` and `src1` are valid at a position, the result is `max(src0, src1)`; if only one input is valid at that position, the result directly takes the value of that input. Other cases where the valid regions do not match are implementation-defined.

## Mathematical Semantics

For each element `(i, j)` in the destination valid region:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\max(\mathrm{src0}_{i,j}, \mathrm{src1}_{i,j}) & \text{if both inputs are defined at } (i,j) \text{ } \\\\
\mathrm{src0}_{i,j} & \text{if only src0 is defined at } (i,j) \text{ } \\\\
\mathrm{src1}_{i,j} & \text{if only src1 is defined at } (i,j) \text{ }
\end{cases}
$$

## Assembly Syntax

Synchronous form:

```text
%dst = tpartmax %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
```

### AS Level 1 (SSA)

```text
%dst = pto.tpartmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### AS Level 2 (DPS)

```text
pto.tpartmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Built-in APIs

Declared in `include/pto/common/pto_instr.hpp`:
> The public include header is `<pto/pto-inst.hpp>`, and the internal declaration is located in `pto/common/pto_instr.hpp`.

```cpp
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMAX(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, WaitEvents &... events);
```

## Constraints

### General Constraints or Checks

- The element types of `dst`, `src0`, and `src1` must be consistent.
- The destination valid region defines the computation range of the result.
- For each element within the destination valid region:
    - If both inputs are valid, the element-wise maximum value operation is performed;
    - If only one input is valid, the result directly takes the value of that input.
- If the valid region of `dst` is zero, the instruction returns directly.
- The supported partial valid region modes require that the valid region of at least one source tile is exactly the same as that of `dst`, and the valid area of the other source tile cannot exceed that of `dst` in either dimension.
- For valid region combinations outside the above range, the behavior is implementation-defined.

### Implementation Check for Atlas A2/A3 Training Products/Atlas A2/A3 Inference Products

- Supported element types: `int32_t`, `int`, `int16_t`, `half`, `float16_t`, `float`, `float32_t`.
- `dst`, `src0`, and `src1` must all be row-major (`isRowMajor`).

### Ascend 950PR/Ascend 950DT Implementation Check

- Supported element types: `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `half`, `bfloat16_t`, `float`.

## Examples

### Automatic

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TPARTMAX(dst, src0, src1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TPARTMAX(dst, src0, src1);
}
```

## ASM Examples

### Automatic Mode

```text
# Automatic mode: the compiler/runtime is responsible for resource placement and scheduling.
%dst = pto.tpartmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: explicitly bind resources first, then issue the instruction.
# Optional (when the instruction contains tile operands):
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
%dst = pto.tpartmax %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = tpartmax %src0, %src1 : !pto.tile<...> -> !pto.tile<...>
# AS Level 2 (DPS)
pto.tpartmax ins(%src0, %src1 : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```
