# SET_IMG2COL_RPT

## Tile Operation Diagram

![SET_IMG2COL_RPT tile operation](../figures/isa/SET_IMG2COL_RPT.svg)

## Introduction

Set IMG2COL repeat metadata from an IMG2COL configuration tile (implementation-defined).

## Math Interpretation

No direct tensor arithmetic is produced by this instruction. It updates IMG2COL control state used by subsequent data-movement operations.

## Assembly Syntax

Schematic form:

```text
SET_IMG2COL_RPT %cfg
```

### AS Level 1 (SSA)

```text
pto.SET_IMG2COL_RPT %cfg : !pto.fmatrix_config -> ()
```

### AS Level 2 (DPS)

```text
pto.SET_IMG2COL_RPT ins(%cfg : !pto.fmatrix_config) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_RPT(ConvTileData &src, WaitEvents &... events);
```

This overload is provided per target backend under: `#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRINX90)` and `#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_KIRIN9030) || defined(__CPU_SIM)`.

## Constraints

- This instruction is backend-specific and available only for backends that expose IMG2COL configuration state.
- `src` must be a valid IMG2COL configuration tile type accepted by the backend implementation.
- The exact register/metadata fields updated by this instruction are implementation-defined.
- Use this instruction before dependent `TIMG2COL` operations in the same execution stream.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_set_img2col_rpt() {
  // IMG2COL configuration tile: a ConvTile (Mat, NC1HWC0 layout)
  using CfgTile = ConvTile<TileType::Mat, half, 1 * 1 * 16 * 16 * 16, Layout::NC1HWC0,
                           ConvTileShape<1, 1, 16, 16, 16>>;
  CfgTile cfg;
  TASSIGN(cfg, 0x0);
  // After setting fmapH/fmapW, padList, stride, etc., set the repeat metadata:
  SET_IMG2COL_RPT(cfg);
}
```

## ASM Form Examples

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.SET_IMG2COL_RPT %cfg : !pto.fmatrix_config -> ()
```

### Manual Mode

```text
# Manual mode: resources must be bound explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
# pto.tassign %arg1, @tile(0x2000)
pto.SET_IMG2COL_RPT %cfg : !pto.fmatrix_config -> ()
```

### PTO Assembly Form

```text
pto.SET_IMG2COL_RPT %cfg : !pto.fmatrix_config -> ()
# AS Level 2 (DPS)
pto.SET_IMG2COL_RPT ins(%cfg : !pto.fmatrix_config) outs()
```

