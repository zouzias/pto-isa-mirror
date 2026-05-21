# pto.sethf32mode

`pto.sethf32mode` is part of the [Sync And Config](../../sync-and-config.md) instruction set.

## Summary

Configure HF32 transform mode for supported matrix-multiplication and convolution paths.

This instruction controls backend-specific HF32 transformation behavior used by supported compute paths. It is part of the tile synchronization or configuration shell, so the visible effect is ordering or state setup rather than arithmetic payload transformation.

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

## Syntax

Textual spelling is defined by the PTO ISA syntax-and-operands pages.

Schematic form:

```text
pto.sethf32mode {enable = true, mode = ...}
```

### IR Level 1 (SSA)

```text
pto.sethf32mode {enable = true, mode = ...}
```

### IR Level 2 (DPS)

```text
pto.sethf32mode ins({enable = true, mode = ...}) outs()
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <bool isEnable, RoundMode hf32TransMode = RoundMode::CAST_ROUND, typename... WaitEvents>
PTO_INST RecordEvent SETHF32MODE(WaitEvents &... events);
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `enable` | `bool` | Enables (`true`) or disables (`false`) the HF32 transform mode |
| `mode` | `RoundMode` | HF32 rounding mode; supported values are target-profile defined |

## Expected Outputs

This form is defined primarily by its ordering or configuration effect. It does not introduce a new payload tile.

## Side Effects

- **A2/A3 and A5**: Configures implementation-defined HF32 transform mode state when the target profile supports it.
- **CPU simulator**: Updates software-simulation mode state when modeled; otherwise functional no-op.

## Constraints

!!! warning "Constraints"
    - The exact mode values and downstream hardware behavior are target-defined.
    - This instruction has control-state side effects and should be ordered appropriately relative to dependent compute instructions.
    - Programs must not treat this operation as tile-payload transformation; it only changes mode state.

## Cases That Are Not Allowed

!!! danger "Cases That Are Not Allowed"
    - Relying on HF32 behavior on a target profile that does not support HF32 transform mode.
    - Using mode values outside the selected target profile's supported set.

## Examples

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void example_enable_hf32() {
  SETHF32MODE<true, RoundMode::CAST_ROUND>();
}
```

## See Also

- Instruction set overview: [Sync And Config](../../sync-and-config.md)
- Previous op: [pto.tassign](./tassign.md)
- Related mode op: [pto.settf32mode](./settf32mode.md)
- Next op: [pto.setfmatrix](./setfmatrix.md)
