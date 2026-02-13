# A2A3 Subtile (1D/2D) Simplified ISA

This folder contains **minimal, subtile** versions of A2/A3 PTO ISA ops with a **new API**.

## Subtile API
- `Subtile1D<T>`: runtime length, **counter mode**, length can be > VL
- `Subtile2D<T>`: runtime rows/cols, **mask + hw repeat**, constraints:
  - `cols <= VL`
  - `rows <= REPEAT_MAX` (<= 255)

## Design goals
- **No row/col loops** in op code
- **Single-instruction path** (counter mode for 1D, repeat+mask for 2D)

## Assumptions
- Row-major only
- For 2D, `rowStride` is in elements
- 2D uses `SetContMaskByDType(cols)` + repeat

Use this for algorithm prototyping or teaching, **not** for general tiling.
