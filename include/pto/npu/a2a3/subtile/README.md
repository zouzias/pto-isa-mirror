# A2A3 Subtile (1:1) Simplified ISA

This folder contains **minimal, subtile 1:1** versions of A2/A3 PTO ISA ops.

Design goals:
- **No row/col loops**
- **No tail/mask handling** (assume full VL)
- **Single vxxx intrinsic per subtile**

Assumptions:
- `validRows == TileData::Rows` and `validCols == TileData::Cols`
- `TileData::Rows * TileData::Cols == REPEAT_BYTE / sizeof(T)` (exact VL)
- Row-major layout only

Use this for algorithm prototyping or teaching, **not** for general tiling.
