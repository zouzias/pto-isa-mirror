# MGATHER

## Introduction

Gather-load elements from global memory into a tile using per-element indices.

## Math Interpretation

For each element `(i, j)` in the destination valid region:

$$ \mathrm{dst}_{i,j} = \mathrm{mem}[\mathrm{idx}_{i,j}] $$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form - Level1:

```text
%dst = pto.mgather %mem, %idx : (!pto.partition_tensor_view<MxNx dtype>, !pto.tile<...>) 
                              -> !pto.tile<loc, dtype, rows, cols, blayout, slayout, fractal, pad>
```

Synchronous form - Level2:

```text
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<MxNx dtype>, !pto.tile_buf<MxNx dtype>) 
            outs(%dst : !pto.tile_buf<MxNx dtype>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& src, TileInd& indexes, WaitEvents&... events);
```

## Constraints

- Index interpretation is target-defined. The CPU simulator treats indices as linear element indices into `src.data()`.
- No bounds checks are enforced on `indexes` by the CPU simulator.

