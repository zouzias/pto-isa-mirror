# dispatch_ffn_combine_v3 A5 rebuild

This directory is rebuilt from the committed A3/A2A3 baseline:

- Source repository: `/home/ntlab/zy/code/zhangyuan/pto-isa-zy`
- Source tree: `HEAD:kernels/manual/a2a3/dispatch_ffn_combine_v3`
- Source HEAD: `58dca63f3b1726d624834c6496ae529ddc35644e`

The old repository worktree is intentionally not used as a baseline because it contains uncommitted changes.

## A5 deltas

Only the following A5 deltas are carried on top of the A3 committed tree:

1. A5 build wrapper
   - `CMakeLists.txt` builds with `PTO_NPU_ARCH_A5` and `--cce-aicore-arch=dav-c310`.
2. Kernel entry ABI
   - The device entry uses typed `__gm__ uint8_t *` arguments.
   - Host launch passes `static_cast<uint8_t *>` pointers.
3. A5 architecture policy
   - Active dispatch, matmul, and epilogue policies use `AtlasA5` / `MmadAtlasA5*` / `EpilogueAtlasA5*`.
4. A5 vector synchronization
   - Unsupported `pipe_barrier(PIPE_V)` and `TSYNC<TROWMAX/TMAX>` paths are replaced with `AscendC::PipeBarrier<PIPE_V>()`.
5. A5 PTO cast compatibility
   - The generic PTO vector cast path remains PTO-based.
   - A5 `TCVT` is only emitted in `__DAV_VEC__` compilation.
   - `int32_t -> half` uses a PTO two-step path through `float`, because A5 `TCvt.hpp` does not provide a direct overload for that conversion.
6. Soft-flag protocol cleanup
   - The stale `PtoLoadSoftFlagL1` / `PtoStoreSoftFlagL1` GM↔L1 bridge was removed after confirming no `BlockMmad` call passes a soft-flag GM pointer.
   - The active synchronization path uses cross-core flags; production source no longer keeps `AscendC::LocalTensor<TPosition::A1>` / `DataCopy` for this path.
7. HCCL remote window padding / platform runtime
   - Host tiling validates the A5 remote-window layout with padded per-token-scale, dispatch-output, token-count, and signal regions.
   - Runtime SoC selection is passed through `DISPATCH_FFN_COMBINE_V3_SOC_VERSION` when supplied by `run.sh`.
8. Namespace qualification
   - Ambiguous `TPipe` and `GlobalTensor` references are qualified where A5 PTO headers introduce competing names.

## Current verification

This machine can compile A5 `dav-c310` code but is not treated as an A5 runtime validation target. Current evidence is compile-only.

Verified command:

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:${LD_LIBRARY_PATH:-}
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
cmake -S kernels/manual/a5/dispatch_ffn_combine_v3 -B /tmp/dispatch_ffn_combine_v3_a5_rebuild6 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/dispatch_ffn_combine_v3_a5_rebuild6 -j1
```

Observed result:

```text
[100%] Built target dispatch_ffn_combine_v3
```

## Runtime smoke command

Run this only on an A5-capable environment:

```bash
bash kernels/manual/a5/dispatch_ffn_combine_v3/run.sh \
  --soc-version Ascend910_950 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

If `--soc-version` is omitted, host tiling uses the default platform manager.
