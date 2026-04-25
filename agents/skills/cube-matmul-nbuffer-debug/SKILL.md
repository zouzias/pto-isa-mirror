---
name: Cube Matmul N-Buffer Kernel Debugging
description: 'Debug PTO cube/matmul N-buffer kernels (PTO/Ascend A5 sim, dav-c310-cube). USE WHEN: a cube_matmul_* sim test produces ~99% wrong outputs (e.g. bad=8189/8192, max diff ~57), garbage output despite correct compile, ping-pong/N-buffer + outer-loop-preload designs failing identically across structural rewrites, debugging TLOAD/TMOV/TMATMUL pipe-sync (PIPE_MTE2/MTE1/M/FIX) and get_buffer/rls_buffer buf-id allocation. Covers verifying build env (cann_9b2 vs broken ascend-toolkit), forcing rebuild, sanity-routing through a known-good kernel, and minimal-delta bisection of structural changes.'
license: CANN Open Software License Agreement Version 2.0
---

# Cube Matmul N-Buffer Kernel Debugging

This skill captures the workflow used to fix `CubeMatmulNBufTest.buf4_alarge_K128`
(an outer-K-group + inner-sub-tile matmul with M=32, K=1024, N=256, fp16→fp32),
where v1–v14 of a "single large [32,128] A TLOAD + dynamic sub-view" design all
produced byte-identical garbage (`bad=8189/8192, maxDiff=57.0063`) despite very
different code structures.

## Decision: when to use this skill

| Symptom | Apply this skill? |
|---------|-------------------|
| ~99% of outputs wrong, suspiciously stable across rewrites | YES |
| `libnpu_drv_camodel.so: cannot open shared object file` | YES (env section) |
| `Mismatch at 0..9` huge magnitudes, no NaN/Inf | YES |
| Single-output garbage (one TASSIGN-typo) | NO — use a normal review |
| Compile errors / unresolved symbols | NO — use compiler messages |
| Need to add a brand-new op | NO — see `vector-fusion-operator-generate` |

## Build environment (gotcha #1)

The toolchain pointed to by the default `set_env.sh` is broken for these tests
on the dev box (aarch64 SVE/glibc errors). Always:

```bash
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
```

**Never** `source /usr/local/Ascend/ascend-toolkit/set_env.sh` for A5 sim ST runs.

Run a single test from repo root:

```bash
python3 tests/script/run_st.py -r sim -v a5 -t cube_matmul_nbuf \
    -g 'CubeMatmulNBufTest.buf4_alarge_K128'
```

`run_st.py` rebuilds incrementally; do not invoke `cmake`/`make` directly.

## Verify the new code is actually running (gotcha #2)

If sim output looks identical between two clearly-different source versions,
the build may not have picked up the change. Check timestamps:

```bash
cd ~/pto-isa-ptoas-st/tests/npu/a5/src/st/build
stat -c '%y %n' \
  testcase/<test>/CMakeFiles/<test>_kernel.dir/<test>_kernel.cpp.o \
  ../testcase/<test>/<test>_kernel.cpp \
  bin/<test>
```

If the source mtime is **newer** than the `.cpp.o` mtime, the build is stale.
Force a rebuild:

```bash
touch ~/pto-isa-ptoas-st/tests/npu/a5/src/st/testcase/<test>/<test>_kernel.cpp
rm -f ~/pto-isa-ptoas-st/tests/npu/a5/src/st/build/testcase/<test>/CMakeFiles/<test>_kernel.dir/<test>_kernel.cpp.o
rm -f ~/pto-isa-ptoas-st/tests/npu/a5/src/st/build/bin/<test>
```

Then re-run `run_st.py`.

## Bisection workflow

### Step 1 — Sanity-route the launcher

Before deeply debugging the kernel, prove the launcher / build / harness work
by routing the failing launcher through a known-good kernel body:

```cpp
void LaunchCubeMatmul4BufALarge(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream) {
    RunCubeMatmul8Buf8K<float, half><<<1, nullptr, stream>>>(    // proven kernel
        reinterpret_cast<float*>(out),
        reinterpret_cast<half*>(src0),
        reinterpret_cast<half*>(src1));
}
```

If this **fails**, the bug is outside the kernel (build, launcher, golden,
template instantiation, symbol collision). If it **passes**, the bug is in
the kernel body — proceed to Step 2.

### Step 2 — Verbatim-copy a proven kernel into the failing template

Replace the failing kernel body with a literal copy of a passing kernel
(e.g. `RunCubeMatmul8Buf8K`'s body) under the failing template name. Re-run.

If it **fails**, there is a per-template/per-symbol issue (rare — check for
duplicate symbol IDs, link order, or `#if 0` brace mismatches). If it
**passes**, the failing template is fundamentally fine and the bug is in
your structural rewrite — proceed to Step 3.

### Step 3 — Add structure back one increment at a time

Make the smallest possible delta from the verbatim-passing kernel toward the
target design and re-run after each change. Order from cheapest to most
invasive:

1. Wrap the body in a trivial outer loop (e.g. `for (outer=0; outer<2; outer++)`
   and rewrite `k = outer*32 + kk`). Must still pass — proves loop nesting
   is fine.
2. Restructure A loads into a burst (one TLOAD per `cb`) at the top of the
   outer iter, separately get/rls'd on **distinct MTE2 buf ids**.
3. Restructure inner loop to consume one preloaded A slot per inner step.
4. (Optional) Reduce the number of A slots to a ping-pong design — only
   after the burst-preload design works.

Re-run after each step. The first failing step is the bug.

## Root cause patterns observed

### Pattern A — Reusing one buf-id for multiple TLOADs (causes the 8189/8192 bug)

```cpp
// BAD: 8 TLOADs share one MTE2 buf id; sim treats them as one event
get_buffer<PIPE_MTE2>(a_id);
TLOAD(aM0, g0); TLOAD(aM1, g1); ... TLOAD(aM7, g7);
rls_buffer<PIPE_MTE2>(a_id);

// GOOD: one buf id per TLOAD
get_buffer<PIPE_MTE2>(0); TLOAD(aM0, g0); rls_buffer<PIPE_MTE2>(0);
get_buffer<PIPE_MTE2>(1); TLOAD(aM1, g1); rls_buffer<PIPE_MTE2>(1);
...
```

Both v12 (single id) and v13 (hoisted MTE1 around inner loop) produced
**byte-identical garbage** — a strong signal the synchronization is being
collapsed in the sim.

### Pattern B — Sub-view tiles with dynamic `TASSIGN`

```cpp
// FAILED across v6–v11: TASSIGN a small view tile to slot_base + cb*stride
// inside the inner loop, then TMOV to L0A. NZ K1M1M0K0 stride math was
// theoretically correct (cb * 0x400 for [32,16] sub-blocks of [32,128] fp16),
// but every variant produced identical wrong output.
```

Avoid sub-view dynamic `TASSIGN`. Use N statically-`TASSIGN`'d, fixed-address
slots and load the data directly into those slots from GM.

### Pattern C — Single large `TLOAD` of a big tile

A single `TLOAD` of `[32, 128]` (8 K-blocks worth) likewise never produced
correct output in this test, even though `TLoadCubeND2NZ` claims to handle
it. Burst of 8 small TLOADs into 8 dedicated slots is the working
equivalent.

## Working pattern — outer K-group + burst A preload

This is the design that finally passed `buf4_alarge_K128`:

```text
Outer loop (8 iters): preload 8 A tiles into 8 dedicated L1 slots,
                       each TLOAD on its own MTE2 buf id (0..7).
Inner loop (8 iters): for each cb=0..7,
                       - TLOAD B into one of 8 B slots (id = cb)
                       - TMOV A[cb] -> L0A[cb], B[cb] -> L0B[cb]
                       - TMATMUL_ACC into single AccTile
After all 64 inner steps: TSTORE result.
```

Buf-id plan: A=0..7 (MTE2 reused for B in inner), L0=8..15, C=16. Single C
accumulator over all 64 K-blocks.

## Known-good reference kernels in this file

`RunCubeMatmul{2Buf8K, 4Buf8K, 8Buf8K, 2Buf16K, 4Buf16K}` — DO NOT modify.

## Quick verify script

```bash
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
cd ~/pto-isa-ptoas-st
python3 tests/script/run_st.py -r sim -v a5 -t cube_matmul_nbuf 2>&1 \
  | grep -E 'PASSED|FAILED|max diff|bad count'
```

Expect `bad count: 0` for all 6 configs and `[  PASSED  ] 6 tests.`.

## Anti-patterns

- Iterating new structural variants without first sanity-routing through a
  known-good kernel body.
- Sharing one MTE2/MTE1 buf id across many distinct `TLOAD`/`TMOV`
  consumers in the same iteration.
- Dynamic `TASSIGN` of sub-view tiles inside an inner loop.
- Inferring "the build is broken" before checking object timestamps.
