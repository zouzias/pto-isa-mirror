# Auto-mode A3 kernels — testing guide

This directory holds the auto-mode A3 kernel prototypes. This README is for
developers who want to **build and run the kernels and check pass/fail**, not
for editing kernel source. For kernel-level detail (shapes, optimization
notes, layout) read the per-kernel `README.md` in each subdirectory.

The single entry point is [`run_all.sh`](run_all.sh) — it `cd`s into each
selected kernel directory and invokes that kernel's own `run.sh` with the
same `-r` / `-v` / `-C` / `-n` flags forwarded.

## Prerequisites

- A working compiler server / environment where each kernel's own `run.sh`
  already runs (i.e. `bisheng` toolchain available, CANN env sourced).
- For `-r npu`: a connected Ascend device matching `-v`
  (currently we use `Ascend910B1`).
- For `-r sim`: the simulator wired up for that SoC version. Note that not
  every kernel here is sim-clean; when in doubt, run `-r npu` first.

## Quickstart

Run **everything** in `ALL_KERNELS` against real hardware:

```bash
bash run_all.sh -r npu -v Ascend910B1
```

Run a single kernel:

```bash
bash run_all.sh -r npu -v Ascend910B1 --kernels add_tile_array
```

Run two kernels (CSV, no spaces):

```bash
bash run_all.sh -r npu -v Ascend910B1 --kernels add_tile_array,topk
```

Pick a specific NPU (forwarded as `-n` to each kernel's `run.sh` — only
kernels whose `run.sh` accepts `-n` use it; others ignore it):

```bash
bash run_all.sh -r npu -v Ascend910B1 -n 3
```

Override the compiler (default is `bisheng`):

```bash
bash run_all.sh -r npu -v Ascend910B1 -C bisheng
```

## What it runs

The default kernel set (top of [`run_all.sh`](run_all.sh#L44)):

```
add_tile_array  topk  router_topk_small  mla  flash_atten  MoE  MoEv2
```

Kernels present in this directory but **not** in the default set —
[`gemm`](gemm/), [`topkv2`](topkv2/), [`sparse_flash_attn`](sparse_flash_attn/)
— are not wired into `run_all.sh`. Run them by hand from their own directory
using `bash run.sh -r npu -v Ascend910B1`.

`MoE` and `MoEv2` are **families**: each expands to 8 sub-kernels
(`router_matmul`, `moe_topk`, `moe_topk_padded`, `scatter`, `expert_ffn`,
`gather`, `full_moe_separate`, `full_moe_combined`). Filter them with
`--moe-subkernels` (filter applies to both families when both are selected):

```bash
# Only MoE, only two sub-kernels
bash run_all.sh -r npu -v Ascend910B1 \
    --kernels MoE \
    --moe-subkernels expert_ffn,gather
```

## Per-shape overrides

Most kernels run a single hardcoded default shape. Two kernels accept a
shape override; both are forwarded via the kernel's `-a` flag.

**MoE family** — tuple is `T,H,F,E,TopK`
(tokens, hidden, ffn-intermediate, experts, top-k):

```bash
bash run_all.sh -r npu -v Ascend910B1 --kernels MoE \
    --cases-moe "512,128,128,32,1"
```

`--cases-moe` is only forwarded to the `MoE` family. `MoEv2` sub-kernels
ignore unknown args, so it's a no-op there.

**flash_atten** — tuple is `HEAD_SIZE,S0,S1[,CUBE_S0[,TILE_S1]]`
(CUBE_S1 is fixed at 128 — see
[flash_atten/scripts/gen_data.py](flash_atten/scripts/gen_data.py)):

```bash
bash run_all.sh -r npu -v Ascend910B1 --kernels flash_atten \
    --cases-flash-atten "128,128,1024,128,256"
```

For per-shape sweeps inside a single kernel (FA, MoE), prefer running that
kernel's own `run.sh` directly — `run_all.sh` only forwards one `-a` value.

## Output, logs, results

While running, `run_all.sh` prints a banner per kernel showing the dir and
command. **The kernel's own stdout/stderr are redirected to per-run log
files** under `run_log/`:

```
run_log/<kernel>_<start_unix_ts>.out
run_log/<kernel>_<start_unix_ts>.err
```

For MoE sub-kernels the label includes the family
(`MoE_expert_ffn_<ts>.out`). The console shows only the banner, the log
paths, and the final summary — to debug a failure, open the matching
`.out` / `.err` pair.

At the end you get a summary table:

```
  add_tile_array          PASS              4s
  topk                    FAIL (rc=1)       8s
  ...
  TOTAL                   12/14 passed  (elapsed: 312s)
```

**Pass/fail is by exit code only**: a kernel "passes" if its `run.sh` exits
0. Each kernel's host code is expected to return non-zero on
correctness-check failure (see commit `1d7a06f3` — "standardized host code
return codes"). If a `run.sh` exits 0 despite a numerical mismatch, that's
a bug in that kernel's host code, not in `run_all.sh`.

`run_all.sh` itself exits 0 only when **every** selected kernel passed.

## Common pitfalls

- **CSV must have no spaces**: `--kernels add_tile_array,topk` works,
  `--kernels "add_tile_array, topk"` does not — unknown kernel name error.
- **`-r` and `-v` are required**: omitting either prints usage and exits 1.
- **`-r sim` is not uniformly supported**. Some kernels build/run cleanly
  under the simulator, others do not. If a kernel passes on `-r npu` but
  fails on `-r sim`, that's expected for now; check the kernel's own
  README before filing it as a bug.
- **`run_log/` is not auto-cleaned**. Old logs accumulate. Delete it
  yourself between large sweeps if disk usage matters.
- **MoE single-case-per-binary**: each MoE sub-kernel is rebuilt for the
  first `--cases` shape (see comment in
  [MoE/scripts/generate_cases.py](MoE/scripts/generate_cases.py)). Passing
  multiple `--cases-moe` tuples to `run_all.sh` is not supported — it
  forwards one string.
- **Per-kernel shape sweeps** (multiple shapes in one go) belong in that
  kernel's own `run.sh` invocation, not `run_all.sh`.

## Adding a new kernel to `run_all.sh`

1. Drop the kernel project under `kernels/automode/a2a3/<name>/` with its
   own `run.sh` that accepts at least `-r` / `-v` / `-C` and exits non-zero
   on failure.
2. Add `<name>` to `ALL_KERNELS` at the top of [`run_all.sh`](run_all.sh#L44).
3. If the kernel needs an `-a` shape override forwarded, add a
   `--cases-<name>` arg and a `case` branch in the dispatch loop, mirroring
   the `flash_atten` block ([`run_all.sh`](run_all.sh#L232)).
4. If it's a multi-sub-kernel family, mirror `run_moe_family` instead.

## Where to look next

- Per-kernel detail: `add_tile_array/README.md`, `flash_atten/README.md`,
  `gemm/README.md`, `MoE/README.md`, `MoEv2/README.md`, `mla/README.md`,
  `topk/README.md`, `topkv2/README.md`, `router_topk_small/README.md`.
- Auto-mode constraints, anti-patterns, A3-vs-A5 differences: `../../../docs_for_ai/`.
- Confirmed-good baseline reference: [`add_tile_array/`](add_tile_array/)
  (see project `CLAUDE.md` — "Confirmed baseline auto-mode A3 kernel
  project").
