# Kernel Test Guidance — Multi-Shape Kernels

Use this document when creating a new kernel under `kernels/automode/a2a3/`. It is the canonical place for test-harness contract guidance for future kernel writers when a kernel needs to cover multiple shapes.

## 1. Required test harness files (single-kernel project)

For a new standalone kernel folder, include at least:

1. `scripts/gen_data.py`
2. `main.cpp`
3. `run.sh`
4. `README.md`

Reference examples:

1. MoE sub-kernels: [kernels/automode/a2a3/MoE/gather/](../kernels/automode/a2a3/MoE/gather/)
2. Flash attention: [kernels/automode/a2a3/flash_atten/](../kernels/automode/a2a3/flash_atten/)

Known:

1. The project-local `run.sh` is the stable user entry point used by both families.
2. The golden-data path is Python-first (`gen_data.py` creates expected outputs), then C++ host compares device output against those files.
3. `run_all.sh` at [kernels/automode/a2a3/run_all.sh](../kernels/automode/a2a3/run_all.sh) expects each selected kernel folder to expose `run.sh`.

## 1.5. How to generate cases

Use the same pattern as the MoE and flash-attention `generate_cases.py` scripts when your kernel needs user-selected shapes.

General flow:

1. Accept case descriptions from the CLI.
2. Normalize each case into a structured record.
3. Write a generated header and a JSON manifest into the build directory.
4. Include the generated header from the host/kernel sources so the selected case becomes compile-time input.

Example case-generation pattern from MoE:

1. `kernels/automode/a2a3/MoE/scripts/generate_cases.py` accepts repeated `--cases` values.
2. Each case is passed as `kT,kH,kF,kE,kTopK`.
3. The script writes `build/generated_cases.h` and `build/generated_cases.json`.
4. The generated header defines `MOE_FOR_EACH_CASE(...)` and convenience aliases like `kMoeT`, `kMoeH`, `kMoeF`, `kMoeE`, and `kMoeTopK`.
5. Current MoE sub-kernels still compile against the first generated case, so the script is used to regenerate the header for each selected case set.

Example case-generation pattern from flash attention:

1. `kernels/automode/a2a3/flash_atten/scripts/generate_cases.py` also accepts repeated `--cases` values.
2. Each case is passed as `HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1]`.
3. The script supports shared options such as `--qk-preload` and `--causal-mask`.
4. It writes `build/generated_cases.h` and `build/generated_cases.json`.
5. The generated header defines `TFA_FOR_EACH_CASE(...)` and a `GeneratedTfaCase` table that the host/kernel build reads.
6. `run.sh` forwards the selected case arguments into the generator before `cmake`, so the binary is built against the selected case template.

## 2. `gen_data.py` contract (what to generate)

Minimum outputs:

1. Input tensors consumed by `main.cpp`
2. Golden output tensors computed in Python/Numpy
3. Optional intermediate golden tensors for staged debug

Recommended multi-shape additions:

1. Keep deterministic seeds for reproducibility.
2. For segmented or routed layouts, also dump metadata golden files (`expert_count`, `expert_start`, mapping arrays) so mismatch triage is fast.
3. For top-k or routing, use stable-sort behavior explicitly in Python to avoid tie-order ambiguity across runs.

Common multi-shape pattern:

1. Support generating multiple shapes from one invocation (`--cases`).
2. Support filtered generation for one case (`--case`) when debugging.
3. Keep shape tuples in one canonical source (`generated_cases.*` or equivalent) consumed by both data generation and executable launch.

### 2.1. Shape-source pitfall: `gen_data.py` must read the case manifest

**Failure mode (observed in the MoE family):** `run.sh` invokes `generate_cases.py` with `--cases "..."` to rewrite `build/generated_cases.{h,json}`, then invokes `gen_data.py`. The C++ binary picks up the new shape via the regenerated header, but `gen_data.py` had its own hardcoded `kT/kH/...` constants and silently ignored the manifest. Result: input/golden files are still at the default shape while the binary expects the override shape — bytes-vs-shape mismatch, validator fails or (worse) reads stale bytes from a previous run.

Rules to prevent this:

1. **Single source of truth.** `gen_data.py` MUST read shape constants from the same manifest the C++ side consumes (`build/generated_cases.json` or equivalent). Do not duplicate the default shape as a Python literal that can drift.
2. **First-case wins, mirroring the header.** When the header exposes single-case aliases (e.g. `kMoeT = kGeneratedMoeCases[0].t`), `gen_data.py` must use the first JSON entry too, applying any same-named transforms the header applies (e.g. `kTopK = max(2, _kTopK)` for TSORT32-floored kernels — keep this rule next to the load, with a comment pointing at the matching `main.cpp` line).
3. **Fallback is allowed but must match the header default.** If the JSON is missing, fall back to the hardcoded defaults — but those defaults MUST equal the `DEFAULT_CASES` in `generate_cases.py`. Drift between the two is the bug class this rule prevents.
4. **Path convention.** `gen_data.py` lives at `<kernel>/scripts/gen_data.py`; the family manifest lives at `<family>/build/generated_cases.json`. Resolve with `Path(__file__).resolve().parents[2] / "build" / "generated_cases.json"` for family-style layouts, or `parents[1] / "generated_cases.json"` for self-contained layouts (flash_atten).
5. **Order in `run.sh`.** `generate_cases.py` MUST run before `gen_data.py`; `gen_data.py` MUST run before `rm -rf build` of the per-kernel build dir (or the per-kernel `rm` must target a different dir than the manifest). The MoE family avoids the collision because `<family>/build/` ≠ `<kernel>/build/` — preserve that distinction in new families.

Reference fix: each MoE `gen_data.py` (e.g. [kernels/automode/a2a3/MoE/router_matmul/scripts/gen_data.py](../kernels/automode/a2a3/MoE/router_matmul/scripts/gen_data.py)) now loads the first JSON entry into `kT/kH/kF/kE/kTopK` and falls back to the historical hardcoded defaults only when the JSON is absent.

## 3. `main.cpp` contract (what to validate)

Minimum behavior:

1. Read input/golden files generated by Python.
2. Launch kernel(s) on stream.
3. Synchronize stream.
4. Copy outputs back.
5. Compare against golden and print explicit pass/fail.

Recommended comparison policy:

1. Elementwise exact compare for integer/index outputs.
2. Tolerance-based compare for floating-point outputs.
3. On failure, print first mismatch index and values (or write a detailed diff file).

Example practice from structured kernels:

1. Validate structural outputs and numerical outputs separately (example: routing metadata first, tensor values second).
2. For multi-stage pipelines, optionally compare per-stage intermediates before final output.

Example practice from case-driven kernels:

1. Keep case-name based execution so one binary can run default set, one filtered case, or explicit custom cases.
2. Support optional debug switches for intermediate dumps when investigating numerical drift.

## 4. `run.sh` contract (CLI + flow)

Required CLI baseline:

1. `-r|--run-mode`
2. `-v|--soc-version`
3. Optional `-n|--npu` (default `0`)

Recommended optional flags (copy from existing case-driven kernels when useful):

1. `-c|--case` for single named case
2. `-a|--cases` for semicolon-separated case tuples
3. `--debug` / `--intermediate` style toggles for diagnostics

Execution order (known-good):

1. Parse args + validate SoC/mode compatibility.
2. Generate cases/data (if used).
3. Configure environment (`LD_LIBRARY_PATH` for simulator path).
4. Configure/build (`cmake`, `make`).
5. Run binary with forwarded selectors (`--case`/`--cases`/etc).

Source examples:

1. Flash-attention run script: [kernels/automode/a2a3/flash_atten/run.sh](../kernels/automode/a2a3/flash_atten/run.sh)
2. MoE sub-kernel run script: [kernels/automode/a2a3/MoE/gather/run.sh](../kernels/automode/a2a3/MoE/gather/run.sh)
3. MoE end-to-end script: [kernels/automode/a2a3/MoE/full_moe_separate/run.sh](../kernels/automode/a2a3/MoE/full_moe_separate/run.sh)

## 5. How to update `run_all.sh` when adding a new kernel

File to edit:

1. [kernels/automode/a2a3/run_all.sh](../kernels/automode/a2a3/run_all.sh)

Top-level kernel addition:

1. Add the new kernel folder name to `ALL_KERNELS=(...)`.
2. If the kernel has no special args, no further code change is needed (default path already does `run_one <kernel> <dir>`).

Kernel-specific arg forwarding:

1. If your kernel supports shape overrides, add a dedicated option following existing patterns (`--cases-moe`, `--cases-flash-atten`).
2. Add a variable default near the top (example: `CASES_MY_KERNEL=""`).
3. Parse `--cases-my-kernel` in the arg parser.
4. In the dispatch `case`, route that value into `run_one "my_kernel" "${kernel_dir}" -a "${CASES_MY_KERNEL}"` (or the kernel's own flag).
5. Document the new option in the header usage block at the top of `run_all.sh`.

Family-style addition (for kernels with sub-kernels):

1. Add family name to `ALL_KERNELS`.
2. Add/maintain sub-kernel names list (same shape as `MOE_SUBKERNELS=(...)`).
3. Reuse `run_moe_family` style helper or add a sibling helper if arg semantics differ.
4. Keep summary output unchanged so pass/fail aggregation remains consistent.

Checklist before sending for review:

1. `run_all.sh --help` shows new kernel/options.
2. New kernel appears in `Known: ...` list and usage comments.
3. Running with `--kernels <new_kernel>` reaches that kernel's `run.sh`.
4. Running with full default set still aggregates summary correctly.

## 6. Review-time evidence to include in docs updates

When documenting a newly added kernel test harness in `docs_for_ai/`, record:

1. Exact run command used (example: `bash run.sh -r npu -v Ascend910B1`).
2. Which outputs were compared (final only or per-stage too).
3. Whether comparison was exact or tolerance-based.
4. Any runner wiring added to [kernels/automode/a2a3/run_all.sh](../kernels/automode/a2a3/run_all.sh).

This keeps the doc useful for future kernel authors and avoids repeating trial-and-error.

## 7. Exit codes

Exit-code contract:

1. Host drivers (`main.cpp`) MUST return non-zero when any validation fails. `run_all.sh` treats a child `run.sh` exit code of `0` as PASS and any non-zero as FAIL, so returning `1` (or another non-zero) on failure is required for reliable aggregation.

## 8. `default_cases` guidance:

1. Provide a small, representative `default_cases` set that `run.sh` and `run_all.sh` will run when no `--case`/`--cases` args are supplied. This ensures the broad CI-style `run_all.sh` exercises a meaningful baseline without per-developer arguments.
2. Document the `default_cases` in your kernel's `README.md` and wire `run.sh` so it falls back to the `default_cases` when no cases are provided on the CLI.
3. Keep `default_cases` small (1–3 cases) but representative (covering important tile sizes / edge tails) to keep aggregate runs fast while still exercising important code paths.

Add both requirements to the PR description when proposing new kernels so reviewers can verify `main.cpp` exit behavior and `run.sh` default case behavior quickly.
