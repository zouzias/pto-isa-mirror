# `ptoas`: PTO Assembler (Demo Toolchain)

`ptoas` is a small demo tool that converts a `.pto` file written in the repo’s MLIR-compatible PTO dialect form into a
host executable (`.bin`) compiled with BiSheng (`bisheng` / `ccec`).

## What it does today

- Parses a constrained MLIR-like subset (single `func.func @main`, `pto.alloc_tile`, `pto.tload`, tile ops, `pto.tstore`)
- Generates a C++ kernel using PTO Tile Lib intrinsics (`include/pto/pto-inst.hpp`)
- Builds with CMake + `bisheng` and emits a runnable binary that launches the kernel with ACL

## Usage

From repo root:

```bash
python3 -m ptoas demo/pto/vec_add.pto -o demo/bin/vec_add.bin --soc a3 --run-mode npu
mkdir -p demo/out
demo/bin/vec_add.bin --out-dir demo/out
```

Generate C++ sources without building:

```bash
python3 -m ptoas demo/pto/vec_add.pto --emit-dir build/ptoas-src --kernel-name vec_add
```

Options:

- `--soc a3|a5`: selects A2/A3 vs A5 compile flags
- `--run-mode npu|sim`: links against `runtime` vs `runtime_camodel`
- `--build-dir ...`: where generated sources and build artifacts go
- `--emit-dir ...`: write generated sources and exit (no BiSheng/CANN required)

## Inputs / outputs

- Input: `.pto` (text) using the form documented in `docs/grammar/PTO-AS.md`
- Output: `foo.bin` (host executable) plus intermediate generated sources under `build/ptoas/`
- Run output: `<out-dir>/<memref>.bin` for each output memref (for example `out.bin`)
