# MLIR Development Environment (PTO Dialect)

This repo contains a PTO dialect definition under `mlir/Dialect/PTO/IR/` (`*.td` files). If you want to **build and
iterate on the dialect with upstream MLIR tools** (`mlir-opt`, `mlir-tblgen`, etc.), set up an LLVM/MLIR development
environment first.

## Option A: Install prebuilt LLVM/MLIR (fastest)

If your distro provides MLIR packages, install:

- `mlir-tblgen`
- `mlir-opt`
- `llvm-config`
- `clang`/`clang++` (matching the LLVM version)

Verify:

```bash
mlir-opt --version
mlir-tblgen --version
llvm-config --version
```

## Option B: Build LLVM/MLIR from source (recommended for dialect work)

1. Get `llvm-project` (pick a tag you want to track, e.g. `llvmorg-18.x.y`).
2. Configure a build with MLIR enabled:

```bash
cmake -S llvm-project/llvm -B build-llvm \
  -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-llvm -j
```

3. Add tools to `PATH`:

```bash
export PATH="$PWD/build-llvm/bin:$PATH"
```

Verify:

```bash
mlir-opt --version
mlir-tblgen --version
```

## Next steps in this repo

- PTO dialect TableGen: `mlir/Dialect/PTO/IR/PTODialect.td`, `mlir/Dialect/PTO/IR/PTOTypeDefs.td`,
  `mlir/Dialect/PTO/IR/PTOOps.td`
- Textual `.pto` examples (used by the demo toolchain): `demo/pto/*.pto`
- Demo assembler pipeline: `python/pypto` → `.pto` → `ptoas` → generated C++

