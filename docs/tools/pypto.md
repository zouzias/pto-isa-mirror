# `pypto`: Python Frontend (AST → PTO MLIR text)

`pypto` is a tiny Python frontend that:

- parses a restricted Python kernel function using the Python `ast` module
- emits a `.pto` file in an MLIR-compatible textual form (the PTO MLIR dialect described in `docs/reference/pto-mlir-dialect.md`)
- optionally runs a minimal CPU simulator to dump tile values (“pydb”)

## Emit a `.pto` file

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pypto emit demo/pyPTO/vec_add.py -o demo/pto/vec_add.pto
```

## CPU-sim debug / dumps

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pypto pydb demo/pto/vec_add.pto --dump %t2
```

## Kernel authoring model

The demo frontend expects exactly one function decorated with `@pypto.kernel`, with `MemRefType(...)` annotations on all
arguments, and a restricted body that uses the DSL markers:

- `t0 = tload(a)`
- `t2 = tadd(t0, t1)`
- `tstore(out, t2)`

See `demo/pyPTO/vec_add.py`.

Note: the emitted `.pto` uses a **register-style** form where tile SSA values are allocated once via
`pto.alloc_tile` and subsequent `pto.*` tile ops use an `ins(...) outs(...)` (DPS) spelling.
