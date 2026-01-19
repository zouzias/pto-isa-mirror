# Demo: PyPTO → PTO MLIR → ptoas → NPU

This demo provides a minimal end-to-end flow:

1. `demo/pyPTO/vec_add.py` (Python) emits `demo/pto/vec_add.pto` (PTO MLIR-like text)
2. `ptoas` converts `demo/pto/vec_add.pto` into `demo/bin/vec_add.bin` using BiSheng (`bisheng` / `ccec`)
3. `demo/bin/vec_add.bin` runs the kernel via ACL (NPU mode)

## Quick run (NPU)

Prereqs:

- Ascend CANN installed and `ASCEND_HOME_PATH` set
- `bisheng` available on `PATH` (provided by CANN)

Run:

```bash
chmod +x demo/run_vec_add_npu.sh
./demo/run_vec_add_npu.sh
```

Outputs:

- `demo/out/out.bin` (raw 64x64 `f32` values for the demo)

If you run commands manually, make sure Python can find `python/pypto`:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
```

## CPU-sim debug (tile dumps)

```bash
python3 -m pypto pydb demo/pto/vec_add.pto --dump %t2
```
