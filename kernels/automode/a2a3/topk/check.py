ls -lh input output
python3 - <<'PY'
import numpy as np

src = np.fromfile("input/input_src.bin", dtype=np.float32)
gold = np.fromfile("output/golden_val.bin", dtype=np.float32)
out = np.fromfile("output/output_val.bin", dtype=np.float32)

print("sizes:", src.size, gold.size, out.size)
print("gold first 20:", gold[:20])
print("out  first 20:", out[:20])

bad = np.where(np.abs(gold - out) > 1e-3)[0]
print("num bad:", len(bad))
print("first bad idx:", bad[:20])

if len(bad):
    for i in bad[:20]:
        v = out[i]
        exists = np.any(np.isclose(src, v, atol=1e-3))
        print(i, "gold", gold[i], "out", v, "exists_in_input?", exists)

print("true top first 20:", (-np.sort(-src))[:20])
PY