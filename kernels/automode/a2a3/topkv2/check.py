ls -lh input output
python3 - <<'PY'
import numpy as np

kRows = 4
kCols = 1280
kTopK = 512

src  = np.fromfile("input/input_src.bin",   dtype=np.float32).reshape(kRows, kCols)
goldV = np.fromfile("output/golden_val.bin", dtype=np.float32).reshape(kRows, kTopK)
outV  = np.fromfile("output/output_val.bin", dtype=np.float32).reshape(kRows, kTopK)
goldI = np.fromfile("output/golden_idx.bin", dtype=np.uint32 ).reshape(kRows, kTopK)
outI  = np.fromfile("output/output_idx.bin", dtype=np.uint32 ).reshape(kRows, kTopK)

print("shapes:", src.shape, goldV.shape, outV.shape, goldI.shape, outI.shape)

for r in range(kRows):
    print(f"--- row {r} ---")
    print("gold val first 10:", goldV[r, :10])
    print("out  val first 10:", outV[r,  :10])
    badV = np.where(np.abs(goldV[r] - outV[r]) > 1e-3)[0]
    badI = np.where(goldI[r] != outI[r])[0]
    print("num bad val:", len(badV), " num bad idx:", len(badI))
    if len(badV):
        for i in badV[:10]:
            v = outV[r, i]
            exists = np.any(np.isclose(src[r], v, atol=1e-3))
            print(i, "gold", goldV[r, i], "out", v, "exists_in_input?", exists)
    print("true top first 10:", (-np.sort(-src[r]))[:10])
PY
