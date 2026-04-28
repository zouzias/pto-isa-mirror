# Indexer（MXFP8 GEMM → ReLU → scale → reduce → TopK）規格

依先前共識整理：**單核優先**；GEMM 對齊 **`kernels/manual/a5/matmul_mxfp8_performance`** 的 **`TMATMUL_MX`** 與資料/scale 佈局。

---

## 1. 端到端資料流

| 步驟 | 運算 | 備註 |
|------|------|------|
| 1 | **MXFP8 `TMATMUL_MX`** | 輸出 **bf16** `C`，與 demo 一致 |
| 2 | **ReLU** | 對 `C` elementwise（Vec） |
| 3 | **× scale** | `scale` 形狀 **`[2, 64, 1]`**，廣播到 `L` |
| 4 | **Reduce sum** | 對 **head 維（dim=1）** 求和 → `[2, L]` |
| 5 | **TopK** | `k=2048`，沿 `L` → **indices `[2, 2048]`** |

其中 **`L = 131072`**，`Q` 為 **`[2, 64, 128]`**，`K` 為 **`[131072, 128]`**。

---

## 2. GEMM 與 demo 維度對齊（`C = A×B`，MX 版）

與 **`matmul_mxfp8_performance/README_zh.md`** 一致：

- 數學：**`C = (scaleA ⊗ A) × (scaleB ⊗ B)`**（`⊗` 為按 MX 規則的塊縮放，**`k` 維每 32 列一組 scale**）。
- **Data**
  - **`A`**：`m×k`，FP8 **ND**（e5m2 等，以實際為準）。
  - **`B`**：儲存為 **`n×k`** **ND**（與「數學上的 `k×n`」對應同一組元素，**demo 與 `main.cpp` 中 `bFileSize = k*n`** 一致；實作時以 **`mxmatmul_performance_kernel.cpp`** 與 **`gen_data.py`** 為準）。
- **Scale**
  - **`scaleA`**：`m × (k/32)`（kernel 內 **`SCALE_FACTOR = 32`**）。
  - **`scaleB`**：`(k/32) × n`（README 表為 **`n×scaleK`**，`scaleK = k/32`）。

本 Indexer 取 **`k = 128`** → **`scaleK = 128/32 = 4`**。

---

## 3. 把 `Q`、`K` 映到 `m, k, n`

數學上要算的是（單個 batch）：

\[
\text{scores}[h, \ell] = \sum_{d=0}^{127} Q[h,d]\, K[\ell,d]
\quad\Rightarrow\quad
C = Q \times K^\top
\]

即 **`A = Q`** 為 **`m×k = 64×128`**，**`B = K`** 在矩乘裡充當 **`k×n`** 時，**`n = 131072`**。

因此：

| 角色 | 形狀 | 說明 |
|------|------|------|
| **A（左側）** | **64 × 128** | 單個 batch 的 `Q[b]` |
| **B（右側）** | **128 × 131072** | `K` 的語義（`K` GM 可為 `[131072,128]` 行主序，與 **`k×n`** 一致） |
| **C** | **64 × 131072** | 單 batch 的 bf16 分數 |

**`batch = 2`** 的兩種實作選型（建議先簡單後優化）：

1. **兩次 launch（易對 golden）**  
   - `b=0`：`A = Q[0]`，`C₀` 為 `[64, 131072]`。  
   - `b=1`：`A = Q[1]`，`C₁` 同形。  
   - 拼成 **`[2, 64, 131072]`** 再送 Vec。

2. **一次 launch（m 維合併）**  
   - **`A`**：**`128 × 128`**，上塊 64 行為 `Q[0]`，下塊 64 行為 `Q[1]`。  
   - **`scaleA`**：**`128 × 4`**（`m=128`）。  
   - **`C`**：**`128 × 131072`**，再切成兩個 **`64 × 131072`**。

**Scale 張量形狀（與 demo 一致）**

- **`scaleA`**：`m × 4`（例如 **`64×4`** 或 **`128×4`**）。  
- **`scaleB`**：**`4 × 131072`**（`(k/32)×n`）。

---

## 4. MXFP8 與「後處理 scale `[2,64,1]`」的分工

- **MXFP8 的 `scaleA` / `scaleB`**：塊指數/縮放，**在 `TMATMUL_MX` 內**與 FP8 資料一起參與乘加，輸出 **bf16**。  
- **使用者給的 `scale [2,64,1]`**：作用在 **bf16 分數上**（ReLU 之後），語意獨立；**不要**與 MX 的 `scaleA/B` 混成同一個張量，除非你做過嚴格數學等價證明。

---

## 5. 實作順序建議

1. 跑通 **`matmul_mxfp8_performance`**（`scripts/gen_data.py` + `run.sh`），對齊本機 **GM bin 佈局**與 **`TMATMUL_MX`**。  
2. 把 **`gen_data.py`** 的維度改成 **`m∈{64,128}, k=128, n=131072`**，生成 **`scaleA` / `scaleB`** 與 golden（或先用 PyTorch/CPU 參考）。  
3. Kernel 輸出 bf16 **`C`** 後，在 **Vec** 上接：**ReLU → × `[2,64,1]` → sum over head → TopK(2048)**（可重用 **`topk_ub`** 類思路，鍵類型需按 bf16 分數重選）。

### 5.1 本地跑 A5 sim（實測備註）

| 步驟 | 結果（本倉庫一次實跑） |
|------|-------------------------|
| `python3 scripts/gen_data.py`（在 `matmul_mxfp8_performance/`） | **需依賴 `ml_dtypes`**；可用倉庫根目錄 `python3 -m venv .venv` + `pip install ml_dtypes numpy` 後再執行。 |
| `bash run.sh -r sim -v Ascend910_9599` | **編譯失敗**：`bisheng` 編譯 **`*_kernel.cpp`** 時，`pto/npu/a5/TRowReduce.hpp` → `<cmath>` → 系統 **`math-vector.h`**，在 **aarch64 主機**上出現 **`__neon_vector_type__` is not supported on targets missing 'neon'**（同類錯誤見 **`tests/script/run_st.py -r sim -v a5 -t tmatmul`**）。 |
| 替代 | 在 **x86_64** 或官方推薦的 **CANN 交叉編譯 / 仿真環境** 跑 A5 sim；或先跑 **`python3 tests/run_cpu.py --clean`** 驗證 **CPU 模擬器**（覆蓋通用 `tmatmul`/`tquant` 等，**非** AICore `TMATMUL_MX` 真路徑）。 |

---

## 6. 相關路徑

| 路徑 | 內容 |
|------|------|
| `kernels/manual/a5/matmul_mxfp8_performance/` | MXFP8 GEMM 參考實作 |
| `include/pto/npu/a5/TQuant.hpp` | `TQUANT` MXFP8 |
| `kernels/manual/a5/topk_ub/` | TopK 手動範例（維度與本規格不同，僅供思路） |
