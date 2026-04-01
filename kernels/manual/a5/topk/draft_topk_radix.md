# `draft.cpp`：Radix-Select Top-K 核心說明

本文件說明 `kernels/manual/a5/topk/draft.cpp` 中 **`RunRadixTopKDraft`** 的設計：在 Ascend A5 上以 PTO 指令做 **2 位元組 key（`uint16`）** 的 **radix 風格 Top-K 索引選取**，最後把 **`TopK` 個下標** 寫入全域記憶體。

---

## 1. 目標與假設

| 項目 | 說明 |
|------|------|
| 輸入 | `src`：`N` 個 `uint16` key（範例 `N = 2048`） |
| 輸出 | `outIdx`：`TopK` 個 `uint32_t` **索引**（0 … `N-1`） |
| 排序 | 輸出順序**不要求**與 key 排序一致；若 threshold 上有多個相等 key，EQ 段可含多個同值下標（multiset 語意由 host 驗證） |
| 形狀 | 編譯期固定：`kTileCols = 256`，`kLoop = ceil(N / 256)`（例：8 tile） |

演算法把每個 key 拆成 **高 8 位元（MSB）** 與 **低 8 位元（LSB）**，用兩次 **256-bin 累積直方圖** + **winner 選桶** 定出 **k-th largest 對應的 packed threshold**，再以 **TGATHER 比較收集** 把 GT 與 EQ 的索引分段，最後 **TCONCAT** 合併並 **TSTORE**。

---

## 2. 整體管線（鳥瞰）

```mermaid
flowchart TB
    subgraph pass1["Pass 1：MSB 直方圖"]
        A[THISTOGRAM true 逐 tile] --> B[chistMSB 累積]
    end
    subgraph win_msb["MSB winner + remainK"]
        B --> C[FindWinnerBucketDescending: thr = N-TopK]
        C --> D[msbWinnerSaved: 原始 TROWMIN 桶]
        C --> E[WinnerBinU8FromSelsMin: TADDS -1 等 → TGATHER 用 bin]
        E --> F[RemainK = thr_msb - C_msb[w]]
    end
    subgraph pass2["Pass 2：LSB 直方圖（MSB 篩選）"]
        D --> G[FillIdxMsb: idxFilter = 原始 MSB 位元組]
        G --> H[THISTOGRAM false 逐 tile]
        H --> I[chistLSB 累積]
    end
    subgraph win_lsb["LSB winner + packed threshold"]
        I --> J[LsbHist… + TROWMIN + broadcast]
        L[F remainKTile] --> J
        J --> K[lsbWinnerBin]
        D --> L2[msbWinnerSaved]
        K --> M[PackedThreshold = MSB<<8 | LSB]
        L2 --> M
    end
    subgraph gather["索引收集與輸出"]
        M --> N[GT pass: TGATHER 索引 > thr]
        M --> O[EQ pass: TGATHER 索引 == thr]
        N --> P[TCONCAT mergedIdx]
        O --> P
        P --> Q[TSTORE outIdx]
    end
```

---

## 3. 階段說明

### 3.1 Pass 1：`THISTOGRAM<true>`（MSB）

- 輸入以 tile 載入（`LoadTileU16`），對每個 tile 做 **MSB 直方圖**，累加到 `chistMSB`。
- `chistMSB[b]` 為 **升序累積**：key 的 MSB ≤ `b` 的個數。
- **MTE2 / V 同步**：除第一個 tile 外，每輪在 `TLOAD` 前 `wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2)`，避免覆寫 `inTile`；每輪結尾 `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2)`，供下一輪或 LSB pass 使用。

### 3.2 MSB winner 與 `remainK`

`MsbWinnerRemainKAndSaveTile` 做三件事：

1. **`FindWinnerBucketDescending(chistMSB, thr = N - TopK)`**  
   - `TCMPS(GE)`：累積 `C[b] >= thr` 的 mask；`TCI` 產生 bin id；`TSELS` 填每 lane 的候選值。  
   - 語意上取 **最小滿足 `C[b] >= (N-TopK)` 的 MSB bin**（與「第 k 大 key 所在 MSB bucket」一致）。

2. **`msbWinnerSaved`（`WinnerLsbBinU32RowMinBroadcast`）**  
   - 對上述 lanes 做 **`TROWMIN`**，再 **`TGATHER` 廣播** 到 32 lane。  
   - **不做 `TADDS(-1)`**：保留 **raw TROWMIN 桶**，供 **`THISTOGRAM<false>` 的 `idxFilter`**（只統計該 MSB 的 key）以及 **`PackedThresholdU16ViaShlOr` 的高位元組**。

3. **`msbWinnerBin`（`WinnerBinU8FromSelsMin`）**  
   - 對同一組 `msbWinnerLanes` **先 `TROWMIN`，再 `TADDS(-1)`**，再與常數 256 比較、`TSEL` 等，最後 **`TGATHER` 廣播**。  
   - 得到的是 **「`TGATHER` 讀 `C_msb` 時應使用的 bin 索引」**（對應 **C[winner-1]** 語意），用於 **`RemainKMsbFromTiles`**。

4. **`RemainKMsbFromTiles`**  
   - `thr_msb = (uint32_t)(N - TopK)` 廣播；`TGATHER(cwT, chistMSB, winnerBinU32)` 取 **C[w]**；`TSUB(remainKTile, thr_msb, cw)` 得到 **`remain_k = (N-TopK) - C[winner-1]`**，供 LSB 直方圖上 **與 `chistLSB` 比較**。

### 3.3 Pass 2：`THISTOGRAM<false>`（LSB）

- `FillIdxMsbFromWinnerBin(idxFilter, msbWinnerSaved)`：把 **raw MSB winner** 寫入 idx tile（**不是** `WinnerBinU8` 的 -1 後結果）。  
- 僅統計 **MSB 等於該位元組** 的 key，累積到 `chistLSB`。  
- 同步策略與 MSB pass 類似（`EVENT_ID2` ping-pong）。

### 3.4 LSB winner 與 packed threshold

- **`LsbHistGeRemainKToLanes`**：`TCMPS(chistLSB, remainKTile, CmpMode::GT)`（實作為 **嚴格大於** `remain_k`；若與數學上「≥」語意對齊，需與 golden 腳本一致檢查）。  
- **`WinnerLsbBinU32RowMinBroadcast`**：對 LSB lanes 做 **`TROWMIN` + broadcast**，得到 `lsbWinnerBin`。  
- **`PackedThresholdU16ViaShlOr(msbWinnerSaved, lsbWinnerBin)`**：  
  - `TCVT` MSB/LSB 為 `uint16`，`TSHLS` 把 MSB 左移 8，`TOR` 與 LSB 合併，得到 **與 `uint16` 位元模式一致** 的 **單一 threshold**（host 端用 `int16` 解讀 key 時需與此一致）。

### 3.5 GT / EQ 兩段收集與 `TCONCAT`

- **`GatherCmpToTile`**：`LoadTileI16Gather` 把 key 當 **`int16_t`** 載入，`TGATHER` 的 template 參數 **`offset = tileIndex * ValidCols`**（因 offset 須編譯期常數，故用 **`switch (tileIndex)`** 分 `case 0…7`）。  
- 回傳本 tile 命中個數（寫在 concat scratch）；`tileIndex` 在 `0…7` 時 **不再** 做額外 `+= base`（避免重複加 offset）。  
- GT 段：`AppendIndicesScalar` 把 `gtChunk` 內索引 append 到 `gtSeg`，最多 `TopK` 個。  
- EQ 段：cap 為 `TopK - gtCount`，只補足剩餘名額。  
- **`idxGtCnt[0] = gtCount`、`idxEqCnt[0] = eqCount`**（scalar 寫 UB），供 **`TCONCAT_IMPL(mergedIdx, gtSeg, eqSeg, idxGtCnt, idxEqCnt)`**。  
- **`V / MTE3` 同步**後 `mergedIdx.SetValidCol(TopK)`，**`TSTORE(outGlobal, mergedIdx)`**。

---

## 4. UB 記憶體佈局（摘要）

| 區域（約） | 用途 |
|------------|------|
| `0x00000` | `inTile` |
| `0x10000` / `0x14000` / `0x18000` | tile 直方圖、`chistMSB`、`chistLSB` |
| `0x1C000` | `idxFilter`（THISTOGRAM 索引） |
| `0x20000–` | gather 暫存（`GatherCmpToTile`） |
| `0x23000–0x25000` | winner 路徑（mask、index、TSELS、TROWMIN、remainK、`kMsbWinnerSavedUb` 等） |
| `0x28000–` | `gtSeg`、`eqSeg`、`mergedIdx`、`idxGtCnt`/`idxEqCnt`（TCONCAT） |

細部 offset 以原始碼中 `constexpr uint64_t` 為準；**不同 tile 的 scratch 刻意分開**，避免與直方圖或 winner 路徑 alias。

---

## 5. 對外入口

- **`LaunchRadixTopKDraft<TopK>(src, outIdx, stream)`**：啟動 `RunRadixTopKDraft` kernel（目前 `draft.cpp` 末尾 **explicite instantiate** `TopK = 512`）。

---

## 6. 除錯與驗證建議

- Host 端若以 **key  multiset** 比對 golden，需與 **packed threshold、GT/EQ、`TCONCAT` 長度** 同一套語意；`scripts/radix_topk_golden_stats.py` 可協助對齊 **MSB/LSB winner、remain_k、GT/EQ 數量**。  
- 仿真若出現 **multiset mismatch**，常見檢查點：**MSB/LSB winner、remain_k、`LsbHistGeRemainKToLanes` 的 GE/ GT 與 golden 是否一致**、以及 **idxFilter 用 raw MSB 而非 -1 後 bin**。  
- 仿真器可產生 UB 讀寫 / 向量 dump（如 `core0.*.dump`），用於對照指令序與資料。

---

## 7. 相關檔案

| 檔案 | 角色 |
|------|------|
| `main.cpp` | 載入 `keys.bin`、launch kernel、與 golden 比對 |
| `scripts/gen_data.py` | 產生輸入與參考 golden |
| `scripts/radix_topk_golden_stats.py` | 列印理論 winner / threshold / GT\|EQ 統計 |
| `run.sh` | 以 `-r sim` / `-v Ascend910_9599` 等一鍵建置並執行 |

---

*本文件僅描述 `draft.cpp` 的結構與設計意圖；若 PTO 指令語意或裝置行為有版本差異，以實際硬體/模擬器與 `include/pto/npu/a5/` 為準。*
