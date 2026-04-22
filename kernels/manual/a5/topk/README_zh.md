# A5 TopK 算子（框架）

本目录为 **Ascend A5** 上的 TopK 示例工程；**设备侧**写在 **`draft.cpp`**，与 `kernels/manual/a2a3/topk` 的排序归并路线不同。

与 **`kernels/manual/a5/topk_ub`** 对齐：**共五个 `Phase*`**，所有 PTO 指令（含 `TASSIGN` / `TLOAD` 等）只出现在这些 Phase 中。本工程为 **分块从 GM 读入**（N = 2048，每块 256 列）；`topk_ub` 为 **整段 key 在 UB**、整段比较 `TGATHER`。

## 当前用例

- 输入：**`[1, 2048]`** `uint16` key（`input/keys.bin`）
- 输出：**Top-512** 个下标（按 **uint16 全序**取最大的 512 个 key；**不要求**对输出下标排序）
- 校验：host 上比较「`keys[out[i]]` 的多重集合」是否与 golden 一致（`scripts/gen_data.py` 生成 `golden_topk_multiset.bin`）

## 流程概要（`draft.cpp` 与五个 Phase）

针对 **uint16 可排序 key** 的 **Radix-Select TopK**（2 字节）：

1. **Phase1** — UB `TASSIGN`，按 tile 从 GM `TLOAD`，`THISTOGRAM<HistByte::BYTE_1>`（MSB）累加成 `chistMSB`（`TEXPANDS` 清零后每块 `TADD`）
2. **Phase2** — 在 `chistMSB` 上找 MSB 胜出（`TCMPS` / `TCI` / `TSELS`），`TROWMIN`+`TGATHER` 得到 raw MSB 与 `WinnerBinU8` 路径；`remainK` 用 `TGATHER` 取 `C[w]` 与 `TSUB`（对 u32 减 1 用 `TEXPANDS`+`TSUB`，避免老 CANN 上 `TADDS` 对 u32 的限制）
3. **Phase3** — `TCVT` 将 raw MSB 写入 `idxFilter`；再按 tile `TLOAD`，`THISTOGRAM<BYTE_0>`（LSB，带 MSB 过滤）累加成 `chistLSB`
4. **Phase4** — LSB 侧在 `chistLSB` 与 `remainK` 上找胜出 bin，`TCVT` / `TSHLS` / `TOR` 得到 **16 位 packed 阈值**（与 key 同 bit 含义供比较路径使用）
5. **Phase5** — 每 tile 上 **比较 `TGATHER`（GT/EQ）**：阈值为 **uint16 tile**（UB `kRemainUbOut`），**全局下标**由指令的 **offset 实参**（本 tile 在 GM 上的 `base`）与 tile 内相对下标一起决定；将各 tile 结果 **scalar 追加** 到 GT/EQ 段，再 **五参 `TCONCAT_IMPL`（`TConcatIdx`）**（计数字为 **字节数**）合并，`TSTORE` 输出

## 目录与构建相关

- `CMakeLists.txt` 使用 **`BEFORE` 本仓库 `include/`**，使 `pto-isa` 头文件优先于 `ASCEND_HOME_PATH` 中的旧版头（如 `TCONCAT` 五参、`THISTOGRAM` 的 `HistByte` 等），与 `topk_ub` 一致
- 其余文件：`main.cpp`、`scripts/gen_data.py`、`run.sh` 等见上表

```
kernels/manual/a5/topk/
├── CMakeLists.txt
├── draft.cpp
├── main.cpp
├── scripts/gen_data.py
├── run.sh
├── README.md
└── README_zh.md
```

## 可参考的仓库内代码

- **同算法 UB 全宽版：** `kernels/manual/a5/topk_ub/`
- A5 `TMRGSORT`：`tests/npu/a5/src/st/testcase/tmrgsort/`
- A5 `TSORT32` / `TGATHER`：`tests/npu/a5/src/st/testcase/tsort32/`、`tgather/`
- A2/A3 TopK 示例（仅作思路参考）：`kernels/manual/a2a3/topk/`

## 构建与运行

需已 `source` CANN 的 `set_env.sh`，**`bisheng` 在 `PATH` 中**，`ASCEND_HOME_PATH` 已设置。

**仿真**（与本仓库其他 A5 manual 示例一致，如 **`Ascend910_9599`**，勿用 `Ascend310P*`）：

```bash
cd kernels/manual/a5/topk
chmod +x run.sh
bash run.sh -r sim -v Ascend910_9599
```

**真机**：

```bash
bash run.sh -r npu -v <你的 A5 板卡对应 SOC 字符串>
```

若可执行文件报缺 `.so`，先 `source set_env.sh` 再跑，保证 `lib64` 等在 `LD_LIBRARY_PATH` 中。
