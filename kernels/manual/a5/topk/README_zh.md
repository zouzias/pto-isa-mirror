# A5 TopK 算子（框架）

本目录为 **Ascend A5** 上的 TopK 示例工程；**设备侧算法**写在 **`draft.cpp`**，与 `kernels/manual/a2a3/topk` 的排序归并路线不同。

## 当前用例

- 输入：**`[1, 2048]`** `uint16` key（`input/keys.bin`）
- 输出：**Top-512** 个下标（按 **uint16 全序**取最大的 512 个 key；**不要求**对输出做任何排序，顺序任意即可）
- 校验：仅在 host 上比较「取值多重集合」是否与 golden 一致（`scripts/gen_data.py` 生成 `golden_topk_multiset.bin` 作参考）

## 算法概要（`draft.cpp`）

针对 **uint16 可排序 key** 的 **Radix-Select TopK**（2 字节）：

1. **`THISTOGRAM<true>`**：对全体输入做 **MSB** 直方图，累加到 `chistMSB`。
2. **从 bin 255 向下**做前缀，找到累计个数 ≥ TopK 的 **MSB 胜出桶** `msbWinner`。
3. **`remainK = TopK - sum(H[b] for b > msbWinner)`**（UB tile）；LSB pass 始终执行，LSB 胜出由 tile 上 TCMPS/TSEL 与 `remainK` 合并（如 0→LSB=255）。
4. **`THISTOGRAM<false>`**：用 `idxFilter` 标定 MSB，只对 **MSB == msbWinner** 的 key 做 **LSB** 直方图 → `chistLSB`，再扫出 **LSB 胜出桶** `lsbWinner`。
5. 将 `(msbWinner, lsbWinner)` **打包成 uint16 阈值**，用 **`TGATHER`** 的 **`CmpMode::GT`** 与 **`EQ`**（`int16_t` tile 以匹配 A5 的 b16 比较路径）收集索引；`TADDS` 加上 tile 内 `base` 得到全局下标。
6. 输出 **TopK 个索引**（**不要求**按 key 排序；EQ 阶段可能有重复需调用方裁剪）。

## 目录结构

```
kernels/manual/a5/topk/
├── CMakeLists.txt    # 编译 draft.cpp -> libtopk_kernel.so（dav-c310-vec）
├── draft.cpp         # Radix-Select TopK 设备实现 + `LaunchRadixTopKDraft<512>`
├── main.cpp          # ACL：读 keys、launch、多重集合校验
├── scripts/gen_data.py
├── run.sh            # 先 gen_data，再 cmake/build/run
└── README_zh.md
```

## 可参考的仓库内代码

- A5 `TMRGSORT` 与仅数值 TopK 路径：`tests/npu/a5/src/st/testcase/tmrgsort/`
- A5 `TSORT32` / `TGATHER`：`tests/npu/a5/src/st/testcase/tsort32/`、`tgather/`
- A2/A3 完整 TopK 示例（算法思路参考，勿假定可直接迁移）：`kernels/manual/a2a3/topk/`

## 构建与运行

需已 `source` CANN 的 `set_env.sh`，**`bisheng` 在 `PATH` 中**，`ASCEND_HOME_PATH` 已设置。

**仿真**（与本仓库其他 A5 manual 示例一致，用 **`Ascend910_9599`**，不要用 `Ascend310P*`）：

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
