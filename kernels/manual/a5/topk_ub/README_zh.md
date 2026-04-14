# A5 TopK — 本地 UB radix 变体（`topk_ub`）

本目录与上游仓库里的 **`kernels/manual/a5/topk`** 脚手架**分开维护**，便于在 `git pull cann/pto-isa` 时保留你的实验实现（单次 GM 载入、全宽直方图/GATHER、`TCONCAT` 等）。

- 上游参考：[`../topk/README_zh.md`](../topk/README_zh.md)
- 构建与运行与 `topk` 相同，可执行文件名为 **`topk_ub`**：

```bash
cd kernels/manual/a5/topk_ub
bash run.sh -r sim -v Ascend910_9599
```

- 数据由本目录下 `scripts/gen_data.py` 生成，输出在 **`input/`、`output/`**（与 `../topk` 互不覆盖）。
