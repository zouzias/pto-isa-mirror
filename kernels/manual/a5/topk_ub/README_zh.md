# A5 TopK — 本地 UB radix 变体（`topk_ub`）

本目录与上游仓库里的 **`kernels/manual/a5/topk`** 脚手架**分开维护**，便于在 `git pull cann/pto-isa` 时保留你的实验实现（单次 GM 载入、全宽直方图/GATHER、`TCONCAT` 等）。

- 上游参考：[`../topk/README_zh.md`](../topk/README_zh.md)
- 构建与运行与 `topk` 相同，可执行文件名为 **`topk_ub`**：

```bash
cd kernels/manual/a5/topk_ub
bash run.sh -r sim -v Ascend950PR_9599
```

- 数据由本目录下 `scripts/gen_data.py` 生成，输出在 **`input/`、`output/`**（与 `../topk` 互不覆盖）。

## 性能分析（A5 sim, 2026-05-14）

分析对象：`build/OPPROF_20260514154240_NYBGOWGNYQBFCTNM`（`core0.veccore0`）。

说明：

- `VF01~VF17` 的 IPC 为 cycle-accurate（来自 retire/issue 队列日志）。
- `VF18~VF26` 的 IPC 为 trace window 估算值（`ccu.vec_issque` retire 记录在后段截断）。

### 按 Phase 聚合

| Phase | VF 范围 | 耗时 (us) | IPC | 对应 PTO ISA（主） | 备注 |
|---|---:|---:|---:|---|---|
| Phase1 | VF01-VF02 | 1.358 | 1.503 | `TASSIGN/TLOAD/THISTOGRAM(BYTE_1)/TMOV` | MSB 直方图主算段 |
| Phase2 | VF03-VF13 | 0.838 | 0.447 | `TCMPS/TCI/TSELS/TROWMIN/TGATHER/TSUB` | Winner MSB + remainK（控制流密集） |
| Phase3 | VF14 | 1.464 | 1.481 | `TCVT/THISTOGRAM(BYTE_0)/TMOV` | LSB 直方图主算段 |
| Phase4 | VF15-VF17 | 0.359 | 0.148 | `TCMPS/TSELS/TROWMIN/TGATHER/TSHLS/TOR` | Winner LSB + packed threshold |
| Phase5 | VF18-VF26 | 6.803 | 0.864* | `TGATHER<GT>/TGATHER<EQ>/TCONCAT_IMPL/TSTORE` | 全宽 gather + concat + store |

\* Phase5 的 IPC 含估算段（VF18~VF26）。

### 关键 VF 热点（用于回归定位）

| VF | PC | 耗时 (us) | IPC | 主导 RV 指令簇 | 对应代码语义 |
|---|---|---:|---:|---|---|
| VF02 | `0x10d0d16c` | 1.327 | 1.527 | `RV_VCVT_I2I/RV_VADD/RV_VMOV` | `Phase1` 的 MSB histogram 主体 |
| VF14 | `0x10d0d764` | 1.464 | 1.481 | `RV_VCVT_I2I/RV_VADD/RV_VMOV` | `Phase3` 的 LSB histogram 主体 |
| VF24 | `0x10d0dcc8` | 2.891 | 0.976* | `RV_VLD/RV_VADD/RV_VCMP_GT` | `Phase5` 的 `TGATHER<GT>` 主体 |
| VF25 | `0x10d0dd14` | 2.959 | 0.966* | `RV_VSTUR/RV_VSQZ/RV_VLD` | `Phase5` 的 `TGATHER<EQ>` 主体 |
| VF16 | `0x10d0d85c` | 0.247 | 0.080 | `RV_VLD/RV_PLT/RV_VLOOPv2` | `Phase4` 小段，控制/谓词占比高 |

\* VF24/VF25 IPC 为 trace window 估算值。
