# chevron 用例 vs tilelang 生成的 PTO 后端：差异与问题分析

对比对象：

- **能 PASS 的本地用例**：`moe_token_permute_chevron`（`moe_token_permute_chevron_kernel.cpp` + `moe_token_permute_body.hpp`），`pto_mix_st`(`dav-c220`) + chevron `<<<16>>>`，已实测 `[ PASSED ]`。
- **tilelang 生成的 pto 后端**：来源 https://getnote.top/lwdpto （前端同一份 `moe_token_permute`，后端选 pto），在 NPU 上失败。

> 决定性事实：PTO 中**不带模板参数的 `SYNCALL()` 默认是 `SyncCoreType::AIVOnly`**（见 `include/pto/common/pto_instr.hpp:52`、`include/pto/common/cpu_stub.hpp:248`）。tilelang 生成代码用的正是这种裸 `SYNCALL()`。

---

## 一、整体对比表

| 维度 | tilelang 生成的 pto 后端（lwdpto） | 能 PASS 的 chevron 用例 | 影响 |
|---|---|---|---|
| SYNCALL 模式 | `SYNCALL()` → 默认 **AIVOnly** | `SYNCALL<SyncCoreType::Mix>()` | 决定性 |
| barrier 位置 | **在 `#if defined(__DAV_C220_VEC__)` 之内** | 在 VEC 守卫**之外**（`#if __DAV_CUBE__ \|\| __DAV_VEC__`） | 决定性 |
| AIC 分支 | **完全没有**，AIC 核空转、不进 barrier | AIC 进入 `RunMixSyncBarrier()` 与 AIV 对齐 | 决定性 |
| 跨核 GM cache 处理 | **无任何 dcci/dsb** | 写后 `dcci+dsb`、读前 `InvalidateWorkspaceGm`(全量 dcci)+`dsb` | 决定性 |
| workspace 每核槽位 | `cid*4`（16B/核，**两核共享一条 32B cache line**） | `cid*kMoeWsSlotStride(=8)`（32B/核，独占 cache line） | 决定性 |
| `cid` 取值 | `get_block_idx()`（不 /2） | `get_block_idx()`（不 /2） | **一致，非差异** |
| 启动方式 | `launch_kernel<<<16>>>` + chevron | 同 | 一致 |
| meta 宏 | 无（靠 bisheng 自动拆 `_mix_aic/_mix_aiv`） | 无（靠 bisheng 自动） | 一致 |

> 注意：`cid` 无需 `/2`，tilelang 与通过版在这点上**完全一致**，因此 cid 不是问题。问题集中在**跨核同步语义**与**跨核内存可见性**。

---

## 二、tilelang 生成代码的具体问题（按严重度）

### 问题 1：SYNCALL 是 AIVOnly，kernel 却被当 MIX 1:2 调度 —— 语义错配

tilelang 没有任何 meta 宏，`dav-c220` 下 bisheng 会自动把 `launch_kernel` 拆成 `launch_kernel_mix_aic` / `_mix_aiv` 并打 `KTYPE=MIX_AIC_MAIN, ratio 1:2` 的 `.ascend.meta`（已在「路线2 dump meta」中实测证实）。于是运行时按 **MIX 1:2** 调度，AIC 核被拉起。但：

- `SYNCALL()` 默认是 AIVOnly 语义；
- 且这条 `SYNCALL()` 写在 `#if defined(__DAV_C220_VEC__)` 内 —— **AIC 核根本不会执行到任何 SYNCALL**。

→ 被调度的 AIC 核永远到不了 barrier，跨核屏障与实际调度不一致。这与本用例最初用 `SYNCALL<AIVOnly>` 失败的根因相同。修法：改 `SYNCALL<Mix>` 且让 AIC 也调用（移出 VEC 守卫）。

### 问题 2：完全没有跨核 cache 一致性处理（dcci/dsb）

算子逻辑：每核先把本核 histogram 写进 GM workspace，barrier 后每核再读全部 16 核 histogram 算全局偏移。这是典型跨核 GM 通信，Ascend 上必须显式维护 cache 一致性：

- 写方：`copy_ub_to_gm` 后 `dcci + dsb(DDR)` 把数据刷出到 DDR；
- 读方：读前 `dcci`(invalidate) + `dsb`，保证读到 DDR 最新值而非本核旧 cache。

tilelang 代码**一个 dcci/dsb 都没有**，只有 `set_flag_pipeline/wait_flag_pipeline`（那是**核内**流水同步，管不了**跨核** cache 可见性）。即便 SYNCALL 修对，读到的他核 histogram 仍可能是脏数据 → 全局偏移算错 → sio/perm 全错。通过版用 `InvalidateWorkspaceGm()`（对全 workspace 逐 cache line dcci）+ 每次写后 dcci/dsb 解决。

### 问题 3：workspace 每核槽位仅 4 个 int（16B），两核共享一条 32B cache line

tilelang 用 `workspace_gm_handle + (cid * 4)`，每核 4 个 int = 16B。相邻两核（如 core0/core1）落在**同一条 32B cache line**，跨核 + cache 操作以 cache line 为粒度，会互相覆盖/丢写。通过版把 stride 提到 **8 个 int = 32B**，每核独占一条 cache line（`kMoeWsSlotStride=8`）。

### 问题 4（连带）：缺少 AIC 侧实现

真 MIX kernel 需要 AIC 分支至少参与等量 barrier。tilelang 只有 `#if defined(__DAV_C220_VEC__)` 单分支，无 AIC 路径，AIC 编出来是空壳。与问题 1 是同一处代码的两个表现。

---

## 三、一句话总结

tilelang 生成的 pto 后端**把一个需要跨核同步 + 跨核 GM 通信的 MIX 算子，当成纯 AIV-only 单核思路生成了**：默认 AIVOnly 的 `SYNCALL()`、无 AIC 分支、无任何 dcci/dsb cache 一致性、workspace 槽位小到跨核共享 cache line。四点叠加，在被自动当 MIX 1:2 调度的 `dav-c220` 上必然算错。通过版逐条补齐：`SYNCALL<Mix>` + AIC 参与 barrier + 完整 dcci/dsb + 每核独占 cache line。

---

## 四、对 tilelang codegen 的修复建议

落点在 `tilelang-ascend-pr1042/src/target/codegen_ascend_pto.cc`：

1. **SyncAll 识别为 Mix 并双分支发射**：`SyncAll()` 在 `dav-c220` MIX 场景应生成 `SYNCALL<SyncCoreType::Mix>()`，且在 AIC（`__DAV_CUBE__`）与 AIV（`__DAV_VEC__`）两个分支都发射等量 barrier。
2. **跨核 GM 读写自动插 cache 操作**：对作为跨核通信介质的 GM buffer，写后插 `dcci + dsb(DDR)`，读前插 `dcci(invalidate) + dsb`。
3. **跨核 workspace 按 cache line 对齐分配**：每核槽位按 32B（8×int32）对齐，避免相邻核共享 cache line。

---

## 五、关键代码位置索引

- 裸 `SYNCALL()` 默认模式：`include/pto/common/pto_instr.hpp:52`、`include/pto/common/cpu_stub.hpp:248`
- `SyncCoreType` 定义：`include/pto/common/type.hpp:273`
- 通过版 body：本目录 `moe_token_permute_body.hpp`（`RunMixSyncBarrier` / `InvalidateWorkspaceGm` / `kUbWsSlotStride`）
- 通过版常量：本目录 `moe_token_permute_common.hpp`（`kMoeWsSlotStride=8` 注释）
- meta 实测与根因：本目录 `MIX_ANALYSIS.md`（路线2 dump meta、路线1 改造）
