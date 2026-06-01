# Init Quant / 前重排 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 a8w8 的前重排从当前 correctness-first direct-pack 路线推进到可单阶段验证、可解释性能、可对标
FFN 4097 的 active 多 AIV init-quant 路线。

**Architecture:** 前重排采用 count + prefix + direct expert-major scatter，a8w8 当前不引入 FFN 的多核 sort 子系统。
该路线必须产出与 FFN init-routing 等价的 metadata 和 payload：`expandedRowIdx`、
`tokenPerExpertMatrix`、`cumsumMM`、`preSumBeforeRank`、`dispatchPayload`、`dispatchScale`、
`gmm1InputInt8`、`routingPerTokenScale`。性能收口聚焦 direct scatter 多 AIV 和 PTO Vec quant。

**Tech Stack:** PTO manual A3 mixed AIC/AIV kernel, C++20, CANN runtime, MPI/HCCL peer window, existing
`scripts/run_a3.sh` verification path. No Catlass/AscendC fallback.

---

## 1. Scope

本文档里的“前重排 / init quant”覆盖 MegaMoE 七段中的第一段，以及和 dispatch gather 的边界：

```text
expertIdx / xActiveMask
  -> route count
  -> expert prefix / workerExpertPrefix
  -> expert-major direct pack + dynamic quant
  -> publish/wait tokenPerExpert
  -> cumsumMM / preSumBeforeRank
  -> dispatch gather to gmm1InputInt8
  -> dispatchGroupReady[localExpert]
```

其中 `dispatch gather to gmm1InputInt8` 是前重排和 dispatch/GMM1 的交界。FFN 参考实现也是在 init-routing 和
count/prefix 后做全 AIV 同步，然后按 expert group 搬运并局部放行 GMM1。

## 2. FFN Reference Facts

参考实现路径：

`/mnt/data/ntlab/zy/code/zhangyuan/vllm-ascend-zy/csrc/mc2/dispatch_ffn_combine`

关键事实：

- `moe_init_routing_quant_v2()` 负责 sort、expert token count/cumsum、`srcToDst`、gather dynamic quant。
- `M=4097, topK=2` 时 `totalLength=8194`，大于参考实现的 `sortLoopMaxElement`，不会走 full-load dynamic
  quant 快路径；会走 multi-core sort + srcToDst + gather dynamic quant 路线。
- FFN multi-core sort 是分块局部排序 + 多路归并 + 输出的完整排序子系统，产出全局 expert-major order。
- FFN 的 init-routing 内部多个子阶段使用全 AIV `SyncAll`；跨 rank count/prefix 后也有全 AIV 同步。
- FFN 的 dispatch 到 GMM input 阶段按 local expert group 局部放行 GMM1，不等所有 expert gather 完。

## 3. Current A8W8 Facts

当前 a8w8 不实现 FFN 的多核 sort 子系统，而是使用 counting-sort/direct-scatter 思路：

```text
M3NCountLocalRoutesShard
  -> M3NMergeLocalRouteCounts
  -> M3NRoutePackQuantLocalShardScalar
```

`M3NRoutePackQuantLocalShardScalar` 直接计算：

```cpp
packedRow = blockPrefixPerExpert[globalExpert]
          + workerExpertPrefix[workerId][globalExpert]
          + localOrdinalInWorkerExpert;
```

然后写：

```text
expandedRowIdx[routeIndex] = packedRow
dispatchPayload[packedRow]
dispatchScale[packedRow]
```

已完成 checkpoint 后的当前状态：

- `M3NAssignDispatchWorkers()` 已放开 shape-aware 多 worker；`ffn-v3-4097` debug-stop 11 已验证
  `dispatch_active_aiv_workers=4`、`init_quant_worker_mask=15`。
- dynamic quant 是 scalar row loop，hidden 维没有 PTO Vec tile。
- `M3NMergeLocalRouteCounts()` 在 main AIV 串行 merge worker counts / prefix。
- `M3NGatherDispatchToGmm1InputShard()` 按 localExpert modulo worker 切分，inner tokenOwner 串行。
- 当前 checkpoint 证明了 no-rescan row formula、多 worker correctness 和 debug-stop 16/17 非 overlap gather；
  还没有证明 FFN 4097 前重排性能对标。

## 4. FFN vs A8W8 子阶段对比

| 子阶段 | FFN 参考实现 | 当前 a8w8 实现 | 当前判断 |
| --- | --- | --- | --- |
| sort / expert-major reorder | 按 `M * topK` route element 多 AIV sort；4097 走 multi-core sort | 不实现 FFN sort；用 count + prefix + packed row 公式直接写 expert-major | a8w8 固定走 direct scatter 路线，不把 FFN sort 子系统作为当前实现目标 |
| route count | `MoeV2ExpertTokenOut` 基于 sorted route 生成 expert token count/cumsum | `M3NCountLocalRoutesShard` 按 token range 统计 worker/expert count | 结构可对齐；4097 debug-stop 11 已验证 active 多 worker |
| worker/expert prefix | FFN 通过 expert token/cumsum 和后续 `srcToDst` 建立 row 映射 | `M3NMergeLocalRouteCounts` 在 main AIV 串行生成 `blockPrefixPerExpert` 和 `workerExpertPrefix` | correctness 路线可用；大 shape 可能有串行控制面成本 |
| srcToDst | 独立 `MoeV2SrcToDstOp`，按 route rows 多 AIV 生成 source-to-destination 映射 | 没有独立 srcToDst；`expandedRowIdx` / packed row 直接承担映射语义 | 只要 restore/gather contract 一致就不必机械补；若要复用 FFN contract 再评估 |
| gather + dynamic quant | `MoeV2GatherDynamicQuant` 按 route rows + UB tile 多 AIV，hidden 维 tile/vector 化 | `M3NRoutePackQuantLocalShardScalar` 按 token shard；worker 内 token/topK 串行，row quant 是 scalar loop | 性能差距最大；当前只推进 PTO Vec quant |
| count publish / wait | 跨 rank `tokenPerExpert` all-gather，按 `dstEpIdx = coreIdx; dstEpIdx < EP; dstEpIdx += coreNum` 切分 | `M3NPublishCountRowsShard` / `M3NWaitCountRowsShardScalar` 按 rank modulo workerCount 切分 | 语义接近；需要 worker>1 后验证 active 分摊和信号可见性 |
| cumsum / preSumBeforeRank | `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2` + `GetCumsumForMMAIV` | `M2BuildPrefixMetadata` 构造 `cumsumMM` / `preSumBeforeRank` | 语义必须对齐；当前偏 serial，需要 timeline 判断是否成为瓶颈 |
| dispatch gather to GMM input | 按 local expert group 搬 peermem 到 `gmA/gmPerTokenScale1`，每组 ready 后放行 GMM1 | `M3NGatherDispatchToGmm1InputShard` 按 localExpert modulo worker；overlap path `M3N5...ByExpert` 可按 expert ready | ready 粒度方向对齐；需要多 worker 和 debug-stop 16/17 证明 |

4097 对标结论：

- FFN: `M=4097, topK=2` 时 `totalLength=8194`，走 multi-core sort + srcToDst + gather dynamic quant。
- a8w8: 当前走 direct scatter，多 worker 已通过单阶段验证，但 quant 仍是 scalar。
- 因此当前不是“缺少 FFN sort 子系统导致语义不对”，而是“direct scatter 路线还没有做到 FFN 4097 所需的 vector quant
  性能形态”。

## 5. Requirements

### Correctness Requirements

- `expandedRowIdx` 对每个有效 `(token, topK slot)` 给出 expert-major packed row；invalid 写 `-1`，drop 写
  `>= maxOutputSize` sentinel。
- `tokenPerExpertMatrix`、`blockTokenPerExpert`、`blockPrefixPerExpert`、`cumsumMM`、`preSumBeforeRank` 与 host
  reference 一致。
- `dispatchPayload[packedRow]` 与对应 token 的 int8 dynamic quant 结果一致。
- `dispatchScale[packedRow]` 与对应 token 的 per-token scale 一致。
- `gmm1InputInt8`、`routingPerTokenScale` 与 `GatherDispatchToGmm1Input` reference 一致。
- `dispatchGroupReady[localExpert]` 只能在该 local expert 的 gather rows 全部写入后置位。

### Sync Requirements

- route count、merge/prefix、route pack/quant、count publish/wait、prefix build 之间保留硬同步或等价可证明的
  visibility edge。
- count/prefix 完成前不得放行 GMM1。
- dispatch gather 后可按 local expert / expert group 局部放行 GMM1，不需要等待所有 expert gather 完。
- 单阶段 debug-stop 必须能在 11/12/13/14/15/16/17 切点退出，避免 GMM/SwiGLU 干扰前重排验证。

### Performance Requirements

- `ffn-v3-4097` 必须保持 active dispatch worker > 1 的 evidence。
- route count / pack quant 至少按 token shard 多 AIV active。
- dynamic quant 应改为 PTO Vec tile，避免每 row hidden 维 scalar 两遍扫描成为主瓶颈。
- report 必须打印 requested / available / active 三层事实，不能用 launch AIV 数冒充 active worker。

## 6. Design Direction

路线：保留 direct scatter，不实现 FFN multi-core sort。

理由：

- expert id 范围有限，count + prefix + direct scatter 是 O(M * topK)，理论上比 comparison/merge sort 更适合。
- 最初慢点来自 active worker=1 和 scalar quant；checkpoint 后 active worker 已打开，当前剩余主风险是 scalar
  quant 和串行控制面，不是 direct scatter 语义本身。
- FFN multi-core sort 不是 MegaMoE 必须语义；a8w8 当前任务中不实现、不评估。

## 7. Implementation Tasks

### Task 1: 固化前重排单阶段 verify

**Files:**

- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/host/main.cpp`
- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/reports/M3O.2.md`
- Optional modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/scripts/run_a3.sh`

- [x] **Step 1: 确认 debug-stop 切点语义**

记录并保持以下切点：

```text
11: route count + merge/prefix after local count
12: route/pack/quant completed
13: count rows published
14: count rows waited
15: cumsum/preSum metadata built
16: dispatch gather to gmm1InputInt8 completed
17: dispatchGroupReady and swiglu metadata published
```

- [x] **Step 2: host report 增加 init_quant 单阶段字段**

在现有 structured report 中补充这些字段：

```text
init_quant_debug_stop=<stage>
init_quant_active_workers=<count>
init_quant_worker_mask=<mask>
init_quant_expanded_row_contract=source_local_pre_count_sync|capacity_clipped
init_quant_route_count_match=true|false
init_quant_expanded_row_match=true|false
init_quant_payload_sample_match=true|false
init_quant_gmm1_input_match=true|false
```

- [x] **Step 3: 单阶段验证命令**

Small:

```bash
bash kernels/manual/a2a3/moe_dispatch_combine_a8w8/scripts/run_a3.sh --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 --first-device 4 --ndevices 7 --case-name ffn-v3-small -pes 2 -M 16 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 32 --m2-fused-debug-stop-stage 12
```

Large:

```bash
bash kernels/manual/a2a3/moe_dispatch_combine_a8w8/scripts/run_a3.sh --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 --first-device 4 --ndevices 7 --case-name ffn-v3-4097 -pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194 --m2-fused-debug-stop-stage 12
```

Expected:

```text
exit=0
init_quant_expanded_row_contract=source_local_pre_count_sync
init_quant_route_count_match=true
init_quant_expanded_row_match=true
init_quant_payload_sample_match=true
```

### Task 2: 放开 direct-scatter 多 AIV worker policy

**Files:**

- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`
- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/host/main.cpp`
- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/reports/M3O.2.md`

- [x] **Step 1: 替换 `targetWorkers = 1U`**

在 `M3NAssignDispatchWorkers()` 中使用 shape-aware worker count：

```cpp
uint32_t targetWorkers = logicalAivCount;
if (targetWorkers > kM3NFusedDispatchMaxLaneSlots) {
    targetWorkers = kM3NFusedDispatchMaxLaneSlots;
}
uint32_t routeCount = shape.m * shape.topK;
uint32_t minRoutesPerWorker = 64U;
uint32_t maxUsefulWorkers = (routeCount + minRoutesPerWorker - 1U) / minRoutesPerWorker;
if (maxUsefulWorkers == 0U) {
    maxUsefulWorkers = 1U;
}
if (targetWorkers > maxUsefulWorkers) {
    targetWorkers = maxUsefulWorkers;
}
```

- [x] **Step 2: 验证 route count 多 worker**

Run:

```bash
bash kernels/manual/a2a3/moe_dispatch_combine_a8w8/scripts/run_a3.sh --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 --first-device 4 --ndevices 7 --case-name ffn-v3-4097 -pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194 --m2-fused-debug-stop-stage 11
```

Expected:

```text
exit=0
dispatch_worker_count>1
dispatch_active_aiv_workers>1
worker_expert_count=true
worker_expert_prefix=true
```

- [x] **Step 3: 验证 route/pack/quant 多 worker correctness**

Run debug-stop 12 on small and large. Expected:

```text
exit=0
init_quant_expanded_row_contract=source_local_pre_count_sync
route_shard_rescan=false
dispatch_pack_row_formula=expertBase_plus_workerExpertPrefix_plus_localOrdinal
init_quant_expanded_row_match=true
init_quant_payload_sample_match=true
```

Run debug-stop 14 on large after count rows wait. Expected:

```text
exit=0
init_quant_expanded_row_contract=capacity_clipped
init_quant_token_matrix_full_match=true
init_quant_expanded_row_match=true
```

### Task 3: PTO Vec dynamic quant for direct pack

**Files:**

- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`
- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/kernel/moe_dispatch_combine_a8w8_kernel.cpp`

- [x] **Step 1: 保留 scalar quant 作为 debug fallback**

保留 `M2QuantizeRowToPeerPayloadScalar()`，但 report 中标识：

```text
route_quant_path=scalar|pto_vec
```

- [ ] **Step 2: 增加 PTO Vec row quant helper**

Current blocker: fused direct-pack 中直接复用 PTO Vec route quant helper 后，`ffn-v3-small` debug-stop 12
超时。进一步把 rowmax/GetValue 拆掉、仅保留 scalar max + PTO Vec `TLOAD/TCVT/TQUANT/TSTORE` 仍超时。
当前提交版本回到 scalar fallback，并在 report 中打印：

```text
route_quant_path=scalar
route_quant_scalar_fallback_reason=mixed_fused_direct_pack_pto_vec_probe_timeout
```

下一步需要先做 all-participant-safe 的 mixed AIV Vec microprobe，确认 AIC/AIV `SyncAll` 参与者、event
pair 和 peer-window `TSTORE` 能单独返回，再重新打开 fused direct-pack PTO Vec quant。

2026-06-01 最新 probe 证据：

| Stage | Probe | Result | Meaning |
| --- | --- | --- | --- |
| 99 | no-Vec mixed `SyncAll` baseline | pass | mixed full-kernel debug-stop 基础同步可返回 |
| 102301 | route probe before Vec/UB work | pass | 进入 route probe 本身可返回 |
| 89 | early basic Vec probe | timeout / signal 15 cleanup | 只加基础 Vec store probe 仍会卡住 |
| 102305 | `TEXPANDS` + UB Vec op on main AIV | timeout / signal 15 cleanup | 去掉 route quant 主体后仍会卡住 |
| 102307 | route UB `SetValue` / barrier probe | timeout / signal 15 cleanup | 不限于 peer-window `TSTORE` 对齐问题 |
| 102313 | all-lane basic Vec probe | timeout / signal 15 cleanup | 不是 main AIV 单 lane 特例 |
| 102305 | raw `vector_dup` 替换 `TEXPANDS` | timeout / signal 15 cleanup | raw macro 也不能解除卡住，临时代码已撤回 |

结论：刚才的“卡住”还没有解决；当前证据指向 fused mixed AIV 路径里的 PTO Vec/UB op probe 或其同步参与条件。
这不是已观测到的精度错误，当前可用 correctness 路径仍是 scalar fallback。

目标 helper 语义：

```text
load half row tile
convert to float
abs
row max
store dispatchScale[packedRow]
multiply by invScale
round/saturate int8
store dispatchPayload[packedRow]
zero pad rowBytes tail
```

- [ ] **Step 3: 按 hidden tile loop 支持 K=128**

4097 required case 固定 `K=128`。首版只需支持 `hiddenSize <= 128` 的 tile path；其他 shape 走 scalar fallback 并如实打印。

- [ ] **Step 4: 验证 debug-stop 12**

Run required small/large with debug-stop 12. Expected:

```text
exit=0
route_quant_path=pto_vec
init_quant_payload_sample_match=true
```

### Task 4: dispatch gather worker policy 对齐 expert group ready

**Files:**

- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/kernel/moe_dispatch_combine_a8w8_kernel.cpp`
- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`

- [x] **Step 1: 非 overlap gather 保持 correctness**

`M3NGatherDispatchToGmm1InputShard()` 当前按 localExpert modulo worker 切分。确认 multi-worker 后每个 expert 只由一个
worker 写，且不会重复写 `gmm1InputInt8`。

- [ ] **Step 2: overlap gather 使用 expert ready**

`M3N5GatherDispatchToGmm1InputShardByExpert()` 保持每个 localExpert gather 完即写 `dispatchGroupReady`，并由 AIV
发 `M3N5SignalDispatchExpertReady(localExpert)` 放行 AIC。

- [x] **Step 3: 验证 debug-stop 16/17**

Run large:

```bash
bash kernels/manual/a2a3/moe_dispatch_combine_a8w8/scripts/run_a3.sh --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 --first-device 4 --ndevices 7 --case-name ffn-v3-4097 -pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194 --m2-fused-debug-stop-stage 16
```

Expected:

```text
exit=0
init_quant_gmm1_input_match=true
dispatch_ready_experts=<expertPerRank>
```

### Task 5: 性能采样与 direct-scatter 收口

**Files:**

- Modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/reports/M3O.7.md`
- Optional modify: `kernels/manual/a2a3/moe_dispatch_combine_a8w8/host/main.cpp`

- [ ] **Step 1: timeline 拆分 init_quant 子阶段**

至少记录：

```text
init_quant_count_cycles
init_quant_merge_prefix_cycles
init_quant_pack_quant_cycles
init_quant_count_sync_cycles
init_quant_prefix_cycles
init_quant_gather_cycles
```

- [ ] **Step 2: required large 性能采样**

Run:

```bash
bash kernels/manual/a2a3/moe_dispatch_combine_a8w8/scripts/run_a3.sh --backend int8 --skip-build 1 --clean-build 0 --dry-run 0 --skip-kernel-launch 0 --first-device 4 --ndevices 7 --case-name ffn-v3-4097 -pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 --max-output-size 8194
```

Expected:

```text
final_output.err_count=0
dispatch_active_aiv_workers>1
route_quant_path=pto_vec
```

- [ ] **Step 3: direct-scatter 性能结论**

记录 4097 前重排耗时结论：

```text
direct scatter multi-worker active evidence
PTO Vec quant path/fallback evidence
init_quant 子阶段 timeline
是否满足当前 a8w8 前重排性能目标
```

## 8. Task Acceptance

每个 task 完成后必须留下可复查证据，再进入下一 task：

- Task 1 验收：host report 打印 `init_quant_debug_stop`、`init_quant_active_workers`、
  `init_quant_worker_mask`、`init_quant_expanded_row_contract` 和各项 match 字段；debug-stop
  11/12/13/14/15/16/17 能单独截断；small/large 至少在 debug-stop 12 通过 route count、
  expanded row、payload sample 对比；large debug-stop 14 必须通过 `tokenPerExpertMatrix` 和
  capacity-clipped `expandedRowIdx` 对比。
- Task 2 验收：4097 case 不再被 `targetWorkers=1` 限住；debug-stop 11 报告
  `dispatch_active_aiv_workers>1`；worker expert count/prefix 与 host reference 一致；debug-stop 12 仍保持
  `route_shard_rescan=false` 和 packed row formula 正确；debug-stop 14 仍保持 capacity clipping 后的
  `expandedRowIdx` 正确。
- Task 3 验收：4097 `K=128` 走 `route_quant_path=pto_vec`；scalar fallback 只允许用于未覆盖 shape，并必须在
  report 里说明原因；debug-stop 12 payload/scale sample 对比通过。
- Task 4 验收：debug-stop 16 验证 `gmm1InputInt8` / `routingPerTokenScale` 与 reference 一致；debug-stop 17
  验证 local expert ready 只在对应 expert gather 完后置位；multi-worker 不重复写、不漏写。
- Task 5 验收：4097 full run correctness pass；report 至少拆出 count、merge/prefix、pack/quant、count sync、
  prefix、gather 子阶段耗时；给出 direct-scatter 路线性能结论。

失败判定：

- 任一 debug-stop 卡死、超时、rank 间退出不一致，判定为 sync/blocker，不继续做性能优化。
- 任一 reference match 为 false，判定为 correctness blocker，不用 timeline 结果证明性能。
- debug-stop 12 的 `expandedRowIdx` 按 source-local pre-count-sync contract 验收；debug-stop 14 及之后按
  cross-rank capacity-clipped contract 验收。两者不能混用。
- `dispatch_active_aiv_workers` 仍为 1 时，Task 2 不能验收，后续 PTO Vec 性能数字只能作为 scalar/single-worker
  baseline。

## 8.1 Current Verification Evidence

Checkpoint: Task 1/2/4 correctness path is verified with scalar quant fallback; Task 3 PTO Vec is blocked as above.
Fresh run: 2026-06-01 17:17, logs under `/tmp/a8w8_init_quant_verify_20260601_171712`.

Build:

```bash
cmake --build kernels/manual/a2a3/moe_dispatch_combine_a8w8/build --target moe_dispatch_combine_a8w8 --clean-first -j16
```

Result: exit 0.

Required debug-stop evidence on devices 4-7:

| Case | Stage | Result | Key evidence |
| --- | --- | --- | --- |
| `ffn-v3-small` | 12 | pass | `init_quant_expanded_row_contract=source_local_pre_count_sync`, route count / expanded row / payload sample match |
| `ffn-v3-4097` | 11 | pass | `dispatch_active_aiv_workers=4`, `init_quant_worker_mask=15`, route count match |
| `ffn-v3-4097` | 12 | pass | source-local expanded row and payload/scale sample match |
| `ffn-v3-4097` | 13 | pass | count rows published cut returns with route/pack checks still matching |
| `ffn-v3-4097` | 14 | pass | `init_quant_expanded_row_contract=capacity_clipped`, token matrix full match, expanded row match |
| `ffn-v3-4097` | 15 | pass | `cumsumMM`, `preSumBeforeRank`, `expertTokenNums` match |
| `ffn-v3-4097` | 16 | pass | `gmm1InputInt8` and `routingPerTokenScale` match |
| `ffn-v3-4097` | 17 | pass | dispatch ready match, no missing local expert ready |

Current known limit:

```text
route_quant_path=scalar
route_quant_scalar_fallback_reason=mixed_fused_direct_pack_pto_vec_probe_timeout
```

2026-06-01 的 Vec probe 尝试没有修复 timeout。临时 `102313` all-lane probe 和 18:14 raw `vector_dup`
替换尝试只作为诊断证据记录，不进入提交路线。

## 9. Overall Acceptance

M3O 前重排收口必须同时满足：

- required small/large full run correctness pass。
- debug-stop 11/12/13/14/15/16/17 单阶段验证 pass。
- 4097 reports `dispatch_active_aiv_workers>1`。
- 4097 reports `route_quant_path=pto_vec`，或明确记录 scalar fallback blocker。
- `route_shard_rescan=false` 仍成立。
- no Catlass/AscendC fallback。
- no M3R/M3S or M3N.10 scoreboard/prod-status/min-status restore。

## 10. Handoff Notes

- 不实现 FFN multi-core sort。先把 direct scatter 的 active worker 和 vector quant 做实。
- 不要把 FFN 的具体 `BLOCK_NUM=20`、`aivNumInitRouting=40`、UB 常量硬搬到 a8w8。
- 不要用 launch AIV 数证明 active 并行；必须由 worker counter / worker mask / processed rows 证明。
- 前重排单阶段验证不应受 M3O.4 activation shard hang 影响；使用 debug-stop 11/12/15/16/17 截断。
