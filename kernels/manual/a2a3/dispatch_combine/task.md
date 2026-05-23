# dispatch_combine 任务记录

## 目标

在 `kernels/manual/a2a3/dispatch_combine` 下交付一个基于 PTO-ISA 的 `dispatch_combine` 算法样例，入口为可独立运行的 `main`，输入由命令行参数生成，输出与 CPU golden 对比验收,编程范式要够PTO风味.

## 范围

- 交付 README 中定义的完整算法语义：RoutingExpand、CountExchange、Dispatch、LocalExpertCompute(identity)、CombineReturn、Restore。
- 第一阶段采用单进程/单 NPU 模拟多 rank 张量布局，验证完整索引闭环和 golden 一致性。
- 工程脚手架参考 `kernels/manual/a2a3/gemm_ar` 的 CMake、run.sh、kernel launcher 组织方式，但不在第一阶段引入 HCCL 多进程通信。
- dtype 初版固定为 `float32`，expert compute 固定为 identity。
- 源码必须保持纯 PTO-ISA 风格，kernel 侧禁止出现 Ascend C namespace 调用形式；验收时用 grep 校验。

## 分阶段任务

- [x] 阶段 0：读取 README 需求，确认算法边界与验收标准。
- [x] 阶段 1：对齐交付策略：完整算法范围，分阶段落地，第一阶段必须有 main 参数和 golden 校验。
- [x] 阶段 2：补充工程设计文档 `design.md`。
- [x] 阶段 3：补齐工程脚手架：`CMakeLists.txt`、`run.sh`、launcher header。
- [x] 阶段 4：实现 host 侧输入生成、CPU golden、ACL runtime、设备内存管理和校验。
- [x] 阶段 5：实现 PTO-ISA vector kernel：按 README 六阶段完成单进程多 rank dispatch/combine identity 闭环。
- [x] 阶段 6：编译验证，执行默认 smoke/all case，记录输出证据。

## 验收标准

1. `./run.sh --case all` 能完成编译并运行默认多组用例。
2. 每个用例输出 CPU golden 与 NPU 输出对比结果，失败时打印首个 mismatch 和最大误差。
3. 默认覆盖：最小单 rank、跨 rank offset、inactive token、duplicate expert、invalid expert、empty expert/segment。
4. 代码保留真实通信扩展点，后续可将单进程 dispatch/combine 段搬运替换为 HCCL/PTO comm 原语。
5. 源码 grep 校验 Ascend C namespace 调用形式无命中。

## 验证记录

以下为已完成第一阶段 scaffold 基线的历史验证记录，不作为动态 shape 升级目标；新目标和待验证项见后文“动态 shape 通用化升级”。

- `smoke R=1 E=1 M=3 H=8 topK=1`：覆盖超过 64B 的单 rank packed/dispatch/return/out 边界，已通过。
- `smoke R=1 E=1 M=5 H=4 topK=1`：覆盖 `idx=16` 边界，已通过。
- `smoke R=2 E=1 M=2 H=8 topK=1`：覆盖跨 rank dispatch/combine offset，已通过。
- `--case all`：`minimal/cross/edge` 全部通过，CPU golden 对比 `count/srcExpertOffset/expandedRowIdx/srcPackedX/dispatchX/returnY/out` 全通过。
- DCCI 开关对照：`-cce-aicore-dcci-insert-for-scalar=false` 时 `smoke R=1 E=1 M=3 H=8 topK=1` 在 `srcPackedX idx=16 expected=20 actual=0` 失败；改为 `true` 后同用例通过。
- Ascend C namespace 调用形式源码 grep 无命中。
- host 参数校验已与 launcher 对齐，只允许 `hidden in {1,4,8,16,32,64}`，`hidden=2` 负向用例返回 `[ERROR] hidden must be one of 1, 4, 8, 16, 32, 64`。
- 固定 UB workspace layout 越界参数会被拒绝，`R=8 E=8 M=64 H=64 topK=8` 负向用例返回 `[ERROR] case exceeds fixed UB workspace layout`。

## 调试结论

- 失败根因是当前 kernel 使用 `Tile::GetValue/SetValue` 通过 S 管线读写 UB，再由 MTE3 搬出；CMake 原先关闭 `-cce-aicore-dcci-insert-for-scalar` 会导致超过 64B 后的 scalar UB 写对后续搬运不可见。
- 解决方案是在该工程 kernel 编译选项中启用 `-cce-aicore-dcci-insert-for-scalar=true`，保证 S 管线 UB 写入与 MTE 搬运之间的数据可见性。

## 当前限制

- 第一阶段不是多进程 HCCL 通信实现，而是单 NPU 上用 `[R, ...]` 维度模拟所有 rank。
- 第一阶段不接 GMM/FFN，LocalExpertCompute 为 identity。
- 第一阶段不做 capacity/drop，invalid expert 和 inactive token 统一跳过。

## 动态 shape 通用化升级

### 新目标

- `R/E/M/H/topK` 全部运行时动态。
- 删除 `launchDispatchCombine` 中按 hidden size 展开的 case 分支。
- 删除 host 对 `hidden in {1,4,8,16,32,64}` 的限制。
- 删除固定 UB workspace layout 对 shape 的限制。
- 保留 README 六阶段语义、CPU golden 对比、单进程多 rank 模拟和纯 PTO-ISA 约束。

### 新验收标准

1. `./run.sh --case all` 覆盖非 2 次幂 hidden、大 hidden、topK=4、edge case 并全部通过。
2. `./run.sh --case smoke --hidden 2` 和 `--hidden 65` 必须完成 NPU vs CPU golden 正确性对比并通过，而不只是 host 不再拒绝。
3. `./run.sh --case smoke --ranks 2 --experts-per-rank 2 --tokens 4 --hidden 1024 --topk 2` 通过。
4. `dispatch_combine_kernel.cpp` 中无 `template <int Hidden>`，`launchDispatchCombine` 中无 `switch (hidden)`。
5. 目标目录源码 grep Ascend C namespace 调用形式无命中。

### 待记录验证

- [x] 默认 smoke 通过。
- [x] `--case all` 通过。
- [x] hidden=2 smoke 通过。
- [x] hidden=65 smoke 通过。
- [x] hidden=1024 smoke 通过。
- [x] topK=4 smoke 通过。
- [x] 源码 grep 通过。

### 动态 shape 验证记录

- 默认 smoke：`R=2 E=2 M=4 H=8 topK=2`，`[RESULT] PASS smoke`，`[SUMMARY] All dispatch_combine cases passed.`
- `--case all`：`minimal/hidden_small/hidden_odd/hidden_65/hidden_large/topk4/edge` 全部 `[RESULT] PASS`，最终 `[SUMMARY] All dispatch_combine cases passed.`
- hidden=2 smoke：`R=1 E=1 M=3 H=2 topK=1`，`[RESULT] PASS smoke`，`[SUMMARY] All dispatch_combine cases passed.`
- hidden=65 smoke：`R=2 E=2 M=4 H=65 topK=2`，`[RESULT] PASS smoke`，`[SUMMARY] All dispatch_combine cases passed.`
- hidden=1024 smoke：`R=2 E=2 M=4 H=1024 topK=2`，`[RESULT] PASS smoke`，`[SUMMARY] All dispatch_combine cases passed.`
- topK=4 smoke：`R=2 E=4 M=8 H=128 topK=4`，`[RESULT] PASS smoke`，`[SUMMARY] All dispatch_combine cases passed.`
- 源码 grep：目标目录无 Ascend C namespace 调用形式；`dispatch_combine_kernel.cpp` 无 hidden 模板特化、无 `switch (hidden)`、无 `LaunchPhases<`；`main.cpp` 无 hidden 白名单和固定 UB workspace 报错文本。

### 动态 shape 调试结论

- `launchDispatchCombine` 已改为单一路径，hidden size 完全由 runtime 参数传入。
- `dispatch_combine_kernel.cpp` 不再使用固定整块 UB workspace 保存完整 `x/srcPackedX/dispatchX/returnY/out`。
- GM workspace 作为权威中间存储，UB 仅作为 bounded chunk staging buffer。
- PTO 风味提纯：已删除重复 float scalar GM 清零；float 数据面 row offset 收敛到 `CopyXToPackedRow` / `CopyPackedToDispatchRow` / `CopyDispatchToReturnRow` / `AddReturnRowToOut` helper 边界，主算法循环不再直接拼接 float workspace 裸 GM offset。
- PTO 风味提纯后复验：`--case all` 通过；`H=1024` smoke 通过；`topK=4` smoke 通过；目标目录无 Ascend C namespace 调用形式；kernel 源码无 hidden 模板特化、无 hidden switch、无重复 `ClearVector<float>`。
- PTO-native 逻辑重写：kernel 内部已重写为 `DispatchShape`、GM row views、PTO vector primitives、metadata protocol 和 stage graph；public launcher ABI 与 host golden 验收保持不变。
- PTO-native 重写后复验：`--case all` 通过；hidden=2 smoke 通过；hidden=1024 smoke 通过；topK=4 smoke 通过。
- `H=2/65/1024` 和 `topK=4` 均已通过 CPU golden 对比。
- 性能版本演进设计已写入 `design.md` Section 10，按 P1 vector restore、P2 row copy ping-pong、P3 segment descriptor、P4 多核 segment 并行、P5 真实通信与 expert compute 接入分阶段推进。

## 性能版本实现记录

### 实现范围

- P1：`ScaleAddFloatRow` 使用 PTO `TMULS + TADD` 完成 restore 加权累加，源码中不再出现 `AddScaledFloatRow` / `GetTileValue<float>` / `SetTileValue<float>` restore 累加路径。
- P2：`CopyFloatRow` 改为显式 ping/pong UB staging 边界，UB 使用 `kVecPingUb/kVecPongUb/kAccUb` 固定 tile contract，不随 shape 增长。
- P3：新增 `segmentDesc/segmentReady/returnReady` GM workspace；`segmentDesc` 采用七个 `int32_t` 字段：`src,dst,localExpert,globalExpert,srcRowBase,dispatchRowBase,rows`，host 侧 CPU golden 回传校验。
- P4：stage graph 拆为 `MetadataPackKernel -> DispatchSegmentsKernel -> ReturnSegmentsKernel -> MarkReadyKernel -> RestoreTokensKernel`；segment kernel 以 `segmentId=(dst,localExpert,src)` 为 block 维度，restore kernel 以 `(src,token)` 为 block 维度，避免多个 block 同写同一 out row。
- P5：当前单进程/单 NPU harness 已接入 local simulation backend seam：`DispatchSegmentLocal`、`ExpertIdentitySegmentLocal`、`ReturnSegmentLocal`；真实多进程 HCCL/PTO window 通信尚未实现，不能声明真实通信闭环完成。

### 调试记录

- 初次拆分多 block 后，`srcPackedX/dispatchX/returnY/out/segmentDesc` 均通过 golden，但 `segmentReady/returnReady` 在多 block scalar GM 写路径出现不稳定 0。
- RCA：descriptor single-block scalar 写与数据面 TLOAD/TSTORE 均正确，失败集中在多 block scalar ready 发布的 host 可见控制面；参考 `gemm_ar/ready_queue.hpp` 的 ready 发布需要显式 cache 处理。
- 最终实现：ready 数组由同 stream 的单 block `MarkReadyKernel` 在 segment kernel 完成后统一发布；这符合当前 local simulation 的 kernel launch 同步边界，也避免伪造 segment 内 fine-grained ready 协议。

### 验证证据

- P1 前置 baseline：`bash kernels/manual/a2a3/dispatch_combine/run.sh --case smoke --ranks 2 --experts-per-rank 2 --tokens 4 --hidden 8 --topk 2`，`[RESULT] PASS smoke`，`[SUMMARY] All dispatch_combine cases passed.`
- 实现后默认 smoke：`R=2 E=2 M=4 H=8 topK=2`，`segmentDesc/segmentReady/returnReady/srcPackedX/dispatchX/returnY/out` 全 PASS，`[SUMMARY] All dispatch_combine cases passed.`
- `--case all`：`minimal/hidden_small/hidden_odd/hidden_65/hidden_large/topk4/edge` 全部 PASS；每个 case 均校验 `segmentDesc/segmentReady/returnReady` 与原有中间张量。
- H=2048 smoke：`R=2 E=2 M=4 H=2048 topK=2` 全部 PASS。
- hidden=2 smoke：`R=1 E=1 M=3 H=2 topK=1` 全部 PASS。
- hidden=65 smoke：`R=2 E=2 M=4 H=65 topK=2` 全部 PASS。
- hidden=1024 smoke：`R=2 E=2 M=4 H=1024 topK=2` 全部 PASS。
- topK=4 smoke：`R=2 E=4 M=8 H=128 topK=4` 全部 PASS，`out max_diff=6.10352e-05`，在 `1e-4` 阈值内。
- 源码 grep：目标目录无 Ascend C namespace 调用形式；kernel 源码无 hidden 模板特化、无 hidden switch、无 `LaunchPhases<`、无 `CopyGmVector/AddScaledGmVector/ClearVector<float>`；无 restore scalar 累加路径。
