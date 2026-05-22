# dispatch_combine 任务记录

## 目标

在 `kernels/manual/a2a3/dispatch_combine` 下交付一个基于 PTO-ISA 的 `dispatch_combine` 算法样例，入口为可独立运行的 `main`，输入由命令行参数生成，输出与 CPU golden 对比验收。

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
