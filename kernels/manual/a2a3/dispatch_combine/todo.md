# 任务：A2/A3 dispatch_combine 动态 shape 通用化

## 目标

在已完成的第一阶段 PTO-ISA dispatch_combine correctness scaffold 基础上，升级为 `R/E/M/H/topK` 全运行时动态。去掉 `launchDispatchCombine` hidden switch 和固定 UB workspace 限制，保留单进程多 rank identity 闭环、CPU golden 对比和纯 PTO-ISA 约束。

## 待办事项

- [x] 更新 design.md/task.md，记录动态 shape 架构与验收标准
- [x] 改造 host 参数校验，删除 hidden 白名单和固定 UB layout 校验
- [x] 扩展 all case，覆盖非 2 次幂 hidden、大 hidden 和多 topK
- [x] 改造 kernel，删除 `template <int Hidden>` 和 hidden switch
- [x] 用 GM workspace + UB chunk helper 实现 packed/dispatch/return/out 数据面
- [x] 编译并运行默认 smoke
- [x] 运行 `--case all`
- [x] 运行额外动态 hidden/topK smoke 用例
- [x] grep 校验源码无 Ascend C namespace 调用形式
- [x] 更新 task.md 验证记录

## PTO 编程范式提纯

- [x] 删除重复 float scalar GM 清零
- [x] 将 float 数据面 row offset 封装到 helper 边界
- [x] 更新 design.md/task.md 记录 PTO 风味提纯结论
- [x] 重新运行动态 shape 验证
- [x] 重新运行源码 grep 检查

## PTO-native 逻辑重写

- [x] 重写 kernel 内部为 DispatchShape + GM row views + PTO vector primitives + stage graph
- [x] 保持 launcher ABI 与 host golden 验收不变
- [x] `--case all` 通过
- [x] hidden=2 / hidden=1024 / topK=4 smoke 通过
- [x] 重新运行源码 grep 检查
- [x] 更新 task.md 验证记录

## 性能版本演进设计

- [x] 设计 P1 单核 vector primitive restore
- [x] 设计 P2 row copy ping-pong pipeline
- [x] 设计 P3 metadata kernel + segment descriptor
- [x] 设计 P4 多核 segment 并行
- [x] 设计 P5 真实通信与 expert compute 接入
- [x] 更新 design.md Section 10 和 task.md 记录

## 性能版本实现

- [x] P1：restore 数据面改为 `TMULS + TADD` PTO vector primitive，不再用逐元素 `GetTileValue/SetTileValue` 累加
- [x] P2：row copy helper 显式使用 ping/pong UB staging 边界，并通过 H=2048 smoke
- [x] P3：新增 `segmentDesc/segmentReady/returnReady` workspace，host golden 回传校验 segment descriptor
- [x] P4：拆分为 metadata/pack、segment dispatch、segment return、token restore 多 kernel；segment 与 token 使用多 block 映射
- [x] P5：接入 local simulation backend seam：`DispatchSegmentLocal`、`ExpertIdentitySegmentLocal`、`ReturnSegmentLocal`；真实 HCCL/PTO 多进程通信保留为后续验收项
- [x] 运行黑盒动态 shape 验证与源码 grep 检查
- [x] 更新 design.md/task.md 记录实现边界和验证证据

## 进度

34/34
