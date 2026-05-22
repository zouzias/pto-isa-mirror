# 任务：A2/A3 dispatch_combine PTO-ISA 实现

## 目标

基于 README 的 MoE dispatch/combine 闭环算法，在 `dispatch_combine` 目录交付 design.md、task.md 和可运行工程源码。第一阶段用单进程模拟完整多 rank 布局，main 参数生成输入并与 CPU golden 对比。源码必须保持纯 PTO-ISA 风格，禁止 Ascend C namespace 调用形式。

## 待办事项

- [x] 读取 README 和项目知识路由
- [x] 查看 gemm_ar 与 A2/A3 PTO 示例工程结构
- [x] 确认设计落地位置为 dispatch_combine/design.md 和 dispatch_combine/task.md
- [x] 写入 task.md
- [x] 写入 design.md
- [x] 创建 CMakeLists.txt、run.sh、kernel_launchers.h
- [x] 实现 main.cpp host harness 与 CPU golden
- [x] 实现 dispatch_combine_kernel.cpp PTO-ISA kernel
- [x] 编译/运行验证并记录证据
- [x] grep 校验源码无 Ascend C namespace 调用形式

## 进度

10/10
