# 任务：统一 async workspace 接口第 5 点

## 目标
完成 `include/pto/comm/workspace.hpp` 的统一接口落地，并迁移现有调用点到新 API；验证尺寸、状态与失败路径都符合文档要求。

## 待办事项
- [x] 先完成前置依赖：`DmaEngine` 轻量头拆分、URMA `GetWorkspaceSize()`、RDMA `GetWorkspaceSize()`、URMA `Init()` 幂等
- [x] 新增 `include/pto/comm/workspace.hpp`，定义 `WorkspaceStatus` / `WorkspaceRequest` / `Workspace` / `CreateWorkspace` / `DestroyWorkspace` / `AbandonWorkspace`
- [x] 把 SDMA / URMA / RDMA 的创建与销毁逻辑接进统一门面
- [x] 迁移 `kernels/include/communication/host/detail/transport.hpp` 等内部调用点到新 API
- [x] 迁移 pto-isa 内部测试、demo 与 RDMA 相关调用点到新 API
- [x] 补齐测试：size、状态、失败路径、幂等、非支持架构分支
- [x] 做一次公共头单独编译和对应平台构建验证

## 进度
7/7

## 备注
- 已迁移 a2a3 / a5 的 comm ST、a5 demo 以及 host 侧 transport 到 `Workspace`；仅剩 README / 注释中的旧名和 `transport.hpp` 内部的 URMA peer base 句柄保留。
- 当前环境可编译 a5 / a2a3 的 comm sim 用例；运行仍被 MPI 监听端口权限限制挡住。RDMA HNS_1825 构建被本机 CANN 9.0.0 HCOMM 头字段缺失挡住。
