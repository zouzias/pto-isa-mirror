# 术语表

- **PTO 虚拟 ISA**：PTO 的架构可见虚拟指令集契约。
- **目标 Profile**：对可移植 PTO ISA 子集做出的目标相关缩窄，例如 CPU 仿真、A2/A3 类目标或 A5 类目标。
- **Tile 指令**：`pto.t*` 指令集，以及与 tile 搬运紧密相关的 `pto.mgather` / `pto.mscatter`。
- **向量指令**：`pto.v*` 微指令集。
- **标量与控制指令**：`pto.*` 中负责同步、DMA、谓词、配置与控制流外壳的部分。
- **有效区域（valid region）**：物理 tile 中真正具有架构语义定义的子区域。
- **位置意图（location intent）**：影响合法性的角色或存储意图，例如 `Vec`、`Mat`、`Left`、`Right`、`Acc`。
- **GlobalTensor**：PTO 中面向 GM 的架构可见视图，带有 shape、stride、layout 等元数据。
- **指令集（instruction set）**：按功能角色组织的指令集合，例如 tile、vector、scalar/control、other。
- **指令族（instruction family）**：共享操作数模型、约束和语义模式的一组指令。
- **实现定义（implementation-defined）**：行为存在变化空间，但变化点必须由实现明确说明。
- **未指定（unspecified）**：行为结果不承诺固定，但不应被当作稳定契约使用。
- **非法（illegal）**：程序不满足 PTO 契约，verifier 或 backend 应拒绝。
