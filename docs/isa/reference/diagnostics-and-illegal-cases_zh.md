# 诊断与非法情形

PTO ISA 文档需要区分几类不同的问题来源：

- 类型类别错误
- 合法性检查失败
- 目标 profile 缩窄
- 不应被记录为合法行为的未支持路径

## 诊断分层

### 类型错误

类型错误表示操作数种类或元素类型本身不满足指令要求，例如：

- 把 `!pto.vreg` 传给只接受 tile 的操作
- 在不支持的元素类型上使用某条指令
- 谓词宽度与目标向量宽度不匹配

### 合法性失败

合法性失败表示类型可能正确，但 shape、layout、location intent、valid region 或资源状态不满足要求，例如：

- `Left` / `Right` / `Acc` 角色组合不合法
- 行列范围与目标 tile 不兼容
- 等待一个从未建立的事件

### 目标 profile 限制

目标 profile 限制不是对 PTO 语义的重定义，而是 profile 对可用子集的缩窄，例如：

- 某些 A5 专属 tile / vector 形式
- 仅在特定 profile 上支持的 dtype、layout 或 pipe 空间
- 仅在特定 profile 上存在的对齐或缓冲行为

### 未支持行为

文档不应把“当前 backend 没实现”直接写成 PTO 合法行为。如果某种行为只在个别实现中碰巧可用，而缺少稳定契约，应明确标为：

- backend-specific
- implementation-defined
- unsupported

## 写法要求

每个指令族页和 per-op 页都应明确列出不允许的情形，而不是把这些条件留给 backend 报错或由示例隐含表达。

诊断文本至少应能判断出：

- 是类型错了，还是合法性组合错了
- 是 PTO 层不允许，还是目标 profile 不支持
- 是修正操作数、修正同步，还是需要切换 profile

## 相关页面

- [指令描述格式](./format-of-instruction-descriptions_zh.md)
- [可移植性与目标 Profile](./portability-and-target-profiles_zh.md)
- [规范来源](./source-of-truth_zh.md)
