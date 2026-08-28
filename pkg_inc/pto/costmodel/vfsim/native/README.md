# Native VF 模拟器核心

本目录只保留 PTO-ISA A5 costmodel 主链路需要的 C++ native core：

```text
PTO VF mock -> PTO VfInfo -> CanonicalVfInfo -> runCanonicalVfInfo() -> VfSim cycle
```

当前交付边界：

- `configs/*.json` 是模型参数来源。
- `ParamDB` 负责加载 ISA、uarch、forwarding 和 initiation interval 配置。
- `CanonicalProgramLowering` 将已验证的 `CanonicalVfInfo` 展开为运行时动态指令流。
- `IFU`、`IDU`、`OOO` 和 `SimulatorRunner` 执行 native VF cycle 模拟。
- 正式预测入口只保留 `runCanonicalVfInfo()`。

本目录不交付历史 legacy VfInfo 迁移链路、native JSON runner、fixture parser、
standalone native tests 或 MLIR planner。PTO-ISA 的回归测试位于
`tests/costmodel/st_a5`，通过 PTO adapter 和 `target_enable_a5_vf_mock()` 验证。
