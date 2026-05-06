# include/pto/npu/310p3/

Ascend 310P 系列 PTO 指令实现头文件。

## 概览

- 按指令（或指令族）组织实现，例如：`TAdd.hpp`、`TMatmul.hpp`、`TLoad.hpp`、`TStore.hpp`
- 同时提供一些可复用的算子模式（例如 Reduce/Expand/PartOp 等辅助实现）

## 相关内容

- ISA 语义与示例：`docs/isa/`
- 310P NPU ST 测试：`tests/npu/310p3/src/st/`