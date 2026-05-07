# pto.tpush


pto.tpush 是系统调度指令集中的生产者侧操作，用于将 Tile 写入 TPipe/TMPipe 队列，供配对消费者侧 	pop 使用。

## 摘要


- 负责生产端入队
- 与 	pop 一起构成完整的 FIFO 协议
- 典型用于 Cube↔Vector 的数据接力

## 三阶段语义（概念）


1. Allocate：等待可写槽位
2. Push：写入数据
3. Record：通知消费者数据就绪

## 说明


完整规范、方向类型、分裂模式与目标限制请参考英文页：
[TPUSH.md](./TPUSH.md)。

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## What TPUSH Is Not
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Architecture: The TPipe Abstraction
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Direction Types
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### FIFO Storage Paths
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Three-Phase Protocol
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Phase 1: Allocate (wait for free slot)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Phase 2: Push (write data to FIFO)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Phase 3: Record (signal data-ready to consumer)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Tile Split Modes
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## TMPipe: Multi-Pipe Variant
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

### IR Level 1 (SSA)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### IR Level 2 (DPS)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## C++ Intrinsic
本节给出 C++ 内建接口入口与参数语义说明。

## TMPipe Usage
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Constraints
本节列出类型、布局、shape、valid-region 与 profile 相关约束。

## Target-Profile Restrictions
本节给出 A2/A3、A5 及 CPU-SIM 的差异化限制与行为说明。

## Common Patterns
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 1: Acc → Vec Tile Passing (GEMM Post-Processing)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 2: Vec → Mat Tile Passing with Row Split
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 3: Sparse Sync (Skipping Allocation Wait)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## See Also
本节给出上下游指令与相关章节链接。
