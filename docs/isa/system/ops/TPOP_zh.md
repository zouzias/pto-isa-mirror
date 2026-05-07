# pto.tpop


pto.tpop 是系统调度指令集中的消费者侧操作，用于从 TPipe/TMPipe 队列中弹出可消费数据。

## 摘要


- 与 	push 构成生产者/消费者协议
- 负责等待数据就绪并完成消费端取数
- 常用于 Cube/Vector 间流水接力

## 协议关系


- 生产者：	push 写入并置位 ready
- 消费者：	pop 等待 ready 后读取，并在完成后释放槽位

## 说明


完整行为、同步阶段与目标差异请参考英文页：
[TPOP.md](./TPOP.md)。

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## What TPOP Is Not
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Architecture: The TPipe Abstraction
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Three-Phase Protocol
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Phase 1: Wait (block until data is ready)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Phase 2: Pop (load data from FIFO)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Phase 3: Free (release slot for producer)
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

## TMPipe Consumer Usage
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Constraints
本节列出类型、布局、shape、valid-region 与 profile 相关约束。

## Target-Profile Restrictions
本节给出 A2/A3、A5 及 CPU-SIM 的差异化限制与行为说明。

## Common Patterns
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 1: Consuming Accumulator Tile (GEMM Post-Processing)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 2: Consuming Vector Tile into Matrix (Attention Accumulation)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Pattern 3: Sparse Sync (Consumer Skips Wait Periodically)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Relationship with TFREE
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## See Also
本节给出上下游指令与相关章节链接。
