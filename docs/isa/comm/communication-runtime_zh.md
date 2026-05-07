# 通信与运行时


通信与运行时操作覆盖跨 NPU 的数据交换、集合通信和通知协议。它们属于 PTO 的架构可见行为，但不适合放进 tile、vector 或 scalar/control 主干中。

## 操作概览


| 操作 | 说明 |
| --- | --- |
| `tbroadcast` | 从根 NPU 广播到并行组内所有 rank |
| `tget` | 从远端 NPU 读取数据 |
| `tget_async` | `tget` 的异步形式 |
| `tput` | 向远端 NPU 写入数据 |
| `tput_async` | `tput` 的异步形式 |
| `treduce` | 并行组内的集合归约 |
| `tscatter` | 从根 NPU 向所有 rank 分发数据 |
| `tgather` | 从所有 rank 向根 NPU 聚集数据 |
| `tnotify` | 发送通知事件 |
| `ttest` | 非阻塞测试通知条件 |
| `twait` | 阻塞等待通知条件 |

## 输入


通信与运行时操作通常组合使用以下对象：

- 并行组句柄 `!pto.group<N>`
- 本地或远端的 GM 视图
- UB 暂存 tile
- 归约算子、通知条件或异步事件句柄

## 输出


这些操作会产生：

- 跨 rank 的数据传输结果
- 集合归约结果
- 通知状态变化
- 异步 DMA 或异步通信句柄

## 副作用


通信操作会引入跨 NPU 的排序和可见性要求：

- 发起网络或互连流量
- 修改远端或本地 GM 中的数据
- 建立通知或等待条件
- 在异步形式下引入额外的完成状态

## 约束


!!! warning "约束"
    - 所有参与 rank 必须以匹配的 `ParallelGroup` 调用同一集合操作。
    - 根节点与非根节点的源/目的缓冲区角色必须与操作语义一致。
    - 异步操作使用的会话、暂存区和事件对象必须满足对应接口要求。
    - CPU 仿真器不支持跨 NPU 通信操作。

## 不允许的情形


!!! danger "不允许的情形"
    - 不同 rank 使用不匹配的并行组句柄。
    - 集合操作中源/目的缓冲区尺寸与操作语义不匹配。
    - 在未初始化的异步会话或事件对象上继续推进异步流程。
    - 把 CPU 仿真路径误当作通信操作的可执行 profile。

## 相关页面


- [通信指令集](../instruction-families/communication-families_zh.md)
- [通信指令族](../instruction-families/communication-families_zh.md)
- [PTO 通信参考入口](../comm/README_zh.md)

# Communication And Runtime
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Operations
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

### Data Flow
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Broadcast
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Reduce
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Scatter/Gather
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Point-to-Point (TGET/TPUT)
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## ParallelGroup Handle
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Large Tile Support
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Constraints
本节列出类型、布局、shape、valid-region 与 profile 相关约束。

## Cases That Are Not Allowed
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## See Also
本节给出上下游指令与相关章节链接。
