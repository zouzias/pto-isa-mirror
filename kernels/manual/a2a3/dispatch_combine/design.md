# A2/A3 dispatch_combine PTO-ISA 工程设计

## 1. 目标

在 `kernels/manual/a2a3/dispatch_combine` 下交付一个可独立编译和运行的 kernel 直调工程，使用命令行参数生成 MoE dispatch/combine 输入，在 NPU 上执行 README 定义的完整闭环算法，并与 CPU golden 对比。

硬约束：

1. 源码保持纯 PTO-ISA 风格，kernel 侧只使用 PTO-ISA include、PTO tile/global tensor、基础 CCE 内建搬运能力和 C/C++ 控制流，禁止出现 Ascend C namespace 调用形式。
2. 第一阶段覆盖 README 的算法范围：RoutingExpand、CountExchange、Dispatch、LocalExpertCompute(identity)、CombineReturn、Restore。
3. 第一阶段采用单进程/单 NPU 模拟多 rank 布局，后续再把段搬运替换为真实 HCCL/PTO comm 多进程通信。
4. main 必须有参数入口，验收必须打印每个 case 的 NPU vs CPU golden 对比结果。

## 2. 交付形态

新增文件：

| 文件 | 责任 |
| --- | --- |
| `design.md` | 当前工程设计与分阶段边界。 |
| `task.md` | 任务记录、验收标准、阶段状态。 |
| `todo.md` | 长任务焦点列表。 |
| `CMakeLists.txt` | 参考 `a2a3/gemm_ar` / `a2a3/topk` 的 PTO kernel + host executable 构建。 |
| `run.sh` | source CANN、配置 CMake、编译、运行 smoke/all case。 |
| `kernel_launchers.h` | host 到 kernel launcher 的窄接口。 |
| `dispatch_combine_kernel.cpp` | 纯 PTO-ISA vector kernel，执行完整 dispatch/combine identity 闭环。 |
| `main.cpp` | 参数解析、输入生成、CPU golden、ACL runtime、NPU 运行、结果校验。 |

## 3. 数据模型

参数：

```text
R: rank 数
E: 每 rank expert 数
G = R * E
M: 每 rank token 数
H: hidden size
K: topK
```

Host 侧输入按所有 rank 连续存储：

```text
x:          [R, M, H] float32
expert_idx: [R, M, K] int32，存 global expert id
probs:      [R, M, K] float32
active:     [R, M] int32，0/1
```

Device workspace：

```text
count:            [R, G] int32
srcExpertOffset:  [R, G] int32
expandedRowIdx:   [R, M*K] int32
srcPackedX:       [R, M*K, H] float32
dispatchX:        [R, R*M*K, H] float32
returnY:          [R, M*K, H] float32
out:              [R, M, H] float32
```

`srcPackedX` / `returnY` 每个 rank 预留 `M*K` 行，`dispatchX` 每个 dst rank 预留 `R*M*K` 行，不做 capacity/drop。有效行由 `count` 和 `expandedRowIdx` 标识，无效副本使用 `expandedRowIdx=-1`。

## 4. Kernel 算法

第一阶段用一个 vector kernel 串行执行所有 rank 的控制流，目的是锁定索引公式和闭环正确性。

1. 清零 `count/out/srcPackedX/returnY`，`expandedRowIdx` 初始化为 `-1`。
2. RoutingExpand 第一遍：按 `src -> token -> slot` 稳定顺序统计 `count[src][g]`。
3. Prefix：计算 `srcExpertOffset[src][g]`。
4. RoutingExpand 第二遍：写 `srcPackedX[src, row, :] = x[src, token, :]`，写 `expandedRowIdx[src, flat] = row`。
5. Dispatch：按 `dst -> e -> src` 遍历，读取 `srcPackedX[src, srcExpertOffset[src][g] + r, :]`，写出显式 `dispatchX[dst, dispatch_row, :]` 作为阶段 1 调试面和后续通信替换点。
6. LocalExpertCompute(identity) + CombineReturn：当前 identity 路径从 `dispatchX` 读回，并按 `srcExpertOffset` 写入 `returnY[src, same_row, :]`。
7. Restore：按 `src -> token -> slot` 读取 `expandedRowIdx`，将 `probs[src, token, slot] * returnY[src, row, :]` 累加到 `out[src, token, :]`。

数据面实现要求：

- GM 与 UB 边界使用 PTO vector tile：`GlobalTensor` + `Tile<TileType::Vec>` + `TLOAD/TSTORE`。
- 阶段 1 正确性骨架在 UB 内用 `Tile::GetValue/SetValue` 与 C/C++ 控制流完成动态 routing、dispatch 和 restore 组装；后续性能阶段再替换为更完整的 vector 化 `TMULS/TADD` 路径。
- 非连续或动态小规模 metadata 使用 C/C++ 标量读写 GM，避免引入 Ascend C namespace 调用形式。
- 第一阶段已完成 scaffold 基线中，host 只允许 launcher 支持的 `H in {1,4,8,16,32,64}`，并校验固定 UB workspace layout 不越界；这只是动态 shape 升级前的历史限制，Section 8 将其替换为运行时 `H` 和 GM-backed chunk staging 设计。

## 5. Host 验证设计

`main.cpp` 提供参数：

```text
--case smoke|minimal|cross|edge|all
--ranks R
--experts-per-rank E
--tokens M
--hidden H
--topk K
--device DEVICE_ID
```

输入生成规则：

- `x[src, token, h] = src * 1000 + token * 10 + h * 0.125`。
- `probs` 由 case 固定生成，默认每 token topK 权重和为 1。
- `expert_idx` 覆盖本地 expert、跨 rank expert、空 expert、重复 expert、invalid expert。
- `active` 覆盖 active/inactive token。

CPU golden 按 README 的伪代码实现相同六阶段。校验以 `abs(actual - expected) <= 1e-4` 为准，失败时打印 case、rank/token/hidden、expected、actual、diff。

## 6. 分阶段交付

### 阶段 1：单进程多 rank identity 闭环（已完成基线）

- 已完成的第一阶段 correctness scaffold 基线，不代表动态 shape 最终目标。
- 目标是证明 README 中的 count/prefix/expandedRowIdx/restore 公式正确。
- 基线验收：`./run.sh --case all` 通过，源码 grep 无 Ascend C namespace 调用形式；动态 shape 升级验收见 Section 8.5。

### 阶段 2：dispatchX segment metadata 与通信替换

- 阶段 1 已落地并校验显式 `dispatchX`。
- 阶段 2 可继续增加 segment metadata 回传和打印，用于对齐 `dst/e/src` 段，并将本地 `dispatchX` 搬运替换为真实通信版。

### 阶段 3：真实多进程通信闭环

- 复用 `gemm_ar` 的 HCCL/MPI 初始化、窗口与 run.sh 组织。
- 将阶段 1 的 `Dispatch + CombineReturn` 段搬运替换为 PTO comm / HCCL window 搬运。
- CPU golden 保持不变。

### 阶段 4：真实 expert compute

- 在 `LocalExpertCompute` 中替换 identity 为 GMM/FFN。
- 保持段内 row 顺序不变，combine/restore 逻辑不变。

## 7. 风险与约束

- 第一阶段是正确性骨架，不代表真实跨卡通信性能。
- 单 kernel 串行控制流便于验证，不作为最终性能形态。
- 如果 H 或 M*K 超过 kernel tile/workspace 约束，host 参数校验直接拒绝，避免静默越界。
- 所有纯 PTO-ISA 约束通过源码 grep 和 code review 双重检查。
- 当前 kernel 的 UB 数据面会用 `Tile::GetValue/SetValue` 在 S 管线上组装 packed/dispatch/return/out。A2/A3 编译必须开启 `-cce-aicore-dcci-insert-for-scalar=true`，否则超过 64B 后的 scalar UB 写可能无法被后续 MTE 搬运完整观察到。

## 8. 动态 shape 通用化升级设计

本阶段目标是把第一阶段 correctness scaffold 从固定 hidden 模板和固定 UB workspace，升级为 `R/E/M/H/topK` 全运行时动态的主流 dispatch_combine correctness 版本。

### 8.1 支持范围

- `R/E/M/H/topK` 均由 `main` 命令行参数传入，kernel launch 不再按 `H` 分 case。
- `H` 支持任意正整数，不再限制为 `1/4/8/16/32/64`。
- dtype 仍固定为 `float32`，中间 expert compute 仍固定为 identity。
- 第一阶段仍采用单进程/单 NPU 模拟多 rank 张量布局，不引入真实 HCCL 多进程通信。
- 不做 capacity/drop；inactive token 和 invalid expert 继续用 `expandedRowIdx=-1` 跳过；duplicate expert 保留多个副本。

### 8.2 数据面架构

GM workspace 是权威存储：

```text
count:            [R, R*E]
srcExpertOffset:  [R, R*E]
expandedRowIdx:   [R, M*topK]
srcPackedX:       [R, M*topK, H]
dispatchX:        [R, R*M*topK, H]
returnY:          [R, M*topK, H]
out:              [R, M, H]
```

UB 只作为小块 staging buffer 使用。kernel 按 row 和 hidden chunk 从 GM 读取、写回，不再把 `x/srcPackedX/dispatchX/returnY/out` 整体驻留到固定 UB 区间。float 数据面统一经过 row-level helper（`CopyXToPackedRow`、`CopyPackedToDispatchRow`、`CopyDispatchToReturnRow`、`AddReturnRowToOut`）进入 `CopyGmVector` / `AddScaledGmVector`，主算法循环不直接拼接 float workspace 的裸 GM offset；metadata 仍允许在 correctness scaffold 中使用标量 GM 读写。

### 8.3 Kernel 分阶段

1. `DispatchPhaseKernel` 清零 `count/srcExpertOffset/expandedRowIdx/srcPackedX/dispatchX`。
2. 按 `src -> token -> slot` 统计 `count[src][g]`。
3. 按 `src -> g` 计算 `srcExpertOffset[src][g]`。
4. 第二遍 routing expand：计算 row，使用 runtime `hidden` row copy 把 `x[src, token, :]` 写入 `srcPackedX[src, row, :]`。
5. Dispatch：按 `dst -> e -> src -> row` 使用 runtime `hidden` row copy 从 `srcPackedX` 写入 `dispatchX`。
6. `CombineRestoreKernel` 清零 `returnY/out`。
7. Combine identity：按同一段公式从 `dispatchX` 写回 `returnY`。
8. Restore：按 `src -> token -> slot`，对 runtime `hidden` 做 `out += probs * returnY`。

### 8.4 Host 校验

`ValidateSpec` 只做真实边界校验：

- 维度必须为正。
- `R*E`、`M*topK`、各 workspace element count 和 byte count 必须用 checked multiply 计算，避免 host overflow。
- `M*topK` 和 `R*M*topK` 必须能放入 kernel 当前使用的 `int` row index。
- 固定 UB workspace layout 校验删除。

### 8.5 验收矩阵

默认 `--case all` 覆盖：

```text
minimal:       R=1 E=1 M=2 H=1 topK=1
hidden_small:  R=1 E=2 M=3 H=3 topK=2
hidden_odd:    R=2 E=2 M=4 H=7 topK=2
hidden_65:     R=2 E=2 M=4 H=65 topK=2
hidden_large:  R=2 E=2 M=4 H=257 topK=2
topk4:         R=2 E=4 M=8 H=128 topK=4
edge:          R=3 E=2 M=5 H=17 topK=2
```

额外 smoke 覆盖 `H=2/3/7/65/257/1024` 和 `topK=1/2/4/8`。

## 9. PTO-native 逻辑重写设计

本阶段不再对旧 correctness scaffold 做小修补，而是将 kernel 内部重写为 PTO-native 的分层结构。外部 launcher ABI、host 参数、CPU golden 和 README 六阶段语义保持不变。

### 9.1 顶层结构

重写后的 kernel 按四层组织：

```text
DispatchShape
  -> GM row views
  -> PTO vector primitives
  -> metadata protocol + stage graph
```

- `DispatchShape` 集中保存 runtime `R/E/M/H/topK` 及派生规模：`global_experts/max_rows/max_dispatch_rows/count_elems/packed_elems/dispatch_elems/out_elems`。
- GM row views 用 `Row()` / `At()` / `Set()` 表达逻辑坐标到 GM 地址的映射，主 stage 不再直接拼 `base + Offset(...)`。
- PTO vector primitives 负责唯一的 float 数据面 GM/UB 边界：`LoadVec`、`StoreVec`、`FillVecTile`、`CopyFloatRow`、`FillFloatRows`、`AddScaledFloatRow`。
- stage graph 只表达 README 算法语义，不直接管理 Tile 绑定和 GM offset 细节。

### 9.2 GM row view contract

当前 row view 包括：

| View | 语义 | 访问方式 |
| --- | --- | --- |
| `XRows` | `x: [R,M,H]` | `Row(shape, rank, token)` |
| `ProbRows` | `probs: [R,M,topK]` | `At(shape, rank, token, slot)` |
| `ExpertIdxRows` | `expert_idx: [R,M,topK]` | `At(shape, rank, token, slot)` |
| `ActiveRows` | `active: [R,M]` | `At(shape, rank, token)` |
| `CountRows` | `count/srcExpertOffset: [R,G]` | `At/Set/AddOne` |
| `ExpandedRows` | `expandedRowIdx: [R,M*topK]` | `Flat/AtFlat/Set` |
| `PackedRows` | `srcPackedX: [R,M*topK,H]` | `Row(shape, rank, row)` |
| `DispatchRows` | `dispatchX: [R,R*M*topK,H]` | `Row(shape, dst, row)` |
| `ReturnRows` | `returnY: [R,M*topK,H]` | `Row(shape, rank, row)` |
| `OutRows` | `out: [R,M,H]` | `Row(shape, rank, token)` |

规则：主算法 stage 只使用这些 view，不直接使用旧的 `XOffset/PackedOffset/DispatchOffset` 函数。

### 9.3 PTO Tile / GlobalTensor contract

float 数据面采用固定 staging tile contract：

```text
PtoGlobalNd<Element> = GlobalTensor<Element, dynamic 5D contiguous vector view, Layout::ND>
PtoVecTile<Element>  = Tile<TileType::Vec, Element, 1, 1024, RowMajor>
```

每次 row copy 或 restore 只处理一个 hidden chunk：

```text
GM row chunk -> GlobalTensor -> TLOAD -> PtoVecTile -> TSTORE -> GM row chunk
```

`H > 1024` 时由 `CopyFloatRow` / `AddScaledFloatRow` 按 chunk 循环，不增加 UB 常驻空间。

### 9.4 Data path 与 metadata path

float data path：

```text
x/srcPackedX/dispatchX/returnY/out
```

全部通过 PTO vector primitive 访问：

```text
CopyFloatRow
FillFloatRows
AddScaledFloatRow
```

metadata path：

```text
expert_idx/active/count/srcExpertOffset/expandedRowIdx/probs scale
```

仍保留 correctness scaffold 的 scalar GM 读写。这样把动态路由控制面与 float 数据面隔离，后续可以独立替换 metadata 统计和 prefix 策略。

### 9.5 Stage graph

`DispatchPhaseImpl` 只编排 dispatch 前半段：

```text
ClearDispatchMetadata
  -> CountReplicas
  -> BuildSourceExpertPrefix
  -> ClearDispatchData
  -> PackSourceRows
  -> BuildDispatchRows
```

`CombineRestoreImpl` 只编排 combine/restore 后半段：

```text
ClearCombineData
  -> BuildReturnRows
  -> RestoreOutputRows
```

每个 stage 的职责单一，后续替换通信或 expert compute 时优先替换 stage 内部，不改 host ABI。

### 9.6 同步策略

当前 correctness 版本继续采用单 block / 单 subblock 串行执行，并在 stage 边界使用 `pipe_barrier(PIPE_ALL)` 保守保证顺序。该同步模型只用于正确性闭环，不代表最终性能形态。

性能化版本的目标同步模型：

```text
row tile ready queue / per-segment counter / comm notify-wait / event chain
```

### 9.7 后续性能化路线

1. 将 `AddScaledFloatRow` 中的 scalar `GetTileValue/SetTileValue` 循环替换为 tile-level vector multiply/add primitive。
2. 将 metadata count/prefix 改为按 expert segment 或 block 级并行统计。
3. 将 `BuildDispatchRows` / `BuildReturnRows` 的 GM row copy 替换为 HCCL window 或 PTO comm primitive。
4. 将 identity `BuildReturnRows` 替换为真实 expert compute，并保持 dispatch row 顺序不变。
5. 在多核版本中以 `dst/local_expert/src` segment 为切分单位，使用 ready counter 保证 combine/restore 可见性。

## 10. 性能版本演进设计

本节定义从当前 PTO-native correctness 版本演进到性能版本的路线。目标不是一次性把 vector 化、多核、真实通信和 expert compute 全部揉进一个大改动，而是按可验证边界逐层替换内部实现，并保持 README 六阶段语义、host ABI、CPU golden 协议和中间张量可观测性稳定。

### 10.1 性能演进目标与非目标

目标：

- 将 float data path 从 correctness 版的逐元素 Tile 访问，演进为真正的 PTO vector primitive 数据流。
- 将 row copy 从简单 `load -> store` 演进为 ping-pong / event pipeline。
- 将 dispatch/return 从单 block 串行循环演进为按 segment 并行。
- 将本地模拟 `dispatchX/returnY` copy 替换为 PTO/HCCL remote window 通信。
- 为后续接入 GMM/FFN expert compute 保持稳定的 segment 协议。

非目标：

- 不改变 `count/srcExpertOffset/expandedRowIdx/srcPackedX/dispatchX/returnY/out` 的可观测语义。
- 不在同一阶段同时引入多核、通信、expert compute 和 HOut 泛化。
- 不牺牲 CPU golden 和动态 shape 验收来换性能。

### 10.2 总体路线

性能演进分五个阶段：

```text
P1: 单核 vector primitive restore
P2: row copy ping-pong pipeline
P3: metadata kernel + segment descriptor
P4: 多核 segment 并行
P5: 真实通信 + expert compute 接入
```

每个阶段都必须满足：

```text
correctness matrix 通过
源码 PTO 范式检查通过
性能数据有记录
可以独立回滚
```

### 10.3 P1：单核 vector primitive restore

当前 `RestoreOutputRows` 依赖 `AddScaledFloatRow`，其内部用 `GetTileValue/SetTileValue` 做逐元素累加。P1 的目标是把 restore 的核心计算替换为 tile-level vector primitive：

```text
returnY row chunk -> TLOAD -> returnTile
out row chunk     -> TLOAD -> accTile
returnTile        -> TMULS(scale)
accTile           -> TADD(accTile, returnTile)
accTile           -> TSTORE -> out row chunk
```

设计抓手：

- 新增 `ScaleAddFloatRow`，替换 `AddScaledFloatRow`。
- `ScaleAddFloatRow` 继续使用 `OutRows` / `ReturnRows` view，不改变 stage graph。
- `H` 仍 runtime dynamic，按 `kPtoVectorTileElems` 分 chunk。
- 保留单 block / 单 subblock，先只优化 restore 数据面。

P1 验收：

- `--case all`、`H=2`、`H=65`、`H=1024`、`topK=4` 全部通过。
- restore path 不再出现逐元素 `GetTileValue/SetTileValue` 累加循环。
- 记录 restore-heavy case 的运行时间，作为 P2/P4 对比基线。

### 10.4 P2：row copy ping-pong pipeline

当前 `CopyFloatRow` 的模式是串行 chunk copy：

```text
TLOAD chunk i
TSTORE chunk i
TLOAD chunk i+1
TSTORE chunk i+1
```

P2 将其演进为 ping-pong staging：

```text
TLOAD chunk 0 -> ping
TLOAD chunk 1 -> pong while TSTORE ping
TLOAD chunk 2 -> ping while TSTORE pong
...
```

UB 规划：

```text
kVecPingUb: row copy ping tile
kVecPongUb: row copy pong tile
kAccUb: restore accumulator tile
```

同步策略：

- 用 MTE2/MTE3 event 串联 `TLOAD` 与 `TSTORE`。
- 每个 row copy helper 内部维护 ping/pong state，不泄露到 stage graph。
- 对齐 `gemm_ar/comm_kernel.cpp` 中 `RsPipelineStep` 的思路：当前 tile load、上一 tile store、事件等待、最后 drain。

P2 验收：

- correctness matrix 全通过。
- 大 hidden case（例如 `H=1024/2048`）copy-heavy 路径不回退。
- `CopyFloatRow` 内部有明确 ping/pong tile storage 与 event chain。

### 10.5 P3：metadata kernel 与 segment descriptor

当前 metadata 在 dispatch kernel 内串行完成：

```text
ClearDispatchMetadata
CountReplicas
BuildSourceExpertPrefix
PackSourceRows
BuildDispatchRows
```

P3 将 metadata 预处理拆成独立 kernel，并生成 segment descriptor：

```cpp
struct SegmentDesc {
    int src;
    int dst;
    int localExpert;
    int globalExpert;
    int srcRowBase;
    int dispatchRowBase;
    int rows;
};
```

segment 定义：

```text
segment = (dst rank, local expert, src rank)
rows    = count[src][dst * E + localExpert]
```

新增 workspace：

```text
segmentDesc:  [R * E * R]
segmentReady: [R * E * R]
returnReady:  [R * E * R]
```

P3 后 stage graph 变为：

```text
MetadataKernel
  -> count / srcExpertOffset / expandedRowIdx / segmentDesc
PackKernel
DispatchKernel
ReturnKernel
RestoreKernel
```

P3 验收：

- segmentDesc 可回传 host，与 CPU golden 推导的 segment 完全一致。
- `BuildDispatchRows` / `BuildReturnRows` 改为消费 segmentDesc，而不是在内部重复三重循环推导 segment。
- 输出张量与 correctness 版本一致。

### 10.6 P4：多核 segment 并行

P4 以 segment row tile 为 block 级并行单位：

```text
blockIdx -> segmentId + rowTileId
segmentId -> (dst, localExpert, src)
rowTileId -> rows 内的局部 row range
```

推荐切分：

```text
row_tile = 根据 H 选择
H >= 1024: row_tile = 1~4
H <= 128:  row_tile = 8~16
```

并行 kernel：

```text
PackKernel:     可按 src/token range 或 expanded row range 并行
DispatchKernel: 按 segment row tile 并行
ReturnKernel:   按 segment row tile 并行
RestoreKernel:  按 src/token range 并行，避免多个 block 同写同一 out token
```

同步策略：

- Dispatch 完成后写 `segmentReady[segmentId]`。
- Return 完成后写 `returnReady[segmentId]`。
- Restore 只等待自己 token 依赖的 return segment。
- correctness 版本可先用 kernel launch 边界做全局同步，性能版本再下沉到 ready counter。

P4 验收：

- 多 block 运行 correctness matrix 全通过。
- 性能随 block 数增加有正向趋势，至少在中大 shape 上不回退。
- 无多个 block 对同一 out row 非原子并发累加。

### 10.7 P5：真实通信与 expert compute 接入

P5 将本地模拟 copy 替换为跨 rank通信，并接入 expert compute。当前实现先落地 local simulation backend seam，真实多进程 HCCL/PTO window 通信需要后续在多进程 harness 中验收，不能用当前单进程/单 NPU 用例声明完成。

Dispatch 通信：

```text
srcPackedX[src, row, :]
  -> remote dispatch window on dst rank
```

Return 通信：

```text
computeY[dst, dispatchRow, :]
  -> remote return window on src rank
```

通信协议保持 segmentDesc 不变：

```text
DispatchSegment(desc)
ReturnSegment(desc)
```

内部实现可切换：

```text
local simulation: DispatchSegmentLocal / ReturnSegmentLocal，用 GM row copy 保持当前单进程可验证闭环
identity expert: ExpertIdentitySegmentLocal，保持 dispatch row 原地有序
real comm: HCCL window / PTO comm primitive，后续多进程 harness 接入
```

当前代码已把 segment kernel 路由到 local backend helper：

```text
DispatchSegmentsKernel -> DispatchSegmentLocal
ReturnSegmentsKernel   -> ExpertIdentitySegmentLocal -> ReturnSegmentLocal
```

Expert compute 接入：

```text
dispatchX segment -> Expert(localExpert) -> computeY segment
```

约束：

- Expert compute 不得打乱 segment 内 row 顺序。
- 第一版保持 `HOut = H`，避免同时引入输出维度泛化。
- HOut 泛化作为后续阶段，届时 `returnY/out` workspace 需要从 `H` 参数扩展为 `HOut`。

P5 验收：

- 当前 local simulation backend seam：`segmentDesc/segmentReady/returnReady/srcPackedX/dispatchX/returnY/out` 与 CPU golden 一致。
- 单进程 local backend 不死锁，ready/return 由同 stream finalizer 在 segment kernel 完成后发布。
- 后续 real comm backend：单机多进程 identity 通信闭环与 CPU golden 一致，ready/return counter 不死锁，local simulation 与 real comm 在同 shape 下输出一致。

### 10.8 Tiling 与 workspace 策略

Host 侧 tiling 参数：

```text
hiddenTileElems
rowTile
segmentCount
numMetadataBlocks
numPackBlocks
numDispatchBlocks
numReturnBlocks
numRestoreBlocks
```

推荐默认：

```text
hiddenTileElems = 1024
rowTile = H >= 1024 ? 1~4 : 8~16
segmentCount = R * E * R
```

workspace：

```text
保留：
count
srcExpertOffset
expandedRowIdx
srcPackedX
dispatchX
returnY
out

新增：
segmentDesc
segmentReady
returnReady
optional metrics buffer
```

容量原则：

- UB 只随 tile contract 变化，不随 `R/E/M/topK` 增长。
- GM workspace 随 runtime shape 增长，由 host checked arithmetic 保护。
- segment workspace 规模是 `O(R * E * R)`，远小于 row data workspace。

### 10.9 性能指标与基线

每个性能阶段都记录：

```text
warmup = 5
repeat = 20
avg / p50 / p90
```

推荐 benchmark shape：

```text
R=2 E=2 M=128 H=1024 topK=2
R=4 E=4 M=256 H=1024 topK=2
R=8 E=8 M=512 H=2048 topK=4
```

指标：

```text
end-to-end kernel time
restore path time
dispatch copy GB/s
return copy GB/s
effective total data movement GB/s
scaling with block count
```

对比方式：

```text
P1 vs current PTO-native correctness
P2 vs P1
P4 vs single-block P2
P5 local simulation vs real comm
```

不接受只给绝对耗时；必须给相对变化和 case 规模。

### 10.10 回滚边界

每个阶段必须保留可回滚开关或独立 helper 边界：

- P1 只替换 restore helper。
- P2 只替换 row copy helper。
- P3 只新增 metadata kernel 和 segmentDesc，旧 stage 可保留到验证完成。
- P4 以新 kernel 替换 segment copy，旧单 block kernel 可作为 fallback。
- P5 通过 `local simulation / real comm` 两种 backend 对比验证。

原则：性能化只替换实现层，不改变 README 语义层。
