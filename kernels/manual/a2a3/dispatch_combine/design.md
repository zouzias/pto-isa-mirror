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
- host 只允许 launcher 支持的 `H in {1,4,8,16,32,64}`，并校验固定 UB workspace layout 不越界；默认用例选择小 H 便于 smoke。

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

### 阶段 1：单进程多 rank identity 闭环

- 本次实现。
- 目标是证明 README 中的 count/prefix/expandedRowIdx/restore 公式正确。
- 验收：`./run.sh --case all` 通过，源码 grep 无 Ascend C namespace 调用形式。

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
