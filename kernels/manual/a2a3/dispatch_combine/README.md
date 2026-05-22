# A2/A3 dispatch_combine 算法设计

## 1. 目标与边界

`dispatch_combine` 是 MoE 路由通信的算法骨架，目标是定义清楚 token 在多 rank、多 expert 场景下如何完成：

```text
本 rank 原始 token
  -> dispatch：按 topK expert 路由到目标 expert 所在 rank
  -> local expert compute：目标 rank 对本地 expert 的 token 段做计算
  -> combine：把每个 expert 的结果发回 token 原始 rank
  -> restore：按原 token 顺序和 gate 权重聚合
  -> 本 rank 输出 token
```

本文只描述算法、数据布局、索引关系和同步依赖，为后续编码做准备；不约束实际实现必须使用哪套通信库、矩阵库或 kernel helper。

当前设计默认面向 A2/A3 多卡环境，但算法本身只依赖“rank 间可以交换变长 token 段”这个抽象能力。

## 2. 项目定位：MoE dispatch/combine 闭环

`dispatch_combine` 可以理解为一个不绑定具体 FFN 实现的 MoE 通信闭环。它负责把 token 从原始 rank 送到目标 expert rank，再把 expert 输出送回原始 rank，并在原始 rank 上按 `probs` 聚合。中间的 `Expert()` 可以先用 identity 验证通信闭环，后续再替换成 GMM/FFN。

符号约定：`M` 是每个 rank 本地 token 数，`H` 是输入 token hidden size，`H_out` 是 expert 输出 hidden size；若中间计算是 identity，`H_out = H`。`topK` 是每个 token 路由到的 expert 数，`R` 是 rank 数，`E` 是每 rank 本地 expert 数，`G = R * E` 是全局 expert 数。

算法计算链路：

```text
每个 rank 本地输入 x、expert_idx、probs、active_mask
        |
        v
RoutingExpand：按 topK 展开 token，按 global expert 分组，生成 count 和 expandedRowIdx
        |
        v
CountExchange：交换 count[src][global_expert]，计算 dispatch/combine offset
        |
        v
Dispatch：跨 rank 搬运 token 段，目标 rank 得到 expert-major 的 dispatchX
        |
        v
LocalExpertCompute：目标 rank 对本地 expert 段执行 Expert()
        |
        v
CombineReturn：目标 rank 把 expert 输出按原 source 段写回 owner rank
        |
        v
Restore：owner rank 用 expandedRowIdx 找回 topK 副本，用 probs 加权求和
        |
        v
每个 rank 最终输出 out
```

各阶段主要输入输出形状：

| 阶段 | 主要输入 | 主要输出 | 说明 |
| --- | --- | --- | --- |
| 每 rank 本地输入 | `x: [M, H]`，`expert_idx: [M, topK]`，`probs: [M, topK]`，`active_mask: [M]` | - | `x` 是本 rank 原始 token；`expert_idx` 存 global expert id；`probs` 是 topK gate 权重；`active_mask` 可选。 |
| RoutingExpand | `x`，`expert_idx`，`active_mask` | `srcPackedX: [localExpandedRows, H]`，`localCount: [G]`，`expandedRowIdx: [M * topK]` | 每个有效 token 副本按 global expert 分组写入 source rank 的 packed buffer；`expandedRowIdx[flat]` 记录原 token/topK 副本对应的 packed row。 |
| CountExchange | 各 rank 的 `localCount: [G]` | 逻辑全局表 `count: [R, G]`，`srcExpertOffset: [R, G]`，目标侧 `expertBase/sourceBase` | `count[src][g]` 表示 source rank 有多少 token 副本要发给 global expert `g`；prefix 表决定 dispatch 和 combine 的读写 row。 |
| Dispatch | 各 source rank 的 `srcPackedX` 和 `count/srcExpertOffset` | 每个 dst rank 的 `dispatchX: [dispatchRows[dst], H]` | `dispatchRows[dst] = sum_src sum_e count[src][dst * E + e]`；布局为 local expert-major，每个 expert 内再按 source rank 分段。 |
| LocalExpertCompute | `dispatchX`，每个 local expert 的参数或函数 | `computeY: [dispatchRows[dst], H_out]` | 中间计算不改变 row 顺序；identity 路径下 `H_out = H`，FFN/GMM 路径下以实际 expert 输出维度为准。 |
| CombineReturn | 每个 dst rank 的 `computeY`，`count`，prefix offset | 每个 source rank 的 `returnY: [localExpandedRows, H_out]` | 按 `(src, dst, local_expert)` 段把结果写回 source rank；写回 row 与 `srcPackedX` / `expandedRowIdx` 同构。 |
| Restore | `returnY: [localExpandedRows, H_out]`，`expandedRowIdx: [M * topK]`，`probs: [M, topK]` | `out: [M, H_out]` | 对每个 token 的 topK 副本取对应 `returnY[row]`，乘 `probs` 后累加。 |

几个关键索引和计数关系：

```text
flat = token_id * topK + slot
localExpandedRows[src] = sum_g count[src][g]
dispatchRows[dst] = sum_src sum_e count[src][dst * E + e]
expandedRowIdx[flat] = 该 token 副本在 source packed buffer / returnY 中的 row
```

因此，dispatch buffer 和 computeY 中不需要为每行额外保存 source rank 或原 token id；source rank、目标 expert 和写回位置都由 `count` 与 prefix offset 推导。真正恢复原 token 语义的是最后的 `expandedRowIdx + probs`。

## 3. 符号约定

| 符号 | 含义 |
| --- | --- |
| `R` | rank 数，也就是 expert parallel world size |
| `rank` / `src` / `dst` | 当前 rank / token 原始 rank / expert 所在 rank |
| `M` | 每个 rank 本地 token 数 |
| `H` | token hidden size，也就是每行 token 的列数 |
| `topK` | 每个 token 路由到的 expert 数 |
| `E` | 每个 rank 拥有的本地 expert 数，记作 `expertPerRank` |
| `G = R * E` | 全局 expert 数 |
| `g` | global expert id，范围 `[0, G)` |
| `e` | local expert id，范围 `[0, E)` |
| `flat = token_id * topK + slot` | 本 rank 展开后的 token 副本编号 |

全局 expert 到 rank 的映射：

```text
dst_rank(g)    = g / E
local_expert(g)= g % E
g               = dst_rank * E + local_expert
```

每个 rank 的输入：

```text
x:             [M, H]
expert_idx:    [M, topK]
probs:         [M, topK]
x_active_mask: [M]，可选
```

输出：

```text
out: [M, H]
```

语义：

```text
out[token] = sum_{slot=0}^{topK-1} probs[token, slot]
             * Expert(expert_idx[token, slot], x[token])
```

如果 `x_active_mask[token] == false`，该 token 不参与真实 expert 计算；初版可以约定输出为零，或保持上层定义的默认值。编码时必须把 inactive 行的行为固定下来，并让 golden 与 device 一致。

## 4. 总体数据流

```text
每个 src rank:
  x / expert_idx / probs / active_mask
      |
      v
  1. RoutingExpand
      - token 按 topK 展开
      - 统计每个 global expert 的本地 token 数
      - 生成源 rank 内部的 expert-major packed buffer
      - 生成 expandedRowIdx，用于最后 restore
      |
      v
  2. CountExchange
      - 所有 rank 交换 count[src][global_expert]
      - 每个 dst rank 计算本地 expert 的接收 offset
      - 每个 src rank 计算返回 buffer 的写回 offset
      |
      v
  3. Dispatch
      - 对每个 dst rank，把对应 expert 的 token 段搬到 dst
      - dst rank 得到按 local expert 连续排列的 dispatch buffer
      |
      v
  4. LocalExpertCompute
      - 每个 dst rank 对本地 expert 段执行计算
      - 计算不改变行顺序
      |
      v
  5. CombineReturn
      - dst rank 把每个 source rank 的结果段写回 source
      - 写回位置复用 source routing 阶段的 packed row index
      |
      v
  6. Restore
      - source rank 用 expandedRowIdx 和 probs 做加权累加
      - 得到 out[M, H]
```

核心设计原则：

1. **只在边界保存必要元数据**：每行 dispatch 数据不额外携带 `src_rank/token_id/slot`，而是通过 count 和 prefix offset 推导。
2. **dispatch 后布局直接服务 expert compute**：目标 rank 收到的数据按 `local expert -> source rank` 排列，local expert 段连续。
3. **combine 写回复用 routing 行号**：结果回到 source rank 的 expanded buffer 后，restore 只需要 `expandedRowIdx` 就能找到每个 topK 副本的结果。
4. **中间计算保持行顺序**：local expert compute 可以改变列内容，但不能打乱行顺序，否则 combine offset 会失效。

## 5. RoutingExpand：本地展开、计数和源端 packed 布局

### 5.1 目标

每个 source rank 独立完成本地 token 展开，把 `[M, H]` 变成按 global expert 分组的 packed buffer：

```text
srcPackedX:
  expert 0 的 token 副本
  expert 1 的 token 副本
  ...
  expert G-1 的 token 副本
```

同时生成：

```text
localCount[g]               # 本 src rank 发往 global expert g 的 token 副本数
expandedRowIdx[flat]        # 原始 token/topK 副本在 srcPackedX 中的 row
validExpandedMask[flat]     # 可选；标记 inactive / dropped / invalid expert
```

### 5.2 稳定顺序

为了 CPU golden、device 实现和后续调试一致，所有 rank 都使用同一个稳定遍历顺序：

```text
for token_id in 0 .. M-1:
  for slot in 0 .. topK-1:
    flat = token_id * topK + slot
```

同一个 expert 内，token 副本按上述 `flat` 顺序追加。即使同一个 token 的两个 slot 指向同一个 expert，也当作两个独立副本，保留两份 row，restore 时分别乘各自的 `probs`。

### 5.3 两遍扫描算法

第一遍统计 count：

```text
localCount[g] = 0 for g in [0, G)

for token_id in 0 .. M-1:
  active = x_active_mask 不存在 or x_active_mask[token_id]
  for slot in 0 .. topK-1:
    flat = token_id * topK + slot
    if not active:
      validExpandedMask[flat] = false
      continue

    g = expert_idx[token_id, slot]
    if g < 0 or g >= G:
      validExpandedMask[flat] = false
      continue

    validExpandedMask[flat] = true
    localCount[g] += 1
```

计算 source rank 内每个 expert 的起始 offset：

```text
srcExpertOffset[0] = 0
for g in 1 .. G-1:
  srcExpertOffset[g] = srcExpertOffset[g-1] + localCount[g-1]

localExpandedRows = sum_g localCount[g]
```

第二遍写 packed buffer：

```text
writeCursor[g] = srcExpertOffset[g]

for token_id in 0 .. M-1:
  for slot in 0 .. topK-1:
    flat = token_id * topK + slot
    if not validExpandedMask[flat]:
      expandedRowIdx[flat] = -1
      continue

    g = expert_idx[token_id, slot]
    row = writeCursor[g]
    writeCursor[g] += 1

    srcPackedX[row, :] = x[token_id, :]
    expandedRowIdx[flat] = row
```

这个阶段完成后，source rank 内部有一个可被所有目标 expert rank 按段读取的 outgoing buffer。段地址完全由 `srcExpertOffset[g]` 和 `localCount[g]` 决定。

## 6. CountExchange：全局计数表和 prefix offset

### 6.1 需要交换什么

每个 rank 只需要把自己的 `localCount[g]` 分享给所有 rank。交换完成后，所有 rank 逻辑上都能看到：

```text
count[src][g]
```

等价三维视角：

```text
count3d[src][dst][e] = count[src][dst * E + e]
```

### 6.2 目标 rank 侧 offset

对于目标 rank `dst`，它只关心本地 expert：

```text
g = dst * E + e
```

某个 local expert `e` 总共收到：

```text
expertRows[dst][e] = sum_{src=0}^{R-1} count[src][dst * E + e]
```

目标 rank 的 dispatch buffer 采用 expert-major 布局：

```text
dispatchX on dst:
  local expert 0:
    from src 0
    from src 1
    ...
    from src R-1
  local expert 1:
    from src 0
    from src 1
    ...
  ...
```

对应 prefix：

```text
expertBase[0] = 0
for e in 1 .. E-1:
  expertBase[e] = expertBase[e-1] + expertRows[dst][e-1]

sourceBase[e][0] = 0
for src in 1 .. R-1:
  sourceBase[e][src] = sourceBase[e][src-1] + count[src-1][dst * E + e]
```

因此 source `src` 发给目标 `dst` 的 local expert `e` 的 token 段，在 `dst` rank 的 dispatch buffer 中位置为：

```text
dstDispatchOffset(dst, e, src) = expertBase[e] + sourceBase[e][src]
rows                            = count[src][dst * E + e]
```

### 6.3 source rank 侧回写 offset

combine 时，目标 expert rank 要把结果写回 source rank 的 expanded result buffer。source rank 早在 RoutingExpand 阶段已经定义了每个 global expert 在 source packed buffer 里的位置：

```text
srcReturnOffset(src, dst, e) = srcExpertOffset[src][dst * E + e]
rows                         = count[src][dst * E + e]
```

这里的 `srcExpertOffset[src][g]` 可以由 source 本地保存，也可以由所有 rank 根据 `count[src][*]` 重建。后续编码时二选一即可，关键是公式必须保持一致。

## 7. Dispatch：变长 token 段搬运到目标 expert rank

Dispatch 的算法语义是把每个 source rank 的 `srcPackedX` 中属于某个目标 expert 的连续段，放到目标 rank 的 expert-major dispatch buffer。

对目标 rank `dst`：

```text
for e in 0 .. E-1:
  g = dst * E + e
  for src in 0 .. R-1:
    rows = count[src][g]
    if rows == 0:
      continue

    read_row  = srcExpertOffset[src][g]
    write_row = expertBase[e] + sourceBase[e][src]

    dispatchX_dst[write_row : write_row + rows, :] =
        srcPackedX_src[read_row : read_row + rows, :]
```

Dispatch 输出：

```text
dispatchX_dst: [sum_e expertRows[dst][e], H]
expertRows[e]
segmentOffset[e][src] = expertBase[e] + sourceBase[e][src]
segmentRows[e][src]   = count[src][dst * E + e]
```

`dispatchX_dst` 的每个 local expert 段连续，因此 local expert compute 可以按 expert group 直接消费：

```text
expert e input = dispatchX_dst[expertBase[e] : expertBase[e] + expertRows[e], :]
```

## 8. LocalExpertCompute：中间计算契约

`dispatch_combine` 本身只关心 dispatch 和 combine 的通信闭环，中间的 expert 计算可以是：

1. identity，用于验证 dispatch/combine 自身正确性；
2. 单个 FFN；
3. `GMM1 -> activation -> GMM2`；
4. 任意逐 expert 函数。

算法契约只有两条：

```text
expertInput_e:  [expertRows[e], H_in]
expertOutput_e: [expertRows[e], H_out]
```

1. 同一个 expert 段内，输出行顺序必须和输入行顺序一致。
2. 不同 expert 的输出仍按 `expertBase[e]` 连续存放。

也就是说：

```text
computeY_dst[expertBase[e] + local_row, :]
  = Expert(dst * E + e, dispatchX_dst[expertBase[e] + local_row, :])
```

如果后续接入 GMM/FFN，只需要保证输出 `computeY_dst` 仍使用同一套 row offset，combine 算法不用改变。

## 9. CombineReturn：专家结果写回 token 原始 rank

Combine 的输入是目标 rank 上的 `computeY_dst`，布局与 dispatch 输出完全一致：

```text
local expert e:
  from src 0 segment
  from src 1 segment
  ...
```

Combine 要把每个 `(dst, e, src)` 段写回 source rank 的 expanded result buffer：

```text
for e in 0 .. E-1:
  g = dst * E + e
  for src in 0 .. R-1:
    rows = count[src][g]
    if rows == 0:
      continue

    read_row  = expertBase[e] + sourceBase[e][src]
    write_row = srcExpertOffset[src][g]

    returnY_src[write_row : write_row + rows, :] =
        computeY_dst[read_row : read_row + rows, :]
```

关键点：

1. `read_row` 是 dispatch 写入目标 rank 时使用的位置。
2. `write_row` 是 source rank RoutingExpand 给该 global expert 预留的位置。
3. 因为 dispatch 和 combine 使用同一份 `count[src][g]`，所以每个 row 都能一一对应回 source packed row。
4. 每行不需要保存原 token id；restore 阶段由 `expandedRowIdx` 负责恢复 token/topK 语义。

Combine 完成后，每个 source rank 得到：

```text
returnY_src: [localExpandedRows, H_out]
```

其 row index 与 `srcPackedX`、`expandedRowIdx` 完全同构。

## 10. Restore：按原 token 顺序加权聚合

Restore 只在 token 原始 rank 上执行。输入：

```text
returnY:        [localExpandedRows, H_out]
expandedRowIdx: [M * topK]
probs:          [M, topK]
valid mask:     [M * topK]，可选
```

算法：

```text
for token_id in 0 .. M-1:
  out[token_id, :] = 0

  for slot in 0 .. topK-1:
    flat = token_id * topK + slot
    row = expandedRowIdx[flat]
    if row < 0:
      continue

    out[token_id, :] += probs[token_id, slot] * returnY[row, :]
```

如果 `topK` 中存在重复 expert，重复副本会以不同 `flat` 进入 restore，因此会自然累加多次。是否在 routing 前合并重复 expert，不属于本设计默认行为；初版建议不要合并，先保持语义简单和可验证。

## 11. 必须保持的索引不变量

后续编码时可以用下面的不变量做 assert 或 debug print：

### 11.1 count 守恒

每个 source rank：

```text
localExpandedRows[src] = sum_g count[src][g]
```

全局：

```text
sum_src localExpandedRows[src]
  = 所有 active token 的有效 topK 副本总数
```

每个目标 rank：

```text
dispatchRows[dst] = sum_e expertRows[dst][e]
                  = sum_src sum_e count[src][dst * E + e]
```

### 11.2 dispatch/combine row 对称

对于任意 `(src, dst, e)`：

```text
rows = count[src][dst * E + e]

Dispatch:
  srcPackedX[srcExpertOffset[src][g] : +rows]
    -> dispatchX_dst[expertBase[e] + sourceBase[e][src] : +rows]

Combine:
  computeY_dst[expertBase[e] + sourceBase[e][src] : +rows]
    -> returnY_src[srcExpertOffset[src][g] : +rows]
```

只要 compute 不改变行顺序，`returnY_src[row]` 就是 `srcPackedX[row]` 对应 expert 的计算结果。

### 11.3 expandedRowIdx 完整性

对每个有效 `flat`：

```text
0 <= expandedRowIdx[flat] < localExpandedRows
```

并且同一个 source rank 内所有有效 `flat` 的 `expandedRowIdx` 应互不相同。无效 `flat` 使用 `-1` 或统一 sentinel。

## 12. 容量、截断和异常输入

初版建议采用最简单的容量策略：

```text
maxDispatchRowsPerRank >= M * topK
maxReturnRowsPerRank   >= M * topK
```

也就是不做 drop、不做 expert capacity 截断，先保证 dispatch/combine 闭环正确。

如果后续必须支持 capacity，需要明确以下规则：

1. 截断发生在 RoutingExpand 阶段，而不是 dispatch 之后。
2. 截断顺序使用稳定 `flat` 顺序，确保 CPU golden 和 device 一致。
3. 被截断的副本：
   ```text
   validExpandedMask[flat] = false
   expandedRowIdx[flat] = -1
   ```
4. `count[src][g]` 只统计保留下来的副本。
5. Restore 跳过被截断副本，是否重归一化 `probs` 由上层语义决定；初版建议不重归一化。

异常输入建议：

| 场景 | 初版建议 |
| --- | --- |
| `expert_idx` 越界 | 视为无效副本，`expandedRowIdx=-1` |
| inactive token | 所有 slot 视为无效副本 |
| 某 expert 没有 token | count 为 0，dispatch/combine 跳过 |
| 某 rank 没有任何 token 发给另一个 rank | 对应 segment rows 为 0，offset 仍可计算 |
| 重复 expert | 不合并，保留多个副本 |

## 13. 同步依赖

算法上需要三个同步边界：

### 13.1 Count ready

所有 rank 必须先完成本地 `localCount` 生成，并让其他 rank 可见，之后才能计算 dispatch/combine offset。

```text
RoutingExpand local count done
  -> CountExchange done
  -> prefix offset valid
```

### 13.2 Dispatch payload ready

某个目标 expert 段开始 compute 前，该 expert 需要的所有 source segment 至少要满足对应数据可见。

保守版本：

```text
all dispatch segments for dst rank done
  -> local expert compute starts
```

流水版本：

```text
segment or expert e dispatch done
  -> expert e compute can start
```

初版编码建议先做保守版本，闭环正确后再拆成 expert 粒度或 tile 粒度流水。

### 13.3 Combine payload ready

source rank 开始 restore 前，所有发往该 source 的 combine segment 必须写完并可见。

```text
all combine returns to src done
  -> Restore starts
```

流水 restore 理论上可以按 token/topK readiness 做，但初版不建议这么做；先保留一个 rank 级或 all-rank barrier 更容易验证。

## 14. 推荐的编码阶段

### 阶段 1：单 rank identity 闭环

目标：验证 RoutingExpand 和 Restore 的 row mapping。

```text
R = 1
Expert(x) = x
out[token] = sum_slot probs[token, slot] * x[token]
```

如果 `probs` 每行和为 1，则输出应等于输入；如果不为 1，则输出应等于输入乘以该 token 的 `sum(probs)`。

### 阶段 2：多 rank count + offset 纯 CPU / host golden

目标：验证 `count[src][g]`、`srcExpertOffset`、`expertBase/sourceBase` 的公式。

检查：

```text
sum source segments == dispatchRows[dst]
combine write rows exactly cover returnY_src valid rows
```

### 阶段 3：多 rank identity dispatch/combine

目标：先不接 FFN，只验证跨 rank dispatch/combine。

```text
computeY_dst = dispatchX_dst
```

Restore 结果应与 CPU golden 完全一致。

### 阶段 4：接入真实 expert compute

目标：保持通信布局不变，只替换中间 `Expert()`。

要求：

1. 每个 expert 输入段 `[expertBase[e], expertBase[e] + expertRows[e])` 连续。
2. compute 输出不改变行顺序。
3. combine 继续使用同一份 segment offset。

### 阶段 5：做流水优化

在正确性稳定后再考虑：

1. dispatch 按 expert 粒度释放 compute；
2. compute 按 expert 或 tile 粒度释放 combine；
3. combine 完成后按 rank 粒度释放 restore；
4. 减少全局 barrier。

## 15. CPU golden 伪代码

下面是最直接的参考实现，后续写测试时建议先按这个逻辑生成 golden：

```python
# inputs per rank:
# x[src][M, H], expert_idx[src][M, topK], probs[src][M, topK], active[src][M]

count = zeros([R, G], int)
valid = {}

for src in range(R):
    for token in range(M):
        is_active = active[src][token] if active is not None else True
        for slot in range(topK):
            flat = token * topK + slot
            g = expert_idx[src][token, slot]
            ok = is_active and 0 <= g < G
            valid[(src, flat)] = ok
            if ok:
                count[src, g] += 1

srcExpertOffset = zeros([R, G], int)
for src in range(R):
    running = 0
    for g in range(G):
        srcExpertOffset[src, g] = running
        running += count[src, g]

srcPackedX = [zeros([sum(count[src]), H]) for src in range(R)]
expandedRowIdx = [full([M * topK], -1) for src in range(R)]
writeCursor = srcExpertOffset.copy()

for src in range(R):
    for token in range(M):
        for slot in range(topK):
            flat = token * topK + slot
            if not valid[(src, flat)]:
                continue
            g = expert_idx[src][token, slot]
            row = writeCursor[src, g]
            writeCursor[src, g] += 1
            srcPackedX[src][row] = x[src][token]
            expandedRowIdx[src][flat] = row

returnY = [zeros_like(srcPackedX[src]) for src in range(R)]

for dst in range(R):
    dispatchRows = []
    meta = []

    for e in range(E):
        g = dst * E + e
        for src in range(R):
            rows = count[src, g]
            read = srcExpertOffset[src, g]
            for r in range(rows):
                dispatchRows.append(srcPackedX[src][read + r])
                meta.append((src, g, r))

    computeRows = ExpertBatch(dst, dispatchRows, meta)  # identity 或真实 expert

    cursor = 0
    for e in range(E):
        g = dst * E + e
        for src in range(R):
            rows = count[src, g]
            write = srcExpertOffset[src, g]
            returnY[src][write : write + rows] = computeRows[cursor : cursor + rows]
            cursor += rows

out = [zeros([M, H_out]) for _ in range(R)]
for src in range(R):
    for token in range(M):
        for slot in range(topK):
            flat = token * topK + slot
            row = expandedRowIdx[src][flat]
            if row < 0:
                continue
            out[src][token] += probs[src][token, slot] * returnY[src][row]
```

## 16. 最小测试矩阵

| 用例 | 目的 |
| --- | --- |
| `R=1, E=1, topK=1` | 最小闭环，验证 row mapping |
| `R=1, E>1, topK>1` | 单卡多 expert，验证 expert 分组和重复 expert |
| `R=2, E=1, topK=1` | 最小跨 rank dispatch/combine |
| `R=2, E=2, topK=2` | 覆盖 `src/dst/e` 三维 offset |
| 某个 expert count 为 0 | 验证空段跳过和 offset 连续性 |
| 某个 rank 完全不发给另一个 rank | 验证 zero rows segment |
| inactive token | 验证 sentinel / skip 行为 |
| duplicate expert in topK | 验证重复副本不合并 |
| 非均匀 routing，所有 token 偏向一个 expert | 验证容量和 `max rows` 边界 |
| identity expert + probs 行和为 1 | 输出应回到输入，用于快速 smoke |
| identity expert + probs 行和不为 1 | 输出应等于输入乘以 probs 行和 |

## 17. 后续编码时的关键抓手

1. 先把 `count[src][g]`、`srcExpertOffset[src][g]`、`expertBase[e]`、`sourceBase[e][src]` 打印或回传出来，对照 CPU golden。
2. 先实现 identity expert，避免把通信错误和 FFN 精度错误混在一起。
3. dispatch 和 combine 必须共用同一份 count/prefix 公式，不要分别写两套近似逻辑。
4. 所有无效副本都统一使用 `expandedRowIdx=-1`，restore 只认这一种 skip 规则。
5. 初版用全局同步换正确性；性能优化阶段再把同步拆细。
6. 一旦引入 capacity/drop，必须把 drop 发生点固定在 RoutingExpand，并同步更新 count、row map 和 golden。
