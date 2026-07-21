# Generation Event 稳定性 ST

该用例验证优化后的 `TGET_ASYNC`、`TPUT_ASYNC` Generation Event 在多 Post、Session 复用和
Payload ring 回绕场景下的正确性。

## 覆盖点

| 用例 | Post 检查方式 | Post 数 | Queue | 目的 |
|---|---|---:|---:|---|
| `TGet_Immediate_P4_Q1` | 每次 Post 后立即 Wait、消费 | 4 | 1 | 基础多 Post |
| `TGet_Deferred_P8_Q4` | 连续 Post 8 次，再逐 Event Wait、消费 | 8 | 4 | 检查每个 Generation Event |
| `TGet_Deferred_P16_Q4` | 连续 Post 16 次，再逐 Event Wait、消费 | 16 | 4 | Deferred Event 数量边界及 Payload 槽回环 |
| `TGet_Immediate_512Posts_Q4` | 每次 Post 后立即 Wait、消费 | 512 | 4 | Session 复用及 256 槽 Payload ring 回绕 |
| `TPut_Immediate_P4_Q1` | 每次 Post 后立即 Wait、通知对端消费 | 4 | 1 | 基础多 Post |
| `TPut_Deferred_P8_Q4` | 连续 Post 8 次，再逐 Event Wait、通知 | 8 | 4 | 检查每个 Generation Event |
| `TPut_Deferred_P16_Q4` | 连续 Post 16 次，再逐 Event Wait、通知 | 16 | 4 | Deferred Event 数量边界及 Payload 槽回环 |
| `TPut_Immediate_512Posts_Q4` | 每次 Post 后立即 Wait、通知 | 512 | 4 | Session 复用及 Payload ring 回绕 |
| `TGet_Deferred_P8_Q4_3Ranks` | Root 连续从两个 Peer Post，再逐 Event Wait、消费 | 每 Peer 8 | 4 | 1 打 2 TGET |
| `TPut_Deferred_P8_Q4_3Ranks` | Root 连续向两个 Peer Post，再逐 Event Wait、通知 | 每 Peer 8 | 4 | 1 打 2 TPUT |

每个 Post 使用独立的 4 KiB 数据片段和唯一数据模式。数据消费采用
`TLOAD -> TADDS(+100) -> TSTORE`，Host 同时检查：

1. `AsyncEvent::valid()`；
2. 每个 Event 的 `Wait()` 返回值；
3. 每个 Post 对应的原始目标数据；
4. 每个 Post 对应的消费后数据。

TPUT 接收端按 Post 序号执行有上限的 `TTEST` 轮询。发送端即使遇到 Event/Wait 失败也会发出通知，
使接收端读取预置的 poison 数据并让用例失败；若通知本身未到达，轮询超时后也会返回失败，避免设备因
测试失败永久阻塞。

## 运行

默认从设备 0 开始。使用设备 5/6：

```bash
source /mnt/data/ntlab/ltr/cann/cann/set_env.sh
conda activate ltr_pto
export PTO_COMM_ST_FIRST_DEVICE_ID=5
python3 tests/script/run_st.py -r npu -v a3 -t comm/async_generation_stability -n 2
```

单独运行长稳用例：

```bash
python3 tests/script/run_st.py -r npu -v a3 -t comm/async_generation_stability -n 2 \
  -g 'AsyncGenerationStability.*512Posts*'
```

使用设备 5/6/7 运行 1 打 2：

```bash
export PTO_COMM_ST_FIRST_DEVICE_ID=5
python3 tests/script/run_st.py -r npu -v a3 -t comm/async_generation_stability -n 3 \
  -g 'AsyncGenerationStabilityOneToTwo.*'
```
