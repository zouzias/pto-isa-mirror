# Generation Event 稳定性 ST 验证记录

## 变更基线

- 仓库：`/mnt/data/ntlab/ltr/pto-isa_0608`
- 分支：`a2a3_opi`
- 基线提交：`fc5fe6033361595351235cccf7034f659b75e452`
- 环境：CANN 9.0.1，`ltr_pto`，Ascend 设备 5/6，2 MPI ranks

## 变更范围

- 在 comm ST 注册 `async_generation_stability`。
- 新增 TGET/TPUT 的立即逐 Post 检查、延迟逐 Event 检查和 512 Post 长稳用例。
- 每个 Post 使用独立数据片段；Event 有效性、Wait、原始数据及消费数据均单独检查。
- 数据消费路径为 `TLOAD -> TADDS(+100) -> TSTORE`。
- TPUT 接收端使用有界 `TTEST`，失败时不会永久阻塞设备。
- Host 侧检查 HCCL window 容量，并在 kernel 启动前同步各 rank 的初始化状态。

## 验证命令

```bash
source /home/ntlab/miniconda3/etc/profile.d/conda.sh
conda activate ltr_pto
source /mnt/data/ntlab/ltr/cann/cann/set_env.sh
export PTO_COMM_ST_FIRST_DEVICE_ID=5

python3 tests/script/run_st.py -r npu -v a3 \
  -t comm/async_generation_stability -n 2
```

重复稳定性验证：

```bash
for round in $(seq 1 5); do
  python3 tests/script/run_st.py -r npu -v a3 \
    -t comm/async_generation_stability -n 2 -w || exit 1
done
```

## 结果

- 全量构建成功。
- 6/6 个 ST 用例通过。
- 加固后的全套用例连续运行 5 轮，30/30 个 GTest 均通过。
- 每轮长稳场景分别完成 512 次 TGET Post 和 512 次 TPUT Post。
- 5 轮长稳累计检查：
  - TGET：2560 个 Post/Event；
  - TPUT：2560 个 Post/Event；
  - 合计：5120 个 Post/Event。
- 所有检查均满足：
  - `AsyncEvent::valid() == true`；
  - `AsyncEvent::Wait() == true`；
  - 原始目标数据逐元素正确；
  - 消费后数据逐元素等于原始值加 100。
- IDE lint：无新增诊断。

## 阶段审视

初版 TPUT 接收端采用无界 `TWAIT`，且部分 Host 初始化失败路径可能造成单 rank 提前返回。代码审视后已改为：

1. 有界 `TTEST` 轮询，超时写失败状态并退出；
2. kernel 启动前通过 MPI 汇总各 rank 初始化结果；
3. 校验 HCCL window 容量；
4. 清理部分分配成功后的失败路径；
5. TPUT 接收端为每个 Post 写入并校验消费状态。
6. GTest fixture 在运行前检查设备范围，设备不足时明确标记为 SKIPPED，避免无执行的假 PASS。

上述修正后重新完整构建并全量运行通过；核心安全加固版本另执行 5 轮重复验证通过。

`TestContext::Init()` 内部仍沿用 comm ST 公共框架的 MPI/HCCL 初始化流程；初始化中途的单 rank
硬件故障处理属于公共框架行为，不在本用例内改动。用例自身新增的 TPUT 数据等待均为有界轮询。

## 1 打 2 补充验证

- 运行环境：3 MPI ranks，设备 5/6/7。
- Root rank 0 分别与 peer rank 1、rank 2 通信。
- 每个 peer 连续提交 8 个 Post，共 16 个独立 Generation Event，Q=4。
- TGET：Root 逐 Event Wait，并分别消费来自两个 peer 的独立数据区域。
- TPUT：Root 逐 Event Wait 后分别通知两个 peer；两个 peer 独立消费并校验各自区域。
- 两个用例连续运行 5 轮，10/10 个 GTest 通过。
- 每轮逐元素检查两个 peer 的原始数据及 `TADDS(+100)` 消费结果。
- 泛化后原 2-rank 全套回归：6/6 通过，3-rank 专用用例按预期 SKIPPED。
