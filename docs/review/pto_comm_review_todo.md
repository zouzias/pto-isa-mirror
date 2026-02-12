## pto-comm review 待改清单

说明：以下条目仅为 review 发现的问题/改进点清单，未实际修改代码。请逐项确认是否需要改动。

### 必须修复（会影响构建/正确性）
- 修复 `TGATHER_IMPL` 中 `rootIdx` 未定义（debug 构建会报错）。
  - 位置：`include/pto/comm/TGather.hpp`
  - 建议：在 `PTO_ASSERT` 前补 `const int rootIdx = parallelGroup.GetRootIdx();`
- 修复 `TSCATTER_IMPL` 中 `rootIdx` 未定义（debug 构建会报错）。
  - 位置：`include/pto/comm/TScatter.hpp`
  - 建议：在 `PTO_ASSERT` 前补 `const int rootIdx = parallelGroup.GetRootIdx();`

### 高优先级（可扩展性/语义一致性）
- `TREDUCE` ping-pong 对 rank 数硬编码限制（<=17）。
  - 位置：`include/pto/comm/TReduce.hpp`
  - 建议：去除限制，因为本来不应该有rank数限制。
- 集体通信“仅 root 执行”的语义未在实现中显式防护。
  - 位置：`include/pto/comm/pto_comm_inst.hpp` + 各 `*_IMPL`
  - 建议：在接口层或实现层增加非 root 调用的断言/保护；或文档明确禁止。
- `TGATHER/TSCATTER` 假设所有 rank shape/stride 一致但未校验。
  - 位置：`include/pto/comm/TGather.hpp`, `include/pto/comm/TScatter.hpp`
  - 建议：debug 下增加一致性检查，或文档强调严格一致。

### 文档一致性/完整性
- README 提到 `TGET_ASYNC`，但 `docs/isa/comm` 缺失对应文档。
  - 位置：`docs/isa/comm/README.md`
  - 建议：补 `TGET_ASYNC` 说明（接口、限制、示例）。
- `TPUT_ASYNC` 对 URMA（1D）限制未写清楚。
  - 位置：`docs/isa/comm/TPUT_ASYNC.md`
  - 建议：目前SDMA 也是 1D 的，需要改成这个限制
- `TGATHER/TSCATTER` 对 `shape3 == N*H`/`>` 的行为未明确。
  - 位置：`docs/isa/comm/TGATHER.md`, `docs/isa/comm/TSCATTER.md`
  - 建议：明确等于/大于时的语义与越界处理。

### 测试补齐建议
- `TTEST`/`TWAIT` 子区域测试内核存在但 main 未覆盖。
  - 位置：`tests/npu/a2a3/comm/st/testcase/ttest/main.cpp`, `.../twait/main.cpp`
  - 建议：新增用例调用 `TTestSubRegionKernel` / `TWaitSubRegionKernel`。
- 集体通信 root != 0 场景覆盖不足。
  - 位置：`tests/npu/a2a3/comm/st/testcase/tgather`, `.../tscatter`, `.../treduce`
  - 建议：补 root 变化用例（root=1/2）。
- 空数据/边界分块未覆盖。
  - 位置：同上
  - 建议：`count=0`、tile 边界等案例。
