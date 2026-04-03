# PR 标题

`feat(a2a3): 新增 AllGather+GEMM 通算融合算子（M维切分 + Tile流式流水线）`

---

## 描述

新增 AllGather + GEMM 通算融合示例 `allgather_gemm`（目录 `kernels/manual/a2a3/allgather_gemm`），基于 M 维切分 + Tile 流式流水线架构，实现多卡场景下通信与计算的并发重叠，有效隐藏通信延迟。

核心设计要点：
- **双 Stream 并发架构**：AIV 通信核（TPUT）与 AIC 计算核（GEMM）分别运行在独立 stream 上，实现通信/计算 overlap。
- **Summary 单调计数器 + TWAIT 同步**：通信核每完成一个 tile 传输，通过 AtomicAdd 递增远端 summary 计数器；计算核通过 TWAIT 硬件指令零开销等待，实现"来一块算一块"的流式消费。
- **本地数据零等待优先计算**：计算核先处理本 rank 数据（无需等待通信），再流式处理远端 rank 数据。
- **数据类型**：输入 A/B 为 FP16，输出 C 为 FP32（Cube 累加精度）。
- **通信后端**：Host 侧使用 HCCL 建立 remote buffer，设备侧使用 hcomm/PTO comm 原语（TPUT、TNOTIFY、TWAIT）。
- **关键优化**：连续 K 累加流水（TMATMUL_ACC）、L1/L0 两级双缓冲、动态 tile size 计算、并行 AIV full-mesh 通信。
- **Host 端兼容处理**：`main.cpp` 在 include `common.hpp` 前将 `AICORE`/`__gm__` 重定义为空宏，使 device-only 的 `HcclRemotePtr` 模板在 `-xc++` host 编译下通过，无需修改仓库外部文件。

文件结构：
- `main.cpp`：Host 入口，含 HCCL 初始化、双 stream 调度、warmup、精度验证与性能统计
- `allgather_gemm_comm_kernel.cpp`：AIV 通信核，负责按 tile 粒度 TPUT 到远端 rank
- `allgather_gemm_compute_kernel.cpp`：AIC 计算核，实现 streaming GEMM
- `ready_queue.hpp`：TileFlagMatrix / Summary 计数器元数据定义
- `run.sh`：端到端构建运行脚本，支持多卡设备列表、CSV 批量 shape 测试、精度验证
- `scripts/`：数据生成（gen_data.py）、shape 配置（test_shapes.csv）、结果验证（verify_result.py）

## 代码质量

- 所有 C++ 文件使用 `/**` 版权声明，CMake/shell/python 使用 `# ----` 版权声明
- 所有函数控制在 50 行以内，大函数已拆分为职责清晰的子函数
- 头文件 `ready_queue.hpp` 控制在 250 行（≤500 行上限）
- 删除冗余的 `run_with_msprof.sh`，精简注释
- README 对齐 `gemm_performance` 参考格式（英文、标准章节）

## 关联的Issue

无

## 测试

- 基于上游 `cann/pto-isa` master 分支 rebase，**仅修改算子目录内的文件**，使用 CANN 9.0.0 Toolkit 编译
- 使用 2 卡（910B1）功能验证通过：M=2048, K=2048, N=1024
- 精度验证通过 `verify_result.py` 对比 FP32 golden 结果，rtol=atol=0.001，max diff=0，err count=0
- 支持 `test_shapes.csv` 批量 shape 测试（2048x2048x1024、4096x4096x2048、131072x4096x4096）
- 使用 `--perf` 模式进行性能测试，输出 TFLOPS、MFU（相对 FP16 peak）、通信带宽等指标

## 文档更新

新增 `README.md`（英文），包含 Overview、Supported AI Processors、Directory Layout、Operator Description、Architecture（双 Stream 并发 + 流式流水线图示）、Optimization Notes、Build and Run、Changelog。

## 类型标签

- [ ] Bug修复
- [x] 新特性
- [ ] 性能优化
- [ ] 文档更新
- [ ] 其他，请描述：
