# A5 VfSim Costmodel：PR 评审说明（按变更文件检索）

本文只说明当前未提交 PR **新增、修改、删除了什么，以及为什么这样改**。每项使用同一格式：`变更`、`内容`、`原因`、`评审点`。编号可用于评审会直接定位。

> 按约定，不讨论旧 `include/pto/costmodel/a5/VfSim/` 文件迁移到 `pkg_inc/` 的目录位置本身；下文只说明迁移后新增的能力与接口变化。

## A. 构建与集成

### A01 · `cmake/a5_vf_mock.cmake` 删除；`pkg_inc/pto/costmodel/vfsim/cmake/a5_vf_mock.cmake` 新增

- **变更：** 原仓内 helper 被替换为随安装包携带的 helper；新增 `target_enable_a5_vf_mock()`、native/pass 构建、JSON 同步、LLVM/Clang 检查、安装路径校验。
- **内容：** 用户 target 调用 helper 后，会自动得到 `PtoLoopTracePass`、`pto_a5_vfsim`、`-fpass-plugin`、所需 include/link 和 build 目录下的配置文件。
- **原因：** VfSim 不是 header-only，用户不能只 include PTO 头文件；必须把 pass、静态库和 JSON 作为一个完整能力接入。
- **评审点：** helper 的路径是否完全可重定位；Clang/LLVM major 是否强制一致；配置是否每次同步；多套 PTO 安装被同时 include 时是否明确报错而非静默串用。

### A02 · `pkg_inc/pto/costmodel/vfsim/native/CMakeLists.txt` 新增

- **变更：** 新建 native core 的 object/static library 构建规则，供 A01 的 helper 链接。
- **内容：** 将 native source 汇总为 `vfsim_native_core_objects`，再由 `pto_a5_vfsim` 吸收，消费者只需链接一个 archive。
- **原因：** 避免外部用户分别管理 adapter 和 native core 的链接顺序。
- **评审点：** CMake 最低版本与项目环境是否匹配；target 名称是否会与其他安装实例冲突。

## B. PTO Costmodel 接口、trace 与 Host 兼容

### B01 · `include/pto/costmodel/pto_instr.hpp` 修改

- **变更：** 增加/补齐 A5 的 PTO 接口映射，包括精度模板参数、`SYNCALL`、以及可由 Host mock 接住的 `TDIVS`、`TRSQRT`、`TQUANT`、`TMOV`、`TSCATTER` 和访存入口。
- **内容：** 每次 PTO 调用创建 trace scope，执行对应 `_IMPL` mock，收集周期后写入 `PtoRecorder`；A5 设备 `SyncAll.hpp` 不在 Host costmodel 下直接编译。
- **原因：** 上层 kernel 必须沿用 NPU 的 PTO API；Host 路径只替换最终实现为采集/建模，不能要求业务代码改写。
- **评审点：** 逐项对照 `include/pto/common/pto_instr.hpp` 和 `include/pto/npu/a5/` 的签名、模板默认值、参数顺序和 overload；访存周期只能由带宽模型或本文件的记录逻辑结算一次。

### B02 · `include/pto/costmodel/trace.hpp` 修改

- **变更：** 为 A5 PTO record 增加 `vf_infos`；在顶层 PTO scope 结束时调用 VfSim 并将得到的周期加入当前 PTO 总周期。
- **内容：** 保持原有 CCE pipe queue/tail 机制，同时把一段 `__VEC_SCOPE__` 采集结果与其所属 PTO 指令关联。
- **原因：** VfSim 只输出 vector body 周期，必须在 PTO 指令边界统一汇总，才能交给 perf-sim。
- **评审点：** 嵌套 PTO scope 不得重复触发预测；没有 VF trace 时不得改变原周期；Vector tail 与同步 barrier 的结算顺序是否正确。

### B03 · `include/pto/costmodel/perf_sim/config.hpp`、`costmodel_provider.hpp` 修改

- **变更：** 补充 A5 costmodel 可识别的指令/数据类型/周期提供逻辑。
- **内容：** 让 recorder 能为没有 VfSim body 的路径保留原有估计，并把记录映射到正确 pipe stage。
- **原因：** VfSim 覆盖的是 Vector body，不覆盖所有 PTO 指令和所有输入形式。
- **评审点：** VfSim 命中和公式估计的选择条件是否清晰；不存在 VF trace 时不能得到 0 周期或覆盖访存周期。

### B04 · `include/pto/costmodel/common/qualifiers.hpp`、`include/pto/common/cpu_stub.hpp`、`include/pto/costmodel/common/aclrt_stub.hpp` 修改

- **变更：** 增补 A5 Host 编译需要的 ACL 类型/返回码、qualifier、同步函数选择和 Host runtime 行为。
- **内容：** 让 A5 设备头在普通 C++ 编译器中可解析，并让 allocation/copy/stream 等 Host 测试可运行。
- **原因：** 原 CPU stub 不足以覆盖 A5 设备头所引用的编译期符号和 runtime 接口。
- **评审点：** 兼容代码不得影响 CPU_SIM；同步入口应调用 A5 costmodel mock，不能靠不可追踪的宏改名绕开。

## C. A5 CCE mock 与 VF 采集

### C01 · `include/pto/costmodel/a5/cce_costmodel/cce_costmodel.hpp` 修改

- **变更：** 聚合 A5 vector stub、同步 mock 与 VfSim cost 接口。
- **内容：** 调整 include 顺序，使后续 A5 PTO 设备头看到 Host mock。
- **原因：** A5 costmodel 需要在解析设备实现前替换 CCE intrinsic。
- **评审点：** 此文件只应聚合；高层 PTO 语义不应落在这里。

### C02 · `include/pto/costmodel/a5/cce_costmodel/a5_vf_stub.hpp` 修改

- **变更：** 新增 vector/predicate/地址兼容类型，扩展大量 CCE vector intrinsic 的 Host stub，并在 `__VEC_SCOPE__` 内记录 operand、immediate、config 和 memory location；新增 scope sentinel 将事件组装为 `VfInfo`。
- **内容：** 例如 vector load/store、算术、predicate、gather 等调用不再尝试执行，而是生成接近 ISA 的 VF 事件序列。
- **原因：** VfSim 无法理解 PTO Tile 或 C++ 实现细节，需要 intrinsic 层提供最接近硬件的程序描述。
- **评审点：** op 名必须和 catalog 对齐；dst/src、immediate/config、UB/寄存器/predicate 的 location 必须正确；类型兼容只能服务编译，不得伪造数据结果；高层 `TXXX_IMPL` 不应持续放入 CCE stub。

### C03 · `include/pto/costmodel/a5/cce_costmodel/cce_costmodel_sync.hpp` 新增

- **变更：** 新增 A5 Host 同步 mock：`pipe_barrier`、`set_flag`、`wait_flag`、`wait_flag_dev`、FFTS、intra-block、`mem_bar`、`dsb`。
- **内容：** 普通 pipe/event 同步写入 `SyncRecorder`；VF scope 内 memory barrier 记录为 VfSim event。
- **原因：** A5 同步设备实现涉及原子/设备 runtime，Host 上不可执行，但 perf-sim 仍需要依赖关系。
- **评审点：** 普通同步和 VF memory barrier 的归属是否正确；wait 是否刷新 Vector tail；不得意外进入设备 atomic 路径。

### C04 · `include/pto/costmodel/a5/cce_costmodel/vf_info.hpp` 修改

- **变更：** 扩展 VF 指令树、operand kind/location、loop、memory barrier 等 PTO 侧描述。
- **内容：** 这是 C02 采集结果和 adapter 输入之间的内部数据结构。
- **原因：** 需要在不让 native model 依赖 PTO Tile 的前提下保留足够的 ISA 语义。
- **评审点：** 这是内部格式；不要把 native API 或 perf-sim 调试字段耦合进来。

### C05 · `include/pto/costmodel/a5/cce_costmodel/vf_cost.hpp` 修改；`vfsim_cost_model.hpp` 新增

- **变更：** `vf_cost.hpp` 从简单占位周期改为调用 adapter；新增公开的预测结果/选项声明头。
- **内容：** 输入 `VfInfo`，输出预测周期和状态；公开声明留在 `include/`，实现仍在 `pkg_inc`。
- **原因：** 需要从 trace 实际调用 native VfSim，同时不能让 public header 依赖内部 include 路径。
- **评审点：** 预测失败和未命中的状态语义是否稳定；公共头是否仅依赖公开头；调试状态不应写入每条通用 perf-sim record。

## D. Adapter、native simulator 与配置

### D01 · `pkg_inc/pto/costmodel/vfsim/pto_adapter/pto_canonical_lowering.{hpp,cpp}` 新增

- **变更：** 新增 PTO `VfInfo` 到 native `CanonicalVfInfo` 的 lowering。
- **内容：** 转换指令、operand、循环和 memory barrier，并验证可表示性。
- **原因：** PTO 采集格式会演进；native model 需要稳定、可校验、与 PTO 解耦的输入。
- **评审点：** unknown/unsupported form 不应生成错误程序；所有 C02 记录的 operand 形式均应有明确 lowering 规则。

### D02 · `pkg_inc/pto/costmodel/vfsim/pto_adapter/vfsim_cost_model.{hpp,cpp}` 新增

- **变更：** 新增 adapter 运行入口：lowering、配置定位、native runner 调用、结果转换。
- **内容：** 为调用方返回 cycles、命中/降级状态、忽略指令数和诊断。
- **原因：** 将 PTO 对 native model 的依赖收敛到一个边界。
- **评审点：** 内部 header 不应成为 public dependency；每类失败状态是否可预测、可测试。

### D03 · `pkg_inc/pto/costmodel/vfsim/api/native/{CanonicalVfInfo,RuntimeTypes,RuntimeValue}.{h,cpp}` 新增

- **变更：** 新增 native API 的 canonical program、operand 类型和值表示。
- **内容：** native simulator 通过这些类型读取程序，不再读取 PTO trace 结构。
- **原因：** 防止 simulator 与 PTO mock 直接耦合。
- **评审点：** 类型、单位、默认值和所有权边界稳定；无 PTO include 依赖。

### D04 · `pkg_inc/pto/costmodel/vfsim/api/native/{InstructionCatalog,UarchOverrideSchema}.{h,cpp}` 及 `generated/*.inc` 新增

- **变更：** 新增支持指令的 catalog、operand 校验和 uarch override schema；生成编译期表。
- **内容：** native model 可据此拒绝不支持形式、解释每个 operand 并加载可覆盖参数。
- **原因：** 将“支持什么指令、参数如何解释”从散落 C++ 分支收敛为可校验数据契约。
- **评审点：** JSON、生成 `.inc` 和 C02/C01 op 名一致；新增 intrinsic 必须同步更新 catalog、lowering、测试。

### D05 · `pkg_inc/pto/costmodel/vfsim/native/{Json,ParamCompat,ParamDB,ISATraits,CanonicalProgramLowering,ControlUnit,IFU,IDU,OOO,SimulatorRunner}.{h,cpp}` 新增

- **变更：** 新增 native VfSim 主体：JSON/参数读取、ISA traits、canonical lowering、取指/译码/发射/乱序和运行入口。
- **内容：** 将 canonical program 与 JSON uarch 参数转为周期结果。
- **原因：** 旧 PTO costmodel 无法描述 A5 VF 的流水、forwarding、initiation interval 与资源约束。
- **评审点：** 参数缺失不能静默给出不可解释默认值；cycle 单位一致；native 层不应 include PTO/CCE mock。

### D06 · `pkg_inc/pto/costmodel/vfsim/configs/{instruction_catalog,uarch_override_schema,isa,uarch,forwarding,InitiationInterval}.json` 新增

- **变更：** 新增 ISA、微架构、forwarding、II、catalog/schema 配置。
- **内容：** 模型参数从代码中抽离，可在不改变 runner 框架时校准。
- **原因：** 性能模型需要频繁校准，数据化比在 C++ 中硬编码更可维护。
- **评审点：** 文件名/加载路径/CMake copy 路径一致；字段单位、默认值、生成关系可追溯。

## E. 测试与文档

### E01 · `tests/costmodel/st_a5/CMakeLists.txt`、`testcase/CMakeLists.txt` 修改

- **变更：** 注册 A5 VfSim ST 和 helper 配置。
- **内容：** 将 adapter、Host runtime、memory/sync、full-host 和基础 tileop 纳入统一 CTest。
- **原因：** 该能力横跨 pass、native library、配置和 header，需端到端回归。
- **评审点：** 每类测试是否通过统一入口构建，避免只在仓内特殊路径可运行。

### E02 · `tests/costmodel/st_a5/testcase/common/a5_vfsim_tileop_check.hpp` 修改；`{tadd,tadds,tsub,tsubs,tmul,tmuls,tdiv,tmin,tmins,tmax,tand,tshls,tshrs}/main.cpp` 修改

- **变更：** 现有基础 tileop 用例接入 VfSim trace/cycle 断言。
- **内容：** 验证 PTO API 到 CCE mock、loop pass、adapter、VfSim 的端到端路径。
- **原因：** 单测 native runner 不足以证明真实 PTO tileop 会产生正确输入。
- **评审点：** 断言稳定语义（VfInfo/cycle/结果），不应绑定临时诊断文本；覆盖的数据类型和操作形式应与“支持范围”一致。

### E03 · `tests/costmodel/st_a5/testcase/adapter_golden/*` 新增

- **变更：** 新增 adapter/canonical lowering golden tests。
- **内容：** 覆盖 catalog 命中/不命中、operand 表达、循环、预测状态和异常输入。
- **原因：** adapter 是 PTO 与 native 的协议边界，需要比 tileop ST 更细的契约验证。
- **评审点：** 重点查看不支持形式与非法 trace 的预期，不应默默算出正常结果。

### E04 · `tests/costmodel/st_a5/testcase/memory_sync/*` 新增

- **变更：** 新增访存和同步的专项测试。
- **内容：** 验证 `TLOAD/TSTORE` 进入带宽/pipe 模型；普通同步进入 `SyncRecorder`；VF mem barrier 才进入 `VfInfo`。
- **原因：** 防止引入 VfSim 后访存/同步丢失、错路由或重复计费。
- **评审点：** 这是验证 B01/B02/C03 分层的核心用例。

### E05 · `tests/costmodel/st_a5/testcase/host_runtime/*` 新增

- **变更：** 新增 ACL Host runtime stub 测试。
- **内容：** 覆盖 allocation、copy、stream/device 生命周期和非法 buffer range。
- **原因：** A5 Host path 需要实际执行这些 stub，不能只满足头文件编译。
- **评审点：** 错误处理、边界检查和资源释放是否合理。

### E06 · `tests/costmodel/st_a5/testcase/full_host_tadd/*` 新增

- **变更：** 使用已有 CPU TADD kernel 增加不改用户 kernel 的 full-host 回归。
- **内容：** 验证真实 Host 可编译 kernel 通过 helper 后能加载 pass、产生 VF trace 并调用 VfSim。
- **原因：** 证明这不是只对手工构造 Tile 的测试能力。
- **评审点：** 不应依赖测试专有 header 或硬编码本地 build 路径。

### E07 · `tests/costmodel/st_a5/verify_relocated_vfsim.cmake`、`verify_external_vfsim_project.cmake` 新增

- **变更：** 新增配置重定位与独立消费者工程验证。
- **内容：** 前者检查 build/install 后 JSON 仍能读取；后者安装 PTO 后创建一个新 CMake 项目，验证 helper、pass、archive 和配置可以被仓外 target 使用。
- **原因：** 这是 package 功能，不能只验证当前源码树。
- **评审点：** external project 必须只依赖安装后的 `include/pkg_inc` 与公开 helper；不得偷用仓内路径。

### E08 · `docs/costmodel/vfsim-a5-user-guide.md`、`vfsim-a5-user-guide_zh.md` 新增

- **变更：** 新增中英文使用指南。
- **内容：** 说明能力边界、CMake 接入、Clang/LLVM 要求、支持的测试形态和排障方式。
- **原因：** 该能力需要 pass、静态库与 JSON，用户无法从单个 include 推断正确用法。
- **评审点：** 文档的“支持”必须限定为已验证的指令/控制流/编译选项，不能承诺任意 A5 kernel 或硬件级精度。

## 评审会最终应回答

1. 新增链路是否遵守 PTO → CCE mock → trace → adapter → native → perf-sim 分层？
2. 新增/修改代码是否让 VF、访存、同步各自只计一次周期？
3. 新增的 package/helper/config 是否能在安装和仓外工程中可重复使用，并对不支持情形给出可解释失败？
