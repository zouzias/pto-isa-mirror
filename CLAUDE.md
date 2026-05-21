# Project AGENTS.md

> 本文件是 Cursor / Claude Code / Codex 等客户端共用的项目知识入口。内容与 `CLAUDE.md` 保持一致；两者中任意一方更新后应同步。



## A3于A5差异
- A3和A5的HCCL window的地址都要加一些偏移，否则在A5上会有问题
- 当前机器是A3的，可以直接运行A3的用例，A5的用例只可以跑编译

## MCP 工具使用指引

- `user-local-rag-9.0_a5`：**CANN 9.0 beta2 官方文档向量库**（BASE_DIR=hiascend_cann_900beta2，已入库 4500+ 份官方文档，hybrid 搜索）。查 950 a5 910 a3 的 CANN 的 API 签名、参数、使用教程、Tiling/Kernel 开发细节时**首选**这个。
- `user-cann-rag`：策划过的昇腾多产品线文档（CANN 多版本 + PyTorch + MindIE + MindStudio + Atlas 200I A2 + FAQ），支持 CANN 术语展开和跨产品线检索。查非 CANN 9.0 或跨产品线问题时用。
- `context7`：第三方库/框架的最新文档（React/PyTorch/vLLM 等）。


## 版本与芯片默认对应

- 默认对应关系：**CANN 9.0** 问题优先按 **A5 / 950** 形态理解，**CANN 8.5** 问题优先按 **A3 / 910** 形态理解；若用户明确指定芯片、SoC 或运行环境，以用户指定为准。

## 本地构建环境约定

- 对需要编译 Ascend C / SHMEM 相关工程的任务，默认先加载以下环境：
  - `source /usr/local/Ascend/cann-8.5.0/set_env.sh`
  - `export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH`
  - `export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:$LD_LIBRARY_PATH`
  - `export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so`
- 卡不支持多任务并行，任务要一个个执行，可以执行前先npu-smi info看下是否有人在用
-  对a5的项目，使用source /home/ntlab/liulei/can/cann-9.0.0-beta.1/
-  
## 项目内置知识库（.ai-knowledge）

- ascend的知识库位于 `.ai-knowledge/`，作为本项目的**常驻知识路由层**使用。
- 面对 Ascend / CANN / PTO / MC2 / 芯片规格 / pattern / testcase 相关问题时，**第一步先按需命中 `.ai-knowledge/router/ROUTER.md` 指向的现有资产**，不要先从零搜索或重讲。
- 必须遵守 `.ai-knowledge/router/HARD_RULES.md` 与 `.ai-knowledge/router/ANSWERING_POLICY.md`：
  - 文档事实优先来自外部文档 RAG 或官方材料
  - 仓内当前实现优先来自代码
  - 现有 `cards/wiki/writeback/skills` 命中即止，不为求全继续扩张
- `.ai-knowledge` 的使用边界：
  - `cards/`：短事实、规格、checklist、映射表
  - `wiki/`：主题路由、对比、canonical scope
  - `writeback/`：经验、陷阱、recipe
  - `skills/`：方法论与查询/维护策略
- 对 PTO / MC2 相关任务，优先复用当前已落地的 `.ai-knowledge` 页面与映射表，再回代码核实。
- 对芯片/容量/TileType 可见预算问题，优先看 `.ai-knowledge/cards/tables/chip-memory-specs.md` 与 `wiki/chips/`。
- 若新增或修正 `.ai-knowledge` 内容，必须同步更新对应 `_INDEX.md` / `wiki/index.md`、`wiki/log.md`、`meta/CHANGELOG.md`。
- pto的知识库可以看/home/ntlab/zy/code/zhangyuan/pto-isa-zy/docs/pto-knowledge.md 这个入口