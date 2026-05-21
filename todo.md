# 任务：整理 dispatch_ffn_combine_v3 MC2→PTO 最终转换点

## 目标
结合原始 MC2 `dispatch_ffn_combine_v2`、历史任务文档与当前 A5 `dispatch_ffn_combine_v3` 活树，生成 `mc2_2_pto.md`，只记录最终转换点，不复述过程流水账。

## 待办事项
- [x] 梳理历史任务文档中的最终转换点
- [x] 核对当前 A5 PTO 代码落点与原 MC2 目录差异
- [x] 编写 mc2_2_pto.md 只记录最终转换点
- [x] 校验 mc2_2_pto.md 格式与事实引用

## 进度
4/4

## 验证证据
- `git diff --check -- kernels/manual/a5/dispatch_ffn_combine_v3/mc2_2_pto.md todo.md` 无输出。
- Markdown 基础检查：`lines 243`, `fences 8`, `issues none`。
- A5 compile-only：`/tmp/dispatch_ffn_combine_v3_a5_mc2_2_pto_verify` 构建通过，`[100%] Built target dispatch_ffn_combine_v3`。
