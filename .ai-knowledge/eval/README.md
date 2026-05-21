# Eval README

本目录存放知识库评测样例，当前先建立最小可运行骨架。

## 子目录

- `qa-benchmark/`：事实问答样例
- `coding-tasks/`：代码任务样例
- `debugging-cases/`：调试案例样例

## 当前状态

- 已有每类 1 个最小样例
- `scripts/run-eval.sh` 已提供 inventory 级脚手架
- 后续需要补：yaml 字段校验、wiki/frontmatter 体检、source 检查、回归对比

## 使用方式

```bash
bash .ai-knowledge/scripts/run-eval.sh
```
