---
name: card-error-aclnn-161xxx
description: |
  触发：查 ACLNN 161xxx 参数类错误的含义、常见根因与排查入口。
  Trigger: Lookup ACLNN 161xxx parameter-related errors and common root causes.
card_type: error-codes
version: CANN 9.0 beta2
chip: [950 A5, 910A3]
source: official
last_updated: 2026-04-19
---

# ACLNN 161xxx 速查卡

## 定位方向

161xxx 先按“参数前置条件不满足”来理解，优先从地址、shape、dtype、格式、长度、空指针和版本约束入手，而不是先怀疑调度或性能问题。

## 首轮排查

1. 参数是否为空、未初始化或 host 侧构造错误。
2. shape / dtype / format 是否匹配。
3. 地址、长度、对齐是否满足约束。
4. host 侧传入尺寸、stride、tiling 与 kernel 侧理解是否一致。
5. 当前 API 是否在指定版本/芯片上支持对应组合。

## 常见根因簇

- 地址与长度非法。
- 输入输出张量属性不匹配。
- host 侧参数封装错误。
- 调用路径使用了当前版本不支持的组合。

## 使用方式

- 先用本卡缩小到“参数类”问题。
- 再跳到对应 API 卡检查具体前置条件。
- 若仍无法收敛，再查运行时或精度类 skill，不要一开始就扩散排查面。

## 联动入口

- `cards/api/datacopy.md`
- `cards/api/cast.md`
- `ascendc-runtime-debug`

## 来源

- `mcp:user-local-rag-9.0_a5`，建议 query: `aclnn 161xxx error code CANN 9.0`
