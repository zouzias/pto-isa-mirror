---
name: PTO指令文档更新模式
description: PTO ISA 新增指令时需要更新的文档文件和位置模式总结。触发：新增 PTO 指令（如 TPOW、TPOWS）后需要同步更新文档时。
license: CANN Open Software License Agreement Version 2.0
---

# PTO指令文档更新模式

此 skill 总结了为 PTO ISA 添加新指令（如 TPOW、TPOWS）时需要更新的文档文件和位置模式。

## 适用场景

当新增 PTO 指令后，需要同步更新相关文档时使用此 skill。使用 `git status` 查看当前 staging 的文件列表来确认所有需要修改的文件。

## Git Staged 文件分类

### 新增文件（需预先创建）

| 类型 | 说明 |
|------|------|
| 指令diagrams | `docs/figures/isa/{指令名}.svg` - 指令操作示意图 |
| 指令文档 | `docs/isa/{指令名}.md` - 详细指令文档（英文） |
| 指令文档 | `docs/isa/{指令名}_zh.md` - 详细指令文档（中文） |

### 修改文件（需同步更新）

| 类型 | 说明 |
|------|------|
| ISA主索引 | `docs/PTOISA.md` - ISA索引表格 |
| ISA主索引 | `docs/PTOISA_zh.md` - ISA索引表格（中文） |
| ISA参考目录 | `docs/isa/README.md` - 按分类排序的指令列表 |
| ISA参考目录 | `docs/isa/README_zh.md` - 按分类排序的指令列表（中文） |
| 菜单文档 | `docs/menu_ops_development.md`（API参考 段）- 按分类排序的中文链接 |
| 指令族矩阵 | `docs/mkdocs/src/manual/appendix-d-instruction-family-matrix.md` - 指令族矩阵 |
| 指令族矩阵 | `docs/mkdocs/src/manual/appendix-d-instruction-family-matrix_zh.md` - 指令族矩阵（中文） |
| include索引 | `include/README.md` - 实现状态表格 |
| include索引 | `include/README_zh.md` - 实现状态表格（中文） |

---

## 更新模式详解

### 1. ISA 主索引文件

#### docs/PTOISA.md / docs/PTOISA_zh.md
- **位置**: 指令索引表格
- **分类**: 
  - 逐元素（Tile-Tile）指令 → 插在 `TFMOD` 后
  - Tile-标量 / Tile-立即数 → 插在 `TSUBSC` 后

#### include/README.md / include/README_zh.md
- **位置**: 实现状态表格（按字母序）
- **分类**:
  - TPOW → 插在 `TPRELU` 和 `TPUT` 之间
  - TPOWS → 插在 `TPUT_ASYNC` 和 `TQUANT` 之间

### 2. ISA 参考目录

#### docs/isa/README.md / docs/isa/README_zh.md
- **位置**: 按分类排序的指令列表
- **分类**:
  - Elementwise (Tile-Tile) → 插在 `TFMOD` 后
  - Tile-Scalar / Tile-Immediate → 插在 `TSUBSC` 后

### 3. 菜单文档

#### docs/menu_ops_development.md（API参考 段）
- **位置**: `API参考` 段下按分类排序的中文链接列表
- **同 ISA 参考目录结构**

### 4. 指令族矩阵

#### docs/mkdocs/src/manual/appendix-d-instruction-family-matrix.md
- **位置**: D.2 覆盖统计表 + D.4 家族矩阵表
- **D.2 更新示例**:
  ```
  | Elementwise (Tile-Tile) | 28 → 29 |
  | Tile-Scalar / Tile-Immediate | 19 → 20 |
  | Total | 126 → 128 |
  ```
- **D.4 更新**:
  - 在对应分类的最后一条目后插入新指令

#### docs/mkdocs/src/manual/appendix-d-instruction-family-matrix_zh.md
- **同英文版本**

---

## 常见新增指令分类与插入位置

### Tile-Tile (逐元素双Tile)
- **插入位置**: `TFMOD` 之后
- **示例**: TPOW

### Tile-Scalar (Tile与标量)
- **插入位置**: `TSUBSC` 之后
- **示例**: TPOWS

### Axis Reduce / Expand
- **插入位置**: 最后一个 Axis 指令之后

### Memory (GM ↔ Tile)
- **插入位置**: 最后一个 Memory 指令之后

---

## 更新检查清单

### 新增文件（预先创建）
- [ ] `docs/figures/isa/{新指令}.svg` - 指令操作示意图
- [ ] `docs/isa/{新指令}.md` - 详细指令文档（英文）
- [ ] `docs/isa/{新指令}_zh.md` - 详细指令文档（中文）

### 修改文件（同步更新）
- [ ] `docs/PTOISA.md` - ISA主索引
- [ ] `docs/PTOISA_zh.md` - ISA主索引（中文）
- [ ] `include/README.md` - include索引
- [ ] `include/README_zh.md` - include索引（中文）
- [ ] `docs/isa/README.md` - ISA参考目录
- [ ] `docs/isa/README_zh.md` - ISA参考目录（中文）
- [ ] `docs/menu_ops_development.md` - 菜单文档（API参考 段）
- [ ] `docs/mkdocs/src/manual/appendix-d-instruction-family-matrix.md` - 指令族矩阵
- [ ] `docs/mkdocs/src/manual/appendix-d-instruction-family-matrix_zh.md` - 指令族矩阵（中文）

---

## 注意事项

1. **英文+中文**: 每个文件都有中英文两个版本，需要同步更新
2. **计数变化**: 需要同时更新 Operation Count（分类小计）和 Total（总计）
3. **详细指令文档**: 需要预先创建在 `docs/isa/` 目录下
4. **diagrams**: 需要预先创建在 `docs/figures/isa/` 目录下
5. 使用 `git status` 可以查看当前 staging 的文件列表，这是确认所有需要修改文件的最佳方式
