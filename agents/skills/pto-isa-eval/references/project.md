---
name: pto-isa-project-structure
description: PTO-ISA项目结构参考
---

# PTO-ISA 项目结构

## 目录结构

```
pto-isa/
├── docs/                    # 文档目录
│   ├── getting-started.md  # 入门指南(中英文)
│   ├── PTOISA.md           # ISA规范文档(中英文)
│   ├── PTO-Virtual-ISA-Manual.md  # 虚拟ISA手册(中英文)
│   ├── isa/                # ISA详细文档
│   ├── coding/             # 编码指南
│   ├── reference/          # 参考文档
│   ├── tools/              # 工具文档
│   ├── auto_mode/          # 自动模式文档
│   ├── assembly/           # 汇编文档
│   ├── menu_apis.md        # 菜单API文档
│   └── menu_ops_development.md  # 算子开发文档
├── kernels/                # 算子实现
│   ├── manual/             # 手动实现算子
│   ├── custom/             # 用户自定义算子
│   └── run_kernels.sh      # 算子运行脚本
├── tests/                  # 测试脚本
│   ├── run_st.sh           # Shell测试入口
│   ├── script/             # Python测试脚本
│   └── cases/              # 测试用例
└── README.md
```

## 支持平台

- A2/A3: 入门级/中端平台
- A5: 高端平台
- Kirin9030: Kirin芯片平台
- KirinX90: Kirin X90芯片平台

## 测试命令

### 命令行选项

```bash
bash tests/run_st.sh [选项]
```

| 选项 | 说明 |
|------|------|
| `--a3` | A3平台测试 |
| `--a5` | A5平台测试 |
| `--a3_a5` | A3和A5双平台测试 |
| `--kirin9030` | Kirin9030平台 |
| `--kirinX90` | Kirin X90平台 |
| `--sim` | Simulation模式 |
| `--npu` | NPU模式 |
| `--comm` | 通信测试 |
| `--simple` | 冒烟测试(精简用例集) |
| `--all` | 完整测试(所有用例) |
| `--auto_mode` | 自动模式 |

### 测试模式组合示例

```bash
# A5冒烟测试
bash tests/run_st.sh --a5 --simple --npu

# A3完整测试
bash tests/run_st.sh --a3 --all --npu

# A5完整测试
bash tests/run_st.sh --a5 --all --npu

# A3和A5双平台测试
bash tests/run_st.sh --a3_a5 --simple --npu
```

### 冒烟测试用例数量

- A3 `--simple`: 约260+个用例
- A5 `--simple`: 约170+个用例

## 核心算子列表

### 基础算子
- TADD, TSUB, TMUL, TDIV, TREM
- TCONCAT, TGATHER, TLOAD, TSTORE
- TCVT, TEXP, TLOG, TPOW
- TMAX, TMIN, TAND, TOR, TXOR
- TSHL, TSHR

### 矩阵算子
- TMATMUL, TMOV, TTRANS
- TMRGSORT, TPART, TROW
- TROWEXPAND, TROWSUM, TROWPROD

### 高级算子
- TROWARGMAX, TROWARGMIN
- TINSERT, TQUANT
- MSCATTER, MGATHER
