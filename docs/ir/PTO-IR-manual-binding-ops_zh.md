# 手动/资源绑定

本文档描述手动资源绑定和配置操作。

**操作总数：** 6

---

## 操作

### TASSIGN

**数学解释：**

不适用。

**IR Level 1 (SSA)：**
```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

**IR Level 2 (DPS)：**
```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

---

### TSETHF32MODE

**数学解释：**

此指令不产生直接的张量运算。它更新后续指令使用的目标模式状态。

**IR Level 1 (SSA)：**
```text
pto.tsethf32mode {enable = true, mode = ...}
```

**IR Level 2 (DPS)：**
```text
pto.tsethf32mode ins({enable = true, mode = ...}) outs()
```

---

### TSETTF32MODE

**数学解释：**

此指令不产生直接的张量运算。它更新后续指令使用的目标模式状态。

**IR Level 1 (SSA)：**
```text
pto.tsettf32mode {enable = true, mode = ...}
```

**IR Level 2 (DPS)：**
```text
pto.tsettf32mode ins({enable = true, mode = ...}) outs()
```

---

### TSETFMATRIX

**数学解释：**

除非另有说明，语义在有效区域上定义，目标相关的行为标记为实现定义。

**IR Level 1 (SSA)：**
```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

---

### TSET_IMG2COL_RPT

**数学解释：**

此指令不产生直接的张量运算。它更新后续数据移动操作使用的 IMG2COL 控制状态。

**IR Level 1 (SSA)：**
```text
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tset_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

---

### TSET_IMG2COL_PADDING

**数学解释：**

此指令不产生直接的张量运算。它更新后续数据移动操作使用的 IMG2COL 填充控制状态。

**IR Level 1 (SSA)：**
```text
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS)：**
```text
pto.tset_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```
