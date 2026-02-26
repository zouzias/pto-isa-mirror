# Manual / Resource Binding

This document describes manual resource binding and configuration operations.

**Total Operations:** 6

---

## Operations

### TASSIGN

**Math Interpretation:**

Not applicable.

**IR Level 1 (SSA):**
```text
pto.tassign %tile, %addr : !pto.tile<...>, dtype
```

**IR Level 2 (DPS):**
```text
pto.tassign ins(%tile, %addr : !pto.tile_buf<...>, dtype)
```

---

### TSETHF32MODE

**Math Interpretation:**

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

**IR Level 1 (SSA):**
```text
pto.tsethf32mode {enable = true, mode = ...}
```

**IR Level 2 (DPS):**
```text
pto.tsethf32mode ins({enable = true, mode = ...}) outs()
```

---

### TSETTF32MODE

**Math Interpretation:**

No direct tensor arithmetic is produced by this instruction. It updates target mode state used by subsequent instructions.

**IR Level 1 (SSA):**
```text
pto.tsettf32mode {enable = true, mode = ...}
```

**IR Level 2 (DPS):**
```text
pto.tsettf32mode ins({enable = true, mode = ...}) outs()
```

---

### TSETFMATRIX

**Math Interpretation:**

Unless otherwise specified, semantics are defined over the valid region and target-dependent behavior is marked as implementation-defined.

**IR Level 1 (SSA):**
```text
pto.tsetfmatrix %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS):**
```text
pto.tsetfmatrix ins(%cfg : !pto.fmatrix_config) outs()
```

---

### TSET_IMG2COL_RPT

**Math Interpretation:**

No direct tensor arithmetic is produced by this instruction. It updates IMG2COL control state used by subsequent data-movement operations.

**IR Level 1 (SSA):**
```text
pto.tset_img2col_rpt %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS):**
```text
pto.tset_img2col_rpt ins(%cfg : !pto.fmatrix_config) outs()
```

---

### TSET_IMG2COL_PADDING

**Math Interpretation:**

No direct tensor arithmetic is produced by this instruction. It updates IMG2COL padding control state consumed by subsequent data-movement operations.

**IR Level 1 (SSA):**
```text
pto.tset_img2col_padding %cfg : !pto.fmatrix_config -> ()
```

**IR Level 2 (DPS):**
```text
pto.tset_img2col_padding ins(%cfg : !pto.fmatrix_config) outs()
```

---


