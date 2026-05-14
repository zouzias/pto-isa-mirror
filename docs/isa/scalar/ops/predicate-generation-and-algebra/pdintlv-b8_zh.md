# pto.pdintlv_b8

对谓词做按位去交错。

## 语法

### PTO 汇编形式

```text
pdintlv_b8 %dst0, %dst1, %src : !pto.mask, !pto.mask, !pto.mask
```

### AS Level 1（SSA）

```mlir
%dst0, %dst1 = pto.pdintlv_b8 %src : !pto.mask -> !pto.mask, !pto.mask
```

### AS Level 2（DPS）

```mlir
pto.pdintlv_b8 ins(%src : !pto.mask) outs(%dst0, %dst1 : !pto.mask, !pto.mask)
```

## 关键约束

- 参与操作的谓词宽度必须兼容。
- pattern / partition token 必须属于文档化取值域。

## 相关页面

- [谓词生成与代数](../../predicate-generation-and-algebra_zh.md)
