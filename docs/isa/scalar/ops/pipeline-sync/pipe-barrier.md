# pto.pipe_barrier

`pto.pipe_barrier` is part of the [Pipeline Sync](../../pipeline-sync.md) instruction set.

## Summary

Drain all previously issued work in one pipeline before allowing later work on that same pipeline to begin.

## Mechanism

`pto.pipe_barrier` forces completion of the named pipeline's outstanding work. It is the simplest ordering primitive when the hazard is confined to one pipeline and no explicit cross-pipeline event identity is required.

## Syntax

### PTO Assembly Form

```text
pipe_barrier "PIPE_*"
```

### AS Level 1 (SSA)

```mlir
pto.pipe_barrier "PIPE_*"
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| `PIPE_*` | pipe identifier | Pipeline whose outstanding work must retire before later work begins |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it establishes an ordering point on the named pipeline. |

## Side Effects

Stalls the named pipeline until its previously issued operations retire. Later operations issued to that pipeline are ordered after the barrier.

## Constraints

!!! warning "Constraints"
    - The selected pipe identifier MUST be valid for the selected target profile.
    - This barrier orders work within one pipeline; cross-pipeline producer-consumer edges still require the appropriate event or buffer protocol.
    - Portable code SHOULD use the narrowest ordering primitive that matches the hazard.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported pipe or event identifiers, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - CPU simulation preserves the visible ordering contract but may not expose every pipeline hazard that motivates the barrier on hardware.
    - A2/A3 and A5 may differ in the exact pipe set that can be named.

## Examples

```mlir
pto.pipe_barrier "PIPE_MTE3"
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in instruction set: [pto.wait_flag](./wait-flag.md)
- Next op in instruction set: [pto.mem_bar](./mem-bar.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
