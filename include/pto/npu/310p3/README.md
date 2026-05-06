# include/pto/npu/310p3/

Ascend 310P series PTO instruction implementation headers.

## Overview

- Implementations are organized per instruction (or instruction family), for example: `TAdd.hpp`, `TMatmul.hpp`, `TLoad.hpp`, `TStore.hpp`
- Some shared operator patterns are also provided (for example, Reduce/Expand/PartOp helpers)

## Related

- ISA semantics and examples: `docs/isa/`
- 310P NPU ST tests: `tests/npu/310p3/src/st/`