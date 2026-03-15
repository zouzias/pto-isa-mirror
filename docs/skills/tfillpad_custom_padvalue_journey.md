# TFILLPAD Custom PadValue: A Prompt Engineering Case Study

This document summarizes the iterative development process for implementing custom PadValue support in PTO-ISA's TFILLPAD operation, highlighting prompt engineering lessons and the value of skill documentation.

## Table of Contents
1. [Task Overview](#task-overview)
2. [Prompt Iteration Timeline](#prompt-iteration-timeline)
3. [The Hidden Bug](#the-hidden-bug)
4. [Lessons Learned](#lessons-learned)
5. [Skill Documentation Benefits](#skill-documentation-benefits)

---

## Task Overview

**Goal:** Extend TFILLPAD to support custom float padding values (e.g., `-1.0f`) instead of only `Min`/`Max`/`Zero`/`Null`.

**Challenge:** The `PadValue` enum needed to encode arbitrary 32-bit float values while maintaining backward compatibility with existing enum values.

**Solution:** Use `uint64_t` underlying type where:
- Lower 32 bits: Marker (`0x00000001` for custom values)
- Upper 32 bits: Float bit pattern

---

## Prompt Iteration Timeline

### Iteration 1: Initial Design Request
**User Prompt:**
> "Add support for custom pad values in TFILLPAD"

**AI Response:** Proposed using a separate `customPadValue` field in Tile.

**Problem:** Required changing Tile struct layout, breaking ABI compatibility.

---

### Iteration 2: Encoding Approach
**User Prompt:**
> "Can we encode the custom value in the PadValue enum itself?"

**AI Response:** Proposed `uint64_t` enum with bit encoding:
```cpp
enum class PadValue : uint64_t {
    Null = 0, Zero = 1, Max = 2, Min = 3,
    CustomBase = 0x100000000ULL  // Bit 32 marks custom
};
```

**Problem:** Initial encoding used bit 32 as marker, but extraction logic was unclear.

---

### Iteration 3: Refined Encoding
**User Prompt:**
> "The encoding needs to work with constexpr on NPU side"

**AI Response:** Refined to:
```cpp
// Encoding: (floatBits << 32) | 0x00000001
constexpr PadValue PadCustomNeg1 = static_cast<PadValue>(0xBF80000100000000ULL);
```

**Problem:** ⚠️ **ENCODING WAS WRONG!** Bytes were swapped.
- Wrong: `0xBF80000100000000` (marker in wrong position)
- Correct: `0xBF80000000000001` (floatBits in high 32, marker in low 32)

---

### Iteration 4: Test Implementation
**User Prompt:**
> "Implement test cases 12 and 13 for custom pad values"

**AI Response:** Added tests with golden generation using separate function `get_input_golden_case_custom_float()`.

**Problem:** Duplicated code pattern, didn't follow existing test structure.

---

### Iteration 5: Refactor to Unified Golden Generation
**User Prompt:**
> "Change the PADCUSTOM_NEG1 test case to generate golden data in-code like other test cases"

**AI Response:** Refactored to use unified `get_input_golden_case<>()` with `auto PadVal_` template parameter.

**Problem:** Used `isCustomPadValue()` and `getCustomPadBits()` which are marked `AICORE` - not callable from host code.

---

### Iteration 6: Fix Host Code Compilation
**User Prompt:** (Implicit from compilation error)

**AI Response:** Inlined the bit extraction logic for host code:
```cpp
if constexpr ((static_cast<uint64_t>(PadVal_) & 0xFFFFFFFFULL) == 0x00000001ULL) {
    uint32_t bits = static_cast<uint32_t>(static_cast<uint64_t>(PadVal_) >> 32);
    // ...
}
```

**Result:** Tests compiled and reported "PASSED".

---

### Iteration 7: Path Error Discovery
**User Prompt:**
> "I still get error message 'Failed to get file. Path = ../TFILLPADTest.case_...' for new PADCUSTOM_NEG1 test cases"

**AI Response:** Discovered `gen_data.py` creates test output directories, but new cases weren't added to the list.

**Fix:** Added new test cases to `gen_data.py`:
```python
case_name_list = [
    # ... existing cases
    "TFILLPADTest.case_float_GT_128_64_VT_128_128_PADCUSTOM_NEG1",
    "TFILLPADTest.case_float_GT_128_127_VT_128_160_BLK1_PADCUSTOM_NEG1_PADCUSTOM_NEG1",
]
```

---

### Iteration 8: Encoding Bug Analysis
**User Prompt:**
> "Could you take a look at the TLOAD logic in case fill pad value is not unknown, is it also following the new API?"

**AI Response:** Traced TLOAD flow and discovered the encoding bug:

```python
# -1.0f bit pattern
>>> hex(struct.unpack('<I', struct.pack('<f', -1.0))[0])
'0xbf800000'

# Correct encoding
>>> hex((0xBF800000 << 32) | 0x00000001)
'0xbf80000000000001'

# What was actually used (WRONG!)
>>> hex(0xBF80000100000000)
'0xbf80000100000000'  # Bytes in wrong position!
```

**Root Cause:** The value `0xBF80000100000000` puts `0x00000001` in the HIGH bits and `0xBF800000` split incorrectly.

---

## The Hidden Bug

### Why Tests "Passed" Despite Wrong Encoding

```
┌─────────────────────────────────────────────────────────────┐
│                    Test Execution Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Golden Generator          Kernel                            │
│  ┌─────────────┐          ┌─────────────┐                   │
│  │ PadVal_ =   │          │ PadVal_ =   │                   │
│  │ 0xBF800001..│          │ 0xBF800001..│   Same wrong      │
│  │ extracts:   │          │ extracts:   │   value!          │
│  │ 0xBF800001  │          │ 0xBF800001  │                   │
│  │ (-1.000000..)│         │ (-1.000000..)│                  │
│  └──────┬──────┘          └──────┬──────┘                   │
│         │                        │                           │
│         ▼                        ▼                           │
│  ┌─────────────┐          ┌─────────────┐                   │
│  │ golden[]    │          │ output[]    │                   │
│  │ filled with │          │ filled with │                   │
│  │ -1.0000001  │   ════   │ -1.0000001  │  MATCH!          │
│  └─────────────┘          └─────────────┘                   │
│                                                              │
│  Result: max diff = 0, test PASSED ✓                        │
│  But actual pad value is NOT -1.0f!                         │
└─────────────────────────────────────────────────────────────┘
```

The bug was masked because:
1. Both kernel and golden generator used the same wrong encoding
2. They produced the same (wrong) output
3. Comparison showed `max diff: 0`
4. File I/O errors were ignored

---

## Lessons Learned

### For AI/LLM Users (Prompt Engineering)

| Lesson | Description |
|--------|-------------|
| **Verify bit-level operations** | When working with bit manipulation, ask the AI to show Python verification of the encoding/decoding |
| **Don't ignore errors** | Even if tests "pass", investigate error messages like "Failed to get file" |
| **Request thinking process** | Ask AI to show step-by-step reasoning, especially for low-level code |
| **Cross-reference with tests** | Ask AI to verify the actual runtime values, not just that tests pass |
| **Check all affected files** | New features often require changes in multiple files (kernel, test, gen_data.py, etc.) |

### For AI Assistants

| Lesson | Description |
|--------|-------------|
| **Show your math** | For bit manipulation, always show the calculation |
| **Don't assume test pass = correct** | Tests can pass for wrong reasons |
| **Read error logs completely** | Don't skip ERROR messages even if final result looks OK |
| **Trace code paths** | When asked about behavior, trace the actual execution path |
| **Document encodings precisely** | Include examples with actual hex values |

### Specific to This Task

```cpp
// WRONG - common mistake
constexpr PadValue PadCustomNeg1 = static_cast<PadValue>(0xBF80000100000000ULL);
//                                                        ^^^^^^^^ marker here? NO!

// CORRECT - float bits in HIGH 32, marker in LOW 32
constexpr PadValue PadCustomNeg1 = static_cast<PadValue>(0xBF80000000000001ULL);
//                                                        ^^^^^^^^         ^
//                                                        float bits    marker
```

---

## Skill Documentation Benefits

### What is a Skill File?

A skill file (`SKILL.md`) is a structured document that captures:
- Domain knowledge
- Common patterns and anti-patterns
- Debugging procedures
- Code templates
- Lessons learned

### How Skills Improve AI Task Efficiency

```
┌────────────────────────────────────────────────────────────────┐
│                Without Skill Documentation                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User: "Add test case for TFILLPAD"                            │
│    ↓                                                           │
│  AI: Creates kernel... (missing gen_data.py)                   │
│    ↓                                                           │
│  Error: "Failed to get file"                                   │
│    ↓                                                           │
│  User: "Fix the error"                                         │
│    ↓                                                           │
│  AI: Investigates... finds gen_data.py                         │
│    ↓                                                           │
│  3-4 iterations to complete task                               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                 With Skill Documentation                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User: "Add test case for TFILLPAD"                            │
│    ↓                                                           │
│  AI: Reads SKILL.md → sees 6-step checklist                    │
│    ↓                                                           │
│  AI: Creates kernel + main.cpp + gen_data.py + instantiations  │
│    ↓                                                           │
│  1 iteration to complete task ✓                                │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Key Sections in pto-isa-testing SKILL.md

1. **Repository Structure** - Where to find files
2. **Running Tests** - Exact commands
3. **Test Case Structure** - File layout and patterns
4. **Custom PadValue** - Encoding format with examples
5. **Adding New Test Case** - Complete 6-step checklist
6. **Common Issues** - Known pitfalls and fixes
7. **Thinking Process Template** - Debugging methodology

### Sharing Skills Across Teams

Benefits of committing SKILL.md to the repository:

| Benefit | Description |
|---------|-------------|
| **Onboarding** | New team members (human or AI) can quickly understand patterns |
| **Consistency** | Everyone follows the same procedures |
| **Knowledge Preservation** | Lessons learned aren't lost when people move on |
| **AI Efficiency** | AI assistants can be more helpful with context |
| **Reduced Errors** | Common pitfalls are documented and avoided |

---

## Final Implementation

### Files Changed (Commit `89a060d`)

```
include/pto/common/constants.hpp    # PadValueCustom(), isCustomPadValue(), getCustomPadBits()
include/pto/common/type.hpp         # PadValue enum with uint64_t underlying type
tests/cpu/st/testcase/tfillpad/
  ├── tfillpad_kernel.cpp           # CPU test with custom pad
  └── README.md                     # Documentation
tests/npu/a2a3/src/st/testcase/tfillpad/
  ├── main.cpp                      # Test cases 12, 13
  ├── tfillpad_kernel.cpp           # Kernel + golden generation
  ├── gen_data.py                   # Directory creation list
  └── README.md                     # Documentation
```

### API Summary

```cpp
// Runtime (host/CPU)
PadValue pv = PadValueCustom(-1.0f);

// Compile-time template
using MyTile = Tile<..., PadCustom<-1.0f>>;

// NPU constexpr (manual encoding required)
constexpr PadValue PadCustomNeg1 = static_cast<PadValue>(0xBF80000000000001ULL);

// Detection and extraction
bool isCustom = isCustomPadValue(pv);      // true if custom
uint32_t bits = getCustomPadBits(pv);      // 0xBF800000 for -1.0f
```

---

## Conclusion

This task demonstrated how iterative prompt engineering can lead to a working solution, but also how subtle bugs can hide in "passing" tests. Key takeaways:

1. **Bit manipulation requires verification** - Don't trust intuition, verify with actual calculations
2. **Test success ≠ correctness** - Both sides using the same wrong value will still match
3. **Read all error messages** - They often indicate real problems
4. **Document lessons learned** - Skills prevent repeating mistakes
5. **Share knowledge** - Repository-committed skills help the whole team

The SKILL.md file created from this experience will help future tasks involving PTO-ISA testing be completed faster and with fewer errors.
