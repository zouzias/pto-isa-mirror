/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
*/

// DEPRECATED — kept only to avoid breaking historical references.
//
// The host-side `pto_aiv_treduce_launch(...)` launcher has been merged into
// kernel.cpp (single bisheng translation unit, mirroring gemm_ar's
// `comm_kernel.cpp` pattern). This is required because the bisheng-emitted
// `KernelName<<<...>>>(args)` launch thunk only links correctly when both
// the kernel and the launch dispatch sit in the same CCE TU.
//
// CMakeLists.txt no longer compiles this file. Safe to delete in a future
// pass; preserved here only so existing git references don't 404 immediately.

// (intentionally empty — no symbols defined here)
