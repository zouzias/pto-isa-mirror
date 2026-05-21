/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef PTO_COMM_TGET_ASYNC_HPP
#define PTO_COMM_TGET_ASYNC_HPP

#include "pto/comm/async_common/TGetAsyncCommonDetail.hpp"
#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/npu/comm/async/urma/urma_async_intrin.hpp"
#endif

// A5 uses the common TGET_ASYNC_IMPL and TGET_ASYNC_URMA_IMPL definitions
// provided by TGetAsyncCommonDetail.hpp.

#endif // PTO_COMM_TGET_ASYNC_HPP
