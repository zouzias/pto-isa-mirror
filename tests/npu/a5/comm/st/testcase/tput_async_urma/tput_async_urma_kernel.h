/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstddef>
#include <cstdint>

template <typename T, size_t count>
bool RunPutAsyncUrmaRootPut(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// SharedPool: nAiv AIVs each owning jettiesPerCore consecutive jetties, so the pool
// holds nAiv * jettiesPerCore of them. The kernel never names a jetty; sessions pick
// up their run from get_block_idx(). A put only spreads across that run once the
// payload needs a second 256MB WQE.
template <typename T, size_t count, int nAiv, int jettiesPerCore>
bool RunPutAsyncUrmaPool(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Default-sized SharedPool: Init asks for SHARED_POOL but omits aivCount, so the
// pool is sized from ACL_DEV_ATTR_VECTOR_CORE_NUM with one jetty per AIV. The
// transfer itself only uses two blocks; the assertion is that JettyCount() equals
// the queried device AIV count.
bool RunPutAsyncUrmaPoolAutoAivCount(int n_ranks, int n_devices, int first_rank_id, int first_device_id);
