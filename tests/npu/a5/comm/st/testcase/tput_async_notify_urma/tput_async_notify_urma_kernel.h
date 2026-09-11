/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

enum class UrmaNotifyStMode {
    Set,
    ReceiverConsumeSet,
    AtomicAdd,
    SetBatchDrain65,
    FaaSharedSink65,
    MixedPutSetGetAdd,
};

bool RunTPutAsyncNotifyUrma(int nRanks, int nDevices, int firstRankId, int firstDeviceId, UrmaNotifyStMode mode);
void FinalizeTPutAsyncNotifyUrma();

// SharedPool notify: root AIV i SET-notifies send[i] -> every peer's recv[i] and
// sets that peer's signals[i]. The pool holds nAiv * jettiesPerCore jetties and no
// test names one; each session derives its run from get_block_idx(). Distinct
// per-(aiv) payloads and signal values turn a mis-routed jetty into a wrong-slot
// mismatch. Large count (>256MB payload) additionally exercises payload chunking.
// Every receiver AIV proves ordering on-device: it waits on its signal, then reads
// its payload slot, so a signal that overtakes the payload (or a mis-routed jetty)
// is caught on-device instead of being masked by host-side timing after the barrier.
// jettiesPerCore > 1 covers the case that matters most here: a notify must stay on
// the run's first jetty however wide the run is, because placeOrder only orders
// payload before signal within one SQ.
template <typename T, size_t count, int nAiv, int jettiesPerCore>
bool RunNotifyUrmaPool(int nRanks, int nDevices, int firstRankId, int firstDeviceId);
