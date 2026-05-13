/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0.
 */

// pto::ccu::detail::PtoGatedReduceScatterMesh1D — the actual CCU kernel
// implementation behind `pto::ccu::MakeGatedKernelCreator`.
//
// This subclass extends `hcomm::CcuKernel` directly (NOT
// `ops_hccl::CcuKernelAlgBase`), keeping the build coupling to hcomm pkg_inc
// only. See `pto/ccu/pto_gated_kernel.hpp` for the architectural rationale and
// pilot scope statement.

#ifndef PTO_CCU_DETAIL_PTO_GATED_KERNEL_H
#define PTO_CCU_DETAIL_PTO_GATED_KERNEL_H

#include <cstdint>
#include <vector>

#include "hcomm/ccu/ccu_kernel.h"
#include "pto/ccu/pto_gated_kernel.hpp"

namespace pto {
namespace ccu {
namespace detail {

class PtoGatedReduceScatterMesh1D : public hcomm::CcuKernel {
public:
    explicit PtoGatedReduceScatterMesh1D(const hcomm::CcuKernelArg &arg);
    ~PtoGatedReduceScatterMesh1D() override = default;

    // hcomm::CcuKernel virtual interface. `Algorithm()` is invoked once during
    // `HcclCcuKernelRegister` (after `Init()`) and builds the CCU IR;
    // `GeneArgs()` is invoked on every `HcclCcuKernelLaunch` and packs runtime
    // arg slots.
    HcclResult Algorithm() override;
    std::vector<uint64_t> GeneArgs(const hcomm::CcuTaskArg &arg) override;

private:
    // Snapshot of `PtoGatedKernelArg` fields, captured in the constructor so
    // `Algorithm()` (which has no `arg` parameter) can read them.
    uint32_t rankId_{0};
    uint32_t rankSize_{1};
    uint32_t gateMask_{1u << 0};
    uint32_t doneMask_{1u << 0};
    uint64_t payloadBytes_{0};

    // CCU IR resources allocated during `Algorithm()`.
    //
    // Captured as members because `GeneArgs()` needs access to `gateEvent_`
    // post-translate (to publish `(dieId, ckeId)` to the descriptor registry
    // for the AIV trigger to consume).
    hcomm::CcuRep::Variable        inputVar_;
    hcomm::CcuRep::Variable        outputVar_;
    hcomm::CcuRep::Variable        tokenVar_;
    hcomm::CcuRep::Variable        lengthVar_;
    hcomm::CcuRep::CompletedEvent  gateEvent_;
    hcomm::CcuRep::CompletedEvent  doneEvent_;
    hcomm::CcuRep::CompletedEvent  copyEvent_;
};

}  // namespace detail
}  // namespace ccu
}  // namespace pto

#endif  // PTO_CCU_DETAIL_PTO_GATED_KERNEL_H
