#include "pto/costmodel/a5/cce_costmodel/vf_cost.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace vf = ::pto::mocker::vf;

namespace {

vf::MemInfo Reg(std::string name, std::string dtype = "fp32")
{
    return {std::move(name), vf::MemLocation::PhyRegister, std::move(dtype)};
}

vf::MemInfo Ub(std::string name, std::string dtype = "fp32")
{
    return {std::move(name), vf::MemLocation::UB, std::move(dtype)};
}

vf::VfNode Inst(std::string op, std::vector<vf::MemInfo> dst, std::vector<vf::MemInfo> src)
{
    return vf::MakeInst(vf::VfInst{std::move(op), std::move(dst), std::move(src)});
}

uint64_t Fallback(const vf::VfInfo &info)
{
    uint64_t total = 0;
    const auto &loop = vf::AsLoop(info.tree.front());
    for (const auto &node : loop.body) total += vf::FallbackVecCycle(vf::AsInst(node).opName);
    return total * loop.count;
}

}  // namespace

int main()
{
    vf::VfInfo supported;
    supported.op = "synthetic_supported";
    supported.tree.push_back(vf::MakeLoop(
        4, {Inst("vlds", {Reg("V0")}, {Ub("mem0")}), Inst("vlds", {Reg("V1")}, {Ub("mem1")}),
            Inst("vadd", {Reg("V2")}, {Reg("V0"), Reg("V1")}), Inst("vsts", {Ub("mem2")}, {Reg("V2")})}));

    const uint64_t simulated = vf::PredictVfCycles(supported);
    const uint64_t simple = Fallback(supported);
    if (simulated == 0 || simulated == simple) {
        std::fprintf(stderr, "supported program did not use VfSim: simulated=%llu fallback=%llu\n",
                     static_cast<unsigned long long>(simulated), static_cast<unsigned long long>(simple));
        return 1;
    }

    vf::VfInfo mixedDtype;
    mixedDtype.op = "synthetic_mixed_dtype";
    mixedDtype.tree.push_back(vf::MakeLoop(
        16, {Inst("vlds", {Reg("V0", "fp32")}, {Ub("mem0", "fp32")}),
             Inst("vlds", {Reg("V1", "fp32")}, {Ub("mem1", "fp32")}),
             Inst("vadd", {Reg("V2", "fp32")}, {Reg("V0", "fp32"), Reg("V1", "fp32")}),
             Inst("vcvt_f32_to_f16", {Reg("V3", "fp16")}, {Reg("V2", "fp32")}),
             Inst("vlds", {Reg("V4", "fp16")}, {Ub("mem2", "fp16")}),
             Inst("vadd", {Reg("V5", "fp16")}, {Reg("V3", "fp16"), Reg("V4", "fp16")}),
             Inst("vsts", {Ub("mem3", "fp16")}, {Reg("V5", "fp16")})}));
    const uint64_t mixedSimulated = vf::PredictVfCycles(mixedDtype);
    const uint64_t mixedFallback = Fallback(mixedDtype);
    if (mixedSimulated == 0 || mixedSimulated == mixedFallback) {
        std::fprintf(stderr, "mixed dtype program did not use VfSim: simulated=%llu fallback=%llu\n",
                     static_cast<unsigned long long>(mixedSimulated),
                     static_cast<unsigned long long>(mixedFallback));
        return 1;
    }

    vf::VfInfo unsupported = supported;
    vf::AsLoop(unsupported.tree.front()).body.push_back(vf::MakeInst("unsupported_op"));
    const uint64_t fallback = vf::PredictVfCycles(unsupported);
    const uint64_t expectedFallback = Fallback(unsupported);
    if (fallback != expectedFallback) {
        std::fprintf(stderr, "unsupported program fallback mismatch: actual=%llu expected=%llu\n",
                     static_cast<unsigned long long>(fallback),
                     static_cast<unsigned long long>(expectedFallback));
        return 1;
    }

    const uint64_t combined = vf::PredictVfCycles(std::vector<vf::VfInfo>{supported, unsupported});
    const uint64_t expectedCombined = simple + expectedFallback;
    if (combined != expectedCombined) {
        std::fprintf(stderr, "multi-VF all-or-nothing fallback mismatch: actual=%llu expected=%llu\n",
                     static_cast<unsigned long long>(combined),
                     static_cast<unsigned long long>(expectedCombined));
        return 1;
    }

    std::printf("PASS: vfsim=%llu mixed=%llu supported_fallback=%llu unsupported_fallback=%llu combined=%llu\n",
                static_cast<unsigned long long>(simulated), static_cast<unsigned long long>(mixedSimulated),
                static_cast<unsigned long long>(simple), static_cast<unsigned long long>(fallback),
                static_cast<unsigned long long>(combined));
    return 0;
}
