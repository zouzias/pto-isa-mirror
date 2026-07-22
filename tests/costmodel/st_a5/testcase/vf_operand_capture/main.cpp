#include "pto/costmodel/a5/cce_costmodel/cce_costmodel.hpp"
#include "pto/costmodel/trace.hpp"

#include <cstdint>
#include <cstdio>

namespace vf = ::pto::mocker::vf;

int main()
{
    ::pto::mocker::ResetTrace();
    ::pto::mocker::PtoInstrScope ptoScope("operand_capture");
    __VEC_SCOPE__
    {
        pto::RegTensor<float> src0;
        pto::RegTensor<float> src1;
        pto::RegTensor<float> dst;
        pto::MaskReg predicate;
        uint32_t predicateScalar = 64;
        auto *mem0 = reinterpret_cast<float *>(0x1000);
        auto *mem1 = reinterpret_cast<float *>(0x2000);
        auto *mem2 = reinterpret_cast<float *>(0x3000);
        predicate = plt_b32(predicateScalar, POST_UPDATE);
        vlds(src0, mem0, 0, NORM);
        vlds(src1, mem1, 0, NORM);
        vadd(dst, src0, src1, predicate, MODE_ZEROING);
        vsts(dst, mem2, 0, NORM, predicate);
    }
    ptoScope.Finish();

    const auto &record = ::pto::mocker::GetTrace().executed_pto.back();
    // plt_b32 only sets the predicate and must not add a fifth micro-op.
    if (record.vf_infos.size() != 1 || record.vf_infos[0].tree.size() != 4) return 1;
    const auto &tree = record.vf_infos[0].tree;
    const vf::VfInst &load0 = vf::AsInst(tree[0]);
    const vf::VfInst &load1 = vf::AsInst(tree[1]);
    const vf::VfInst &add = vf::AsInst(tree[2]);
    const vf::VfInst &store = vf::AsInst(tree[3]);

    const bool valid = load0.dst.size() == 1 && load0.src.size() == 1 && load1.dst.size() == 1 &&
                       load1.src.size() == 1 && add.dst.size() == 1 && add.src.size() >= 2 &&
                       store.dst.size() == 1 && store.src.size() == 1 &&
                       load0.dst[0].location == vf::MemLocation::PhyRegister &&
                       load0.src[0].location == vf::MemLocation::UB && load0.dst[0].dtype == "fp32" &&
                       load0.src[0].dtype == "fp32" && add.src[0].name == load0.dst[0].name &&
                       add.src[1].name == load1.dst[0].name && store.src[0].name == add.dst[0].name &&
                       store.dst[0].location == vf::MemLocation::UB;
    if (!valid) {
        std::fprintf(stderr, "captured operand metadata or dependency names are invalid\n");
        return 1;
    }

    std::printf("PASS: %s,%s -> %s -> %s cycles=%llu\n", load0.dst[0].name.c_str(), load1.dst[0].name.c_str(),
                add.dst[0].name.c_str(), store.dst[0].name.c_str(),
                static_cast<unsigned long long>(record.total_cycles));
    return 0;
}
