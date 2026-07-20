// st_a5 multi_vf — 验证一条 PTO-ISA 含多个 __VEC_SCOPE__(VF)时,vf_infos 攒成 vector、
// 后端对 vector<VfInfo> 统一结算(占位先求和),并对每个 VF 的树结构做强断言。
//
// 合成构造:一个 PtoInstrScope 内手动放两个 __VEC_SCOPE__(各含一个 for+vadd),模拟
// 2D_PostUpdate 的 FullRepeats + Tail 双 VF 场景(绕开 2D tile 分派的构造复杂度,聚焦验证
// "多 VF → vector → 后端"机制)。
//
// 强断言:
//   - vf_infos == 2(两个 VF 独立折叠,互不串扰)。
//   - 每个 VF:顶层一个 LOOP、count==4、body==[vadd]。
//   - cycles == 两 VF 之和(各 vadd=6 × 4 = 24 → 共 48)。
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "pto/costmodel/trace.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vf = ::pto::mocker::vf;

int main()
{
    ::pto::mocker::ResetTrace();
    {
        // 一个 PTO 指令周期(模拟一条 PTO 指令,含两个 VF)。
        ::pto::mocker::PtoInstrScope scope("MOCK_2VF");
        // VF #1
        __VEC_SCOPE__
        {
            for (int i = 0; i < 4; ++i) { vadd(); }
        }
        // VF #2
        __VEC_SCOPE__
        {
            for (int j = 0; j < 4; ++j) { vadd(); }
        }
    }  // ~PtoInstrScope → EndPtoInstr → PredictVfCycles(vector<VfInfo>)

    const uint64_t cycles = ::pto::mocker::GetLastPtoInstrCycles();
    const auto &trace = ::pto::mocker::GetTrace();
    const auto &pto_rec = trace.executed_pto.back();
    const std::size_t nvf = pto_rec.vf_infos.size();

    constexpr uint64_t kExpectedCount = 4;  // for (..; i < 4; ..)

    // 单 VF 期望 cycle = count × ΣVecCycle(body);body=[vadd]。两 VF 求和。
    auto vf_expected = [](const vf::VfInfo &info) -> uint64_t {
        if (info.tree.size() != 1 || !vf::IsLoop(info.tree[0])) return 0;
        const vf::VfLoop &lp = vf::AsLoop(info.tree[0]);
        uint64_t s = 0;
        for (const std::string &n : vf::FlattenLeafInstrs(lp.body)) s += vf::VecCycle(n);
        return lp.count * s;
    };

    int fails = 0;
    auto check = [&](bool cond, const char *msg) {
        std::printf("  [%s] %s\n", cond ? " ok " : "FAIL", msg);
        if (!cond) ++fails;
    };

    uint64_t expected_total = 0;
    for (std::size_t i = 0; i < nvf; ++i) expected_total += vf_expected(pto_rec.vf_infos[i]);

    check(cycles > 0, "cycles > 0");
    check(nvf == 2, "vf_infos == 2(两 VF 独立折叠)");
    check(cycles == expected_total, "cycles == Σ(count × ΣVecCycle(body))(占位常数无关)");
    for (std::size_t i = 0; i < nvf; ++i) {
        char msg[80];
        const vf::VfInfo &info = pto_rec.vf_infos[i];
        const bool one_loop = (info.tree.size() == 1 && vf::IsLoop(info.tree[0]));

        std::snprintf(msg, sizeof(msg), "VF#%zu: 顶层一个 LOOP", i);
        check(one_loop, msg);
        std::snprintf(msg, sizeof(msg), "VF#%zu: loop count == 4", i);
        check(one_loop && vf::AsLoop(info.tree[0]).count == kExpectedCount, msg);
        std::snprintf(msg, sizeof(msg), "VF#%zu: body == [vadd]", i);
        const bool body_ok = one_loop &&
            vf::FlattenLeafInstrs(vf::AsLoop(info.tree[0]).body) == std::vector<std::string>{"vadd"};
        check(body_ok, msg);
    }

    std::printf("st_a5 multi_vf: cycles=%llu vf_infos=%zu\n",
                static_cast<unsigned long long>(cycles), nvf);
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
