// st_a5 TADD — 验证 A5 CCE mock 的 PTO 入口 + VF 结算全链路(占位 cycle),并对 VF 树结构做强断言。
//
// 链路:TADD → MAP_INSTR_IMPL(PtoInstrScope + TADD_IMPL + RecordInstr)
//        → a5/TAdd → BinaryInstr → TBinOps_1D_NoPostUpdate(__VEC_SCOPE__{for...vadd...})
//          ├─ pass 拦截 for(测试 TU 经 target_enable_a5_vf_mock 带 pass)
//          ├─ vadd 桩 rec() 记指令(构建 VF)
//          └─ ~ScopeSentinel → BuildVfInfo → push vf_infos
//        → ~PtoInstrScope(EndPtoInstr)→ PredictVfCycles(vector<VfInfo>) → total_cycles
//
// 强断言(cycle 是占位常数不与真机对齐,只验结构):
//   - vf_infos 恰好 1 个 VfInfo;树顶层 1 个 LOOP、嵌套深度 1(1D NoPostUpdate 路径)。
//   - loop count = ceil(640 / 64) = 10(ElementsPerRepeat = CCE_VL/sizeof(float) = 256/4 = 64)。
//   - 典范 body(均匀循环各轮同构)= [plt_b32, vlds, vlds, vadd, vsts]。
//   - cycles == count × ΣVecCycle(body):验证 loop 倍率正确乘进叶子(不硬编码占位常数)。
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "pto/costmodel/trace.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vf = ::pto::mocker::vf;
using namespace pto;

// 单轮 body 叶子的 VecCycle 之和(用占位表自身求和,与 PredictVfCycles 同源 → 只验「倍率×遍历」)。
static uint64_t SumVecCycle(const std::vector<std::string> &leaves)
{
    uint64_t s = 0;
    for (const std::string &n : leaves) s += vf::VecCycle(n);
    return s;
}

int main()
{
    using TileData = Tile<TileType::Vec, float, 1, 640, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(1, 640);
    TileData src1Tile(1, 640);
    TileData dstTile(1, 640);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    ::pto::mocker::ResetTrace();
    TADD(dstTile, src0Tile, src1Tile);

    const uint64_t cycles = ::pto::mocker::GetLastPtoInstrCycles();
    const auto &trace = ::pto::mocker::GetTrace();
    const auto &pto_rec = trace.executed_pto.back();
    const std::size_t nvf = pto_rec.vf_infos.size();

    // TADD 1×640 f32:EPR=256/4=64 → repeat=ceil(640/64)=10;每轮 plt_b32+vlds+vlds+vadd+vsts。
    constexpr uint64_t kExpectedRepeat = 10;
    const std::vector<std::string> kExpectedBody = {"plt_b32", "vlds", "vlds", "vadd", "vsts"};

    int fails = 0;
    auto check = [&](bool cond, const char *msg) {
        std::printf("  [%s] %s\n", cond ? " ok " : "FAIL", msg);
        if (!cond) ++fails;
    };

    check(cycles > 0, "cycles > 0");
    check(nvf == 1, "vf_infos == 1");
    check(nvf >= 1 && vf::LoopDepth(pto_rec.vf_infos[0].tree) == 1, "loop 嵌套深度 == 1(1D 路径)");
    if (nvf >= 1) {
        const vf::VfInfo &info = pto_rec.vf_infos[0];
        const bool one_loop = (info.tree.size() == 1 && vf::IsLoop(info.tree[0]));
        check(one_loop, "顶层恰好一个根 LOOP 节点");
        if (one_loop) {
            const vf::VfLoop &lp = vf::AsLoop(info.tree[0]);
            check(lp.count == kExpectedRepeat, "loop count == 10(1x640 f32 / EPR 64)");
            const std::vector<std::string> body = vf::FlattenLeafInstrs(lp.body);
            check(body == kExpectedBody, "典范 body == [plt_b32,vlds,vlds,vadd,vsts]");
            // cycles 必须等于 loop 倍率 × 单轮 body 之和(验倍率正确乘进叶子,不依赖占位数值)。
            check(cycles == lp.count * SumVecCycle(body), "cycles == count * ΣVecCycle(body)");
        }
    }

    std::printf("st_a5 TADD f32 1x640: cycles=%llu vf_infos=%zu\n",
                static_cast<unsigned long long>(cycles), nvf);
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
