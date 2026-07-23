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
//   - loop count = ceil(cols / 64)(ElementsPerRepeat = CCE_VL/sizeof(float) = 256/4 = 64)。
//   - 典范 body(均匀循环各轮同构)= [vlds, vlds, vadd, vsts]。
//   - plt_b32 是 predicate 设置，不记为 micro-op。
//   - vlds/vadd/vsts 的 dst/src 名称、location 和 dtype 被 mock 完整捕获。
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

namespace {

template <unsigned Cols, uint64_t ExpectedRepeat>
int RunTaddCase()
{
    using CaseTileData = Tile<TileType::Vec, float, 1, Cols, BLayout::RowMajor, -1, -1>;
    CaseTileData src0Tile(1, Cols);
    CaseTileData src1Tile(1, Cols);
    CaseTileData dstTile(1, Cols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x4000);
    TASSIGN(dstTile, 0x8000);

    ::pto::mocker::ResetTrace();
    TADD(dstTile, src0Tile, src1Tile);

    const uint64_t cycles = ::pto::mocker::GetLastPtoInstrCycles();
    const auto &trace = ::pto::mocker::GetTrace();
    const auto &pto_rec = trace.executed_pto.back();
    const std::size_t nvf = pto_rec.vf_infos.size();

    const std::vector<std::string> kExpectedBody = {"vlds", "vlds", "vadd", "vsts"};

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
            check(lp.count == ExpectedRepeat, "loop count == expected repeat(1xCols f32 / EPR 64)");
            const std::vector<std::string> body = vf::FlattenLeafInstrs(lp.body);
            check(body == kExpectedBody, "典范 body == [vlds,vlds,vadd,vsts]");
            check(lp.body.size() == 4, "body 节点数 == 4");
            if (lp.body.size() == 4) {
                const vf::VfInst &load0 = vf::AsInst(lp.body[0]);
                const vf::VfInst &load1 = vf::AsInst(lp.body[1]);
                const vf::VfInst &add = vf::AsInst(lp.body[2]);
                const vf::VfInst &store = vf::AsInst(lp.body[3]);
                check(load0.dst.size() == 1 && load0.src.size() == 1 &&
                          load0.dst[0].location == vf::MemLocation::PhyRegister &&
                          load0.src[0].location == vf::MemLocation::UB && load0.dst[0].dtype == "fp32" &&
                          load0.src[0].dtype == "fp32",
                      "vlds 捕获 PhyRegister dst + UB src + fp32 dtype");
                check(add.dst.size() == 1 && add.src.size() >= 2 && load0.dst.size() == 1 && load1.dst.size() == 1 &&
                          add.src[0].name == load0.dst[0].name && add.src[1].name == load1.dst[0].name,
                      "vadd src 名称延续两条 vlds dst 依赖");
                check(store.dst.size() == 1 && store.src.size() == 1 && add.dst.size() == 1 &&
                          store.dst[0].location == vf::MemLocation::UB &&
                          store.src[0].name == add.dst[0].name,
                      "vsts 捕获 UB dst 且 src 延续 vadd dst 依赖");
            }
            check(cycles > 0, "全部 micro-op 受支持时 VfSim 返回正 cycle");
        }
    }

    std::printf("st_a5 TADD f32 1x%u repeat=%llu: cycles=%llu vf_infos=%zu\n", Cols,
                static_cast<unsigned long long>(ExpectedRepeat), static_cast<unsigned long long>(cycles), nvf);
    return fails;
}

} // namespace

int main()
{
    int fails = 0;
    fails += RunTaddCase<64, 1>();
    fails += RunTaddCase<512, 8>();
    fails += RunTaddCase<1024, 16>();
    fails += RunTaddCase<2048, 32>();
    fails += RunTaddCase<4096, 64>();
    fails += RunTaddCase<6144, 96>();

    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
