// PtoLoopTracePass.cpp — 给 C++ for 循环自动插桩的 LLVM new-PM pass 插件。
// 给每个 natural loop 插运行期 C ABI hook:preheader→loop_enter(id,file,line,col)、
// header→loop_iter(id)(每轮)、exit 边→loop_exit(id)。vf_trace 据事件流折叠成 VfInfo 树。
// loopId=hash(func|file|line|col)。须 -O0 -g 保源码 for 结构。加载:-fpass-plugin=lib...so。
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopUtils.h"  // formDedicatedExitBlocks

#include <cstring>
#include <string>
#include <vector>

using namespace llvm;

namespace {

// 跨 LLVM 版本兼容(14–18+):StringRef::starts_with/startswith 跨版本命名不一致,自写 memcmp 判断。
static bool nameStartsWith(StringRef s, const char *p) {
    size_t n = std::strlen(p);
    return s.size() >= n && std::memcmp(s.data(), p, n) == 0;
}
// 不透明指针:getUnqual(i8) 在 typed-ptr(≤15)给 i8*、opaque(≥16)给 ptr,均自洽。
static PointerType *i8PtrTy(LLVMContext &C) {
    return PointerType::getUnqual(Type::getInt8Ty(C));
}

// hook 函数类型(惰性插入 module)。
static FunctionCallee getOrInsertLoopEnter(Module &M) {
    LLVMContext &C = M.getContext();
    auto *i64 = Type::getInt64Ty(C);
    auto *i8p = i8PtrTy(C);
    auto *i32 = Type::getInt32Ty(C);
    return M.getOrInsertFunction("__pto_trace_loop_enter",
                                 FunctionType::get(Type::getVoidTy(C), {i64, i8p, i32, i32}, false));
}
static FunctionCallee getOrInsertLoopIter(Module &M) {
    LLVMContext &C = M.getContext();
    return M.getOrInsertFunction("__pto_trace_loop_iter",
                                 FunctionType::get(Type::getVoidTy(C), {Type::getInt64Ty(C)}, false));
}
static FunctionCallee getOrInsertLoopExit(Module &M) {
    LLVMContext &C = M.getContext();
    return M.getOrInsertFunction("__pto_trace_loop_exit",
                                 FunctionType::get(Type::getVoidTy(C), {Type::getInt64Ty(C)}, false));
}

// 取 loop DebugLoc(优先 getStartLoc,回退 header 首条带 DebugLoc 的指令)。
static DebugLoc getLoopDebugLoc(Loop *L) {
    if (DebugLoc loc = L->getStartLoc()) return loc;
    for (Instruction &I : *L->getHeader())
        if (I.getDebugLoc()) return I.getDebugLoc();
    return DebugLoc();
}

// loopId = hash(funcName|file|line|col),同一源码 loop 每次执行拿到同一个 id。
static uint64_t computeLoopId(Function &F, Loop *L) {
    DebugLoc loc = getLoopDebugLoc(L);
    std::string key;
    raw_string_ostream os(key);
    os << F.getName() << "|";
    if (loc) {
        if (auto *scope = dyn_cast_or_null<DILocalScope>(loc.getScope()))
            if (auto *file = scope->getFile()) os << file->getFilename() << "|";
        os << loc.getLine() << ":" << loc.getCol();
    }
    os.flush();
    return static_cast<uint64_t>(hash_value(StringRef(key)));
}

// 单 loop 插桩工作项(先快照指针,收集后统一 mutate,避免 SplitEdge 打乱迭代)。
struct LoopWork {
    uint64_t loopId;
    BasicBlock *preheader;   // enter 插在其 terminator 前
    BasicBlock *header;      // iter 插在首非 PHI 前
    std::string file;
    int line;
    int col;
    std::vector<BasicBlock *> exitBlocks;  // dedicated exit blocks(exit hook 插这些块开头)
};

static bool isScopeCtorCall(const CallBase *call) {
    const Function *callee = call->getCalledFunction();
    return callee != nullptr && callee->getName().contains("ScopeSentinelC");
}

static std::vector<const Instruction *> collectScopeEnters(Function &F) {
    std::vector<const Instruction *> scopeEnters;
    for (BasicBlock &BB : F)
        for (Instruction &I : BB)
            if (const auto *call = dyn_cast<CallBase>(&I))
                if (isScopeCtorCall(call))
                    scopeEnters.push_back(&I);
    return scopeEnters;
}

// 递归收集 loop 及内层,只收 inScope 为真者(在 __VEC_SCOPE__ 内,按 header 被 scope_enter 支配判定)。
template <class InScopeFn>
static void collectLoopInScope(Loop *L, Function &F, std::vector<LoopWork> &out,
                               InScopeFn &&inScope) {
    for (Loop *sub : L->getSubLoops()) collectLoopInScope(sub, F, out, inScope);

    if (!inScope(L)) return;  // 不在 __VEC_SCOPE__ 内(host 标量预处理 for 等)→ 跳过

    BasicBlock *pre = L->getLoopPreheader();
    DebugLoc loc = getLoopDebugLoc(L);
    if (!pre) {
        errs() << "[PtoLoopTrace] WARN: loop without preheader skipped ("
               << F.getName() << ")\n";
        return;
    }
    if (!loc) {
        errs() << "[PtoLoopTrace] WARN: loop without DebugLoc skipped ("
               << F.getName() << ")\n";
        return;
    }

    LoopWork w;
    w.loopId = computeLoopId(F, L);
    w.preheader = pre;
    w.header = L->getHeader();
    if (auto *scope = dyn_cast_or_null<DILocalScope>(loc.getScope()))
        if (auto *file = scope->getFile()) w.file = file->getFilename().str();
    w.line = static_cast<int>(loc.getLine());
    w.col = static_cast<int>(loc.getCol());

    // VF 模板的 for 是规范计数循环 → exit block 唯一且 dedicated,exit hook 插其首个插入点。
    if (!L->hasDedicatedExits()) {
        errs() << "[PtoLoopTrace] WARN: loop without dedicated exits skipped ("
               << F.getName() << ")\n";
        return;
    }
    SmallVector<BasicBlock *, 4> exitBlks;
    L->getUniqueExitBlocks(exitBlks);
    for (BasicBlock *eb : exitBlks) w.exitBlocks.push_back(eb);
	    out.push_back(std::move(w));
	}

template <class InScopeFn>
static bool formDedicatedExitsForScopedLoops(LoopInfo &LI, DominatorTree &DT,
                                             InScopeFn &&loopInScope) {
    bool cfgChanged = false;
    for (Loop *L : LI) {
        SmallVector<Loop *, 8> nest;
        nest.push_back(L);
        for (size_t i = 0; i < nest.size(); ++i)
            for (Loop *sub : nest[i]->getSubLoops()) nest.push_back(sub);
        for (Loop *cur : nest)
            if (loopInScope(cur))
                cfgChanged |= formDedicatedExitBlocks(cur, &DT, &LI,
                                                      /*MSSAU=*/nullptr,
                                                      /*PreserveLCSSA=*/false);
    }
    return cfgChanged;
}

static void insertLoopEnterAndIter(LoopWork &w, LLVMContext &C,
                                   FunctionCallee enterFn,
                                   FunctionCallee iterFn) {
    ConstantInt *loopIdC = ConstantInt::get(C, APInt(64, w.loopId));
    IRBuilder<> enterBuilder(w.preheader->getTerminator());
    Value *filePtr =
        enterBuilder.CreateGlobalStringPtr(w.file.empty() ? StringRef("") : StringRef(w.file));
    enterBuilder.CreateCall(
        enterFn, {loopIdC, filePtr,
                  ConstantInt::get(C, APInt(32, static_cast<uint64_t>(w.line))),
                  ConstantInt::get(C, APInt(32, static_cast<uint64_t>(w.col)))});

    IRBuilder<> iterBuilder(w.header, w.header->getFirstInsertionPt());
    iterBuilder.CreateCall(iterFn, {loopIdC});
}

static void insertLoopExits(LoopWork &w, LLVMContext &C,
                            FunctionCallee exitFn) {
    ConstantInt *loopIdC = ConstantInt::get(C, APInt(64, w.loopId));
    for (BasicBlock *eb : w.exitBlocks) {
        IRBuilder<> b(eb, eb->getFirstInsertionPt());
        b.CreateCall(exitFn, {loopIdC});
    }
}

struct PtoLoopTracePass : PassInfoMixin<PtoLoopTracePass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
        if (F.isDeclaration()) return PreservedAnalyses::all();

        Module &M = *F.getParent();
        LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
        DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);

	        std::vector<const Instruction *> scopeEnters = collectScopeEnters(F);

        const bool isDemo = nameStartsWith(F.getName(), "__pto_demo_");
        if (scopeEnters.empty() && !isDemo) {
            return PreservedAnalyses::all();  // 函数内无 VEC scope 且非 demo → 不碰
        }

        // loop 是否在某 scope_enter 支配下(= 在 __VEC_SCOPE__ 内)。
        auto loopInScope = [&](Loop *L) -> bool {
            if (isDemo) return true;  // demo 豁免
            BasicBlock *h = L->getHeader();
            for (const Instruction *en : scopeEnters)
                if (DT.dominates(en, &*h->getFirstInsertionPt())) return true;
            return false;
        };

        // VF 驱动内联进 if-init 块后,loop exit 常与哨兵析构/if-false 合流 → 非 dedicated。
        // 先对 in-scope loop 形成专用出口块(改 CFG,重算 DT),之后 getUniqueExitBlocks 才稳。
	        bool cfgChanged = formDedicatedExitsForScopedLoops(LI, DT, loopInScope);
	        if (cfgChanged) DT.recalculate(F);

        std::vector<LoopWork> work;
        for (Loop *L : LI) collectLoopInScope(L, F, work, loopInScope);
        if (work.empty()) return PreservedAnalyses::all();

        FunctionCallee enterFn = getOrInsertLoopEnter(M);
        FunctionCallee iterFn = getOrInsertLoopIter(M);
        FunctionCallee exitFn = getOrInsertLoopExit(M);
        LLVMContext &C = M.getContext();

	        bool changed = false;
	        for (LoopWork &w : work) {
	            insertLoopEnterAndIter(w, C, enterFn, iterFn);
	            changed = true;
	        }

        // exit:每个 dedicated exit block 首插入点。dedicated ⇒ 块只由本 loop 进入 → 跳出必经一次。
        // 嵌套循环里内层 exit block 可能是外层 latch,exit hook 落外层 body → 正确反映"内层结束回外层"。
	        for (LoopWork &w : work) {
	            insertLoopExits(w, C, exitFn);
	        }

        return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }

    static bool isRequired() { return true; }  // 即使 -O0 也保留
};

}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "PtoLoopTrace", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                // 显式 -passes=pto-loop-trace 时挂 function pipeline。
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name != "pto-loop-trace") return false;
                        FPM.addPass(PtoLoopTracePass());
                        return true;
                    });
                // 默认 pipeline 起点自动挂(覆盖 -O0/-O1),用 module→function adaptor 跑遍每函数。
                PB.registerPipelineStartEPCallback(
                    [](ModulePassManager &MPM, OptimizationLevel) {
                        MPM.addPass(createModuleToFunctionPassAdaptor(PtoLoopTracePass()));
                    });
            }};
}
