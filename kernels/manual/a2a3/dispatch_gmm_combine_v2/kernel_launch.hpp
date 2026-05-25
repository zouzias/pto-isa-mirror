#pragma once

#include <cstdint>
#include <vector>


struct DispatchRangeLaunchArgs {
    void* remoteWindow = nullptr;
    void* dispatchRanges = nullptr;
    void* tiling = nullptr;
    uint32_t blockDim = 1;
};

struct CombineOnlyLaunchArgs {
    void* remoteWindow = nullptr;
    void* params = nullptr;
    void* tiling = nullptr;
    uint32_t blockDim = 1;
};

struct CommVecQueueLaunchArgs {
    void* remoteWindow = nullptr;
    void* quantInput = nullptr;
    void* scale1 = nullptr;
    void* gmm1Out = nullptr;
    void* swigluQ = nullptr;
    void* scale2 = nullptr;
    void* gmm2Out = nullptr;
    void* signalBase = nullptr;
    void* computeTiles = nullptr;
    void* expertSourceSegments = nullptr;
    void* expertGroupRanges = nullptr;
    void* combineParams = nullptr;
    void* combineTiling = nullptr;
    void* dispatchRanges = nullptr;
    void* dispatchTiling = nullptr;
    void* dispatchToGmm1Queue = nullptr;
    void* gmm1ToSwiGluQueue = nullptr;
    void* swiGluToGmm2Queue = nullptr;
    void* gmm2ToCombineQueue = nullptr;
    uint32_t expertSafeRowsIndex = 0;
    uint32_t requiredSafeRows = 0;
    uint32_t expertGroupCount = 0;
    uint32_t vecDispatchProducerCount = 1;
    uint32_t vecSwiGluProducerCount = 1;
    uint32_t computeTileCount = 0;
    uint32_t rowCount = 0;
    uint32_t inputStrideBytes = 0;
    uint32_t scaleStrideBytes = 0;
    uint32_t outputElems = 0;
    uint32_t outputStrideElems = 0;
    uint32_t hiddenK = 0;
    uint32_t gmm1N = 0;
    uint32_t blockDim = 1;
};

struct ComputeCubeQueueLaunchArgs {
    void* quantInput = nullptr;
    void* weight1 = nullptr;
    void* gmm1Out = nullptr;
    void* scale1Channel = nullptr;
    void* swigluQ = nullptr;
    void* weight2 = nullptr;
    void* gmm2Out = nullptr;
    void* scale2Channel = nullptr;
    void* signalBase = nullptr;
    void* dispatchToGmm1Queue = nullptr;
    void* gmm1ToSwiGluQueue = nullptr;
    void* swiGluToGmm2Queue = nullptr;
    void* gmm2ToCombineQueue = nullptr;
    uint32_t gmm2TileReadyIndex = 0;
    uint32_t gmm2TileReadyCount = 0;
    uint32_t rowCount = 0;
    uint32_t outputElems = 0;
    uint32_t outputStrideElems = 0;
    uint32_t weight2StrideElems = 0;
    uint32_t hiddenK = 0;
    uint32_t gmm1N = 0;
    uint32_t blockDim = 1;
};

struct ModeRunResult {
    bool pass = false;
    std::vector<unsigned char> bytes;
    std::vector<float> output;
};

void launchDispatchRange(const DispatchRangeLaunchArgs& args, void* stream);
void launchCombineOnly(const CombineOnlyLaunchArgs& args, void* stream);
void launchCommVecQueuePipeline(const CommVecQueueLaunchArgs& args, void* stream);
void launchComputeCubeQueuePipeline(const ComputeCubeQueueLaunchArgs& args, void* stream);

inline int ExitCodeFromResult(const ModeRunResult& result) {
    return result.pass ? 0 : 1;
}
