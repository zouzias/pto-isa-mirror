#include "pto/comm/comm_types.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "pto/common/cpu_stub.hpp"

class UrmaWorkspaceManager {
public:
    UrmaWorkspaceManager() = default;
    ~UrmaWorkspaceManager()
    {
        Finalize();
    }

    void Finalize()
    {
    }

    void *GetWorkspaceAddr() const
    {
        return urmaInfoDevice_;
    }

    void *urmaInfoDevice_{nullptr};
};


struct UrmaInfo {
    uint32_t qpNum;
    uint32_t localTokenId;
    uint32_t rankCount;
    uint64_t sqPtr;
    uint64_t rqPtr;
    uint64_t scqPtr;
    uint64_t rcqPtr;
    uint64_t memPtr;
};

struct UrmaMemInfo {
    bool tokenValueValid;
    uint32_t rmtJettyType : 2;
    uint8_t targetHint;
    uint32_t tpn;
    uint32_t tid;
    uint32_t rmtTokenValue;
    uint32_t len;
    uint64_t addr;
    uint64_t eidAddr;
};

struct UrmaTestContext {
    int deviceId{-1};
    void *devBuf{nullptr};
    UrmaWorkspaceManager urmaMgr;

    bool AllocHugePageBuffer(size_t commBytesNeeded)
    {
        return true;
    }

    bool Setup(int rank_id, int n_ranks, int n_devices, int first_device_id, int root_rank, size_t commBytesNeeded)
    {
        return true;
    }

    void Cleanup()
    {
        urmaMgr.Finalize();
        if (devBuf) {
            aclrtFree(devBuf);
            devBuf = nullptr;
        }
    }
};

AICORE inline uint64_t UrmaPeerMrBaseAddr(__gm__ uint8_t *urmaWorkspace, uint32_t peerRank)
{
    __gm__ UrmaInfo *info = (__gm__ UrmaInfo *)urmaWorkspace;
    PTO_ASSERT(peerRank < info->rankCount, "UrmaPeerMrBaseAddr: peerRank out of range");
    __gm__ UrmaMemInfo *memRow = reinterpret_cast<__gm__ UrmaMemInfo *>(info->memPtr) + peerRank;
    return memRow->addr;
}


template <pto::comm::DmaEngine engine>
PTO_INTERNAL bool BuildAsyncSession(__gm__ uint8_t *workspace, uint32_t destRankId, pto::comm::AsyncSession &session)
{
    static_assert(engine == pto::comm::DmaEngine::URMA, "This overload is for URMA only");
    return true;
}

using UrmaKernelFn = bool (*)(int, int, int, int, int, int);

inline bool RunUrmaTestMpiLaunch(int n_ranks, int n_devices, int first_rank_id, int first_device_id,
                                 UrmaKernelFn kernelFn)
{
    int root = 0;
    bool res = false;
    for (size_t i = 0; i < n_ranks; i++) {
        res = res || kernelFn(i, n_ranks, n_devices, first_device_id, first_rank_id, root);
    }
    return res;
}
