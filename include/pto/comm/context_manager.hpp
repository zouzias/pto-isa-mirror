#ifndef PTO_COMM_CONTEXT_MANAGER_HPP
#define PTO_COMM_CONTEXT_MANAGER_HPP

#include <cstddef>
#include <cstdint>

#include "pto/comm/backend/shmem/shmem_backend.hpp"
#include "pto/comm/comm_types.hpp"

namespace pto {
namespace comm {

struct InitOptions {
    BackendKind backend {BackendKind::Shmem};
    int rank {0};
    int size {1};
    uint64_t symmetricHeapBytes {0};
    const char *ipPort {nullptr};
};

class ContextManager {
public:
    static int Init(const InitOptions &opts)
    {
        backend_ = opts.backend;
        int ret = 0;
        switch (backend_) {
            case BackendKind::Shmem:
                ret = backend::ShmemBackend::Init(
                    {opts.rank, opts.size, opts.symmetricHeapBytes, opts.ipPort});
                break;
            default:
                ret = -1;
                break;
        }
        initialized_ = (ret == 0);
        return ret;
    }

    static void Finalize()
    {
        if (!initialized_) {
            return;
        }
        switch (backend_) {
            case BackendKind::Shmem:
                backend::ShmemBackend::Finalize();
                break;
            default:
                break;
        }
        initialized_ = false;
    }

    static void *SymmetricAlloc(std::size_t bytes)
    {
        switch (backend_) {
            case BackendKind::Shmem:
                return backend::ShmemBackend::SymmetricAlloc(bytes);
            default:
                return nullptr;
        }
    }

    static void SymmetricFree(void *ptr)
    {
        if (ptr == nullptr) {
            return;
        }
        switch (backend_) {
            case BackendKind::Shmem:
                backend::ShmemBackend::SymmetricFree(ptr);
                break;
            default:
                break;
        }
    }

    static BackendKind Backend()
    {
        return backend_;
    }

    static int GetRankID()
    {
        switch (backend_) {
            case BackendKind::Shmem:
                return backend::ShmemBackend::GetRankID();
            default:
                return -1;
        }
    }

    static int GetRankSize()
    {
        switch (backend_) {
            case BackendKind::Shmem:
                return backend::ShmemBackend::GetRankSize();
            default:
                return -1;
        }
    }

private:
    static inline BackendKind backend_ = BackendKind::Shmem;
    static inline bool initialized_ = false;
};

// Convenience functions for direct access
inline int GetRankID()
{
    return ContextManager::GetRankID();
}

inline int GetRankSize()
{
    return ContextManager::GetRankSize();
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_CONTEXT_MANAGER_HPP

