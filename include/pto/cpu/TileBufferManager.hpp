#ifndef PTO_TILE_BUFFER_MANAGER_HPP
#define PTO_TILE_BUFFER_MANAGER_HPP

#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <vector>
#include <memory>

namespace pto {

class TileBufferManager {
public:
    static TileBufferManager& Instance() {
        static TileBufferManager instance;
        return instance;
    }

    template<typename DType>
    DType* Allocate(std::size_t numElements) {
        std::size_t bytes = numElements * sizeof(DType);
        auto buffer = std::make_unique<std::vector<char>>(bytes, 0);
        DType* ptr = reinterpret_cast<DType*>(buffer->data());
        ownedBuffers_.push_back(std::move(buffer));
        return ptr;
    }

    // Check if offset has already had a tile assigned this phase
    bool IsFirstAssign(std::size_t ubOffset) {
        bool first = assignedThisPhase_.find(ubOffset) == assignedThisPhase_.end();
        if (first) {
            assignedThisPhase_.insert(ubOffset);
        }
        return first;
    }

    // Call at start of each loop iteration to reset first-assign tracking
    void ResetPhase() {
        assignedThisPhase_.clear();
    }

    template<typename DType>
    DType* GetSharedBuffer(std::size_t ubOffset, std::size_t numElements) {
        auto it = sharedBuffers_.find(ubOffset);
        if (it == sharedBuffers_.end()) {
            std::size_t bytes = numElements * sizeof(DType);
            sharedBuffers_[ubOffset].resize(bytes, 0);
        }
        return reinterpret_cast<DType*>(sharedBuffers_[ubOffset].data());
    }

    void Clear() {
        ownedBuffers_.clear();
        sharedBuffers_.clear();
        assignedThisPhase_.clear();
    }

private:
    TileBufferManager() = default;
    std::vector<std::unique_ptr<std::vector<char>>> ownedBuffers_;
    std::map<std::size_t, std::vector<char>> sharedBuffers_;
    std::set<std::size_t> assignedThisPhase_;  // Track first assign per phase
};

} // namespace pto
#endif
