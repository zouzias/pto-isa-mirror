#include "acl/acl.h"
#include "kernel_launchers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string case_name = "smoke";
    int device = 0;
    int ranks = 2;
    int experts_per_rank = 2;
    int tokens = 4;
    int hidden = 8;
    int topk = 2;
};

struct CaseSpec {
    std::string name;
    int ranks;
    int experts_per_rank;
    int tokens;
    int hidden;
    int topk;
};

struct HostData {
    std::vector<float> x;
    std::vector<int32_t> expert_idx;
    std::vector<float> probs;
    std::vector<int32_t> active;
};

struct GoldenData {
    std::vector<float> out;
    std::vector<int32_t> count;
    std::vector<int32_t> src_expert_offset;
    std::vector<int32_t> expanded_row_idx;
    std::vector<float> src_packed_x;
    std::vector<float> dispatch_x;
    std::vector<float> return_y;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        if (bytes_ == 0) {
            return;
        }
        Check(aclrtMalloc(&ptr_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_)
    {
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other) {
            Release();
            ptr_ = other.ptr_;
            bytes_ = other.bytes_;
            other.ptr_ = nullptr;
            other.bytes_ = 0;
        }
        return *this;
    }
    ~DeviceBuffer()
    {
        Release();
    }
    void *ptr() const
    {
        return ptr_;
    }
    size_t bytes() const
    {
        return bytes_;
    }

private:
    static void Check(aclError ret, const char *op)
    {
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error(std::string(op) + " failed: " + std::to_string(static_cast<int>(ret)));
        }
    }
    void Release()
    {
        if (ptr_ != nullptr) {
            aclrtFree(ptr_);
            ptr_ = nullptr;
            bytes_ = 0;
        }
    }

    void *ptr_ = nullptr;
    size_t bytes_ = 0;
};

void CheckAcl(aclError ret, const char *op)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(op) + " failed: " + std::to_string(static_cast<int>(ret)));
    }
}

int XOffset(int rank, int token, int h, int tokens, int hidden)
{
    return (rank * tokens + token) * hidden + h;
}

int RouteOffset(int rank, int token, int slot, int tokens, int topk)
{
    return (rank * tokens + token) * topk + slot;
}

int CountOffset(int rank, int expert, int global_experts)
{
    return rank * global_experts + expert;
}

int PackedOffset(int rank, int row, int h, int max_rows, int hidden)
{
    return (rank * max_rows + row) * hidden + h;
}

int DispatchOffset(int dst, int row, int h, int max_dispatch_rows, int hidden)
{
    return (dst * max_dispatch_rows + row) * hidden + h;
}

Options ParseOptions(int argc, char **argv)
{
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto require_value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (key == "--case") {
            opt.case_name = require_value("--case");
        } else if (key == "--device") {
            opt.device = std::stoi(require_value("--device"));
        } else if (key == "--ranks") {
            opt.ranks = std::stoi(require_value("--ranks"));
        } else if (key == "--experts-per-rank") {
            opt.experts_per_rank = std::stoi(require_value("--experts-per-rank"));
        } else if (key == "--tokens") {
            opt.tokens = std::stoi(require_value("--tokens"));
        } else if (key == "--hidden") {
            opt.hidden = std::stoi(require_value("--hidden"));
        } else if (key == "--topk") {
            opt.topk = std::stoi(require_value("--topk"));
        } else if (key == "--help" || key == "-h") {
            std::cout << "Usage: dispatch_combine [--case smoke|minimal|cross|edge|all] [--device N] "
                      << "[--ranks R] [--experts-per-rank E] [--tokens M] [--hidden H] [--topk K]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }
    return opt;
}

void ValidateSpec(const CaseSpec &spec)
{
    if (spec.ranks <= 0 || spec.experts_per_rank <= 0 || spec.tokens <= 0 || spec.hidden <= 0 || spec.topk <= 0) {
        throw std::invalid_argument("all dimensions must be positive");
    }
    if (!(spec.hidden == 1 || spec.hidden == 4 || spec.hidden == 8 || spec.hidden == 16 || spec.hidden == 32 ||
          spec.hidden == 64)) {
        throw std::invalid_argument("hidden must be one of 1, 4, 8, 16, 32, 64");
    }
    if (spec.ranks > 8 || spec.experts_per_rank > 8 || spec.tokens > 64 || spec.topk > 8) {
        throw std::invalid_argument("case is larger than this correctness scaffold allows");
    }

    const size_t float_bytes = sizeof(float);
    const size_t x_bytes = static_cast<size_t>(spec.ranks) * spec.tokens * spec.hidden * float_bytes;
    const size_t packed_bytes = static_cast<size_t>(spec.ranks) * spec.tokens * spec.topk * spec.hidden * float_bytes;
    const size_t dispatch_bytes = static_cast<size_t>(spec.ranks) * spec.ranks * spec.tokens * spec.topk * spec.hidden * float_bytes;
    constexpr size_t kXUb = 0x00000;
    constexpr size_t kPackedUb = 0x04000;
    constexpr size_t kDispatchUb = 0x0C000;
    constexpr size_t kReturnUb = 0x1C000;
    constexpr size_t kOutUb = 0x24000;
    constexpr size_t kUbLimit = 0x28000;
    if (kXUb + x_bytes > kPackedUb || kPackedUb + packed_bytes > kDispatchUb ||
        kDispatchUb + dispatch_bytes > kReturnUb || kReturnUb + packed_bytes > kOutUb ||
        kOutUb + x_bytes > kUbLimit) {
        throw std::invalid_argument("case exceeds fixed UB workspace layout");
    }
}

HostData MakeInput(const CaseSpec &spec)
{
    const int g = spec.ranks * spec.experts_per_rank;
    HostData data;
    data.x.resize(static_cast<size_t>(spec.ranks) * spec.tokens * spec.hidden);
    data.expert_idx.resize(static_cast<size_t>(spec.ranks) * spec.tokens * spec.topk);
    data.probs.resize(static_cast<size_t>(spec.ranks) * spec.tokens * spec.topk);
    data.active.assign(static_cast<size_t>(spec.ranks) * spec.tokens, 1);

    for (int rank = 0; rank < spec.ranks; ++rank) {
        for (int token = 0; token < spec.tokens; ++token) {
            for (int h = 0; h < spec.hidden; ++h) {
                data.x[XOffset(rank, token, h, spec.tokens, spec.hidden)] =
                    static_cast<float>(rank * 1000 + token * 10) + static_cast<float>(h) * 0.125f;
            }
            float denom = static_cast<float>(spec.topk * (spec.topk + 1) / 2);
            for (int slot = 0; slot < spec.topk; ++slot) {
                int offset = RouteOffset(rank, token, slot, spec.tokens, spec.topk);
                data.expert_idx[offset] = (rank + token + slot) % g;
                data.probs[offset] = static_cast<float>(slot + 1) / denom;
            }
        }
    }

    if (spec.name == "cross" || spec.name == "smoke" || spec.name == "edge") {
        for (int rank = 0; rank < spec.ranks; ++rank) {
            for (int token = 0; token < spec.tokens; ++token) {
                for (int slot = 0; slot < spec.topk; ++slot) {
                    int dst_rank = (rank + slot + 1) % spec.ranks;
                    int local_e = (token + slot) % spec.experts_per_rank;
                    data.expert_idx[RouteOffset(rank, token, slot, spec.tokens, spec.topk)] =
                        dst_rank * spec.experts_per_rank + local_e;
                }
            }
        }
    }

    if (spec.name == "edge") {
        if (spec.tokens > 1) {
            data.active[1] = 0;
        }
        if (spec.topk > 1) {
            data.expert_idx[RouteOffset(0, 0, 1, spec.tokens, spec.topk)] =
                data.expert_idx[RouteOffset(0, 0, 0, spec.tokens, spec.topk)];
        }
        if (spec.ranks > 1 && spec.tokens > 2) {
            data.expert_idx[RouteOffset(1, 2, 0, spec.tokens, spec.topk)] = g + 3;
        }
        if (spec.topk > 1 && spec.tokens > 3) {
            data.probs[RouteOffset(0, 3, 0, spec.tokens, spec.topk)] = 0.25f;
            data.probs[RouteOffset(0, 3, 1, spec.tokens, spec.topk)] = 0.50f;
        }
    }

    return data;
}

GoldenData ComputeGolden(const CaseSpec &spec, const HostData &data)
{
    const int global_experts = spec.ranks * spec.experts_per_rank;
    const int max_rows = spec.tokens * spec.topk;
    const int max_dispatch_rows = spec.ranks * max_rows;

    GoldenData golden;
    golden.out.assign(static_cast<size_t>(spec.ranks) * spec.tokens * spec.hidden, 0.0f);
    golden.count.assign(static_cast<size_t>(spec.ranks) * global_experts, 0);
    golden.src_expert_offset.assign(static_cast<size_t>(spec.ranks) * global_experts, 0);
    golden.expanded_row_idx.assign(static_cast<size_t>(spec.ranks) * max_rows, -1);
    golden.src_packed_x.assign(static_cast<size_t>(spec.ranks) * max_rows * spec.hidden, 0.0f);
    golden.dispatch_x.assign(static_cast<size_t>(spec.ranks) * max_dispatch_rows * spec.hidden, 0.0f);
    golden.return_y.assign(static_cast<size_t>(spec.ranks) * max_rows * spec.hidden, 0.0f);

    auto valid = [&](int src, int token, int slot) {
        if (data.active[src * spec.tokens + token] == 0) {
            return false;
        }
        int expert = data.expert_idx[RouteOffset(src, token, slot, spec.tokens, spec.topk)];
        return expert >= 0 && expert < global_experts;
    };

    for (int src = 0; src < spec.ranks; ++src) {
        for (int token = 0; token < spec.tokens; ++token) {
            for (int slot = 0; slot < spec.topk; ++slot) {
                if (!valid(src, token, slot)) {
                    continue;
                }
                int expert = data.expert_idx[RouteOffset(src, token, slot, spec.tokens, spec.topk)];
                golden.count[CountOffset(src, expert, global_experts)] += 1;
            }
        }
    }

    for (int src = 0; src < spec.ranks; ++src) {
        int running = 0;
        for (int expert = 0; expert < global_experts; ++expert) {
            int idx = CountOffset(src, expert, global_experts);
            golden.src_expert_offset[idx] = running;
            running += golden.count[idx];
        }
    }

    std::vector<int32_t> write_cursor = golden.src_expert_offset;
    for (int src = 0; src < spec.ranks; ++src) {
        for (int token = 0; token < spec.tokens; ++token) {
            for (int slot = 0; slot < spec.topk; ++slot) {
                int flat = token * spec.topk + slot;
                if (!valid(src, token, slot)) {
                    golden.expanded_row_idx[src * max_rows + flat] = -1;
                    continue;
                }
                int expert = data.expert_idx[RouteOffset(src, token, slot, spec.tokens, spec.topk)];
                int cursor_idx = CountOffset(src, expert, global_experts);
                int row = write_cursor[cursor_idx]++;
                golden.expanded_row_idx[src * max_rows + flat] = row;
                for (int h = 0; h < spec.hidden; ++h) {
                    golden.src_packed_x[PackedOffset(src, row, h, max_rows, spec.hidden)] =
                        data.x[XOffset(src, token, h, spec.tokens, spec.hidden)];
                }
            }
        }
    }

    for (int dst = 0; dst < spec.ranks; ++dst) {
        int dispatch_cursor = 0;
        for (int e = 0; e < spec.experts_per_rank; ++e) {
            int expert = dst * spec.experts_per_rank + e;
            for (int src = 0; src < spec.ranks; ++src) {
                int rows = golden.count[CountOffset(src, expert, global_experts)];
                int read_base = golden.src_expert_offset[CountOffset(src, expert, global_experts)];
                for (int r = 0; r < rows; ++r) {
                    for (int h = 0; h < spec.hidden; ++h) {
                        golden.dispatch_x[DispatchOffset(dst, dispatch_cursor + r, h, max_dispatch_rows, spec.hidden)] =
                            golden.src_packed_x[PackedOffset(src, read_base + r, h, max_rows, spec.hidden)];
                    }
                }
                dispatch_cursor += rows;
            }
        }
    }

    for (int dst = 0; dst < spec.ranks; ++dst) {
        int dispatch_cursor = 0;
        for (int e = 0; e < spec.experts_per_rank; ++e) {
            int expert = dst * spec.experts_per_rank + e;
            for (int src = 0; src < spec.ranks; ++src) {
                int rows = golden.count[CountOffset(src, expert, global_experts)];
                int write_base = golden.src_expert_offset[CountOffset(src, expert, global_experts)];
                for (int r = 0; r < rows; ++r) {
                    for (int h = 0; h < spec.hidden; ++h) {
                        golden.return_y[PackedOffset(src, write_base + r, h, max_rows, spec.hidden)] =
                            golden.dispatch_x[DispatchOffset(dst, dispatch_cursor + r, h, max_dispatch_rows, spec.hidden)];
                    }
                }
                dispatch_cursor += rows;
            }
        }
    }

    for (int src = 0; src < spec.ranks; ++src) {
        for (int token = 0; token < spec.tokens; ++token) {
            for (int slot = 0; slot < spec.topk; ++slot) {
                int flat = token * spec.topk + slot;
                int row = golden.expanded_row_idx[src * max_rows + flat];
                if (row < 0) {
                    continue;
                }
                float scale = data.probs[RouteOffset(src, token, slot, spec.tokens, spec.topk)];
                for (int h = 0; h < spec.hidden; ++h) {
                    golden.out[XOffset(src, token, h, spec.tokens, spec.hidden)] +=
                        scale * golden.return_y[PackedOffset(src, row, h, max_rows, spec.hidden)];
                }
            }
        }
    }

    return golden;
}

template <typename T>
void CopyHostToDevice(DeviceBuffer &dst, const std::vector<T> &src, const char *name)
{
    CheckAcl(aclrtMemcpy(dst.ptr(), dst.bytes(), src.data(), src.size() * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE), name);
}

template <typename T>
std::vector<T> CopyDeviceToHost(const DeviceBuffer &src, size_t count, const char *name)
{
    std::vector<T> host(count);
    CheckAcl(aclrtMemcpy(host.data(), host.size() * sizeof(T), src.ptr(), src.bytes(), ACL_MEMCPY_DEVICE_TO_HOST), name);
    return host;
}

bool CompareFloatVector(const std::string &case_name, const std::string &label, const std::vector<float> &expected,
                        const std::vector<float> &actual, float atol)
{
    if (expected.size() != actual.size()) {
        std::cout << "[FAIL] " << case_name << " " << label << " size mismatch expected=" << expected.size()
                  << " actual=" << actual.size() << "\n";
        return false;
    }
    float max_diff = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        float diff = std::fabs(expected[i] - actual[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
        if (diff > atol) {
            std::cout << "[FAIL] " << case_name << " " << label << " mismatch idx=" << i
                      << " expected=" << expected[i] << " actual=" << actual[i] << " diff=" << diff << "\n";
            return false;
        }
    }
    std::cout << "[PASS] " << case_name << " " << label << " max_diff=" << max_diff << " idx=" << max_idx << "\n";
    return true;
}

bool CompareIntVector(const std::string &case_name, const std::string &label, const std::vector<int32_t> &expected,
                      const std::vector<int32_t> &actual)
{
    if (expected.size() != actual.size()) {
        std::cout << "[FAIL] " << case_name << " " << label << " size mismatch expected=" << expected.size()
                  << " actual=" << actual.size() << "\n";
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::cout << "[FAIL] " << case_name << " " << label << " mismatch idx=" << i
                      << " expected=" << expected[i] << " actual=" << actual[i] << "\n";
            return false;
        }
    }
    std::cout << "[PASS] " << case_name << " " << label << "\n";
    return true;
}

bool RunOneCase(const CaseSpec &spec, int device)
{
    ValidateSpec(spec);
    HostData input = MakeInput(spec);
    GoldenData golden = ComputeGolden(spec, input);

    const int global_experts = spec.ranks * spec.experts_per_rank;
    const int max_rows = spec.tokens * spec.topk;
    const int max_dispatch_rows = spec.ranks * max_rows;
    const size_t x_count = static_cast<size_t>(spec.ranks) * spec.tokens * spec.hidden;
    const size_t route_count = static_cast<size_t>(spec.ranks) * spec.tokens * spec.topk;
    const size_t active_count = static_cast<size_t>(spec.ranks) * spec.tokens;
    const size_t count_count = static_cast<size_t>(spec.ranks) * global_experts;
    const size_t expanded_count = static_cast<size_t>(spec.ranks) * max_rows;
    const size_t packed_count = static_cast<size_t>(spec.ranks) * max_rows * spec.hidden;
    const size_t dispatch_count = static_cast<size_t>(spec.ranks) * max_dispatch_rows * spec.hidden;

    DeviceBuffer out_dev(x_count * sizeof(float));
    DeviceBuffer x_dev(x_count * sizeof(float));
    DeviceBuffer expert_dev(route_count * sizeof(int32_t));
    DeviceBuffer probs_dev(route_count * sizeof(float));
    DeviceBuffer active_dev(active_count * sizeof(int32_t));
    DeviceBuffer count_dev(count_count * sizeof(int32_t));
    DeviceBuffer offset_dev(count_count * sizeof(int32_t));
    DeviceBuffer expanded_dev(expanded_count * sizeof(int32_t));
    DeviceBuffer packed_dev(packed_count * sizeof(float));
    DeviceBuffer dispatch_dev(dispatch_count * sizeof(float));
    DeviceBuffer return_dev(packed_count * sizeof(float));

    CopyHostToDevice(x_dev, input.x, "copy x host->device");
    CopyHostToDevice(expert_dev, input.expert_idx, "copy expert host->device");
    CopyHostToDevice(probs_dev, input.probs, "copy probs host->device");
    CopyHostToDevice(active_dev, input.active, "copy active host->device");

    aclrtStream stream = nullptr;
    CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
    try {
        launchDispatchCombine(static_cast<float *>(out_dev.ptr()), static_cast<float *>(x_dev.ptr()),
                              static_cast<int32_t *>(expert_dev.ptr()), static_cast<float *>(probs_dev.ptr()),
                              static_cast<int32_t *>(active_dev.ptr()), static_cast<int32_t *>(count_dev.ptr()),
                              static_cast<int32_t *>(offset_dev.ptr()), static_cast<int32_t *>(expanded_dev.ptr()),
                              static_cast<float *>(packed_dev.ptr()), static_cast<float *>(dispatch_dev.ptr()),
                              static_cast<float *>(return_dev.ptr()), spec.ranks, spec.experts_per_rank, spec.tokens,
                              spec.hidden, spec.topk, stream);
        CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    } catch (...) {
        aclrtDestroyStream(stream);
        throw;
    }
    CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");

    auto actual_out = CopyDeviceToHost<float>(out_dev, x_count, "copy out device->host");
    auto actual_count = CopyDeviceToHost<int32_t>(count_dev, count_count, "copy count device->host");
    auto actual_offset = CopyDeviceToHost<int32_t>(offset_dev, count_count, "copy offset device->host");
    auto actual_expanded = CopyDeviceToHost<int32_t>(expanded_dev, expanded_count, "copy expanded device->host");
    auto actual_packed = CopyDeviceToHost<float>(packed_dev, packed_count, "copy packed device->host");
    auto actual_dispatch = CopyDeviceToHost<float>(dispatch_dev, dispatch_count, "copy dispatch device->host");
    auto actual_return = CopyDeviceToHost<float>(return_dev, packed_count, "copy return device->host");

    std::cout << "[CASE] " << spec.name << " R=" << spec.ranks << " E=" << spec.experts_per_rank
              << " M=" << spec.tokens << " H=" << spec.hidden << " topK=" << spec.topk << " device=" << device
              << "\n";
    bool ok = true;
    ok = CompareIntVector(spec.name, "count", golden.count, actual_count) && ok;
    ok = CompareIntVector(spec.name, "srcExpertOffset", golden.src_expert_offset, actual_offset) && ok;
    ok = CompareIntVector(spec.name, "expandedRowIdx", golden.expanded_row_idx, actual_expanded) && ok;
    ok = CompareFloatVector(spec.name, "srcPackedX", golden.src_packed_x, actual_packed, 1e-4f) && ok;
    ok = CompareFloatVector(spec.name, "dispatchX", golden.dispatch_x, actual_dispatch, 1e-4f) && ok;
    ok = CompareFloatVector(spec.name, "returnY", golden.return_y, actual_return, 1e-4f) && ok;
    ok = CompareFloatVector(spec.name, "out", golden.out, actual_out, 1e-4f) && ok;
    std::cout << (ok ? "[RESULT] PASS " : "[RESULT] FAIL ") << spec.name << "\n";
    return ok;
}

std::vector<CaseSpec> BuildCases(const Options &opt)
{
    if (opt.case_name == "all") {
        return {CaseSpec{"minimal", 1, 1, 2, 4, 1}, CaseSpec{"cross", 2, 2, 4, 8, 2},
                CaseSpec{"edge", 3, 2, 5, 8, 2}};
    }
    if (opt.case_name == "minimal") {
        return {CaseSpec{"minimal", 1, 1, 2, 4, 1}};
    }
    if (opt.case_name == "cross") {
        return {CaseSpec{"cross", 2, 2, 4, 8, 2}};
    }
    if (opt.case_name == "edge") {
        return {CaseSpec{"edge", 3, 2, 5, 8, 2}};
    }
    if (opt.case_name == "smoke") {
        return {CaseSpec{"smoke", opt.ranks, opt.experts_per_rank, opt.tokens, opt.hidden, opt.topk}};
    }
    throw std::invalid_argument("unknown case: " + opt.case_name);
}

} // namespace

int main(int argc, char **argv)
{
    try {
        Options opt = ParseOptions(argc, argv);
        std::vector<CaseSpec> cases = BuildCases(opt);

        CheckAcl(aclInit(nullptr), "aclInit");
        bool device_set = false;
        try {
            CheckAcl(aclrtSetDevice(opt.device), "aclrtSetDevice");
            device_set = true;
            bool all_ok = true;
            for (const auto &spec : cases) {
                all_ok = RunOneCase(spec, opt.device) && all_ok;
            }
            if (device_set) {
                CheckAcl(aclrtResetDevice(opt.device), "aclrtResetDevice");
                device_set = false;
            }
            CheckAcl(aclFinalize(), "aclFinalize");
            if (all_ok) {
                std::cout << "[SUMMARY] All dispatch_combine cases passed.\n";
                return 0;
            }
            std::cout << "[SUMMARY] dispatch_combine has failed cases.\n";
            return 1;
        } catch (...) {
            if (device_set) {
                aclrtResetDevice(opt.device);
            }
            aclFinalize();
            throw;
        }
    } catch (const std::exception &ex) {
        std::cerr << "[ERROR] " << ex.what() << "\n";
        return 1;
    }
}
