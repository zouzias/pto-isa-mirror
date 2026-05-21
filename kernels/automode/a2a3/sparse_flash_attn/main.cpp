#include <acl/acl.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template <int H, int D, int BLOCK>
void call(uint8_t *q_handle,
          uint8_t *kv_handle,
          uint8_t *output_handle,
          uint8_t *attn_sink_handle,
          uint8_t *topk_idxs_handle,
          uint8_t *workspace_1_handle,
          uint8_t *workspace_2_handle,
          uint8_t *workspace_3_handle,
          uint8_t *workspace_4_handle,
          int64_t b,
          int64_t m,
          int64_t n,
          int64_t topk,
          void *stream);

namespace {

// struct ShapeConfig {
//     int64_t b = 1;
//     int64_t m = 256;
//     int64_t n = 256;
//     int64_t h = 64;
//     int64_t d = 512;
//     int64_t topk = 128;
// };
struct ShapeConfig {
    int64_t b = 1;
    int64_t m = 6;
    int64_t n = 6;
    int64_t h = 16;
    int64_t d = 256;
    int64_t topk = 6;
};

// Updated to a constexpr variadic template using C++17 fold expressions
template <typename... Args>
constexpr size_t ElemCount(Args... args) {
    return (... * static_cast<size_t>(args));
}

bool ReadBinaryFile(const std::string &path, void *data, size_t bytes) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }
    ifs.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(bytes));
    if (!ifs) {
        std::cerr << "Failed to read " << bytes << " bytes from: " << path << std::endl;
        return false;
    }
    return true;
}

bool WriteBinaryFile(const std::string &path, const void *data, size_t bytes) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file for write: " << path << std::endl;
        return false;
    }
    ofs.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!ofs) {
        std::cerr << "Failed to write " << bytes << " bytes to: " << path << std::endl;
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> [device_id]" << std::endl;
        return 1;
    }

    const std::string data_dir = argv[1];
    const int device_id = (argc == 3) ? std::stoi(argv[2]) : 0;

    // Define cfg as a compile-time constant
    constexpr ShapeConfig cfg{};

    // All memory sizes are now computed at compile-time
    constexpr size_t q_bytes = ElemCount(cfg.b, cfg.m, cfg.h, cfg.d) * sizeof(uint16_t);
    constexpr size_t kv_bytes = ElemCount(cfg.b, cfg.n, cfg.d) * sizeof(uint16_t);
    constexpr size_t output_bytes = ElemCount(cfg.b, cfg.m, cfg.h, cfg.d) * sizeof(float);
    constexpr size_t attn_sink_bytes = ElemCount(cfg.h) * sizeof(float);
    constexpr size_t topk_idxs_bytes = ElemCount(cfg.b, cfg.m, cfg.topk) * sizeof(int32_t);

    constexpr size_t block_num = static_cast<size_t>(cfg.b * cfg.m);
    constexpr size_t workspace_1_bytes = block_num * 64ULL * static_cast<size_t>(cfg.d) * sizeof(uint16_t);
    constexpr size_t workspace_2_bytes = block_num * static_cast<size_t>(cfg.h) * 64ULL * sizeof(float);
    constexpr size_t workspace_3_bytes = block_num * static_cast<size_t>(cfg.h) * 64ULL * sizeof(uint16_t);
    constexpr size_t workspace_4_bytes = block_num * static_cast<size_t>(cfg.h) * static_cast<size_t>(cfg.d) * sizeof(float);

    std::vector<uint16_t> q_host(q_bytes / sizeof(uint16_t));
    std::vector<uint16_t> kv_host(kv_bytes / sizeof(uint16_t));
    std::vector<float> attn_sink_host(attn_sink_bytes / sizeof(float));
    std::vector<int32_t> topk_idxs_host(topk_idxs_bytes / sizeof(int32_t));
    std::vector<float> output_host(output_bytes / sizeof(float), 0.0f);

    if (!ReadBinaryFile(data_dir + "/q.bin", q_host.data(), q_bytes) ||
        !ReadBinaryFile(data_dir + "/kv.bin", kv_host.data(), kv_bytes) ||
        !ReadBinaryFile(data_dir + "/attn_sink.bin", attn_sink_host.data(), attn_sink_bytes) ||
        !ReadBinaryFile(data_dir + "/topk_idxs.bin", topk_idxs_host.data(), topk_idxs_bytes)) {
        return 1;
    }

    aclError ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclInit failed: " << ret << std::endl;
        return 1;
    }

    ret = aclrtSetDevice(device_id);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtSetDevice failed: " << ret << std::endl;
        aclFinalize();
        return 1;
    }

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtCreateStream failed: " << ret << std::endl;
        aclrtResetDevice(device_id);
        aclFinalize();
        return 1;
    }

    auto cleanup = [&](void *q_dev,
                       void *kv_dev,
                       void *out_dev,
                       void *sink_dev,
                       void *idx_dev,
                       void *w1_dev,
                       void *w2_dev,
                       void *w3_dev,
                       void *w4_dev) {
        if (w4_dev) aclrtFree(w4_dev);
        if (w3_dev) aclrtFree(w3_dev);
        if (w2_dev) aclrtFree(w2_dev);
        if (w1_dev) aclrtFree(w1_dev);
        if (idx_dev) aclrtFree(idx_dev);
        if (sink_dev) aclrtFree(sink_dev);
        if (out_dev) aclrtFree(out_dev);
        if (kv_dev) aclrtFree(kv_dev);
        if (q_dev) aclrtFree(q_dev);
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
    };

    void *q_dev = nullptr;
    void *kv_dev = nullptr;
    void *out_dev = nullptr;
    void *sink_dev = nullptr;
    void *idx_dev = nullptr;
    void *w1_dev = nullptr;
    void *w2_dev = nullptr;
    void *w3_dev = nullptr;
    void *w4_dev = nullptr;

    auto malloc_dev = [&](void **ptr, size_t bytes, const char *name) -> bool {
        aclError e = aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (e != ACL_ERROR_NONE) {
            std::cerr << "aclrtMalloc failed for " << name << ": " << e << std::endl;
            return false;
        }
        return true;
    };

    if (!malloc_dev(&q_dev, q_bytes, "q") ||
        !malloc_dev(&kv_dev, kv_bytes, "kv") ||
        !malloc_dev(&out_dev, output_bytes, "output") ||
        !malloc_dev(&sink_dev, attn_sink_bytes, "attn_sink") ||
        !malloc_dev(&idx_dev, topk_idxs_bytes, "topk_idxs") ||
        !malloc_dev(&w1_dev, workspace_1_bytes, "workspace_1") ||
        !malloc_dev(&w2_dev, workspace_2_bytes, "workspace_2") ||
        !malloc_dev(&w3_dev, workspace_3_bytes, "workspace_3") ||
        !malloc_dev(&w4_dev, workspace_4_bytes, "workspace_4")) {
        cleanup(q_dev, kv_dev, out_dev, sink_dev, idx_dev, w1_dev, w2_dev, w3_dev, w4_dev);
        return 1;
    }

    auto memcpy_h2d = [&](void *dst, const void *src, size_t bytes, const char *name) -> bool {
        aclError e = aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (e != ACL_ERROR_NONE) {
            std::cerr << "aclrtMemcpy H2D failed for " << name << ": " << e << std::endl;
            return false;
        }
        return true;
    };

    if (!memcpy_h2d(q_dev, q_host.data(), q_bytes, "q") ||
        !memcpy_h2d(kv_dev, kv_host.data(), kv_bytes, "kv") ||
        !memcpy_h2d(sink_dev, attn_sink_host.data(), attn_sink_bytes, "attn_sink") ||
        !memcpy_h2d(idx_dev, topk_idxs_host.data(), topk_idxs_bytes, "topk_idxs")) {
        cleanup(q_dev, kv_dev, out_dev, sink_dev, idx_dev, w1_dev, w2_dev, w3_dev, w4_dev);
        return 1;
    }

    // Now valid because cfg.h and cfg.d are evaluated at compile time
    call<cfg.h, cfg.d, 64>(reinterpret_cast<uint8_t *>(q_dev),
                           reinterpret_cast<uint8_t *>(kv_dev),
                           reinterpret_cast<uint8_t *>(out_dev),
                           reinterpret_cast<uint8_t *>(sink_dev),
                           reinterpret_cast<uint8_t *>(idx_dev),
                           reinterpret_cast<uint8_t *>(w1_dev),
                           reinterpret_cast<uint8_t *>(w2_dev),
                           reinterpret_cast<uint8_t *>(w3_dev),
                           reinterpret_cast<uint8_t *>(w4_dev),
                           cfg.b,
                           cfg.m,
                           cfg.n,
                           cfg.topk,
                           stream);

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtSynchronizeStream failed: " << ret << std::endl;
        cleanup(q_dev, kv_dev, out_dev, sink_dev, idx_dev, w1_dev, w2_dev, w3_dev, w4_dev);
        return 1;
    }

    ret = aclrtMemcpy(output_host.data(), output_bytes, out_dev, output_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "aclrtMemcpy D2H failed for output: " << ret << std::endl;
        cleanup(q_dev, kv_dev, out_dev, sink_dev, idx_dev, w1_dev, w2_dev, w3_dev, w4_dev);
        return 1;
    }

    if (!WriteBinaryFile(data_dir + "/output.bin", output_host.data(), output_bytes)) {
        cleanup(q_dev, kv_dev, out_dev, sink_dev, idx_dev, w1_dev, w2_dev, w3_dev, w4_dev);
        return 1;
    }

    std::cout << "Saved output to: " << data_dir + "/output.bin" << std::endl;
    std::cout << "Shape: b=" << cfg.b
              << " m=" << cfg.m
              << " n=" << cfg.n
              << " h=" << cfg.h
              << " d=" << cfg.d
              << " topk=" << cfg.topk << std::endl;

    cleanup(q_dev, kv_dev, out_dev, sink_dev, idx_dev, w1_dev, w2_dev, w3_dev, w4_dev);
    return 0;
}