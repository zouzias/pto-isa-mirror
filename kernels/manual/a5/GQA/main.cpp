/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/*
 * Standalone driver for TGQA (Grouped Query Attention)
 * GQA runs the same per-head flash attention kernel as MHA,
 * but iterates over multiple Q heads sharing each K/V head.
 */

#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <set>

#include "test_common.h"
#include "runtime/rt.h"
#include "gqa_performance_kernel.h"
#include "generated_cases.h"

using namespace std;
using namespace PtoTestCommon;

#define GOP_PRECISION 6
#define TIME_PRECISION 3

static std::vector<std::string> Split(const std::string &s, char delim)
{
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(s);
    while (std::getline(ss, item, delim)) {
        if (!item.empty())
            out.push_back(item);
    }
    return out;
}

static std::string Trim(const std::string &s)
{
    const auto start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos)
        return "";
    const auto end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> SplitAny(const std::string &s, const std::string &delims)
{
    std::vector<std::string> out;
    std::string token;
    for (char ch : s) {
        if (delims.find(ch) != std::string::npos) {
            if (!token.empty()) {
                out.push_back(token);
                token.clear();
            }
        } else {
            token.push_back(ch);
        }
    }
    if (!token.empty())
        out.push_back(token);
    return out;
}

static thread_local std::string g_case_name;
static bool g_enable_intermediate = false;
static thread_local std::string g_fifo_summary;
static int g_chip_id = 0;
static const std::string kReportCsv = "./report.csv";
static double g_sys_cnt_multiple = 20.0;

static void AppendReportRow(const std::string &case_name, int num_q_heads, int num_kv_heads, int head_size, int s0,
                            int s1, int cube_s0, int cube_s1, int tile_s1, uint64_t start_time, uint64_t end_time,
                            double duration_us, double avg_block_us, double gops, const std::string &tflops_str, bool ok)
{
    const bool exists = std::ifstream(kReportCsv).good();
    std::ofstream ofs(kReportCsv, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[WARN] Unable to open report file: " << kReportCsv << std::endl;
        return;
    }
    if (!exists) {
        ofs << "case,NUM_Q,NUM_KV,HEAD,S0,S1,CUBE_S0,CUBE_S1,TILE_S1,start_time,end_time,duration_us,avg_block_us,"
               "GOPS,TFLOPS,result\n";
    }
    ofs << case_name << ',' << num_q_heads << ',' << num_kv_heads << ',' << head_size << ',' << s0 << ',' << s1 << ','
        << cube_s0 << ',' << cube_s1 << ',' << tile_s1 << ',' << start_time << ',' << end_time << ',' << std::fixed
        << std::setprecision(TIME_PRECISION) << duration_us << ',' << std::setprecision(TIME_PRECISION) << avg_block_us
        << ',' << std::setprecision(GOP_PRECISION) << gops << ',' << tflops_str << ',' << (ok ? "OK" : "NOK") << '\n';
}

std::string GetGoldenDir()
{
    return "./" + g_case_name;
}

template <typename T, int NUM_Q_HEADS, int NUM_KV_HEADS, int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1,
          int TILE_S1, int QK_PRELOAD, bool INTERMEDIATE_CHECK, bool CAUSAL_MASK>
void run_tgqa()
{
    constexpr int NUM_QUERIES_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS;
    constexpr int tile_factor = TILE_S1 / CUBE_S1;
    constexpr size_t qk_fifo_stride = static_cast<size_t>(kGqaCvFifoSize) * static_cast<size_t>(CUBE_S0) *
                                      static_cast<size_t>(tile_factor) * static_cast<size_t>(CUBE_S1);
    constexpr size_t p_fifo_stride = qk_fifo_stride;
    constexpr size_t p_max_fifo_stride = static_cast<size_t>(kGqaCvFifoSize) * static_cast<size_t>(CUBE_S0);
    constexpr size_t pv_fifo_stride =
        static_cast<size_t>(kGqaCvFifoSize) * static_cast<size_t>(CUBE_S0) * static_cast<size_t>(HEAD_SIZE);
    const size_t block_rows = S0 / CUBE_S0;
    constexpr size_t cv_comm_slots = static_cast<size_t>(S0) / static_cast<size_t>(CUBE_S0);
    constexpr size_t cv_comm_bytes = cv_comm_slots * kGqaCvCommSlotBytes;
    constexpr size_t profile_bytes_per_block = kGqaProfileBytesPerBlock;
    const size_t profile_bytes = profile_bytes_per_block * block_rows;

    g_fifo_summary.clear();

    size_t qSize = static_cast<size_t>(NUM_Q_HEADS) * static_cast<size_t>(S0) *
                   static_cast<size_t>(HEAD_SIZE) * sizeof(aclFloat16);
    size_t kSize = static_cast<size_t>(NUM_KV_HEADS) * static_cast<size_t>(HEAD_SIZE) *
                   static_cast<size_t>(S1) * sizeof(aclFloat16);
    size_t vSize = static_cast<size_t>(NUM_KV_HEADS) * static_cast<size_t>(S1) *
                   static_cast<size_t>(HEAD_SIZE) * sizeof(aclFloat16);
    size_t oSize = static_cast<size_t>(NUM_Q_HEADS) * static_cast<size_t>(S0) *
                   static_cast<size_t>(HEAD_SIZE) * sizeof(T);

    const size_t qk_fifo_bytes = qk_fifo_stride * block_rows * sizeof(T);
    const size_t p_fifo_bytes_half = p_fifo_stride * block_rows * sizeof(aclFloat16);
    const size_t p_fifo_bytes_float = p_max_fifo_stride * block_rows * sizeof(float);
    const size_t pv_fifo_bytes = pv_fifo_stride * block_rows * sizeof(T);
    const size_t singleHeadQSize = static_cast<size_t>(S0) * static_cast<size_t>(HEAD_SIZE) * sizeof(aclFloat16);
    const size_t singleHeadKSize = static_cast<size_t>(HEAD_SIZE) * static_cast<size_t>(S1) * sizeof(aclFloat16);
    const size_t singleHeadVSize = static_cast<size_t>(S1) * static_cast<size_t>(HEAD_SIZE) * sizeof(aclFloat16);
    const size_t singleHeadOSize = static_cast<size_t>(S0) * static_cast<size_t>(HEAD_SIZE) * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(g_chip_id);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclFloat16 *qHost = nullptr;
    aclFloat16 *kHost = nullptr;
    aclFloat16 *vHost = nullptr;
    T *oHost = nullptr;
    aclFloat16 *qDevice = nullptr;
    aclFloat16 *kDevice = nullptr;
    aclFloat16 *vDevice = nullptr;
    T *oDevice = nullptr;

    aclrtMallocHost((void **)(&qHost), qSize);
    aclrtMallocHost((void **)(&kHost), kSize);
    aclrtMallocHost((void **)(&vHost), vSize);
    aclrtMallocHost((void **)(&oHost), oSize);

    aclrtMalloc((void **)&qDevice, qSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&kDevice, kSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&vDevice, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&oDevice, oSize, ACL_MEM_MALLOC_HUGE_FIRST);

    std::string golden_dir = GetGoldenDir();

    ReadFile(golden_dir + "/q_all.bin", qSize, qHost, qSize);
    ReadFile(golden_dir + "/kt_all.bin", kSize, kHost, kSize);
    ReadFile(golden_dir + "/v_all.bin", vSize, vHost, vSize);

    aclrtMemcpy(qDevice, qSize, qHost, qSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDevice, kSize, kHost, kSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(vDevice, vSize, vHost, vSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    if constexpr (INTERMEDIATE_CHECK) {
        std::cout << "[INFO] Intermediate checking is enabled" << std::endl;
    } else {
        std::cout << "[INFO] Intermediate checking is disabled" << std::endl;
    }

    bool all_heads_ok = true;
    int num_tiles = S1 / TILE_S1;

    for (int kv_group = 0; kv_group < NUM_KV_HEADS; ++kv_group) {
        aclFloat16 *kHeadDevice = kDevice + static_cast<size_t>(kv_group) * static_cast<size_t>(HEAD_SIZE) *
                                              static_cast<size_t>(S1);
        aclFloat16 *vHeadDevice = vDevice + static_cast<size_t>(kv_group) * static_cast<size_t>(S1) *
                                              static_cast<size_t>(HEAD_SIZE);

        for (int q_idx = 0; q_idx < NUM_QUERIES_PER_KV; ++q_idx) {
            int q_head = kv_group * NUM_QUERIES_PER_KV + q_idx;

            aclFloat16 *qHeadDevice = qDevice + static_cast<size_t>(q_head) * static_cast<size_t>(S0) *
                                                  static_cast<size_t>(HEAD_SIZE);
            T *oHeadDevice = oDevice + static_cast<size_t>(q_head) * static_cast<size_t>(S0) *
                                       static_cast<size_t>(HEAD_SIZE);

            std::string head_dir = golden_dir + "/h_q" + std::to_string(q_head) + "_g_kv" + std::to_string(kv_group);
            std::string head_name = g_case_name + "_h_q" + std::to_string(q_head) + "_g_kv" + std::to_string(kv_group);

            size_t qk_fifo_bytes_head = qk_fifo_bytes;
            size_t p_fifo_bytes_half_head = p_fifo_bytes_half;
            size_t p_fifo_bytes_float_head = p_fifo_bytes_float;
            size_t pv_fifo_bytes_head = pv_fifo_bytes;

            T *outHost = nullptr;
            aclFloat16 *xexpHost = nullptr;
            float *tmpFloatExpHost = nullptr;
            T *out2Host = nullptr;
            T *outDevice_head = nullptr;
            aclFloat16 *xexpDevice = nullptr;
            void *expMaxIfifoDevice = nullptr;
            uint8_t *profileDevice = nullptr;
            uint8_t *cvCommDevice = nullptr;
            T *out2Device = nullptr;

            aclrtMallocHost((void **)(&outHost), qk_fifo_bytes_head);
            aclrtMalloc((void **)&outDevice_head, qk_fifo_bytes_head, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc((void **)&xexpDevice, p_fifo_bytes_half_head, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc((void **)&expMaxIfifoDevice, p_fifo_bytes_float_head, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc((void **)&profileDevice, profile_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc((void **)&cvCommDevice, cv_comm_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMalloc((void **)&out2Device, pv_fifo_bytes_head, ACL_MEM_MALLOC_HUGE_FIRST);

            size_t gsumTotalElems = static_cast<size_t>(S0) * static_cast<size_t>(num_tiles);
            size_t gsumSize = gsumTotalElems * sizeof(float);
            float *gSumDevice = nullptr;
            aclrtMalloc((void **)&gSumDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
            float *expMaxDevice = nullptr;
            aclrtMalloc((void **)&expMaxDevice, gsumSize, ACL_MEM_MALLOC_HUGE_FIRST);
            size_t oPartsTotalSize = singleHeadOSize * num_tiles;
            T *oPartsDevice = nullptr;
            aclrtMalloc((void **)&oPartsDevice, oPartsTotalSize, ACL_MEM_MALLOC_HUGE_FIRST);

            LaunchTGQA<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kGqaCvFifoSize, INTERMEDIATE_CHECK,
                       CAUSAL_MASK, kGqaCvFifoConsSyncPeriod>(
                (uint16_t *)ffts, (aclFloat16 *)qHeadDevice, (aclFloat16 *)kHeadDevice, (aclFloat16 *)vHeadDevice,
                (aclFloat16 *)xexpDevice, (float *)expMaxIfifoDevice, (float *)gSumDevice, (float *)expMaxDevice,
                (float *)oHeadDevice, (float *)oPartsDevice, (float *)outDevice_head, (float *)out2Device,
                profileDevice, stream, cvCommDevice);

            aclrtSynchronizeStream(stream);

            std::cout << "[CHECK] Head q=" << q_head << " kv_group=" << kv_group << " kernel completed" << std::endl;

            aclrtFree(outDevice_head);
            aclrtFree(xexpDevice);
            aclrtFree(expMaxIfifoDevice);
            aclrtFree(out2Device);
            aclrtFree(gSumDevice);
            aclrtFree(expMaxDevice);
            aclrtFree(profileDevice);
            aclrtFree(cvCommDevice);
            aclrtFree(oPartsDevice);
            aclrtFreeHost(outHost);
        }
    }

    aclrtMemcpy(oHost, oSize, oDevice, oSize, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> golden_o_all(static_cast<size_t>(NUM_Q_HEADS) * static_cast<size_t>(S0) *
                                    static_cast<size_t>(HEAD_SIZE));
    std::vector<float> dev_o_all(static_cast<size_t>(NUM_Q_HEADS) * static_cast<size_t>(S0) *
                                 static_cast<size_t>(HEAD_SIZE));
    size_t golden_o_file_size = 0;
    ReadFile(golden_dir + "/o_all.bin", golden_o_file_size, golden_o_all.data(), oSize);
    memcpy(dev_o_all.data(), oHost, oSize);

    for (int q_head = 0; q_head < NUM_Q_HEADS; ++q_head) {
        std::vector<float> golden_o_head(S0 * HEAD_SIZE);
        std::vector<float> dev_o_head(S0 * HEAD_SIZE);
        for (int i = 0; i < S0 * HEAD_SIZE; ++i) {
            golden_o_head[i] = golden_o_all[static_cast<size_t>(q_head) * S0 * HEAD_SIZE + i];
            dev_o_head[i] = dev_o_all[static_cast<size_t>(q_head) * S0 * HEAD_SIZE + i];
        }
        bool head_ok = ResultCmp<float>(golden_o_head, dev_o_head, 0.001f);
        std::cout << "[CHECK] O running output compare for head q=" << q_head << ": "
                  << (head_ok ? "OK" : "FAIL") << std::endl;
        all_heads_ok = all_heads_ok && head_ok;
    }

    uint64_t start_min = 0;
    uint64_t end_max = 0;
    double total_gops = static_cast<double>(NUM_Q_HEADS) * static_cast<double>(S0) * static_cast<double>(S1) *
                        static_cast<double>(HEAD_SIZE) * 4.0 / 1e6;
    std::string tflops_str = "NA";

    AppendReportRow(g_case_name, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE, S0, S1, CUBE_S0, CUBE_S1, TILE_S1, start_min,
                    end_max, 0.0, 0.0, total_gops, tflops_str, all_heads_ok);

    std::cout << (all_heads_ok ? "test success" : "test failed") << std::endl;
    if (!g_fifo_summary.empty()) {
        std::cout << g_fifo_summary << std::endl;
    }
    std::cout << "[SUMMARY] o_out status: " << (all_heads_ok ? "OK" : "FAIL") << std::endl;

    aclrtFreeHost(qHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(vHost);
    aclrtFreeHost(oHost);
    aclrtFree(qDevice);
    aclrtFree(kDevice);
    aclrtFree(vDevice);
    aclrtFree(oDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(g_chip_id);
    aclFinalize();
}

template <typename T, int NUM_Q_HEADS, int NUM_KV_HEADS, int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1,
          int TILE_S1, int QK_PRELOAD, bool CAUSAL_MASK>
void run_gqa_case(const std::string &case_name)
{
    g_case_name = case_name;
    if (g_enable_intermediate) {
        run_tgqa<T, NUM_Q_HEADS, NUM_KV_HEADS, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, true,
                 CAUSAL_MASK>();
    } else {
        run_tgqa<T, NUM_Q_HEADS, NUM_KV_HEADS, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, false,
                 CAUSAL_MASK>();
    }
}

int main(int argc, char **argv)
{
    struct CaseEntry {
        std::string name;
        std::function<void()> run;
    };

    std::vector<CaseEntry> cases = {
#define TGQA_CASE_ENTRY(NUM_Q, NUM_KV, HEAD, S0_VAL, S1_VAL, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CAUSAL_MASK) \
    {"case_float_GQA_Q" #NUM_Q "_K" #NUM_KV "_H" #HEAD "_S0_" #S0_VAL "_S1_" #S1_VAL, []() {                  \
         run_gqa_case<float, NUM_Q, NUM_KV, S0_VAL, HEAD, S1_VAL, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD,       \
                      CAUSAL_MASK>("case_float_GQA_Q" #NUM_Q "_K" #NUM_KV "_H" #HEAD "_S0_" #S0_VAL "_S1_"      \
                                   #S1_VAL);                                                                     \
     }},
        TGQA_FOR_EACH_CASE(TGQA_CASE_ENTRY)
#undef TGQA_CASE_ENTRY
    };

    std::vector<std::string> filters;
    std::string filter_arg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const std::string prefix_case = "--case=";
        const std::string prefix_cases = "--cases=";
        const std::string prefix_chip = "--chip=";
        const std::string prefix_npu = "--npu=";
        const std::string prefix_sys_cnt_mtp = "--sys_cnt_multiple=";

        if (arg.rfind(prefix_case, 0) == 0) {
            filter_arg = arg.substr(prefix_case.size());
            continue;
        }
        if (arg.rfind(prefix_cases, 0) == 0) {
            filter_arg = arg.substr(prefix_cases.size());
            continue;
        }
        if (arg.rfind(prefix_chip, 0) == 0) {
            g_chip_id = std::stoi(arg.substr(prefix_chip.size()));
            continue;
        }
        if (arg.rfind(prefix_npu, 0) == 0) {
            g_chip_id = std::stoi(arg.substr(prefix_npu.size()));
            continue;
        }
        if (arg.rfind(prefix_sys_cnt_mtp, 0) == 0) {
            g_sys_cnt_multiple = std::atof(arg.substr(prefix_sys_cnt_mtp.size()).c_str());
            continue;
        }
        if ((arg == "--case" || arg == "--cases") && (i + 1) < argc) {
            filter_arg = argv[++i];
            continue;
        }
        if ((arg == "--chip" || arg == "-c") && (i + 1) < argc) {
            g_chip_id = std::stoi(argv[++i]);
            continue;
        }
        if ((arg == "--npu" || arg == "-n") && (i + 1) < argc) {
            g_chip_id = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--intermediate" || arg == "-i" || arg == "-I") {
            g_enable_intermediate = true;
            continue;
        }
        if (arg.rfind("--intermediate=", 0) == 0) {
            std::string val = arg.substr(std::strlen("--intermediate="));
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            g_enable_intermediate = (val == "1" || val == "true" || val == "yes");
            continue;
        }
        if ((arg == "--sys_cnt_multiple") && (i + 1) < argc) {
            g_sys_cnt_multiple = std::atof(argv[++i]);
            continue;
        }
    }
    if (!filter_arg.empty()) {
        std::vector<std::string> raw_filters = Split(filter_arg, ';');
        auto normalize_filter = [](const std::string &f) {
            const std::string trimmed = Trim(f);
            if (trimmed.rfind("case_float", 0) == 0)
                return trimmed;
            std::vector<std::string> parts = Split(trimmed, ',');
            if (parts.size() >= 5) {
                try {
                    int num_q = std::stoi(Trim(parts[0]));
                    int num_kv = std::stoi(Trim(parts[1]));
                    int head = std::stoi(Trim(parts[2]));
                    int s0 = std::stoi(Trim(parts[3]));
                    int s1 = std::stoi(Trim(parts[4]));
                    return std::string("case_float_GQA_Q") + std::to_string(num_q) + "_K" + std::to_string(num_kv) +
                           "_H" + std::to_string(head) + "_S0_" + std::to_string(s0) + "_S1_" + std::to_string(s1);
                } catch (...) {
                    return trimmed;
                }
            }
            return trimmed;
        };
        for (auto &f : raw_filters) {
            const std::string norm = normalize_filter(f);
            if (!norm.empty())
                filters.push_back(norm);
        }
    }

    std::cout << "[DEBUG] Available cases (" << cases.size() << "): ";
    for (size_t i = 0; i < cases.size(); ++i) {
        std::cout << cases[i].name;
        if (i + 1 != cases.size())
            std::cout << ",";
    }
    std::cout << std::endl;

    if (!filters.empty()) {
        std::cout << "[DEBUG] Requested filters: ";
        for (size_t i = 0; i < filters.size(); ++i) {
            std::cout << filters[i];
            if (i + 1 != filters.size())
                std::cout << ",";
        }
        std::cout << std::endl;
    } else {
        std::cout << "[DEBUG] No filters provided; running all cases" << std::endl;
    }

    auto should_run = [&](const std::string &name) {
        if (filters.empty())
            return true;
        return std::find(filters.begin(), filters.end(), name) != filters.end();
    };

    std::vector<std::string> to_run;
    for (const auto &c : cases) {
        if (should_run(c.name)) {
            to_run.push_back(c.name);
        }
    }

    if (to_run.empty()) {
        if (!filters.empty()) {
            std::cerr << "[WARN] No cases matched filters; check --case/--cases values." << std::endl;
        } else {
            std::cerr << "[WARN] No cases available to run." << std::endl;
        }
        return 1;
    }

    std::cout << "[DEBUG] Will run cases (" << to_run.size() << "): ";
    for (size_t i = 0; i < to_run.size(); ++i) {
        std::cout << to_run[i];
        if (i + 1 != to_run.size())
            std::cout << ",";
    }
    std::cout << std::endl;

    for (const auto &c : cases) {
        if (should_run(c.name)) {
            c.run();
        }
    }

    return 0;
}
