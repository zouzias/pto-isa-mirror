#include "tput_mc_perf_test.h"

// ============================================================================
// Command-line argument parsing
// ============================================================================
struct TestArgs {
    int n_ranks = 8;
    int n_devices = 8;
    int first_rank_id = 0;
    int first_device_id = 0;
    int warmup_iters = 5;
    int repeat_iters = 20;
    int block_num = TPUT_BLOCK_NUM;
    bool verbose = true;
    bool help = false;
};

void PrintUsage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -r, --ranks N          Number of ranks (default: 8)\n"
              << "  -d, --devices N        Number of devices (default: 8)\n"
              << "  -b, --blocks N         Number of blocks/cores (default: " << TPUT_BLOCK_NUM << ")\n"
              << "  -w, --warmup N         Warmup kernel launches (default: 5)\n"
              << "  -n, --repeat N         TPUT repeat iters inside kernel (default: 20)\n"
              << "  -q, --quiet            Disable verbose output\n"
              << "  -h, --help             Show this help message\n"
              << "\nMeasurement Strategy:\n"
              << "  1. Warmup: launch kernel W times (each runs TPUT N times in a loop)\n"
              << "  2. Measure: launch kernel once (TPUT runs N times), wall clock the whole thing\n"
              << "  3. BW = (tensor_size * N) / wall_time  -> steady-state bandwidth\n"
              << "\nTraffic Pattern:\n"
              << "  Each rank distributes blocks evenly across all (R-1) peer ranks,\n"
              << "  utilizing all HCCS links simultaneously for full-mesh bandwidth.\n"
              << "\nTest Configurations:\n"
              << "  1. 4096x4096 (64MB),  Tile 64x64 (16KB)  - baseline\n"
              << "  2. 4096x4096 (64MB),  Tile 64x128 (32KB) - wider tile\n"
              << "  3. 4096x4096 (64MB),  Tile 32x64 (8KB)   - smaller tile\n"
              << "  4. 8192x8192 (256MB), Tile 64x128 (32KB) - large data\n"
              << std::endl;
}

TestArgs ParseArgs(int argc, char **argv) {
    TestArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if ((arg == "-r" || arg == "--ranks") && i + 1 < argc) {
            args.n_ranks = std::atoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--devices") && i + 1 < argc) {
            args.n_devices = std::atoi(argv[++i]);
        } else if ((arg == "-b" || arg == "--blocks") && i + 1 < argc) {
            args.block_num = std::atoi(argv[++i]);
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            args.warmup_iters = std::atoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--repeat") && i + 1 < argc) {
            args.repeat_iters = std::atoi(argv[++i]);
        } else if (arg == "-q" || arg == "--quiet") {
            args.verbose = false;
        }
    }
    return args;
}

// ============================================================================
// Test runners: each runs AutoChunk + PingPong back-to-back
// ============================================================================

bool RunTest_64MB_64x64(const TestArgs &args, PerfTestConfig config) {
    constexpr int kTRows = 64, kTCols = 64;
    config.total_rows = 4096;
    config.total_cols = 4096;

    size_t bytes = static_cast<size_t>(config.total_rows) * config.total_cols * sizeof(float);
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  Test: " << config.total_rows << "x" << config.total_cols
              << " (" << (bytes / (1024*1024)) << "MB), Tile "
              << kTRows << "x" << kTCols << " (" << (kTRows*kTCols*4/1024) << "KB)"
              << ", " << config.repeat_iters << " iters" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::cout << "\n--- AutoChunk ---" << std::endl;
    bool ok1 = RunPutAutoChunkPerf<float, kTRows, kTCols>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);

    std::cout << "--- PingPong ---" << std::endl;
    bool ok2 = RunPutPingPongPerf<float, kTRows, kTCols>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);

    return ok1 && ok2;
}

bool RunTest_64MB_64x128(const TestArgs &args, PerfTestConfig config) {
    constexpr int kTRows = 64, kTCols = 128;
    config.total_rows = 4096;
    config.total_cols = 4096;

    size_t bytes = static_cast<size_t>(config.total_rows) * config.total_cols * sizeof(float);
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  Test: " << config.total_rows << "x" << config.total_cols
              << " (" << (bytes / (1024*1024)) << "MB), Tile "
              << kTRows << "x" << kTCols << " (" << (kTRows*kTCols*4/1024) << "KB)"
              << ", " << config.repeat_iters << " iters" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::cout << "\n--- AutoChunk ---" << std::endl;
    bool ok1 = RunPutAutoChunkPerf<float, kTRows, kTCols>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);

    std::cout << "--- PingPong ---" << std::endl;
    bool ok2 = RunPutPingPongPerf<float, kTRows, kTCols>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);

    return ok1 && ok2;
}

bool RunTest_64MB_32x64(const TestArgs &args, PerfTestConfig config) {
    constexpr int kTRows = 32, kTCols = 64;
    config.total_rows = 4096;
    config.total_cols = 4096;

    size_t bytes = static_cast<size_t>(config.total_rows) * config.total_cols * sizeof(float);
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  Test: " << config.total_rows << "x" << config.total_cols
              << " (" << (bytes / (1024*1024)) << "MB), Tile "
              << kTRows << "x" << kTCols << " (" << (kTRows*kTCols*4/1024) << "KB)"
              << ", " << config.repeat_iters << " iters" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::cout << "\n--- AutoChunk ---" << std::endl;
    bool ok1 = RunPutAutoChunkPerf<float, kTRows, kTCols>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);

    std::cout << "--- PingPong ---" << std::endl;
    bool ok2 = RunPutPingPongPerf<float, kTRows, kTCols>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);

    return ok1 && ok2;
}

bool RunTest_256MB_64x128(const TestArgs &args, PerfTestConfig config) {
    constexpr int kTRows = 64, kTCols = 128;
    config.total_rows = 8192;
    config.total_cols = 8192;

    size_t bytes = static_cast<size_t>(config.total_rows) * config.total_cols * sizeof(float);
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  Test: " << config.total_rows << "x" << config.total_cols
              << " (" << (bytes / (1024*1024)) << "MB), Tile "
              << kTRows << "x" << kTCols << " (" << (kTRows*kTCols*4/1024) << "KB)"
              << ", " << config.repeat_iters << " iters" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::cout << "\n--- AutoChunk ---" << std::endl;
    bool ok1 = RunPutAutoChunkPerf<float, kTRows, kTCols>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);

    std::cout << "--- PingPong ---" << std::endl;
    bool ok2 = RunPutPingPongPerf<float, kTRows, kTCols>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);

    return ok1 && ok2;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv) {
    TestArgs args = ParseArgs(argc, argv);
    if (args.help) {
        PrintUsage(argv[0]);
        return 0;
    }

    PerfTestConfig config;
    config.warmup_iters = args.warmup_iters;
    config.repeat_iters = args.repeat_iters;
    config.block_num = args.block_num;
    config.verbose = args.verbose;

    const int peer_count = args.n_ranks - 1;

    std::cout << "============================================================" << std::endl;
    std::cout << "  TPUT Full-Mesh Steady-State Bandwidth Test" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Ranks:            " << args.n_ranks << std::endl;
    std::cout << "  Devices:          " << args.n_devices << std::endl;
    std::cout << "  Blocks (cores):   " << args.block_num << std::endl;
    std::cout << "  Warmup launches:  " << args.warmup_iters << std::endl;
    std::cout << "  Repeat iters:     " << args.repeat_iters
              << " (TPUT calls per kernel)" << std::endl;
    std::cout << "  Peer links/rank:  " << peer_count << std::endl;
    std::cout << "  Traffic pattern:  Full-mesh (blocks evenly -> all peers)" << std::endl;
    std::cout << "============================================================\n" << std::endl;

    bool all_ok = true;

    all_ok &= RunTest_64MB_64x64(args, config);
    all_ok &= RunTest_64MB_64x128(args, config);
    all_ok &= RunTest_64MB_32x64(args, config);
    all_ok &= RunTest_256MB_64x128(args, config);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  All Tests " << (all_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================================\n" << std::endl;

    return all_ok ? 0 : 1;
}
