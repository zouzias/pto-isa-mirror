#include "treduce_pingpong_perf_test.h"

// ============================================================================
// Command Line Argument Parser
// ============================================================================
struct TestArgs {
    int n_ranks = 4;
    int n_devices = 4;
    int first_rank_id = 0;
    int first_device_id = 0;
    int warmup_iters = 20;
    int measure_iters = 20;
    bool verbose = true;
    std::string test_size = "all";  // small, large, all
    bool help = false;
};

void PrintUsage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -r, --ranks N          Number of ranks (default: 4)\n"
              << "  -d, --devices N        Number of devices (default: 4)\n"
              << "  -w, --warmup N         Warmup iterations (default: 20)\n"
              << "  -m, --measure N        Measurement iterations (default: 50)\n"
              << "  -s, --size SIZE        Test size: small, large, all (default: all)\n"
              << "  -q, --quiet            Disable per-iteration output\n"
              << "  -h, --help             Show this help message\n"
              << "\nTest Sizes:\n"
              << "  small:   256, 1024 elements\n"
              << "  large:   16384, 32768 elements\n"
              << "  all:     Run all sizes\n"
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
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            args.warmup_iters = std::atoi(argv[++i]);
        } else if ((arg == "-m" || arg == "--measure") && i + 1 < argc) {
            args.measure_iters = std::atoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--size") && i + 1 < argc) {
            args.test_size = argv[++i];
        } else if (arg == "-q" || arg == "--quiet") {
            args.verbose = false;
        }
    }
    
    return args;
}

// ============================================================================
// Test Runners for Different Sizes
// ============================================================================
bool RunSmallTests(const TestArgs &args, const PerfTestConfig &config) {
    std::cout << "\n========================================" << std::endl;
    std::cout << " Running SMALL size tests" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    bool success = true;
    
    std::cout << "[TEST] float x 64x64 elements (16 KB)" << std::endl;
    success &= RunReducePingPongPerf<float, 64, 64, 64, 64>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);
    
    std::cout << "[TEST] float x 16x256 elements (16 KB)" << std::endl;
    success &= RunReducePingPongPerf<float, 16, 256, 16, 256>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);
    
    return success;
}

bool RunLargeTests(const TestArgs &args, const PerfTestConfig &config) {
    std::cout << "\n========================================" << std::endl;
    std::cout << " Running LARGE size tests" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    bool success = true;
    
    std::cout << "[TEST] int32_t x 128x128 elements (64 KB)" << std::endl;
    success &= RunReducePingPongPerf<int32_t, 128, 128, 128, 128>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);
    
    std::cout << "[TEST] int32_t x 64x256 elements (64 KB)" << std::endl;
    success &= RunReducePingPongPerf<int32_t, 64, 256, 64, 256>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);
    
    return success;
}

// ============================================================================
// Main Entry Point
// ============================================================================
int main(int argc, char **argv) {
    TestArgs args = ParseArgs(argc, argv);
    
    if (args.help) {
        PrintUsage(argv[0]);
        return 0;
    }
    
    // Print test configuration
    std::cout << "============================================================" << std::endl;
    std::cout << "  TREDUCE_PINGPONG Performance Test Suite" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Configuration:" << std::endl;
    std::cout << "    Ranks:            " << args.n_ranks << std::endl;
    std::cout << "    Devices:          " << args.n_devices << std::endl;
    std::cout << "    Warmup iters:     " << args.warmup_iters << std::endl;
    std::cout << "    Measurement iters:" << args.measure_iters << std::endl;
    std::cout << "    Test size:        " << args.test_size << std::endl;
    std::cout << "    Verbose:          " << (args.verbose ? "yes" : "no") << std::endl;
    std::cout << "============================================================\n" << std::endl;
    
    PerfTestConfig config;
    config.warmup_iters = args.warmup_iters;
    config.measure_iters = args.measure_iters;
    config.verbose = args.verbose;
    
    bool success = true;
    
    if (args.test_size == "small" || args.test_size == "all") {
        success &= RunSmallTests(args, config);
    }
    
    if (args.test_size == "large" || args.test_size == "all") {
        success &= RunLargeTests(args, config);
    }
    
    // Final summary
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Test Suite " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================================\n" << std::endl;
    
    return success ? 0 : 1;
}
