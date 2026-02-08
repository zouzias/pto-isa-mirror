#include "treduce_mc_perf_test.h"

// ============================================================================
// Command Line Argument Parser
// ============================================================================
struct TestArgs {
    int n_ranks = 8;       // Default to 8 ranks (8 NPU cards)
    int n_devices = 8;     // Default to 8 devices
    int first_rank_id = 0;
    int first_device_id = 0;
    int warmup_iters = 3;
    int block_num = AR_BLOCK_NUM;
    bool verbose = true;
    bool help = false;
};

void PrintUsage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -r, --ranks N          Number of ranks (default: 2)\n"
              << "  -d, --devices N        Number of devices (default: 2)\n"
              << "  -b, --blocks N         Number of blocks/cores (default: " << AR_BLOCK_NUM << ")\n"
              << "  -w, --warmup N         Warmup iterations (default: 3)\n"
              << "  -q, --quiet            Disable verbose output\n"
              << "  -h, --help             Show this help message\n"
              << "\nFixed Configuration:\n"
              << "  Tensor size:  32 MB (131072 x 64 float elements)\n"
              << "  Tile size:    64 x 64 = 16 KB per tile\n"
              << "\nExample:\n"
              << "  " << prog << " -r 2 -d 2 -b 8\n"
              << "  This runs 2 ranks on 2 devices, 8 cores per rank,\n"
              << "  with a 32MB tensor, tiled at 64x64 (16KB tiles)\n"
              << "  Each core processes 16384 rows = 256 tiles\n"
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
        } else if (arg == "-q" || arg == "--quiet") {
            args.verbose = false;
        }
    }
    
    return args;
}

// ============================================================================
// Test Runners for Different Tile Sizes
// ============================================================================

bool RunWithTile64x64(const TestArgs &args, PerfTestConfig config) {
    // Tile: 64x64 = 16KB (float)
    // 8MB tensor: rows = 8*1024*1024 / (64*4) = 32768 rows
    // 32768 rows / 32 cores = 1024 rows per core = 16 tiles per core
    constexpr int kTRows = 64;
    constexpr int kTCols = 64;
    constexpr size_t TARGET_SIZE_MB = 64;  // 128MB max due to shmem limits
    
    config.total_cols = kTCols;
    config.total_rows = (TARGET_SIZE_MB * 1024 * 1024) / (kTCols * sizeof(float));
    
    size_t tensor_bytes = static_cast<size_t>(config.total_rows) * kTCols * sizeof(float);
    size_t tile_bytes = kTRows * kTCols * sizeof(float);
    int rows_per_block = (config.total_rows + config.block_num - 1) / config.block_num;
    int tiles_per_block = (rows_per_block + kTRows - 1) / kTRows;
    
    std::cout << "\n[TEST] Tile " << kTRows << "x" << kTCols << " (" << (tile_bytes/1024) << "KB)"
              << ", Tensor " << config.total_rows << "x" << kTCols 
              << " (" << (tensor_bytes / (1024*1024)) << " MB)" << std::endl;
    std::cout << "       Rows/block: " << rows_per_block << ", Tiles/block: " << tiles_per_block << std::endl;
    
    return RunReduceTilingPerf<float, 64, 64>(
        args.n_ranks, args.n_devices, args.first_rank_id, args.first_device_id, config);
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
    
    // Build config (tensor size will be set in RunWithTile64x64)
    PerfTestConfig config;
    config.warmup_iters = args.warmup_iters;
    config.block_num = args.block_num;
    config.verbose = args.verbose;
    
    // Print test configuration
    std::cout << "============================================================" << std::endl;
    std::cout << "  TREDUCE Multi-core Tiling Bandwidth Test" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Configuration:" << std::endl;
    std::cout << "    Ranks:            " << args.n_ranks << std::endl;
    std::cout << "    Devices:          " << args.n_devices << std::endl;
    std::cout << "    Blocks (cores):   " << args.block_num << std::endl;
    std::cout << "    Warmup iters:     " << args.warmup_iters << std::endl;
    std::cout << "    Tensor size:      8 MB (fixed)" << std::endl;
    std::cout << "    Tile size:        64 x 64 = 16 KB (fixed)" << std::endl;
    std::cout << "    Verbose:          " << (args.verbose ? "yes" : "no") << std::endl;
    std::cout << "============================================================\n" << std::endl;
    
    bool success = RunWithTile64x64(args, config);
    
    // Final summary
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  Test " << (success ? "PASSED" : "FAILED") << std::endl;
    std::cout << "============================================================\n" << std::endl;
    
    return success ? 0 : 1;
}
