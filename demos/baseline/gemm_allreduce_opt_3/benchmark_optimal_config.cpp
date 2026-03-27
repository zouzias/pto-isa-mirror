/**
 * Benchmark: Optimal Core Configuration
 * 
 * This program:
 * 1. Tests single comm core TPUT efficiency
 * 2. Tests single compute core GEMM efficiency
 * 3. Calculates optimal comm/compute core ratio based on total cores = 24
 * 4. Runs full benchmark with optimal configuration
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <sys/wait.h>

// Forward declarations
extern bool RunGemmAllReduce(int n_ranks, int first_device_id);

// ============================================================================
// Helper: Run command and capture output
// ============================================================================
static std::string runCommand(const std::string& cmd)
{
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return "";
    }
    
    char buffer[128];
    std::string result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

// ============================================================================
// Parse bandwidth from comm core benchmark output
// ============================================================================
static double parseCommBandwidth(const std::string& output)
{
    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.find("Bandwidth:") != std::string::npos) {
            // Extract number before "GB/s"
            size_t pos = line.find("Bandwidth:");
            if (pos != std::string::npos) {
                std::string num_str = line.substr(pos + 10);
                size_t end_pos = num_str.find("GB/s");
                if (end_pos != std::string::npos) {
                    num_str = num_str.substr(0, end_pos);
                    // Trim whitespace
                    num_str.erase(0, num_str.find_first_not_of(" \t"));
                    num_str.erase(num_str.find_last_not_of(" \t") + 1);
                    return std::stod(num_str);
                }
            }
        }
    }
    return -1.0;
}

// ============================================================================
// Parse GFLOPS from compute core benchmark output
// ============================================================================
static double parseComputeGFLOPS(const std::string& output)
{
    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.find("Performance:") != std::string::npos) {
            // Extract number before "GFLOPS"
            size_t pos = line.find("Performance:");
            if (pos != std::string::npos) {
                std::string num_str = line.substr(pos + 12);
                size_t end_pos = num_str.find("GFLOPS");
                if (end_pos != std::string::npos) {
                    num_str = num_str.substr(0, end_pos);
                    // Trim whitespace
                    num_str.erase(0, num_str.find_first_not_of(" \t"));
                    num_str.erase(num_str.find_last_not_of(" \t") + 1);
                    return std::stod(num_str);
                }
            }
        }
    }
    return -1.0;
}

// ============================================================================
// Calculate optimal core configuration
// ============================================================================
static void calculateOptimalConfig(
    double comm_bandwidth_gb_s,      // Single comm core bandwidth
    double compute_gflops,           // Single compute core GFLOPS
    int total_cores,                 // Total cores available (24)
    int& optimal_comm_cores,         // Output: optimal comm cores
    int& optimal_compute_cores)      // Output: optimal compute cores
{
    // From the main program, we know:
    // - Matrix: M=16384, K=4096, N=4096
    // - Total tiles: 2048 (128x16)
    // - Each tile: 128x256 float = 128KB
    // - Total output: 16384 * 4096 * 4 bytes = 256 MB per rank
    // - In AllReduce, each rank sends 256 MB to all other ranks
    
    const int M = 16384;
    const int K = 4096;
    const int N = 4096;
    const int nranks = 8;  // From terminal output
    
    // Total computation: M * K * N * 2 ops per rank
    double total_compute_ops = (double)M * K * N * 2.0;
    double total_compute_gflop = total_compute_ops / 1e9;
    
    // Total communication: each rank sends output to all ranks
    // Output size per rank: M * N * sizeof(float) = 256 MB
    // Total data sent per rank: 256 MB * nranks = 2 GB (from terminal output)
    double total_comm_bytes = (double)M * N * sizeof(float) * nranks;
    double total_comm_gb = total_comm_bytes / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "\n=== Performance Analysis ===\n";
    std::cout << "Single Comm Core Bandwidth: " << std::fixed << std::setprecision(2) 
              << comm_bandwidth_gb_s << " GB/s\n";
    std::cout << "Single Compute Core Performance: " << std::fixed << std::setprecision(2) 
              << compute_gflops << " GFLOPS\n";
    std::cout << "Total Computation: " << std::fixed << std::setprecision(2) 
              << total_compute_gflop << " GFLOP\n";
    std::cout << "Total Communication: " << std::fixed << std::setprecision(2) 
              << total_comm_gb << " GB\n";
    
    // Estimate time for different configurations
    // We want: compute_time ≈ comm_time for optimal overlap
    
    double best_ratio = 0.0;
    double best_overlap_efficiency = 0.0;
    
    // Try different comm/compute ratios
    for (int comm_cores = 1; comm_cores < total_cores; ++comm_cores) {
        int compute_cores = total_cores - comm_cores;
        
        // Estimate compute time
        double compute_time = total_compute_gflop / (compute_gflops * compute_cores);
        
        // Estimate comm time
        double comm_time = total_comm_gb / (comm_bandwidth_gb_s * comm_cores);
        
        // Overlap efficiency: how well they match
        double max_time = std::max(compute_time, comm_time);
        double min_time = std::min(compute_time, comm_time);
        double overlap_efficiency = min_time / max_time;
        
        std::cout << "  Comm=" << comm_cores << ", Compute=" << compute_cores 
                  << ": Compute=" << std::fixed << std::setprecision(2) << (compute_time * 1000) << "ms, "
                  << "Comm=" << (comm_time * 1000) << "ms, "
                  << "Overlap=" << std::fixed << std::setprecision(2) << (overlap_efficiency * 100) << "%\n";
        
        if (overlap_efficiency > best_overlap_efficiency) {
            best_overlap_efficiency = overlap_efficiency;
            optimal_comm_cores = comm_cores;
            optimal_compute_cores = compute_cores;
        }
    }
    
    std::cout << "\n=== Optimal Configuration ===\n";
    std::cout << "Comm Cores: " << optimal_comm_cores << "\n";
    std::cout << "Compute Cores: " << optimal_compute_cores << "\n";
    std::cout << "Total Cores: " << (optimal_comm_cores + optimal_compute_cores) << "\n";
    std::cout << "Expected Overlap Efficiency: " << std::fixed << std::setprecision(2) 
              << (best_overlap_efficiency * 100) << "%\n";
}

// ============================================================================
// Update CMakeLists.txt with optimal configuration
// ============================================================================
static void updateCMakeConfig(int comm_cores, int compute_cores)
{
    std::string cmake_file = "CMakeLists.txt";
    std::ifstream in(cmake_file);
    std::string content;
    std::string line;
    
    while (std::getline(in, line)) {
        if (line.find("CONFIG_COMM_BLOCK_NUM") != std::string::npos) {
            // Find the next line with the default value
            std::string next_line;
            if (std::getline(in, next_line)) {
                if (next_line.find("#define") != std::string::npos) {
                    content += "#ifndef CONFIG_COMM_BLOCK_NUM\n";
                    content += "#define CONFIG_COMM_BLOCK_NUM " + std::to_string(comm_cores) + "\n";
                    content += "#endif\n";
                    continue;
                }
            }
        }
        if (line.find("CONFIG_COMPUTE_BLOCK_NUM") != std::string::npos) {
            // Find the next line with the default value
            std::string next_line;
            if (std::getline(in, next_line)) {
                if (next_line.find("#define") != std::string::npos) {
                    content += "#ifndef CONFIG_COMPUTE_BLOCK_NUM\n";
                    content += "#define CONFIG_COMPUTE_BLOCK_NUM " + std::to_string(compute_cores) + "\n";
                    content += "#endif\n";
                    continue;
                }
            }
        }
        content += line + "\n";
    }
    in.close();
    
    // Actually, we should modify the source files directly
    // Let's update comm_kernel.cpp and gemm_compute_kernel.cpp
}

// ============================================================================
// Update source files with optimal configuration
// ============================================================================
static void updateSourceConfig(int comm_cores, int compute_cores)
{
    // Update comm_kernel.cpp - need to handle #ifndef pattern
    std::ifstream comm_in("comm_kernel.cpp");
    std::string comm_content;
    std::string line;
    bool in_comm_block = false;
    bool in_compute_block = false;
    int comm_block_lines = 0;
    int compute_block_lines = 0;
    
    while (std::getline(comm_in, line)) {
        if (line.find("#ifndef CONFIG_COMM_BLOCK_NUM") != std::string::npos) {
            in_comm_block = true;
            comm_block_lines = 0;
        }
        if (line.find("#ifndef CONFIG_COMPUTE_BLOCK_NUM") != std::string::npos) {
            in_compute_block = true;
            compute_block_lines = 0;
        }
        
        if (in_comm_block) {
            comm_block_lines++;
            if (line.find("#define CONFIG_COMM_BLOCK_NUM") != std::string::npos) {
                comm_content += "#define CONFIG_COMM_BLOCK_NUM " + std::to_string(comm_cores) + "\n";
                continue;
            }
            if (line.find("#endif") != std::string::npos && comm_block_lines >= 2) {
                in_comm_block = false;
            }
        } else if (in_compute_block) {
            compute_block_lines++;
            if (line.find("#define CONFIG_COMPUTE_BLOCK_NUM") != std::string::npos) {
                comm_content += "#define CONFIG_COMPUTE_BLOCK_NUM " + std::to_string(compute_cores) + "\n";
                continue;
            }
            if (line.find("#endif") != std::string::npos && compute_block_lines >= 2) {
                in_compute_block = false;
            }
        }
        
        comm_content += line + "\n";
    }
    comm_in.close();
    
    std::ofstream comm_out("comm_kernel.cpp");
    comm_out << comm_content;
    comm_out.close();
    
    // Update gemm_compute_kernel.cpp
    std::ifstream compute_in("gemm_compute_kernel.cpp");
    std::string compute_content;
    bool in_compute_block2 = false;
    int compute_block_lines2 = 0;
    
    while (std::getline(compute_in, line)) {
        if (line.find("#ifndef CONFIG_COMPUTE_BLOCK_NUM") != std::string::npos) {
            in_compute_block2 = true;
            compute_block_lines2 = 0;
        }
        
        if (in_compute_block2) {
            compute_block_lines2++;
            if (line.find("#define CONFIG_COMPUTE_BLOCK_NUM") != std::string::npos) {
                compute_content += "#define CONFIG_COMPUTE_BLOCK_NUM " + std::to_string(compute_cores) + "\n";
                continue;
            }
            if (line.find("#endif") != std::string::npos && compute_block_lines2 >= 2) {
                in_compute_block2 = false;
            }
        }
        
        compute_content += line + "\n";
    }
    compute_in.close();
    
    std::ofstream compute_out("gemm_compute_kernel.cpp");
    compute_out << compute_content;
    compute_out.close();
    
    std::cout << "\n[INFO] Updated source files with optimal configuration:\n";
    std::cout << "  comm_kernel.cpp: COMM_BLOCK_NUM=" << comm_cores 
              << ", COMPUTE_BLOCK_NUM=" << compute_cores << "\n";
    std::cout << "  gemm_compute_kernel.cpp: COMPUTE_BLOCK_NUM=" << compute_cores << "\n";
}

// ============================================================================
// Main benchmark program
// ============================================================================
int main(int argc, char* argv[])
{
    int n_ranks = 8;
    int first_device = 0;
    int total_cores = 24;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--nranks") == 0 && i + 1 < argc) {
            n_ranks = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--first-device") == 0 && i + 1 < argc) {
            first_device = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--total-cores") == 0 && i + 1 < argc) {
            total_cores = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--skip-benchmark") == 0) {
            // Skip individual benchmarks, just calculate from known values
            // This is useful if benchmarks are already run
            std::cout << "[INFO] Skipping individual benchmarks, using known values\n";
            
            // Use values from terminal output as reference
            // From terminal: Comm Bandwidth: 68.069 GB/s (for all comm cores)
            // From terminal: Compute: 279257.786 GFLOPS (for all compute cores)
            // We need to estimate per-core values
            
            // Estimate: if 50 comm cores give 68 GB/s, single core ≈ 1.36 GB/s
            // Estimate: if 24 compute cores give 279 TFLOPS, single core ≈ 11.6 TFLOPS
            double comm_bandwidth = 1.36;  // GB/s per core (estimated)
            double compute_gflops = 11600.0;  // GFLOPS per core (estimated)
            
            int optimal_comm, optimal_compute;
            calculateOptimalConfig(comm_bandwidth, compute_gflops, total_cores, 
                                 optimal_comm, optimal_compute);
            
            updateSourceConfig(optimal_comm, optimal_compute);
            
            std::cout << "\n[INFO] Please rebuild and run the main program to test the new configuration.\n";
            return 0;
        }
    }
    
    std::cout << "=== Core Performance Benchmark ===\n";
    std::cout << "Testing individual core performance to determine optimal configuration...\n\n";
    
    // Step 1: Benchmark single comm core
    std::cout << "Step 1: Benchmarking single comm core...\n";
    std::string comm_cmd = "./benchmark_single_comm_core --nranks " + std::to_string(n_ranks) 
                          + " --first-device " + std::to_string(first_device) 
                          + " --num-tiles 100 2>&1";
    std::string comm_output = runCommand(comm_cmd);
    std::cout << comm_output;
    double comm_bandwidth = parseCommBandwidth(comm_output);
    
    if (comm_bandwidth < 0) {
        std::cerr << "[ERROR] Failed to parse comm core bandwidth\n";
        return 1;
    }
    
    // Step 2: Benchmark single compute core
    std::cout << "\nStep 2: Benchmarking single compute core...\n";
    std::string compute_cmd = "./benchmark_single_compute_core --device " 
                             + std::to_string(first_device) + " 2>&1";
    std::string compute_output = runCommand(compute_cmd);
    std::cout << compute_output;
    double compute_gflops = parseComputeGFLOPS(compute_output);
    
    if (compute_gflops < 0) {
        std::cerr << "[ERROR] Failed to parse compute core GFLOPS\n";
        return 1;
    }
    
    // Step 3: Calculate optimal configuration
    int optimal_comm_cores, optimal_compute_cores;
    calculateOptimalConfig(comm_bandwidth, compute_gflops, total_cores,
                          optimal_comm_cores, optimal_compute_cores);
    
    // Step 4: Update source files
    updateSourceConfig(optimal_comm_cores, optimal_compute_cores);
    
    std::cout << "\n[INFO] Configuration updated. Please rebuild and run the main program.\n";
    std::cout << "  Run: cd build && cmake .. && make -j16 && ./gemm_allreduce --nranks " 
              << n_ranks << "\n";
    
    return 0;
}
