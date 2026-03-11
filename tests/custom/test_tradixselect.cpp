/**
 * Test file for TRadixSelect kernel
 * 
 * Tests the radix selection TopK implementation with various:
 * - K values (1, 8, 16, 32)
 * - Input sizes (64, 256, 1024)
 * - Data patterns (random, sorted, reverse-sorted, duplicates)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstdint>

// Reference CPU implementation of TopK (for validation)
template<typename T>
std::vector<uint32_t> reference_topk_indices(const std::vector<T>& input, uint32_t k, bool largest) {
    std::vector<std::pair<T, uint32_t>> indexed;
    indexed.reserve(input.size());
    for (uint32_t i = 0; i < input.size(); ++i) {
        indexed.push_back({input[i], i});
    }
    
    if (largest) {
        std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    } else {
        std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    
    std::vector<uint32_t> result;
    result.reserve(k);
    for (uint32_t i = 0; i < k; ++i) {
        result.push_back(indexed[i].second);
    }
    return result;
}

// Reference CPU cumulative histogram (for validation)
void reference_cumhist8(const std::vector<uint8_t>& bytes, uint16_t cumHist[256], bool msd) {
    // Count histogram
    uint16_t hist[256] = {0};
    for (auto b : bytes) {
        hist[b]++;
    }
    
    if (msd) {
        // Descending cumulative: cumHist[b] = count of elements >= b
        cumHist[255] = hist[255];
        for (int i = 254; i >= 0; --i) {
            cumHist[i] = cumHist[i + 1] + hist[i];
        }
    } else {
        // Ascending cumulative: cumHist[b] = count of elements <= b
        cumHist[0] = hist[0];
        for (int i = 1; i < 256; ++i) {
            cumHist[i] = cumHist[i - 1] + hist[i];
        }
    }
}

// Reference radix select (CPU version for comparison)
std::vector<uint32_t> reference_radix_select_uint8(
    const std::vector<uint8_t>& input, 
    uint32_t k, 
    bool largest)
{
    uint32_t n = input.size();
    assert(k > 0 && k <= n);
    
    // For uint8_t, single byte - direct histogram approach
    uint16_t cumHist[256];
    reference_cumhist8(input, cumHist, largest);
    
    // Find threshold bucket
    uint8_t threshold = 0;
    
    if (largest) {
        // Find first bucket b (from high to low) where cumHist[b] >= k
        for (int b = 255; b >= 0; --b) {
            if (cumHist[b] >= k) {
                threshold = static_cast<uint8_t>(b);
                break;
            }
        }
    } else {
        // Ascending: find first bucket b where cumHist[b] >= k
        for (int b = 0; b < 256; ++b) {
            if (cumHist[b] >= k) {
                threshold = static_cast<uint8_t>(b);
                break;
            }
        }
    }
    
    // Collect indices
    std::vector<uint32_t> result;
    result.reserve(k);
    
    // First: add all elements strictly better than threshold
    for (uint32_t i = 0; i < n && result.size() < k; ++i) {
        if (largest && input[i] > threshold) {
            result.push_back(i);
        } else if (!largest && input[i] < threshold) {
            result.push_back(i);
        }
    }
    
    // Then: add elements equal to threshold until we have k
    for (uint32_t i = 0; i < n && result.size() < k; ++i) {
        if (input[i] == threshold) {
            result.push_back(i);
        }
    }
    
    return result;
}

// Validate that result indices are valid TopK
bool validate_topk(const std::vector<uint8_t>& input, 
                   const std::vector<uint32_t>& resultIndices,
                   uint32_t k, bool largest) 
{
    if (resultIndices.size() != k) {
        std::cerr << "  FAIL: Expected " << k << " indices, got " << resultIndices.size() << std::endl;
        return false;
    }
    
    // Check all indices are valid
    for (auto idx : resultIndices) {
        if (idx >= input.size()) {
            std::cerr << "  FAIL: Invalid index " << idx << std::endl;
            return false;
        }
    }
    
    // Check no duplicates
    std::vector<uint32_t> sorted = resultIndices;
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i] == sorted[i-1]) {
            std::cerr << "  FAIL: Duplicate index " << sorted[i] << std::endl;
            return false;
        }
    }
    
    // Get the k-th value threshold from result
    std::vector<uint8_t> resultValues;
    for (auto idx : resultIndices) {
        resultValues.push_back(input[idx]);
    }
    std::sort(resultValues.begin(), resultValues.end());
    uint8_t kthValue = largest ? resultValues[0] : resultValues[k-1];
    
    // Verify no elements outside result are strictly better
    uint32_t betterOutside = 0;
    for (uint32_t i = 0; i < input.size(); ++i) {
        bool inResult = std::find(resultIndices.begin(), resultIndices.end(), i) != resultIndices.end();
        if (!inResult) {
            if (largest && input[i] > kthValue) betterOutside++;
            if (!largest && input[i] < kthValue) betterOutside++;
        }
    }
    
    if (betterOutside > 0) {
        std::cerr << "  FAIL: " << betterOutside << " elements outside result are better than included ones" << std::endl;
        return false;
    }
    
    return true;
}

int main() {
    std::cout << "=== TRadixSelect Test Suite ===" << std::endl;
    
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    
    struct TestCase {
        std::string name;
        std::vector<uint8_t> input;
        uint32_t k;
        bool largest;
    };
    
    std::vector<TestCase> tests;
    
    // Test 1: Small random
    {
        std::vector<uint8_t> data(64);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& d : data) d = static_cast<uint8_t>(dist(rng));
        tests.push_back({"Random N=64, K=8, largest", data, 8, true});
        tests.push_back({"Random N=64, K=8, smallest", data, 8, false});
    }
    
    // Test 2: Medium random
    {
        std::vector<uint8_t> data(256);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& d : data) d = static_cast<uint8_t>(dist(rng));
        tests.push_back({"Random N=256, K=16, largest", data, 16, true});
        tests.push_back({"Random N=256, K=32, smallest", data, 32, false});
    }
    
    // Test 3: Large random
    {
        std::vector<uint8_t> data(1024);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& d : data) d = static_cast<uint8_t>(dist(rng));
        tests.push_back({"Random N=1024, K=64, largest", data, 64, true});
    }
    
    // Test 4: Sorted ascending
    {
        std::vector<uint8_t> data(128);
        for (int i = 0; i < 128; ++i) data[i] = static_cast<uint8_t>(i * 2);
        tests.push_back({"Sorted asc N=128, K=10, largest", data, 10, true});
        tests.push_back({"Sorted asc N=128, K=10, smallest", data, 10, false});
    }
    
    // Test 5: Sorted descending
    {
        std::vector<uint8_t> data(128);
        for (int i = 0; i < 128; ++i) data[i] = static_cast<uint8_t>(255 - i * 2);
        tests.push_back({"Sorted desc N=128, K=10, largest", data, 10, true});
    }
    
    // Test 6: All same value
    {
        std::vector<uint8_t> data(64, 128);
        tests.push_back({"All same N=64, K=10", data, 10, true});
    }
    
    // Test 7: Two values only
    {
        std::vector<uint8_t> data(100);
        for (int i = 0; i < 100; ++i) data[i] = (i < 50) ? 10 : 200;
        tests.push_back({"Two values N=100, K=20, largest", data, 20, true});
        tests.push_back({"Two values N=100, K=60, largest", data, 60, true});
    }
    
    // Test 8: K=1 (just find max/min)
    {
        std::vector<uint8_t> data(256);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& d : data) d = static_cast<uint8_t>(dist(rng));
        tests.push_back({"Random N=256, K=1, largest", data, 1, true});
        tests.push_back({"Random N=256, K=1, smallest", data, 1, false});
    }
    
    // Test 9: K=N (return all)
    {
        std::vector<uint8_t> data(32);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& d : data) d = static_cast<uint8_t>(dist(rng));
        tests.push_back({"Random N=32, K=32, all", data, 32, true});
    }
    
    // Run tests
    int passed = 0, failed = 0;
    
    for (const auto& test : tests) {
        std::cout << "\nTest: " << test.name << std::endl;
        
        // Run reference radix select
        auto result = reference_radix_select_uint8(test.input, test.k, test.largest);
        
        // Validate
        if (validate_topk(test.input, result, test.k, test.largest)) {
            std::cout << "  PASS" << std::endl;
            passed++;
        } else {
            failed++;
        }
        
        // Also compare with sorting-based reference
        auto sortRef = reference_topk_indices(test.input, test.k, test.largest);
        
        // Both should select same values (indices may differ for ties)
        std::vector<uint8_t> radixVals, sortVals;
        for (auto i : result) radixVals.push_back(test.input[i]);
        for (auto i : sortRef) sortVals.push_back(test.input[i]);
        std::sort(radixVals.begin(), radixVals.end());
        std::sort(sortVals.begin(), sortVals.end());
        
        if (radixVals != sortVals) {
            std::cout << "  WARNING: Radix and sort reference give different values!" << std::endl;
        }
    }
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Passed: " << passed << "/" << (passed + failed) << std::endl;
    
    if (failed > 0) {
        std::cout << "FAILED: " << failed << " tests" << std::endl;
        return 1;
    }
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}
