TREDUCE Performance Test
========================

This test measures the latency and bandwidth of the TREDUCE collective.

Usage:
  ./treduce_perf_test [options]

Options:
  -r, --ranks N          Number of ranks (default: 4)
  -d, --devices N        Number of devices (default: 4)
  -w, --warmup N         Warmup iterations (default: 20)
  -m, --measure N        Measurement iterations (default: 50)
  -s, --size SIZE        Test size: small, large, all (default: all)
  -q, --quiet            Disable per-iteration output
  -h, --help             Show this help message

Test sizes:
  small:  64x64, 16x256 elements
  large:  128x128, 64x256 elements
