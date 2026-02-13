#!/usr/bin/python3
# coding=utf-8
# Script to analyze and compare saturation mode test outputs

import os
import sys
import numpy as np
from pathlib import Path

# Type mapping for test cases
TYPE_MAP = {
    'fp16': np.float16,
    'fp32': np.float32,
    'int8': np.int8,
    'uint8': np.uint8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
}


def parse_case_name(case_name):
    """Extract source and destination types from case name like 'saturation_fp16_int8_1x32'"""
    # Strip "TCVTTest." prefix if present
    if case_name.startswith('TCVTTest.'):
        case_name = case_name[9:]  # Remove "TCVTTest."
    
    parts = case_name.split('_')
    
    # Handle case names like "saturation_fp16_int8_1x32"
    if len(parts) >= 4 and parts[0] == 'saturation':
        src_type_str = parts[1]
        dst_type_str = parts[2]
        shape_str = parts[3]
        
        src_type = TYPE_MAP.get(src_type_str)
        dst_type = TYPE_MAP.get(dst_type_str)
        
        # Parse shape (e.g., "1x32")
        shape_parts = shape_str.split('x')
        m, n = int(shape_parts[0]), int(shape_parts[1])
        
        return src_type, dst_type, m, n, src_type_str, dst_type_str
    
    return None, None, None, None, None, None


def read_bin_file(filepath, dtype, shape):
    """Read binary file and return as numpy array"""
    if not os.path.exists(filepath):
        return None
    return np.fromfile(filepath, dtype=dtype).reshape(shape)


def format_value(val, dtype):
    """Format a value for display, handling special cases"""
    if np.issubdtype(dtype, np.floating):
        if np.isnan(val):
            return "NaN"
        elif np.isinf(val):
            return "+Inf" if val > 0 else "-Inf"
        else:
            # Format floats with appropriate precision
            if abs(val) < 0.01 and val != 0:
                return f"{val:.6e}"
            else:
                return f"{val:.4f}"
    else:
        return f"{int(val):>6d}"


def is_special_value(val, dtype):
    """Check if value is special (inf, nan, or overflow candidate)"""
    if np.issubdtype(dtype, np.floating):
        return np.isnan(val) or np.isinf(val) or abs(val) > 100
    else:
        # For integers, consider it special if it's non-zero (likely overflow test values)
        # since test data typically has special values first followed by zeros
        return val != 0


def analyze_case(case_dir):
    """Analyze a single test case directory"""
    case_name = os.path.basename(case_dir)
    print(f"\n{'═'*100}")
    print(f"  TEST CASE: {case_name}")
    print(f"{'═'*100}\n")
    
    src_type, dst_type, m, n, src_str, dst_str = parse_case_name(case_name)
    
    if src_type is None or dst_type is None:
        print(f"❌ Error: Could not parse case name: {case_name}")
        return
    
    print(f"  Conversion: {src_str.upper()} → {dst_str.upper()}")
    print(f"  Shape:      {m} × {n} elements")
    print(f"  Total size: {m * n} elements\n")
    
    shape = (m, n)
    
    # Read all binary files
    files = {
        'input': ('x1_gm.bin', src_type),
        'golden_trunc': ('golden_truncated.bin', dst_type),
        'output_sat': ('output_saturated.bin', dst_type),
        'output_trunc': ('output_truncated.bin', dst_type),
        'output_default': ('output_default.bin', dst_type),
    }
    
    data = {}
    missing_files = []
    for key, (filename, dtype) in files.items():
        filepath = os.path.join(case_dir, filename)
        arr = read_bin_file(filepath, dtype, shape)
        data[key] = arr
        
        if arr is None:
            missing_files.append(filename)
    
    if missing_files:
        print(f"⚠️  Missing files: {', '.join(missing_files)}\n")
    
    # Print full data comparison
    print("\n" + "┌" + "─"*98 + "┐")
    print("│" + " "*35 + "FULL DATA COMPARISON" + " "*43 + "│")
    print("├" + "─"*98 + "┤")
    
    header = f"│ {'#':<5} │ {'Input':<18} │ {'Golden(T)':<12} │ {'Out_Sat':<12} │ {'Out_Trunc':<12} │ {'Out_Def':<12} │"
    print(header)
    print("├" + "─"*98 + "┤")
    
    flat_size = m * n
    
    # Separate indices into special values and normal values
    special_indices = []
    normal_indices = []
    
    for i in range(flat_size):
        if data['input'] is not None and is_special_value(data['input'].flat[i], src_type):
            special_indices.append(i)
        else:
            normal_indices.append(i)
    
    # Display special values first, then normal values
    all_indices = special_indices + normal_indices
    
    for idx, i in enumerate(all_indices):
        input_val = data['input'].flat[i] if data['input'] is not None else "N/A"
        golden_val = data['golden_trunc'].flat[i] if data['golden_trunc'] is not None else "N/A"
        sat_val = data['output_sat'].flat[i] if data['output_sat'] is not None else "N/A"
        trunc_val = data['output_trunc'].flat[i] if data['output_trunc'] is not None else "N/A"
        def_val = data['output_default'].flat[i] if data['output_default'] is not None else "N/A"
        
        input_str = format_value(input_val, src_type) if data['input'] is not None else "N/A"
        golden_str = format_value(golden_val, dst_type) if data['golden_trunc'] is not None else "N/A"
        sat_str = format_value(sat_val, dst_type) if data['output_sat'] is not None else "N/A"
        trunc_str = format_value(trunc_val, dst_type) if data['output_trunc'] is not None else "N/A"
        def_str = format_value(def_val, dst_type) if data['output_default'] is not None else "N/A"
        
        # Add separator between special and normal values
        if idx == len(special_indices) and len(special_indices) > 0:
            print("├" + "─"*98 + "┤")
        
        print(f"│ {i:<5} │ {input_str:<18} │ {golden_str:<12} │ {sat_str:<12} │ {trunc_str:<12} │ {def_str:<12} │")
    
    print("└" + "─"*98 + "┘")
    
    # Special values table
    print("\n" + "┌" + "─"*98 + "┐")
    print("│" + " "*24 + "SPECIAL VALUES ANALYSIS (Inf, NaN, Overflow)" + " "*30 + "│")
    print("├" + "─"*98 + "┤")
    
    if data['input'] is None:
        print("│  ⚠️  No input data available" + " "*67 + "│")
        print("└" + "─"*98 + "┘")
        return
    
    # Identify special value indices
    special_indices = []
    for i in range(flat_size):
        if is_special_value(data['input'].flat[i], src_type):
            special_indices.append(i)
    
    if not special_indices:
        print("│  ℹ️  No special values detected in input" + " "*53 + "│")
        print("└" + "─"*98 + "┘")
    else:
        header = f"│ {'#':<5} │ {'Type':<8} │ {'Input Value':<18} │ {'Golden':<12} │ {'Out_Trunc':<12} │ {'Out_Def':<12} │ {'Match':<8} │"
        print(header)
        print("├" + "─"*98 + "┤")
        
        for i in special_indices:
            input_val = data['input'].flat[i]
            golden_val = data['golden_trunc'].flat[i] if data['golden_trunc'] is not None else None
            trunc_val = data['output_trunc'].flat[i] if data['output_trunc'] is not None else None
            def_val = data['output_default'].flat[i] if data['output_default'] is not None else None
            
            # Determine input type
            if np.isnan(input_val):
                input_type = "NaN"
            elif np.isinf(input_val):
                input_type = "+Inf" if input_val > 0 else "-Inf"
            elif np.issubdtype(src_type, np.floating):
                input_type = "Normal"
            else:
                input_type = "Overflow?" if is_special_value(input_val, src_type) else "Normal"
            
            input_str = format_value(input_val, src_type)
            golden_str = format_value(golden_val, dst_type) if golden_val is not None else "N/A"
            trunc_str = format_value(trunc_val, dst_type) if trunc_val is not None else "N/A"
            def_str = format_value(def_val, dst_type) if def_val is not None else "N/A"
            
            # Check if truncated and default match
            if trunc_val is not None and def_val is not None:
                match = "✓" if trunc_val == def_val else "✗"
            else:
                match = "-"
            
            print(f"│ {i:<5} │ {input_type:<8} │ {input_str:<18} │ {golden_str:<12} │ {trunc_str:<12} │ {def_str:<12} │ {match:^8} │")
        
        print("└" + "─"*98 + "┘")
    
    # Overall comparison
    print("\n" + "┌" + "─"*98 + "┐")
    print("│" + " "*22 + "COMPARISON: output_default.bin vs output_truncated.bin" + " "*22 + "│")
    print("├" + "─"*98 + "┤")
    
    if data['output_trunc'] is not None and data['output_default'] is not None:
        diff = data['output_default'] != data['output_trunc']
        num_diff = np.sum(diff)
        total = flat_size
        match_pct = 100.0 * (total - num_diff) / total
        
        status_icon = "✅" if num_diff == 0 else "❌"
        print(f"│  {status_icon} Total elements:      {total:<70} │")
        print(f"│     Matching elements:   {total - num_diff:<70} │")
        print(f"│     Different elements:  {num_diff:<70} │")
        print(f"│     Match rate:          {match_pct:.2f}%{' '*67} │")
        
        if num_diff > 0:
            indices = np.where(diff.flat)[0].tolist()
            indices_str = str(indices) if len(indices) <= 10 else f"{indices[:10]}... ({len(indices)} total)"
            print(f"│     Differences at:      {indices_str:<70} │")
        print("└" + "─"*98 + "┘")
    else:
        print("│  ⚠️  Cannot compare - one or both files missing" + " "*48 + "│")
        print("└" + "─"*98 + "┘")
    
    # Golden comparison
    if data['golden_trunc'] is not None and data['output_trunc'] is not None:
        print("\n" + "┌" + "─"*98 + "┐")
        print("│" + " "*20 + "GOLDEN VALIDATION: output_truncated.bin vs golden_truncated.bin" + " "*16 + "│")
        print("├" + "─"*98 + "┤")
        
        diff = data['output_trunc'] != data['golden_trunc']
        num_diff = np.sum(diff)
        total = flat_size
        match_pct = 100.0 * (total - num_diff) / total
        
        status_icon = "✅" if num_diff == 0 else "❌"
        print(f"│  {status_icon} Total elements:      {total:<70} │")
        print(f"│     Matching elements:   {total - num_diff:<70} │")
        print(f"│     Different elements:  {num_diff:<70} │")
        print(f"│     Match rate:          {match_pct:.2f}%{' '*67} │")
        
        if num_diff > 0:
            indices = np.where(diff.flat)[0].tolist()
            indices_str = str(indices) if len(indices) <= 10 else f"{indices[:10]}... ({len(indices)} total)"
            print(f"│     Differences at:      {indices_str:<70} │")
        print("└" + "─"*98 + "┘")


def main():
    if len(sys.argv) > 1:
        # Analyze specific test case directory
        case_dir = sys.argv[1]
        if not os.path.isdir(case_dir):
            print(f"❌ Error: Directory not found: {case_dir}")
            sys.exit(1)
        analyze_case(case_dir)
    else:
        # Show usage and list available test cases
        current_dir = os.getcwd()
        
        # Look for directories matching saturation test pattern
        saturation_cases = []
        for item in os.listdir(current_dir):
            if os.path.isdir(item) and item.startswith('TCVTTest.saturation_'):
                saturation_cases.append(item)
        
        print("\n" + "═"*100)
        print("  SATURATION MODE TEST ANALYZER")
        print("═"*100)
        print("\n📖 Usage:")
        print(f"  {sys.argv[0]} <test_case_directory>")
        print(f"\n💡 Example:")
        print(f"  {sys.argv[0]} TCVTTest.saturation_fp16_int8_1x32")
        
        if saturation_cases:
            saturation_cases.sort()
            print(f"\n✅ Available test cases in current directory ({len(saturation_cases)}):")
            print("┌" + "─"*98 + "┐")
            for i, case in enumerate(saturation_cases, 1):
                print(f"│  {i}. {case:<93} │")
            print("└" + "─"*98 + "┘")
            print("\n⚠️  Note: Analyze ONE test case at a time by specifying the directory name.")
        else:
            print("\n⚠️  No saturation test case directories found in current directory.")
        
        print()


if __name__ == "__main__":
    main()
