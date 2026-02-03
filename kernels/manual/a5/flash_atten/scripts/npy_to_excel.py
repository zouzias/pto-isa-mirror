#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""
Convert .npy files to Excel format for easier viewing and analysis.
Supports both individual files and batch conversion of entire directories.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas openpyxl")
    sys.exit(1)

try:
    import openpyxl
except ImportError:
    print("Error: openpyxl is required. Install with: pip install openpyxl")
    sys.exit(1)


def npy_to_excel(npy_path, excel_path=None, sheet_name='Data', max_rows=1000000, max_cols=16384):
    """
    Convert a single .npy file to Excel format.
    
    Args:
        npy_path: Path to the .npy file
        excel_path: Path for the output Excel file (defaults to same name with .xlsx extension)
        sheet_name: Name of the Excel sheet
        max_rows: Maximum rows to export (Excel limit is ~1M)
        max_cols: Maximum columns to export (Excel limit is 16384)
    
    Returns:
        Path to the created Excel file
    """
    npy_path = Path(npy_path)
    
    if not npy_path.exists():
        raise FileNotFoundError(f"File not found: {npy_path}")
    
    # Load numpy array
    try:
        data = np.load(npy_path)
    except Exception as e:
        raise ValueError(f"Failed to load {npy_path}: {e}")
    
    # Default output path
    if excel_path is None:
        excel_path = npy_path.with_suffix('.xlsx')
    else:
        excel_path = Path(excel_path)
    
    print(f"Converting {npy_path.name} (shape: {data.shape}, dtype: {data.dtype})")
    
    # Handle different array shapes
    if data.ndim == 0:
        # Scalar
        df = pd.DataFrame({'Value': [data.item()]})
    elif data.ndim == 1:
        # 1D array - create single column
        if len(data) > max_rows:
            print(f"  Warning: Array has {len(data)} elements, truncating to {max_rows}")
            data = data[:max_rows]
        df = pd.DataFrame({'Value': data})
    elif data.ndim == 2:
        # 2D array
        rows, cols = data.shape
        if rows > max_rows:
            print(f"  Warning: Array has {rows} rows, truncating to {max_rows}")
            data = data[:max_rows, :]
            rows = max_rows
        if cols > max_cols:
            print(f"  Warning: Array has {cols} columns, truncating to {max_cols}")
            data = data[:, :max_cols]
            cols = max_cols
        df = pd.DataFrame(data)
    else:
        # 3D or higher - flatten to 2D with labeled rows
        print(f"  Info: Reshaping {data.ndim}D array to 2D")
        original_shape = data.shape
        # Flatten all but last dimension
        reshaped = data.reshape(-1, data.shape[-1])
        if reshaped.shape[0] > max_rows:
            print(f"  Warning: Reshaped array has {reshaped.shape[0]} rows, truncating to {max_rows}")
            reshaped = reshaped[:max_rows, :]
        if reshaped.shape[1] > max_cols:
            print(f"  Warning: Reshaped array has {reshaped.shape[1]} columns, truncating to {max_cols}")
            reshaped = reshaped[:, :max_cols]
        df = pd.DataFrame(reshaped)
        df.index.name = f'Index (original shape: {original_shape})'
    
    # Write to Excel
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=(data.ndim > 2))
        print(f"  ✓ Created: {excel_path}")
        return excel_path
    except Exception as e:
        raise RuntimeError(f"Failed to write Excel file: {e}")


def convert_directory(directory, pattern='*.npy', recursive=False, output_dir=None):
    """
    Convert all .npy files in a directory to Excel format.
    
    Args:
        directory: Directory to search for .npy files
        pattern: File pattern to match (default: '*.npy')
        recursive: Whether to search recursively
        output_dir: Output directory (defaults to same as input)
    
    Returns:
        List of created Excel files
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all .npy files
    if recursive:
        npy_files = list(directory.rglob(pattern))
    else:
        npy_files = list(directory.glob(pattern))
    
    if not npy_files:
        print(f"No .npy files found in {directory}")
        return []
    
    print(f"Found {len(npy_files)} .npy file(s) in {directory}")
    
    excel_files = []
    for npy_file in sorted(npy_files):
        try:
            if output_dir:
                output_path = Path(output_dir) / npy_file.with_suffix('.xlsx').name
            else:
                output_path = npy_file.with_suffix('.xlsx')
            
            excel_path = npy_to_excel(npy_file, output_path)
            excel_files.append(excel_path)
        except Exception as e:
            print(f"  ✗ Error converting {npy_file.name}: {e}")
    
    print(f"\nSuccessfully converted {len(excel_files)}/{len(npy_files)} file(s)")
    return excel_files


def create_summary_workbook(npy_files, output_path, max_arrays=50):
    """
    Create a single Excel workbook with multiple .npy files as separate sheets.
    
    Args:
        npy_files: List of .npy file paths
        output_path: Path for the output Excel workbook
        max_arrays: Maximum number of arrays to include (Excel has sheet limit)
    
    Returns:
        Path to the created workbook
    """
    output_path = Path(output_path)
    
    if len(npy_files) > max_arrays:
        print(f"Warning: {len(npy_files)} files requested, limiting to {max_arrays} sheets")
        npy_files = npy_files[:max_arrays]
    
    print(f"Creating summary workbook with {len(npy_files)} sheets: {output_path}")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for i, npy_file in enumerate(npy_files):
            npy_file = Path(npy_file)
            try:
                data = np.load(npy_file)
                sheet_name = npy_file.stem[:31]  # Excel sheet name limit is 31 chars
                
                # Handle different dimensions
                if data.ndim <= 2:
                    df = pd.DataFrame(data) if data.ndim == 2 else pd.DataFrame({'Value': data.ravel()})
                else:
                    # Flatten to 2D
                    reshaped = data.reshape(-1, data.shape[-1])
                    df = pd.DataFrame(reshaped)
                
                # Limit size for performance
                if df.shape[0] > 10000:
                    print(f"  {sheet_name}: truncating {df.shape[0]} rows to 10000")
                    df = df.head(10000)
                if df.shape[1] > 100:
                    print(f"  {sheet_name}: truncating {df.shape[1]} cols to 100")
                    df = df.iloc[:, :100]
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"  ✓ Added sheet: {sheet_name} (shape: {data.shape})")
            except Exception as e:
                print(f"  ✗ Error adding {npy_file.name}: {e}")
    
    print(f"Created: {output_path}")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert .npy files to Excel format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file
  python npy_to_excel.py data.npy
  
  # Convert a single file with custom output name
  python npy_to_excel.py data.npy -o output.xlsx
  
  # Convert all .npy files in a directory
  python npy_to_excel.py -d ./build/case_float_H_128_S0_128_S1_1024/
  
  # Convert all .npy files recursively
  python npy_to_excel.py -d ./build/ -r
  
  # Create a summary workbook with multiple arrays
  python npy_to_excel.py -d ./build/case_float_H_128_S0_128_S1_1024/ --summary summary.xlsx
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input .npy file or directory')
    parser.add_argument('-o', '--output', help='Output Excel file path')
    parser.add_argument('-d', '--directory', help='Convert all .npy files in directory')
    parser.add_argument('-r', '--recursive', action='store_true', help='Search directories recursively')
    parser.add_argument('--pattern', default='*.npy', help='File pattern to match (default: *.npy)')
    parser.add_argument('--output-dir', help='Output directory for batch conversion')
    parser.add_argument('--summary', help='Create summary workbook with multiple sheets')
    parser.add_argument('--max-rows', type=int, default=1000000, help='Maximum rows per sheet (default: 1000000)')
    parser.add_argument('--max-cols', type=int, default=16384, help='Maximum columns per sheet (default: 16384)')
    
    args = parser.parse_args()
    
    if not args.input and not args.directory:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.directory:
            # Batch conversion
            npy_files = convert_directory(
                args.directory,
                pattern=args.pattern,
                recursive=args.recursive,
                output_dir=args.output_dir
            )
            
            # Optionally create summary workbook
            if args.summary and npy_files:
                # Re-find the .npy files for summary
                dir_path = Path(args.directory)
                if args.recursive:
                    all_npy = list(dir_path.rglob(args.pattern))
                else:
                    all_npy = list(dir_path.glob(args.pattern))
                create_summary_workbook(all_npy, args.summary)
        
        elif args.input:
            # Single file conversion
            npy_to_excel(
                args.input,
                excel_path=args.output,
                max_rows=args.max_rows,
                max_cols=args.max_cols
            )
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
