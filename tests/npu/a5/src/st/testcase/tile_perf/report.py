#!/usr/bin/env python3
"""
PTO Regression Report Generator

Generates comparison reports from regression results.
"""

import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "results.db"


def format_epc(epc):
    """Format EPC value with color indicator."""
    if epc is None:
        return "N/A"
    return f"{epc:.2f}"


def generate_report(run_id: int = None, compare_run: int = None, format: str = "text"):
    """Generate a performance report."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get latest run if not specified
    if run_id is None:
        cursor.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            print("No runs found")
            return
        run_id = row["id"]
    
    # Get run info
    cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        print(f"Run #{run_id} not found")
        return
    
    # Get results
    cursor.execute("""
        SELECT * FROM results 
        WHERE run_id = ? 
        ORDER BY arch, test_name
    """, (run_id,))
    results = cursor.fetchall()
    
    # Get comparison results if specified
    compare_results = {}
    if compare_run:
        cursor.execute("SELECT * FROM results WHERE run_id = ?", (compare_run,))
        for r in cursor.fetchall():
            key = (r["test_name"], r["arch"])
            compare_results[key] = r
    
    # Generate report
    print(f"# PTO Performance Report")
    print(f"")
    print(f"**Run:** #{run_id}")
    print(f"**Commit:** {run['git_commit']}")
    print(f"**Branch:** {run['git_branch']}")
    print(f"**Date:** {run['submitted_at']}")
    print(f"**Status:** {run['status']} ({run['passed']}/{run['total_tests']} passed)")
    print()
    
    if compare_run:
        print(f"**Compared to:** Run #{compare_run}")
        print()
    
    # Group by arch
    by_arch = {}
    for r in results:
        arch = r["arch"]
        if arch not in by_arch:
            by_arch[arch] = []
        by_arch[arch].append(r)
    
    for arch, arch_results in by_arch.items():
        print(f"## {arch.upper()}")
        print()
        print(f"| Test | Elements | Tick | EPC | Status |" + (" Δ |" if compare_run else ""))
        print(f"|------|----------|------|-----|--------|" + ("---|" if compare_run else ""))
        
        for r in arch_results:
            test = r["test_name"]
            elements = r["elements"] or "?"
            tick = r["total_tick"] or "N/A"
            epc = format_epc(r["epc"])
            status = "✅" if r["status"] == "pass" else "❌"
            
            delta = ""
            if compare_run:
                key = (test, arch)
                if key in compare_results:
                    old_epc = compare_results[key]["epc"]
                    new_epc = r["epc"]
                    if old_epc and new_epc:
                        change = ((new_epc - old_epc) / old_epc) * 100
                        if change > 5:
                            delta = f"🚀 +{change:.1f}%"
                        elif change < -5:
                            delta = f"⚠️ {change:.1f}%"
                        else:
                            delta = f"{change:+.1f}%"
                    else:
                        delta = "N/A"
                else:
                    delta = "NEW"
            
            row = f"| {test} | {elements} | {tick} | {epc} | {status} |"
            if compare_run:
                row += f" {delta} |"
            print(row)
        
        print()
    
    # Summary stats
    print("## Summary")
    print()
    
    for arch, arch_results in by_arch.items():
        epcs = [r["epc"] for r in arch_results if r["epc"]]
        if epcs:
            avg_epc = sum(epcs) / len(epcs)
            max_epc = max(epcs)
            min_epc = min(epcs)
            print(f"**{arch.upper()}:** Avg EPC: {avg_epc:.2f} | Max: {max_epc:.2f} | Min: {min_epc:.2f}")
    
    conn.close()


def list_runs():
    """List all runs."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, git_commit, git_branch, submitted_at, status, 
               total_tests, passed, failed
        FROM runs ORDER BY id DESC LIMIT 20
    """)
    
    print("| Run | Commit | Branch | Date | Status | Tests |")
    print("|-----|--------|--------|------|--------|-------|")
    
    for run in cursor.fetchall():
        print(f"| #{run['id']} | {run['git_commit'][:8]} | {run['git_branch']} | "
              f"{run['submitted_at'][:16]} | {run['status']} | "
              f"{run['passed']}/{run['total_tests']} |")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="PTO Regression Report")
    parser.add_argument("--run", type=int, help="Run ID (default: latest)")
    parser.add_argument("--compare", type=int, help="Compare with run ID")
    parser.add_argument("--list", action="store_true", help="List all runs")
    parser.add_argument("--format", choices=["text", "html", "csv"], default="text")
    args = parser.parse_args()
    
    if args.list:
        list_runs()
    else:
        generate_report(args.run, args.compare, args.format)


if __name__ == "__main__":
    main()
