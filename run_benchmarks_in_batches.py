#!/usr/bin/env python3
"""Run A2A3/A5 benchmark collectors in batches and update progress files.

Creates `a2a3_progress.md` and `a5_progress.md` in the repo root after
each batch with minimal content: soc, done list, left list.

Requires: working `python3` and the collect_a2a3_benchmark.py / collect_a5_benchmark.py
in the repository root. For real simulator runs, source the CANN env beforehand.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
A2_SCRIPT = REPO / "collect_a2a3_benchmark.py"
A5_SCRIPT = REPO / "collect_a5_benchmark.py"
A2_TESTDIR = REPO / "tests" / "npu" / "a2a3" / "src" / "st" / "testcase"
A5_TESTDIR = REPO / "tests" / "npu" / "a5" / "src" / "st" / "testcase"


def list_testcases(testdir: Path):
    if not testdir.exists():
        return []
    return sorted([p.name for p in testdir.iterdir() if p.is_dir() and (p / "main.cpp").exists()])


def batches(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def write_progress(path: Path, soc: str, done: list, left: list):
    # Minimal overwrite with exactly the three lines requested
    content = f"soc: {soc}\ndone: {', '.join(done)}\nleft: {', '.join(left)}\n"
    path.write_text(content)


def run_batch(script: Path, soc: str, batch: list, no_run: bool):
    cmd = [sys.executable, str(script), "--testcases"] + batch
    if no_run:
        cmd.append("--no-run")
    print(f"Running {soc} batch: {batch}")
    r = subprocess.run(cmd, cwd=str(REPO))
    return r.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--no-run", action="store_true", help="Do not run simulator; only parse existing logs")
    p.add_argument("--only", choices=["a2a3", "a5", "both"], default="both")
    args = p.parse_args()

    todo = []
    if args.only in ("a2a3", "both"):
        todo.append(("a2a3", A2_SCRIPT, A2_TESTDIR, REPO / "a2a3_progress.md"))
    if args.only in ("a5", "both"):
        todo.append(("a5", A5_SCRIPT, A5_TESTDIR, REPO / "a5_progress.md"))

    for soc, script, testdir, prog in todo:
        testcases = list_testcases(testdir)
        if not testcases:
            print(f"[ERROR] No testcases found for {soc} at {testdir}")
            continue

        done = []
        left = list(testcases)

        for batch in batches(testcases, args.batch_size):
            run_batch(script, soc, batch, args.no_run)
            # mark as done regardless of success so orchestration moves forward
            done.extend(batch)
            left = [t for t in testcases if t not in done]
            write_progress(prog, soc, done, left)
            print(f"Wrote progress to {prog} — done {len(done)}/{len(testcases)}")
            # continue to next batch

    print("All requested runs submitted.")


if __name__ == "__main__":
    main()
