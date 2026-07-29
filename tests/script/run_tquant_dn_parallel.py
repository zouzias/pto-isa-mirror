# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

from run_st import build_project, run_gen_data, set_env_variables


SIMULATOR_SOC = "Ascend950PR_9599"
SIMULATOR_THREADS_PER_PROCESS = 32
DEFAULT_JOBS = 5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run all A5 TQUANT DN simulator tests in parallel.")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help="Number of parallel simulator shards (default: 5, assuming 32 simulator threads per shard).",
    )
    parser.add_argument(
        "-g",
        "--gtest-filter",
        default="TQUANTDNTest.*",
        help="GoogleTest filter applied before sharding (default: TQUANTDNTest.*).",
    )
    parser.add_argument(
        "-w",
        "--without-build",
        action="store_true",
        help="Reuse the existing A5 simulator build. Golden data is still regenerated once.",
    )
    parser.add_argument(
        "--without-golden",
        action="store_true",
        help="Reuse existing golden data instead of regenerating it.",
    )
    parser.add_argument(
        "--no-affinity",
        action="store_true",
        help="Do not bind simulator shards to disjoint host CPU sets.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.jobs <= 0:
        raise ValueError("--jobs must be greater than zero")
    if args.jobs > (os.cpu_count() or 1):
        raise ValueError("--jobs must not exceed the number of available host CPUs")
    if not args.gtest_filter:
        raise ValueError("--gtest-filter must not be empty")


def prepare_build(st_dir: Path, args: argparse.Namespace) -> Path:
    original_dir = Path.cwd()
    try:
        os.chdir(st_dir)
        set_env_variables("sim", SIMULATOR_SOC)
        if not args.without_build:
            build_project("sim", SIMULATOR_SOC, "tquant_dn")
        if not args.without_golden:
            run_gen_data("testcase/tquant_dn/gen_data.py")
    finally:
        os.chdir(original_dir)

    build_dir = st_dir / "build"
    binary = build_dir / "bin" / "tquant_dn"
    if not binary.is_file():
        raise FileNotFoundError(f"TQUANT DN simulator binary was not found: {binary}")
    return build_dir


def create_run_root(build_dir: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = build_dir / "tquant_dn_parallel" / f"{timestamp}_{os.getpid()}"
    run_root.mkdir(parents=True)
    return run_root


def link_case_directories(build_dir: Path, shard_dir: Path) -> int:
    count = 0
    for case_dir in sorted(build_dir.glob("TQUANTDNTest.*")):
        if not case_dir.is_dir():
            continue
        (shard_dir / case_dir.name).symlink_to(case_dir, target_is_directory=True)
        count += 1
    return count


def shard_cpu_set(shard_index: int, jobs: int) -> list[int]:
    cpu_count = os.cpu_count() or 1
    cpu_start = shard_index * cpu_count // jobs
    cpu_end = (shard_index + 1) * cpu_count // jobs
    return list(range(cpu_start, max(cpu_start + 1, cpu_end)))


def affinity_preexec(cpu_set: list[int]):
    def set_affinity() -> None:
        os.sched_setaffinity(0, cpu_set)

    return set_affinity


def start_shard(
    build_dir: Path,
    run_root: Path,
    shard_index: int,
    jobs: int,
    gtest_filter: str,
    bind_affinity: bool,
) -> tuple[subprocess.Popen[str], TextIO, Path]:
    shard_dir = run_root / f"shard_{shard_index}"
    bin_dir = shard_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "tquant_dn").symlink_to(build_dir / "bin" / "tquant_dn")
    case_count = link_case_directories(build_dir, shard_dir)
    if case_count == 0:
        raise RuntimeError(f"No generated TQUANTDNTest case directories were found under {build_dir}")

    camodel_log_dir = bin_dir / "camodel_log"
    (bin_dir / "log" / "ub_log").mkdir(parents=True)
    camodel_log_dir.mkdir()

    env = os.environ.copy()
    env["CAMODEL_LOG_PATH"] = str(camodel_log_dir)
    env["GTEST_TOTAL_SHARDS"] = str(jobs)
    env["GTEST_SHARD_INDEX"] = str(shard_index)
    env["GTEST_SHARD_STATUS_FILE"] = str(shard_dir / "shard.status")

    log_path = shard_dir / "run.log"
    log_file = log_path.open("w", encoding="utf-8")
    command = [str(bin_dir / "tquant_dn"), f"--gtest_filter={gtest_filter}"]
    preexec_fn = None
    if bind_affinity and hasattr(os, "sched_setaffinity"):
        preexec_fn = affinity_preexec(shard_cpu_set(shard_index, jobs))

    process = subprocess.Popen(
        command,
        cwd=bin_dir,
        env=env,
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=preexec_fn,
    )
    return process, log_file, log_path


def failed_cases(log_path: Path) -> list[str]:
    content = log_path.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"^\[  FAILED  \] (TQUANTDNTest\.[^\s]+)(?: \(|$)", content, re.MULTILINE)


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def run_shards(build_dir: Path, run_root: Path, args: argparse.Namespace) -> int:
    processes: list[subprocess.Popen[str]] = []
    log_files: list[TextIO] = []
    log_paths: list[Path] = []
    try:
        for shard_index in range(args.jobs):
            process, log_file, log_path = start_shard(
                build_dir,
                run_root,
                shard_index,
                args.jobs,
                args.gtest_filter,
                not args.no_affinity,
            )
            processes.append(process)
            log_files.append(log_file)
            log_paths.append(log_path)
            print(f"[START] shard {shard_index}/{args.jobs}: pid={process.pid}, log={log_path}")

        return_codes = [process.wait() for process in processes]
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Terminating simulator shards...", file=sys.stderr)
        terminate_processes(processes)
        return 130
    except Exception:
        terminate_processes(processes)
        raise
    finally:
        for log_file in log_files:
            log_file.close()

    all_failed_cases: list[str] = []
    for shard_index, (return_code, log_path) in enumerate(zip(return_codes, log_paths)):
        status = "PASS" if return_code == 0 else "FAIL"
        print(f"[{status}] shard {shard_index}: rc={return_code}, log={log_path}")
        all_failed_cases.extend(failed_cases(log_path))

    if all_failed_cases:
        print("\nFailed cases:")
        for case_name in sorted(set(all_failed_cases)):
            print(f"  {case_name}")

    failed_shards = sum(return_code != 0 for return_code in return_codes)
    print(f"\nRun directory: {run_root}")
    print(f"Shard summary: total={args.jobs}, passed={args.jobs - failed_shards}, failed={failed_shards}")
    return 0 if failed_shards == 0 else 1


def main() -> int:
    args = parse_arguments()
    validate_arguments(args)
    repo_root = Path(__file__).resolve().parents[2]
    st_dir = repo_root / "tests" / "npu" / "a5" / "src" / "st"

    build_dir = prepare_build(st_dir, args)
    run_root = create_run_root(build_dir)
    print(f"Running {args.jobs} parallel A5 simulator shard(s), filter={args.gtest_filter}")
    return run_shards(build_dir, run_root, args)


if __name__ == "__main__":
    sys.exit(main())
