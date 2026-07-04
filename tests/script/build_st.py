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

import os
import sys
import subprocess
import shutil
import argparse

def run_command(command, cwd=None, check=True):
    try:
        print(f"run command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd,
            check=check,
            stdout=None,
            stderr=None,
            text=True
        )
        return ""
    except subprocess.CalledProcessError as e:
        print(f"run command failed with return code {e.returncode}")
        raise


def get_simulator_info(ascend_home, soc_version):
    simulator_home = os.path.join(ascend_home, "tools", "simulator")
    soc_candidates = [soc_version]
    if soc_version == "Ascend950PR_9599":
        soc_candidates.extend(["Ascend910_9599"])
    for candidate in soc_candidates:
        camodel_path = os.path.join(simulator_home, candidate, "camodel")
        lib_path = os.path.join(simulator_home, candidate, "lib")
        if os.path.isdir(camodel_path):
            return candidate, camodel_path
        elif os.path.isdir(lib_path):
            return candidate, lib_path
    print(f"Warning: Neither 'camodel' nor 'lib' found in {os.path.join(simulator_home, soc_version)}")
    return soc_version, os.path.join(simulator_home, soc_version, "lib")


def build_project(run_mode, soc_version, auto_enable=False, testcase="all"):
    original_dir = os.getcwd()
    # 清理并创建build目录
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    # Resolve SOC_VERSION for simulator path (e.g. Ascend950PR_9599 -> Ascend910_9599)
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if run_mode == "sim" and ascend_home:
        cmake_soc, _ = get_simulator_info(ascend_home, soc_version)
    else:
        cmake_soc = soc_version

    try:
        cmake_cmd = [
            "cmake",
            f"-DRUN_MODE={run_mode}",
            f"-DSOC_VERSION={cmake_soc}",
            f"-DTEST_CASE={testcase}",
            ".."
        ]

        if auto_enable:
            cmake_cmd.append("-DAUTO_MODE=ON")

        subprocess.run(
            cmake_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # make_cmd = ["make", "VERBOSE=1"] # print compile log for debug
        make_cmd = ["make"]
        cpu_count = os.cpu_count() or 4
        make_cmd.extend(["-j", str(cpu_count)])

        result = subprocess.Popen(
            make_cmd,
            cwd=build_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        while result.poll() is None:
            line = result.stdout.readline()
            line.strip()
            if line:
                print(line)
        if result.returncode == 0:
            print("build success")
        else:
            raise RuntimeError("build failed")

    except subprocess.CalledProcessError as e:
        print(f"build failed: {e.stdout}")
        raise
    except RuntimeError as e:
        print("build failed")
        raise
    finally:
        os.chdir(original_dir)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="执行st脚本")
    parser.add_argument("-r", "--run-mode", required=True, help="运行模式（如 sim or npu)")
    parser.add_argument("-v", "--soc-version", required=True, help="SOC版本 只支持 a3 / a5 / kirinX90 / kirin9030")
    parser.add_argument("-t", "--testcase", required=True, help="需要执行的用例")
    parser.add_argument("-g", "--gtest_filter", required=False, help="可选 需要执行的具体case名")
    parser.add_argument("-a", "--auto-mode-enable", action='store_true', help="开启auto模式")

    args = parser.parse_args()
    default_soc_version = "Ascend910B1"
    if args.soc_version == "a5":
        default_soc_version = "Ascend950PR_9599"
    elif args.soc_version == "kirinX90":
        default_soc_version = "KirinX90"
    elif args.soc_version == "kirin9030":
        default_soc_version = "Kirin9030"
    default_cases = "all"
    if args.gtest_filter != None:
        default_cases = args.gtest_filter

    original_dir = os.getcwd()
    try:
        # 获取当前脚本（run_st.py）的绝对路径
        script_path = os.path.abspath(__file__)
        target_dir = os.path.dirname(os.path.dirname(script_path))

        if args.soc_version == "a3":
            target_dir = target_dir + "/npu/a2a3/src/st"
        elif args.soc_version == "kirin9030": # kirin9030 与 kirinX90 共享代码
            target_dir = target_dir + "/npu/kirin9030/src/st"
        elif args.soc_version == "kirinX90":
            target_dir = target_dir + "/npu/kirinX90/src/st"
        else : # a5
            target_dir = target_dir + "/npu/a5/src/st"

        print(f"target_dir: {target_dir}")
        os.chdir(target_dir)

        # 执行构建
        build_project(args.run_mode, default_soc_version, args.auto_mode_enable, args.testcase)

    except Exception as e:
        print(f"run failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    os.chdir(original_dir)

if __name__ == "__main__":
    main()