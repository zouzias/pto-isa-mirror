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
import signal
import re

def _prepend_env_path(var: str, path: str):
    if not path:
        return
    cur = os.environ.get(var, "")
    if not cur:
        os.environ[var] = path
    else:
        os.environ[var] = f"{path}:{cur}"

def _first_existing_dir(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None

def _source_ascend_env():
    """
    Best-effort loader for Ascend environment variables.

    Prefer sourcing `ASCEND_HOME_PATH/bin/setenv.bash` when available. Fall back
    to common user install locations. If nothing is found, assume the caller
    already exported required env vars.
    """
    candidates = []
    ascend_home = (os.environ.get("ASCEND_HOME_PATH") or "").strip()
    if ascend_home:
        candidates.extend(
            [
                os.path.join(ascend_home, "bin", "setenv.bash"),
                os.path.join(ascend_home, "set_env.sh"),
            ]
        )
    candidates.extend(
        [
            os.path.expanduser("~/Ascend/ascend-toolkit/set_env.sh"),
            os.path.expanduser("~/Ascend/ascend-toolkit/bin/setenv.bash"),
            os.path.expanduser("~/Ascend/ascend-toolkit/latest/bin/setenv.bash"),
        ]
    )

    script = None
    for p in candidates:
        if os.path.exists(p):
            script = p
            break
    if script is None:
        return

    bash = shutil.which("bash") or "bash"
    print(f"run env shell: {script}")
    result = subprocess.run(
        [bash, "-lc", f"source {script} >/dev/null 2>&1 && env -0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"warning: failed sourcing env script: {script}\n{result.stderr}")
        return
    for item in result.stdout.split("\0"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        os.environ[key] = value

def ensure_python_module(module_name: str, pip_spec: str = None, timeout_sec: int = 1800):
    try:
        __import__(module_name)
        return
    except Exception:
        pass
    spec = pip_spec or module_name
    print(f"python dep missing: {module_name}; installing {spec} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", "wheel"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--user", spec],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )
    __import__(module_name)

def run_command(command, cwd=None, check=True, capture_output=False, timeout_sec=None):
    try:
        print(f"run command: {' '.join(command)}")
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
            text=True,
            preexec_fn=os.setsid,
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            raise TimeoutError(f"command timeout after {timeout_sec}s: {' '.join(command)}")
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command, output=stdout)
        return stdout if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"run command failed with return code {e.returncode}")
        raise

def set_env_variables(run_mode, soc_version):
    _source_ascend_env()

    if run_mode != "sim":
        return

    ascend_home = (os.environ.get("ASCEND_HOME_PATH") or "").strip()
    if not ascend_home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set (required for sim mode)")

    # Prefer real runtime libs for camodel; opt-in stub usage only when needed.
    use_stub = os.environ.get("PTO_ST_USE_STUB", "0") == "1"
    if use_stub:
        stub_lib_dir = _first_existing_dir(
            [
                os.path.join(ascend_home, "runtime", "lib64", "stub"),
                os.path.join(ascend_home, "acllib", "lib64", "stub", "linux", "aarch64"),
                os.path.join(ascend_home, "acllib", "lib64", "stub"),
            ]
        )
        if stub_lib_dir:
            _prepend_env_path("LD_LIBRARY_PATH", stub_lib_dir)
        else:
            print("warning: PTO_ST_USE_STUB=1 but stub lib dir not found")

    simulator_lib_dir = _first_existing_dir(
        [
            os.path.join(ascend_home, "aarch64-linux", "simulator", soc_version, "lib"),
            os.path.join(ascend_home, "arm64-linux", "simulator", soc_version, "lib"),
            os.path.join(ascend_home, "tools", "simulator", soc_version, "lib"),  # legacy
        ]
    )
    if simulator_lib_dir:
        _prepend_env_path("LD_LIBRARY_PATH", simulator_lib_dir)
    else:
        print(f"warning: simulator lib path not found for soc `{soc_version}` under {ascend_home}")

def build_project(run_mode, soc_version, testcase = "all", debug_enable = False):
    original_dir = os.getcwd()
    # 清理并创建build目录
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    try:
        cmake_cmd = [
            "cmake",
            f"-DRUN_MODE={run_mode}",
            f"-DSOC_VERSION={soc_version}",
            ".."
        ]
        if testcase != "all":
            cmake_cmd.insert(-1, f"-DTEST_CASE={testcase}")
        if debug_enable :
            cmake_cmd.append("-DDEBUG_MODE=ON")

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
        max_jobs = int(os.environ.get("PTO_ST_JOBS", str(min(cpu_count, 32))))
        make_cmd.extend(["-j", str(max_jobs)])

        result = subprocess.run(
            make_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print("compile process:\n", result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"build failed: {e.stdout}")
        raise
    finally:
        os.chdir(original_dir)

def run_gen_data(golden_path):
    original_dir = os.getcwd()
    try:
        cmd = ["cp", golden_path, "build/gen_data.py"]
        run_command(cmd)

        build_dir = "build/"
        os.chdir(build_dir)

        gloden_gen_cmd = [sys.executable, "gen_data.py"]
        output = run_command(gloden_gen_cmd)
        print(output)
    except Exception as e:
        print(f"gen golden failed: {e}")
        raise
    finally:
        os.chdir(original_dir)

def run_binary(testcase, run_mode, args="all", timeout_sec=None):
    original_dir = os.getcwd()
    try:
        build_dir = "build/bin/"
        os.chdir(build_dir)

        def _sanitize_dir_name(name: str) -> str:
            name = (name or "").strip()
            if not name:
                return "unknown"
            # Avoid characters that can confuse downstream simulators/loggers.
            name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
            return name[:128]

        def _parse_gtest_list(text: str):
            tests = []
            current_suite = None
            for line in text.splitlines():
                if not line.strip():
                    continue
                if not line.startswith(" "):  # suite line like "TMULSTest."
                    current_suite = line.strip()
                    continue
                if current_suite is None:
                    continue
                test_name = line.strip().split("#", 1)[0].strip()
                if not test_name:
                    continue
                tests.append(f"{current_suite}{test_name}")
            return tests

        def list_gtests(gtest_filter=None):
            cmd = ["./" + testcase, "--gtest_list_tests"]
            if gtest_filter:
                cmd.append("--gtest_filter=" + gtest_filter)
            out = run_command(cmd, capture_output=True, timeout_sec=30)
            return _parse_gtest_list(out or "")

        # Guard against a stale/incorrect gtest filter silently running 0 tests.
        def ensure_gtest_has_tests(gtest_filter=None):
            return len(list_gtests(gtest_filter)) > 0

        if run_mode == "sim":
            tag = "all" if args == "all" else _sanitize_dir_name(args)
            camodel_log_dir = os.path.join("..", "camodel_logs", testcase, tag)
            os.makedirs(camodel_log_dir, exist_ok=True)
            os.environ["CAMODEL_LOG_PATH"] = camodel_log_dir

        base_timeout_sec = int(timeout_sec or 30)
        if args == "all":
            num = len(list_gtests())
            if num <= 0:
                raise RuntimeError(f"no gtest cases found in binary: {testcase}")
            effective_timeout_sec = base_timeout_sec * max(1, num)
        else:
            if not ensure_gtest_has_tests(args):
                raise RuntimeError(f"gtest_filter matched no tests: {args}")
            effective_timeout_sec = base_timeout_sec

        if args != "all":
            single_case = "--gtest_filter=" + args
            cmd = ["./" + testcase, single_case]
            print(f"run single testcase : {args}")
            output = run_command(cmd, timeout_sec=effective_timeout_sec)
            print(output)
        else : # all
            cmd = ["./" + testcase]
            print(f"run testcase : {testcase}")
            output = run_command(cmd, timeout_sec=effective_timeout_sec)
            print(output)

    except Exception as e:
        print(f"run binary failed: {e}")
        raise
    finally:
        os.chdir(original_dir)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="执行st脚本")
    parser.add_argument("-r", "--run-mode", required=True, help="运行模式（如 sim or npu)")
    parser.add_argument("-v", "--soc-version", required=True, help="SOC版本: a2, a3, or a5")
    parser.add_argument("-t", "--testcase", required=True, help="需要执行的用例 (or 'all')")
    parser.add_argument("-g", "--gtest_filter", required=False, help="可选 需要执行的具体case名")
    parser.add_argument("-d", "--debug-enable", action='store_true', help="开启debug检查")
    parser.add_argument("--timeout-sec", type=int, default=None, help="base timeout per gtest in seconds (whole binary uses base*#gtests)")

    args = parser.parse_args()
    if args.soc_version == "a2":
        default_soc_version = "Ascend910"
    elif args.soc_version == "a3":
        default_soc_version = "Ascend910B1"
    elif args.soc_version == "a5":
        default_soc_version = "Ascend910_9599"
        ensure_python_module("en_dtypes", pip_spec="en_dtypes==0.0.4")
    else:
        raise ValueError(f"unsupported soc-version: {args.soc_version}")
    default_cases = "all"
    if args.gtest_filter != None:
        default_cases = args.gtest_filter

    original_dir = os.getcwd()
    try:
        # 获取当前脚本（run_st.py）的绝对路径
        script_path = os.path.abspath(__file__)

        if args.soc_version in ("a2", "a3"):
            target_dir = os.path.dirname(os.path.dirname(script_path))
            target_dir = target_dir + "/npu/a2a3/src/st"
        else : # a5
            target_dir = os.path.dirname(os.path.dirname(script_path))
            target_dir = target_dir + "/npu/a5/src/st"

        print(f"target_dir: {target_dir}")
        os.chdir(target_dir)

        # 设置环境变量
        set_env_variables(args.run_mode, default_soc_version)

        # 执行构建
        build_project(args.run_mode, default_soc_version, args.testcase, args.debug_enable)

        if args.timeout_sec is not None:
            timeout_sec = args.timeout_sec
        else:
            timeout_sec = int(os.environ.get("PTO_ST_TIMEOUT_SEC", "30"))

        if args.testcase == "all":
            if args.gtest_filter is not None:
                raise ValueError("cannot use -g/--gtest_filter when -t all")

            testcase_cmake = os.path.join("testcase", "CMakeLists.txt")
            with open(testcase_cmake, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            if "set(ALL_TESTCASES" not in content:
                raise RuntimeError(f"cannot find ALL_TESTCASES in {testcase_cmake}")
            block = content.split("set(ALL_TESTCASES", 1)[1].split(")", 1)[0]
            testcases = []
            for line in block.splitlines():
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                testcases.append(line)

            for tc in testcases:
                golden_path = os.path.join("testcase", tc, "gen_data.py")
                if not os.path.exists(golden_path):
                    raise FileNotFoundError(f"missing gen_data.py for testcase: {tc} ({golden_path})")
                run_gen_data(golden_path)
                run_binary(tc, args.run_mode, "all", timeout_sec=timeout_sec)
        else:
            # 生成标杆
            golden_path = "testcase/" + args.testcase + "/gen_data.py"
            run_gen_data(golden_path)

            # 执行二进制文件
            run_binary(args.testcase, args.run_mode, default_cases, timeout_sec=timeout_sec)

    except Exception as e:
        print(f"run failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    os.chdir(original_dir)

if __name__ == "__main__":
    main()
