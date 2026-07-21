#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# A6 (dav-9201) ST testcase build + golden-gen + deploy-to-platform driver.
#
# This is a Python driver (it replaced build.sh) because the case's artifact
# names, dtypes, sizes and kernel para_offsets are declared ONCE, in gen_data.py
# (CASE_NAME / ARTIFACTS / GOLDEN_REF). build.py imports that manifest and
# derives config.toml and run_<folder>.toml from it, so nothing is hardcoded
# twice and renaming a buffer in gen_data.py automatically flows through to the
# regression-platform config.
#
# The private-zone regression platform runs on the SAME machine, so deploy is
# just a local copy of the case folder + a local run.py invocation — no ssh.
#
# Usage:
#   python3 build.py                       # build + golden + local deploy + run (default)
#   python3 build.py build                 # only compile + golden + tomls (no deploy)
#   python3 build.py deploy                # only copy to platform + run
#   python3 build.py --platform-root PATH  # override platform root
#   python3 build.py --no-deploy           # alias for `build`
# --------------------------------------------------------------------------------

import argparse
import os
import shutil
import subprocess
import sys

# gen_data.py lives next to this script; import it for the manifest.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_data  # noqa: E402  (single source of truth for names/sizes)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.basename(SCRIPT_DIR)  # e.g. tquant
KERNEL_SRC = f"{FOLDER}_kernel.cpp"

# bisheng pipeline -> the .bin the platform loads.
KERNEL_OBJ = f"{FOLDER}_kernel.o"
KERNEL_BIN = f"{FOLDER}_kernel.bin"

# ccec pipeline -> separate object used ONLY to produce the disassembly dump.
# The bisheng object is not directly objdump-able in a useful way, so we
# recompile the same source with ccec (-g for debug info, --cce-aicore-only)
# and disassemble that. The ccec .o exists solely to produce {KERNEL_DUMP}
# and is not shipped.
KERNEL_DUMP_OBJ = f"{FOLDER}_kernel.dump.o"
KERNEL_DUMP = f"{FOLDER}_kernel.dump"

# HBM base addresses for the regression platform. Para_offset 0 is the output
# (per the kernel launch order runTQuantHif4A6(out, src0, src1)); inputs follow.
# Each artifact gets a 16 MB slot, clear of the binary + hbm_para_addr (0x40000000).
HBM_INPUT_BASE = 0x41000000
HBM_INPUT_STRIDE = 0x01000000
HBM_OUTPUT_BASE = 0x43000000
HBM_PARA_ADDR = 0x40000000


def log(stage, msg):
    print(f"[{stage}] {msg}")


def run(cmd, cwd=None):
    log("run", " ".join(cmd) if isinstance(cmd, list) else cmd)
    subprocess.run(cmd, cwd=cwd, check=True)


def kernel_include_dirs():
    """Mirror the include paths declared in the CMakeLists.

    tests/npu/a6/src/st/CMakeLists.txt adds (via PROJECT_SOURCE_DIR = .../src/st):
      <repo>/include, $ASCEND_HOME_PATH/include, /usr/local/Ascend/driver/kernel/inc
    tests/npu/a6/src/st/testcase/CMakeLists.txt (pto_vec_st) adds for the kernel:
      $ASCEND_HOME_PATH/pkg_inc/, .../profiling/, .../runtime/runtime
    From this case folder (.../st/testcase/<case>) the repo include/ is 7 levels up.
    """
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        log("build", "ERROR: ASCEND_HOME_PATH is not set (run set_env.sh first).")
        raise SystemExit(1)

    repo_include = os.path.abspath(os.path.join(SCRIPT_DIR, *[".."] * 7, "include"))
    dirs = [
        repo_include,
        os.path.join(ascend_home, "include"),
        os.path.join(ascend_home, "pkg_inc"),
        os.path.join(ascend_home, "pkg_inc", "profiling"),
        os.path.join(ascend_home, "pkg_inc", "runtime", "runtime"),
        "/usr/local/Ascend/driver/kernel/inc",
    ]
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        log("build", f"WARNING: include dirs not found: {missing}")
    return dirs


def kernel_common_flags():
    """Flags shared by BOTH the bisheng (shipping .bin) and ccec (dump) pipelines.

    These EXACTLY mirror the proven-working cmake build of the kernel .o, as
    recorded in tests/npu/a6/src/st/build/testcase/tquant/CMakeFiles/
    tquant_kernel.dir/flags.make (CXX_FLAGS line). Note in particular that
    --cce-aicore-only is intentionally NOT here: with it, bisheng emits an empty
    .text and the .bin comes out empty. The cmake .so build never uses it either.
    """
    flags = []
    for d in kernel_include_dirs():
        flags += ["-I", d]
    flags += [
        "-std=gnu++17",
        "-fPIC",
        "-D_FORTIFY_SOURCE=2",
        "-O2",
        "-std=c++17",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-Wno-unknown-attributes",
        "-fstack-protector-strong",
        "-xcce",
        "-Xhost-start",
        "-Xhost-end",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "--cce-aicore-arch=dav-920r1-vec",
        "-DREGISTER_BASE",
    ]
    return flags


def compile_kernel(case_kind="hif4", case_rows=128, case_cols=128):
    """Pipeline 1 (bisheng) -> the .bin the platform loads.

    Only ONE kernel instantiation is compiled per build — the platform loads
    a single .bin and calls its sole entry point.
    """
    if case_kind == "hif4":
        defines = ["-DTQUANT_KERNEL_HIF4", f"-DTQUANT_KERNEL_ROWS={case_rows}", f"-DTQUANT_KERNEL_COLS={case_cols}"]
    else:
        defines = ["-DTQUANT_KERNEL_VADD", f"-DTQUANT_KERNEL_ROWS={case_rows}", f"-DTQUANT_KERNEL_COLS={case_cols}"]
    log("build", f"compiling {KERNEL_SRC} -> {KERNEL_OBJ} (bisheng, {case_kind} {case_rows}x{case_cols})")
    cmd = (
        ["bisheng", f"-D{FOLDER}_kernel_EXPORTS"]
        + defines
        + kernel_common_flags()
        + [
            "-MD",
            "-MT",
            f"{FOLDER}_kernel.cpp.o",
            "-MF",
            f"{FOLDER}_kernel.cpp.o.d",
            "-o",
            KERNEL_OBJ,
            "-c",
            KERNEL_SRC,
        ]
    )
    run(cmd)
    # The device code lives in the .aicore_binary section (NOT .text, which is
    # x86 host code: the <<<>>> launch wrapper + runtime launch stubs land there).
    # objcopy ONLY .aicore_binary so the .bin is pure device code — otherwise the
    # platform fetches x86 bytes as AICORE instructions ("instruction from the
    # host code side").
    run(["llvm-objcopy", "-O", "binary", "--only-section=.aicore_binary", KERNEL_OBJ, KERNEL_BIN])


def gen_disasm_dump(platform_root):
    """Pipeline 2 (ccec) -> disassembly dump, for inspecting the device code.

    A SEPARATE compilation from the bisheng pipeline: the bisheng object is not
    convenient to objdump, so we recompile the same source with ccec (-g for
    debug info, --cce-aicore-only) and disassemble it directly. The bisheng .bin
    (what the platform loads) is unaffected; the ccec .o exists solely to
    produce {KERNEL_DUMP} and is not shipped.

    Under ccec --cce-aicore-only the device code lands in a per-function
    .text.<mangled-symbol> COMDAT section (the plain .text is just a stub), and
    llvm-objdump -d disassembles all .text* sections — so we objdump the object
    directly rather than flattening a single section.

    The disassembler is the platform's own bundled llvm-objdump
    (<platform_root>/llvm-objdump), which understands the hiipu64/dav-920r1-vec
    target that the system llvm-objdump does not.
    """
    log("build", f"compiling {KERNEL_SRC} -> {KERNEL_DUMP_OBJ} (ccec, for disasm)")
    # ccec-specific: -c (compile-only), -g (debug info), -x cce (treat input as
    # CCE source BEFORE the file). --cce-aicore-only per the documented disasm
    # flow. NOTE: do NOT pass the shared -xcce again (it would land after the
    # input file -> "no effect" warning, and ccec rejects stray flags).
    ccec_flags = ["-I" + d for d in kernel_include_dirs()]
    ccec_flags += [
        "-std=gnu++17",
        "-fPIC",
        "-D_FORTIFY_SOURCE=2",
        "-O2",
        "-std=c++17",
        "-Wno-macro-redefined",
        "-Wno-ignored-attributes",
        "-Wno-unknown-attributes",
        "-fstack-protector-strong",
        "-Xhost-start",
        "-Xhost-end",
        "-mllvm",
        "-cce-aicore-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-function-stack-size=0x8000",
        "-mllvm",
        "-cce-aicore-record-overflow=true",
        "-mllvm",
        "-cce-aicore-addr-transform",
        "-mllvm",
        "-cce-aicore-dcci-insert-for-scalar=false",
        "--cce-aicore-arch=dav-920r1-vec",
        "-DREGISTER_BASE",
        "--cce-aicore-only",
    ]
    cmd = ["ccec", "-c", "-g", "-x", "cce"] + ccec_flags + ["-o", KERNEL_DUMP_OBJ, KERNEL_SRC]
    run(cmd)

    objdump = os.path.join(platform_root, "llvm-objdump")
    if not os.path.exists(objdump):
        log("build", f"WARNING: {objdump} not found, skipping disassembly dump")
        return
    log("build", f"disassembling {KERNEL_DUMP_OBJ} -> {KERNEL_DUMP} (via {objdump})")
    # objdump flags: -l -g -d plus the target. NOTE the double-dash forms
    # (--triple, --mcpu) — the single-dash -mcpu is parsed as -m (Mach-O) and
    # errors with a misleading "unknown argument '-c'". Exit code ignored: an
    # unrecognised cpu is non-fatal; we still want best-effort disasm text.
    with open(KERNEL_DUMP, "w") as f:
        subprocess.run(
            [objdump, "-l", "-g", "-d", "--triple=hiipu64", "--mcpu=dav-920r1-vec", KERNEL_DUMP_OBJ],
            stdout=f,
            check=False,
        )


def gen_golden(case_idx=None):
    if case_idx is not None:
        log("build", f"generating artifacts via gen_data.py --case {case_idx}")
        run([sys.executable, "gen_data.py", "--case", str(case_idx)], cwd=SCRIPT_DIR)
    else:
        log("build", "generating artifacts via gen_data.py --regression")
        run([sys.executable, "gen_data.py", "--regression"], cwd=SCRIPT_DIR)


def size_hex(artifact):
    return "0x%x" % artifact.bytes_size


def gen_config_toml(case_params=None):
    """config.toml, derived from the case params manifest.

    Each output gets its own stepped HBM address slot (no aliasing) and its own
    [[check_array]] entry comparing the sim output to golden_<name>.bin.
    """
    if case_params is None:
        case_params = gen_data.DEFAULT_PARAMS
    artifacts = case_params.artifacts
    case_name = case_params.case_name
    outputs = [a for a in artifacts if a.role == "output"]
    inputs = [a for a in artifacts if a.role == "input"]
    threshold = 0

    lines = []
    lines.append(f'name = "{case_name}"')
    lines.append('path = "./"')
    lines.append("blockdim = 1")
    lines.append("subtasktype = 0")
    lines.append("subcore_id = 1")
    lines.append(f"hbm_para_addr = 0x{HBM_PARA_ADDR:x}")
    lines.append("")

    for i, a in enumerate(inputs):
        addr = HBM_INPUT_BASE + i * HBM_INPUT_STRIDE
        lines.append("[[input_para_array]]")
        lines.append(f'name = "{a.name}.bin"              # param #{a.para_offset}')
        lines.append(f"addr = 0x{addr:x}")
        lines.append(f"para_offset = {a.para_offset}")
        lines.append("")

    for i, a in enumerate(outputs):
        addr = HBM_OUTPUT_BASE + i * HBM_INPUT_STRIDE
        lines.append("[[output_para_array]]")
        lines.append(f'name = "{a.name}.bin"              # param #{a.para_offset}')
        lines.append(f"addr = 0x{addr:x}")
        lines.append(f"para_offset = {a.para_offset}")
        lines.append(f"size = {size_hex(a)}          # bytes to copy out")
        lines.append("valid = 1")
        lines.append("")

    for a in outputs:
        golden_name = "golden.bin" if len(outputs) == 1 else f"golden_{a.name}.bin"
        lines.append("[[check_array]]")
        lines.append(f'OutData = "./{a.name}.bin"      # sim-produced output')
        lines.append(f'goldData = "./{golden_name}"      # golden')
        lines.append(f"size = {size_hex(a)}          # bytes compared")
        lines.append("valid = 1")
        lines.append(f"threshold = {threshold}                  # bit-exact expected")
        lines.append(f'type = "{a.dtype}"')
        lines.append("")

    lines.append("[BIN]")
    lines.append(f'name = "{KERNEL_BIN}"')
    lines.append("addr = 0")
    lines.append("")

    return "\n".join(lines)


def gen_run_toml(case_params=None):
    if case_params is None:
        case_params = gen_data.DEFAULT_PARAMS
    case_name = case_params.case_name
    lines = []
    lines.append("[[case]]")
    lines.append(f'name = "{case_name}"')
    lines.append('config = "config.toml"')
    lines.append('version = "david_v121"')
    lines.append('architecture = "v310"')
    lines.append('type = "ca"')
    lines.append('mode = "st"')
    lines.append('spec = "etc/davidV121_cloud_config.toml"')
    lines.append("upload = false")
    lines.append("")
    lines.append("[run]")
    lines.append('author = "OmarZohir"')
    lines.append('eid = "z00969266"')
    lines.append("")
    return "\n".join(lines)


def write_tomls(case_params=None):
    log("build", "writing config.toml (from manifest)")
    with open(os.path.join(SCRIPT_DIR, "config.toml"), "w") as f:
        f.write(gen_config_toml(case_params))
    run_toml_name = f"run_{FOLDER}.toml"
    log("build", f"writing {run_toml_name}")
    with open(os.path.join(SCRIPT_DIR, run_toml_name), "w") as f:
        f.write(gen_run_toml(case_params))


def deploy_and_run(platform_root, case_params=None):
    if case_params is None:
        case_params = gen_data.DEFAULT_PARAMS
    case_name = case_params.case_name
    dest = os.path.join(platform_root, case_name)
    run_toml_rel = f"{case_name}/run_{FOLDER}.toml"

    if not os.path.isdir(platform_root):
        log("deploy", f"ERROR: platform root not found: {platform_root}")
        log("deploy", "       set --platform-root and re-run, or use 'python3 build.py build'.")
        raise SystemExit(1)

    log("deploy", f"copy case folder -> {dest}")
    os.makedirs(dest, exist_ok=True)
    deploy_allow = {KERNEL_BIN, KERNEL_DUMP, "config.toml", f"run_{FOLDER}.toml"}
    outputs = [a for a in case_params.artifacts if a.role == "output"]
    for a in case_params.artifacts:
        if a.role == "input":
            deploy_allow.add(a.name + ".bin")
        elif a.role == "output":
            if len(outputs) == 1:
                deploy_allow.add("golden.bin")
            else:
                deploy_allow.add("golden_" + a.name + ".bin")
    log("deploy", f"deploy_allow = {sorted(deploy_allow)}")
    for entry in os.listdir(SCRIPT_DIR):
        if entry in deploy_allow or entry.startswith("out"):
            src = os.path.join(SCRIPT_DIR, entry)
            dst = os.path.join(dest, entry)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                log("deploy", f"  [dir]  {entry}/")
            else:
                sz = os.path.getsize(src)
                shutil.copy2(src, dst)
                log("deploy", f"  [{sz:6d} B]  {entry} -> {dst}")
    # Defensive: strip any stale build cruft / host sources / ccec intermediates
    # that slipped through. (Only the allow-listed files above should remain.)
    for pat in ("*.o", "*.o.d", "*.cpp", "*.dump.o"):
        import glob

        for f in glob.glob(os.path.join(dest, pat)):
            os.remove(f)

    # The platform needs its env sourced first (sourceme david_v121 exports the
    # runtime/sim env that scripts/run.py depends on). Source it in the same
    # shell as run.py so the exports are visible to it.
    sourceme = os.path.join(platform_root, "sourceme")
    if not os.path.exists(sourceme):
        log("deploy", f"ERROR: sourceme not found at {sourceme}")
        raise SystemExit(1)

    # Clear the platform's output/log dir (keep the folder itself) so only the
    # newest run's logs survive.
    out_dir = os.path.join(platform_root, "out")
    if os.path.isdir(out_dir):
        log("deploy", f"clearing {out_dir}")
        for entry in os.listdir(out_dir):
            p = os.path.join(out_dir, entry)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass

    log("deploy", f"source sourceme david_v121  (in {platform_root})")
    log("deploy", f"python3 scripts/run.py -r ./{run_toml_rel} -f ./{case_name}  (in {platform_root})")
    shell_cmd = f"source sourceme david_v121 && {sys.executable} scripts/run.py -r ./{run_toml_rel} -f ./{case_name}"
    run(["bash", "-c", shell_cmd], cwd=platform_root)

    # After the run, remove empty log files the platform left behind in out/
    # (clutter). Equivalent to: find out/ -size 0 -delete. Keep non-empty logs.
    out_dir = os.path.join(platform_root, "out")
    if os.path.isdir(out_dir):
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for name in files:
                p = os.path.join(root, name)
                try:
                    if os.path.getsize(p) == 0:
                        os.remove(p)
                except OSError:
                    pass
        log("deploy", f"removed empty files from {out_dir}")

    log("deploy", f"{case_name} done")


def main():
    parser = argparse.ArgumentParser(description="A6 TQUANT build + deploy driver")
    parser.add_argument(
        "action",
        nargs="?",
        default="all",
        choices=["build", "deploy", "all"],
        help="build = compile+golden+tomls; deploy = copy+run; all = both (default)",
    )
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        help="case index (0=vadd 128x128, 1=hif4 128x128, 2=hif4 64x128, ...). "
        "Default: 1 (hif4 128x128). Use --list to see all.",
    )
    parser.add_argument("--list", action="store_true", help="list all available cases and exit")
    # Search multiple known locations for the regression platform
    _platform_candidates = [
        os.path.expanduser("~/regression_platform/davinci_model_platform"),
        "/home/z84371014/regression_platform/davinci_model_platform",
        os.path.expanduser("~z84371014/regression_platform/davinci_model_platform"),
    ]
    _default_platform = next((p for p in _platform_candidates if os.path.isdir(p)), _platform_candidates[0])
    parser.add_argument(
        "--platform-root", default=_default_platform, help=f"regression platform root (default: {_default_platform})"
    )
    parser.add_argument("--no-deploy", action="store_true", help="alias for action=build (skip deploy+run)")
    args = parser.parse_args()

    if args.list:
        print("Available cases:")
        for i, p in enumerate(gen_data.CASE_PARAMS):
            tag = " (default)" if i == 1 else ""
            print(f"  [{i}] {p.kind:5s} {p.valid_rows}×{p.valid_cols}{tag}")
        return

    case_idx = args.case if args.case is not None else 1  # default: hif4 128x128
    case_params = gen_data.CASE_PARAMS[case_idx]

    action = "build" if args.no_deploy else args.action

    case_name = case_params.case_name
    n_out = sum(1 for a in case_params.artifacts if a.role == "output")
    log(
        "build",
        f"folder={FOLDER}  case=[{case_idx}] {case_params.kind} "
        f"{case_params.valid_rows}×{case_params.valid_cols}  "
        f"outputs={n_out}  case_name={case_name}",
    )

    os.chdir(SCRIPT_DIR)

    do_build = action in ("build", "all")
    do_deploy = action in ("deploy", "all")

    if do_build:
        compile_kernel(case_params.kind, case_params.valid_rows, case_params.valid_cols)
        gen_disasm_dump(args.platform_root)  # pipeline 2: ccec -> disassembly dump
        gen_golden(case_idx)  # generate goldens for selected case
        write_tomls(case_params)  # config.toml for selected case
        log("build", f"build [{case_idx}] {case_params.kind} {case_params.valid_rows}×{case_params.valid_cols} done")
        log("build", f"artifacts in {SCRIPT_DIR}:")
        n_out = len([a for a in case_params.artifacts if a.role == "output"])
        golden_names = (
            ["golden.bin"]
            if n_out == 1
            else ["golden_" + a.name + ".bin" for a in case_params.artifacts if a.role == "output"]
        )
        names = (
            [KERNEL_BIN, KERNEL_DUMP, "config.toml", f"run_{FOLDER}.toml"]
            + [a.name + ".bin" for a in case_params.artifacts if a.role == "input"]
            + golden_names
        )
        for n in names:
            if os.path.exists(n):
                print(f"    {n}")

    if do_deploy:
        deploy_and_run(args.platform_root, case_params)


if __name__ == "__main__":
    main()
