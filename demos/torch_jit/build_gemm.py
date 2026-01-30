import os
import subprocess
import tempfile

def build_gemm_so():
    root = os.path.dirname(__file__)
    build_dir = tempfile.mkdtemp(prefix="jit_gemm_")
    env = os.environ.copy()

    # Pick up the same SOC_VERSION your working project uses
    soc = env.get("SOC_VERSION", "ascend910b2")

    subprocess.check_call([
        "cmake",
        "-S", root,
        "-B", build_dir,
        f"-DSOC_VERSION={soc}",
        f"-DASCEND_CANN_PACKAGE_PATH={env.get('ASCEND_TOOLKIT_HOME','')}",
    ], env=env)
    

    subprocess.check_call([
        "cmake",
        "--build", build_dir,
        "-j"
    ], env=env)

    
    lib_path = os.path.join(build_dir, "libgemm_jit.so")
    if not os.path.exists(lib_path):
        raise RuntimeError(f"Build finished but {lib_path} not found. Directory contains: {os.listdir(build_dir)}")
    return lib_path
