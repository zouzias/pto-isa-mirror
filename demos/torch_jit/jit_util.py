import os
import subprocess
import ctypes

import torch


def compile_cpp(src_path, verbose=False, timeout=10):
    assert src_path.endswith(".cpp")
    lib_path = src_path.removesuffix(".cpp") + ".so"

    ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
    PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "--npu-arch=dav-2201",
        "-O2",
        "-std=c++17",
        f"-I{ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw",
        f"-I{ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl",
        f"-I{ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{PTO_LIB_PATH}/include",
        f"-I{PTO_LIB_PATH}/include/common",
    ]

    command = ["bisheng", *flags, src_path, "-o", lib_path]
    if verbose:
        print(f"compile {src_path} with command: \n", command)

    try:
        ret = subprocess.run(command, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, check_type=True):
    lib = ctypes.CDLL(lib_path)

    if check_type:  # otherwise will get segfault for mismatched type
        # TODO: generate checker according to cpp `void call_kernel` signature
        lib.call_kernel.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_void_p,  # z
            ctypes.c_int,     # N
        ]
        lib.call_kernel.restype = None

    default_block_dim = 20  # 910B4, TODO: query platform information
    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def add_func(
        x,
        y,
        z,
        block_dim=default_block_dim,
        stream_ptr=default_stream_ptr
        ):
        N = x.numel()
        # TODO: customize call args according to cpp `void call_kernel` signature
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(z),
            N
        )

    return add_func


def jit_compile(src_path, clean_up=True):
    lib_path = compile_cpp(src_path)
    func = load_lib(lib_path)
    if clean_up:
        os.remove(lib_path)
    return func
