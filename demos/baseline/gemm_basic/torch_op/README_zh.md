# 自定义 PyTorch 算子（CUBE GEMM）示例

本示例展示如何把 `demos/baseline/gemm_basic` 的 GEMM（CUBE）kernel 暴露为 PyTorch 自定义算子，并通过 `torch_npu` 从 Python 使用 `torch.ops.npu.pto_gemm_basic` 调用。

## 目录结构

```
demos/baseline/gemm_basic/torch_op/
├── gemm_extension/            # Python 包入口（模块加载）
├── csrc/
│   ├── kernel/                # PTO CUBE kernel 实现（复用 ../gemm_basic_impl.hpp）
│   └── host/                  # Host 侧 PyTorch 算子注册
├── test/                      # 最小化 Python 测试
├── CMakeLists.txt             # 构建配置
└── setup.py                   # Wheel 构建脚本
```

## 输入约定

对齐 `demos/baseline/gemm_basic` 的数据布局：`b` 采用 **DN**（等价于 `b.t().contiguous()`）。

- `a`：`[512, 2048]`，`float16`，ND
- `b_dn`：`[1536, 2048]`，`float16`（即原始 `b[k,n]` 的 `b.t().contiguous()`）
- 输出：`[512, 1536]`，`float32`

## 构建与运行（NPU）

```bash
source $HOME/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa

cd demos/baseline/gemm_basic/torch_op
python3 setup.py bdist_wheel
python3 -m pip install -U dist/*.whl

cd test
python3 test.py
```

