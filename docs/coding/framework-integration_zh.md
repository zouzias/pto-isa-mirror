# 推理框架集成

本文档说明如何将 PTO 算子集成到主流推理框架（PyTorch、TensorFlow、ONNX Runtime 等）。

---

## 1. 集成概览

### 1.1 集成架构

```
应用层 (Python/C++)
    ↓
框架层 (PyTorch/TensorFlow/ONNX)
    ↓
PTO 算子层 (C++/CUDA)
    ↓
硬件层 (NPU/GPU/CPU)
```

### 1.2 集成方式

| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Python 扩展** | 开发快 | 存在性能开销 | 原型验证 |
| **C++ 扩展** | 性能高 | 开发复杂 | 生产环境 |
| **JIT 编译** | 灵活 | 首次运行慢 | 动态图 |
| **AOT 编译** | 启动快 | 灵活性较低 | 静态图 |

---

## 2. PyTorch 集成

### 2.1 通过 torch_npu 集成

#### 步骤 1：定义算子 Schema

```cpp
// my_ops.cpp
#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

// 定义算子 schema
TORCH_LIBRARY_FRAGMENT(npu, m) {
  // 基础算子
  m.def("my_add(Tensor x, Tensor y) -> Tensor");

  // 带标量参数
  m.def("my_mul(Tensor x, Scalar alpha) -> Tensor");

  // 多输出
  m.def("my_split(Tensor x, int dim) -> (Tensor, Tensor)");

  // 原地算子
  m.def("my_relu_(Tensor(a!) self) -> Tensor(a!)");
}
```

#### 步骤 2：实现算子

```cpp
#include <pto/pto-inst.hpp>

// PTO Kernel 实现
__global__ __aicore__ void MyAddKernel(
    __gm__ float* out,
    __gm__ const float* x,
    __gm__ const float* y,
    uint32_t length) {

  int block_idx = get_block_idx();
  int block_num = get_block_num();

  int elements_per_block = (length + block_num - 1) / block_num;
  int start = block_idx * elements_per_block;
  int end = min(start + elements_per_block, length);

  using TileT = Tile<TileType::Vec, float, 16, 256>;

  for (int i = start; i < end; i += 16 * 256) {
    TileT tile_x, tile_y, tile_out;

    TLOAD(tile_x, GlobalTensor(x + i));
    TLOAD(tile_y, GlobalTensor(y + i));
    TADD(tile_out, tile_x, tile_y);
    TSTORE(GlobalTensor(out + i), tile_out);
  }
}

// PyTorch 算子实现
at::Tensor my_add_impl(const at::Tensor& x, const at::Tensor& y) {
  // 检查输入
  TORCH_CHECK(x.device() == y.device(), "Inputs must be on same device");
  TORCH_CHECK(x.sizes() == y.sizes(), "Inputs must have same shape");

  // 分配输出
  at::Tensor out = at::empty_like(x);

  // 获取数据指针
  float* out_ptr = out.data_ptr<float>();
  const float* x_ptr = x.data_ptr<float>();
  const float* y_ptr = y.data_ptr<float>();
  uint32_t length = x.numel();

  // 启动 kernel
  int block_num = 24;  // A3 核心数
  EXEC_KERNEL_CMD(MyAddKernel, block_num, out_ptr, x_ptr, y_ptr, length);

  return out;
}
```

#### 步骤 3：注册实现

```cpp
// 注册到 NPU 后端
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("my_add", TORCH_FN(my_add_impl));
}
```

#### 步骤 4：编译 Python 扩展

**setup.py**：
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='my_pto_ops',
    ext_modules=[
        CppExtension(
            name='my_pto_ops',
            sources=['my_ops.cpp'],
            include_dirs=[
                '/path/to/pto-isa/include',
                '/path/to/torch_npu/include',
            ],
            library_dirs=['/path/to/pto-isa/lib'],
            libraries=['pto'],
            extra_compile_args=['-std=c++20', '-O3'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

**编译**：
```bash
python setup.py install
```

#### 步骤 5：Python 使用

```python
import torch
import torch_npu
import my_pto_ops

# 创建输入
x = torch.randn(1024, 1024).npu()
y = torch.randn(1024, 1024).npu()

# 调用自定义算子
z = torch.ops.npu.my_add(x, y)

# 验证结果
expected = x + y
assert torch.allclose(z, expected, rtol=1e-5)

print("✓ Custom op works correctly!")
```

---

## 3. TensorFlow 集成

### 3.1 自定义 Op

#### 步骤 1：定义 Op

```cpp
// my_ops.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("MyAdd")
    .Input("x: float")
    .Input("y: float")
    .Output("z: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Custom add operator

Args:
  x: First input tensor
  y: Second input tensor

Returns:
  z: x + y
)doc");
```

#### 步骤 2：实现 Kernel

```cpp
#include "tensorflow/core/framework/op_kernel.h"
#include <pto/pto-inst.hpp>

class MyAddOp : public tensorflow::OpKernel {
 public:
  explicit MyAddOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    // 获取输入
    const tensorflow::Tensor& x = context->input(0);
    const tensorflow::Tensor& y = context->input(1);

    // 检查形状
    OP_REQUIRES(context, x.shape() == y.shape(),
                tensorflow::errors::InvalidArgument("Inputs must have same shape"));

    // 分配输出
    tensorflow::Tensor* z = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &z));

    // 调用 PTO kernel
    const float* x_ptr = x.flat<float>().data();
    const float* y_ptr = y.flat<float>().data();
    float* z_ptr = z->flat<float>().data();
    uint32_t length = x.NumElements();

    EXEC_KERNEL_CMD(MyAddKernel, 24, z_ptr, x_ptr, y_ptr, length);
  }
};

// 注册 kernel
REGISTER_KERNEL_BUILDER(
    Name("MyAdd").Device(tensorflow::DEVICE_NPU),
    MyAddOp);
```

#### 步骤 3：编译

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++ -std=c++17 -shared my_ops.cc -o my_ops.so \
    ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} \
    -I/path/to/pto-isa/include \
    -L/path/to/pto-isa/lib -lpto \
    -fPIC -O3
```

#### 步骤 4：Python 使用

```python
import tensorflow as tf

# 加载自定义 op
my_ops = tf.load_op_library('./my_ops.so')

# 使用
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.constant([[5.0, 6.0], [7.0, 8.0]])
z = my_ops.my_add(x, y)

print(z.numpy())
# [[6. 8.]
#  [10. 12.]]
```

---

## 4. ONNX Runtime 集成

### 4.1 自定义 Execution Provider

#### 步骤 1：定义 Kernel

```cpp
// my_onnx_ops.cc
#include "onnxruntime/core/framework/op_kernel.h"

class MyAddKernel : public onnxruntime::OpKernel {
 public:
  MyAddKernel(const onnxruntime::OpKernelInfo& info) : OpKernel(info) {}

  onnxruntime::Status Compute(onnxruntime::OpKernelContext* context) const override {
    // 获取输入
    const onnxruntime::Tensor* X = context->Input<onnxruntime::Tensor>(0);
    const onnxruntime::Tensor* Y = context->Input<onnxruntime::Tensor>(1);

    // 分配输出
    onnxruntime::Tensor* Z = context->Output(0, X->Shape());

    // 调用 PTO kernel
    const float* x_data = X->Data<float>();
    const float* y_data = Y->Data<float>();
    float* z_data = Z->MutableData<float>();
    size_t length = X->Shape().Size();

    EXEC_KERNEL_CMD(MyAddKernel, 24, z_data, x_data, y_data, length);

    return onnxruntime::Status::OK();
  }
};
```

#### 步骤 2：注册 Kernel

```cpp
ONNX_OPERATOR_KERNEL_EX(
    Add,
    kOnnxDomain,
    7,  // opset version
    kNpuExecutionProvider,
    MyAddKernel);
```

#### 步骤 3：Python 使用

```python
import onnxruntime as ort

# 注册自定义 EP
session_options = ort.SessionOptions()
session_options.register_custom_ops_library('my_onnx_ops.so')

# 创建会话
session = ort.InferenceSession(
    'model.onnx',
    session_options,
    providers=['NpuExecutionProvider', 'CPUExecutionProvider']
)

# 推理
outputs = session.run(None, {'input': input_data})
```

---

## 5. 性能优化

### 5.1 算子融合

```python
# PyTorch 示例：融合 Add + ReLU
@torch.jit.script
def fused_add_relu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(x + y)

# 使用自定义融合算子
torch.ops.npu.fused_add_relu(x, y)
```

### 5.2 内存优化

```cpp
// 原地算子
at::Tensor& my_add_inplace(at::Tensor& x, const at::Tensor& y) {
  // 直接修改 x，避免分配新内存
  float* x_ptr = x.data_ptr<float>();
  const float* y_ptr = y.data_ptr<float>();
  uint32_t length = x.numel();

  EXEC_KERNEL_CMD(MyAddInplaceKernel, 24, x_ptr, y_ptr, length);

  return x;
}
```

### 5.3 异步执行

```cpp
// 使用 CUDA Stream（或 NPU Stream）
at::Tensor my_add_async(const at::Tensor& x, const at::Tensor& y) {
  at::Tensor out = at::empty_like(x);

  // 获取当前 stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // 异步启动 kernel
  EXEC_KERNEL_ASYNC(MyAddKernel, 24, stream,
                    out.data_ptr<float>(),
                    x.data_ptr<float>(),
                    y.data_ptr<float>(),
                    x.numel());

  return out;
}
```

---

## 参考资料

- [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [TensorFlow Custom Ops](https://www.tensorflow.org/guide/create_op)
- [ONNX Runtime Custom Ops](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
- [Add 算子示例](../../demos/baseline/add/README.md)
- [调试指南](debug.md)
- [性能优化](opt.md)
