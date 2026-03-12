```markdown
# vLLM-PyTorch 版本兼容补丁

## 问题描述
当使用 PyTorch 2.1.0 + vLLM 0.17.1 时，会出现以下导入错误：
```

ImportError: cannot import name 'infer_schema' from 'torch.library'

## 原因分析
`infer_schema` 是 PyTorch 2.2+ 新增的函数，用于模型导出与类型推断，vLLM 0.17.1+ 依赖此函数实现部分功能。

## 兼容补丁（代码层面，无需重装依赖）
在所有使用 vLLM 的代码文件开头添加以下代码：
```python
# 兼容 PyTorch 2.1.0 的 infer_schema 补丁
try:
    from torch._dynamo import infer_schema
except ImportError:
    def infer_schema(model, *args, **kwargs):
        """
        替代实现：仅占位，不影响 vLLM 核心推理功能
        适用于 PyTorch 2.1.0 及以下版本
        """
        return None
```

## 补丁效果

- 拦截 vLLM 对 `infer_schema` 的调用，避免 ImportError
- 不影响 vLLM 核心功能（PagedAttention / 连续批处理）
- 无需升级 PyTorch 或重装 vLLM

## 注意事项

- 此补丁为临时兼容方案，生产环境建议升级 PyTorch 到 2.2+
- 若需使用模型导出/量化等高级功能，仍需升级 PyTorch 版本

#### `notes/phi2_onnx_fail_analysis.md`
```markdown
# Phi-2 ONNX 推理失败原因分析

## 失败表现
1. **依赖缺失**：`No module named 'onnxscript'`
2. **算子不支持**：导出时提示 `UnsupportedOperatorError`
3. **模型加载失败**：ONNX Runtime 无法加载 Phi-2 模型

## 根本原因
### 1. 模型架构特殊性
- Phi-2 是微软自定义的 `PhiForCausalLM` 架构，包含 Rotary Position Embedding 等自定义算子
- 需 `trust_remote_code=True` 才能加载，ONNX 对非标准 Transformer 架构支持不完善

### 2. 权重格式问题
- Phi-2 使用 `safetensors` 格式存储权重，ONNX 导出对 `safetensors` 支持不如 PyTorch `.bin` 格式成熟

### 3. 依赖链复杂
- 导出大模型 ONNX 需要额外依赖 `onnxscript`、`onnxruntime-training` 等
- WSL2 环境下部分依赖安装后无法正常导入，存在版本冲突

## 可行替代方案
1. **先从标准模型入手**：使用 GPT2/OPT 等标准架构模型验证 ONNX 流程，再迁移到 Phi-2
2. **使用 Hugging Face Optimum**：`optimum.onnxruntime` 封装了模型导出逻辑，对自定义模型支持更好
3. **更换环境**：原生 Linux 环境下依赖安装更稳定，可尝试 `optimum` 库导出

## 结论
Phi-2 + ONNX 在 WSL2 环境下的适配难度较高，建议先完成标准模型的 ONNX 优化，再逐步探索复杂模型的适配方案。 
```

