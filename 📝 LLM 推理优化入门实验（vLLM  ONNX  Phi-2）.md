# 📝 LLM 推理优化入门实验（vLLM / ONNX / Phi-2）

> 一个面向新手的 LLM 推理优化入门项目，记录了在 WSL2 环境下对比 vLLM / ONNX Runtime 对 GPT2 / Phi-2 模型的推理性能优化过程，包含完整踩坑记录与工程化实践。

---

## 🎯 实验目标
1. 验证 **vLLM（PagedAttention + 连续批处理）** 对 LLM 推理的性能加速效果
2. 对比 **Phi-2（2.7B）** 与 GPT2 模型在精度、显存占用上的差异
3. 尝试 **ONNX Runtime** 静态图优化方案，探索跨框架部署
4. 记录 WSL2 环境下大模型推理的常见兼容性问题与解决方案

---

## 📊 核心成果速览

| 模型         | 优化方案     | 吞吐率 (tokens/s) | 加速比     | 精度 (PPL) | 状态                |
| ------------ | ------------ | ----------------- | ---------- | ---------- | ------------------- |
| GPT2         | 原生 PyTorch | 143.13            | 1x         | 290.4      | ✅ 完全跑通          |
| GPT2         | vLLM 优化    | 4411.76           | **30.82x** | 290.4      | ✅ 完全跑通          |
| Phi-2 (2.7B) | 原生 PyTorch | 32.08             | 1x         | **13.01**  | ✅ 完全跑通          |
| Phi-2 (2.7B) | vLLM 优化    | —                 | —          | —          | ❌ 环境兼容失败      |
| Phi-2 (2.7B) | ONNX Runtime | —                 | —          | —          | ❌ 依赖/架构适配失败 |

> **关键结论**：vLLM 在 GPT2 上实现了 30 倍以上推理加速；Phi-2 精度远超 GPT2，显存仅占 ~1GB，是消费级 GPU 部署的优质小模型选择。

---

## 🛠️ 环境配置

### 硬件与系统
- **GPU**：NVIDIA RTX 4070 12GB
- **系统**：Windows 11 + WSL2 (Ubuntu 22.04)
- **Python**：3.10
- **PyTorch**：2.1.0 (CUDA 12.1)

### 核心依赖
```bash
pip install vllm==0.17.1 transformers>=4.35.0 onnx onnxruntime-gpu matplotlib seaborn pynvml
```

### 模型信息
| 模型         | 来源                    | 特殊配置                 |
| ------------ | ----------------------- | ------------------------ |
| GPT2         | `openai-community/gpt2` | 无                       |
| Phi-2 (2.7B) | `microsoft/phi-2`       | `trust_remote_code=True` |

---

## 📁 仓库结构

```
llm-inference-optimization/
├── README.md                  # 项目说明文档
├── code/
│   ├── gpt2_vllm_final.py     # GPT2 + vLLM 完整优化代码
│   ├── phi2_baseline.py       # Phi-2 原生 PyTorch 基线代码
│   └── phi2_onnx_attempt.py   # Phi-2 + ONNX 尝试代码（含兼容补丁）
├── results/
│   ├── throughput_comparison.png  # GPT2 性能对比图
│   ├── phi2_baseline_metrics.png   # Phi-2 基线指标图
│   └── experiment_logs.md         # 关键实验日志摘录
└── notes/
    ├── wsl2_compatibility.md      # WSL2 环境踩坑笔记
    ├── version_conflict_fix.md    # vLLM-PyTorch 版本兼容补丁
    └── phi2_onnx_fail_analysis.md # Phi-2 ONNX 失败原因分析
```

---

## 🚀 快速运行

### 1. GPT2 + vLLM（推荐，100% 跑通）
```bash
# 激活环境
conda activate mlsys
export HF_ENDPOINT=https://hf-mirror.com

# 运行代码
python code/gpt2_vllm_final.py
```

### 2. Phi-2 基线（100% 跑通）
```bash
export TRANSFORMERS_TRUST_REMOTE_CODE=1
python code/phi2_baseline.py
```

### 3. Phi-2 + ONNX（仅尝试，可能失败）
```bash
pip install onnxscript  # 补全依赖
python code/phi2_onnx_attempt.py
```

---

## ⚠️ 踩坑笔记（核心避坑指南）

### 1. vLLM 与 PyTorch 2.1.0 版本冲突
- **问题**：`ImportError: cannot import name 'infer_schema' from 'torch.library'`
- **原因**：vLLM 0.17.1+ 依赖 PyTorch 2.2+ 新增的 `infer_schema` 函数，而 PyTorch 2.1.0 无此函数
- **解决方案**：在代码开头手动添加兼容补丁：
  ```python
  try:
      from torch._dynamo import infer_schema
  except ImportError:
      def infer_schema(model, *args, **kwargs):
          return None
  ```

### 2. WSL2 下 vLLM 引擎初始化失败
- **问题**：`Engine core initialization failed`
- **原因**：WSL2 对 NVIDIA GPU 多进程/显存管理支持不完善，vLLM 引擎依赖的 NCCL 通信、显存分配在虚拟机环境下易崩溃
- **解决方案**：
  - 强制单卡模式：`tensor_parallel_size=1`
  - 降低显存占用：`gpu_memory_utilization=0.7`
  - 终极方案：更换为原生 Linux 环境（推荐 Ubuntu 22.04）

### 3. Phi-2 ONNX 导出失败
- **问题**：`No module named 'onnxscript'` / 算子不支持
- **原因**：
  1. Phi-2 是微软自定义架构，需 `trust_remote_code=True`，ONNX 对非标准模型导出支持不完善
  2. 大模型动态维度与复杂算子需要额外依赖 `onnxscript`
- **解决方案**：
  - 补全依赖：`pip install onnxscript`
  - 简化模型：先从 GPT2 等标准模型练手，再尝试 Phi-2

### 4. Seaborn 可视化警告
- **问题**：`Passing 'palette' without assigning 'hue' is deprecated`
- **解决方案**：修改绘图代码，指定 `hue=None` 并关闭图例：
  ```python
  sns.barplot(x=methods_valid, y=tp_valid, palette="viridis", hue=None, legend=False)
  ```

---

## 📌 核心结论

1. **vLLM 优化效果显著**：在 GPT2 上实现 30.82 倍吞吐率提升，验证了 PagedAttention + 连续批处理是解决 LLM 推理瓶颈的核心方案
2. **Phi-2 是优质小模型**：PPL 从 GPT2 的 290.4 降至 13.01，显存仅占 ~1GB，适合消费级 GPU 部署
3. **WSL2 不适合大模型优化**：虚拟机环境存在多进程/显存管理缺陷，建议使用原生 Linux 或云 GPU 实例
4. **工程化思维重要性**：完整的日志、监控、可视化流程比单纯的性能提升更能体现工程能力

---

## 📝 后续方向
- 更换原生 Linux 环境，验证 vLLM/ONNX 在 Phi-2 上的优化效果
- 引入 AWQ/GPTQ 量化技术，进一步降低显存占用、提升推理速度
- 补充 TensorRT-LLM 编译优化，对比“动态批处理（vLLM）+ 静态编译（TensorRT）”的组合效果
- 完善 ONNX Runtime 推理 pipeline，实现完整 Token 生成逻辑

---

## 📜 License
MIT License — 欢迎 Star & Fork，一起完善新手友好的 LLM 优化教程！

---

### 📂 配套踩坑笔记文件

#### `notes/wsl2_compatibility.md`
```markdown
# WSL2 环境大模型推理踩坑笔记

## 核心问题
WSL2 本质是 Hyper-V 虚拟机，对 NVIDIA GPU 的直通支持不完善，存在以下限制：
1. **NCCL 通信问题**：多进程/多卡通信易失败，vLLM 引擎初始化依赖 NCCL
2. **显存管理缺陷**：显存分配/释放效率低，大模型易触发 OOM 或引擎崩溃
3. **依赖链割裂**：Python/CUDA/库版本与 Windows 半隔离，部分底层库（如 pynvml）功能异常

## 临时解决方案
1. **强制单卡模式**：vLLM 配置 `tensor_parallel_size=1`，避免多进程通信
2. **降低显存利用率**：设置 `gpu_memory_utilization=0.7`，预留更多显存缓冲
3. **避免复杂模型**：优先使用 GPT2/OPT 等标准架构模型，Phi-2 等自定义模型易适配失败
4. **重启环境**：若出现资源泄漏，重启 WSL2 终端释放显存

## 终极建议
生产级大模型推理优化实验请使用：
- 原生 Linux 系统（推荐 Ubuntu 22.04）
- 云 GPU 实例（如 AWS G5 / 阿里云 GPU 实例）
```

#### `notes/version_conflict_fix.md`
```markdown
# vLLM-PyTorch 版本兼容补丁

## 问题描述
当使用 PyTorch 2.1.0 + vLLM 0.17.1 时，会出现以下导入错误：
```
ImportError: cannot import name 'infer_schema' from 'torch.library'
```

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
```

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