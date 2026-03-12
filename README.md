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

## 📌 结论

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
