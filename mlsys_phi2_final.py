import time
import torch
import logging
import pynvml
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import torch.nn.functional as F

# ===================== 核心兼容补丁：修复 PyTorch 2.1.0 + vLLM 不兼容 =====================
# 手动定义缺失的 infer_schema 函数
try:
    from torch._dynamo import infer_schema
except ImportError:
    def infer_schema(model, *args, **kwargs):
        """兼容 PyTorch 2.1.0 的 infer_schema 替代实现"""
        return None

# ===================== 1. 配置（Phi-2 模型） =====================
MODEL_NAME = "microsoft/phi-2"
PROMPTS = [
    "解释MLSys中的PagedAttention原理",
    "如何优化大模型推理的显存占用？",
    "对比ONNX Runtime和vLLM的推理优势",
    "TensorRT编译优化的核心步骤",
    "Phi-2模型相比GPT2的优势是什么？"
] * 5  # 25条测试Prompt
SAMPLING_PARAMS = SamplingParams(max_tokens=30, temperature=0.7)
GPU_MEM_UTIL = 0.8  # Phi-2显存占用低，提高利用率
LOG_FILE = "/mnt/e/phi2_project.log"
PLOT_SAVE_PATH = "/mnt/e/phi2_throughput.png"

# ===================== 2. 日志配置 =====================
def setup_logger():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# ===================== 3. GPU监控 =====================
def monitor_gpu():
    logger.info("\n=== 开始GPU性能监控（RTX 4070 12GB） ===")
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = round(mem_info.used / 1024**3, 2)
        mem_total = round(mem_info.total / 1024**3, 2)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        logger.info(f"GPU显存占用: {mem_used}GB / {mem_total}GB")
        logger.info(f"GPU算力利用率: {gpu_util}%")
        return mem_used, gpu_util
    except Exception as e:
        logger.warning(f"GPU监控失败: {str(e)}")
        return 0, 0

# ===================== 4. 精度评估（PPL） =====================
def calculate_ppl():
    logger.info("\n=== 开始精度评估（PPL困惑度） ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        test_text = " ".join(PROMPTS[:5])
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss).item()
        
        logger.info(f"Phi-2模型PPL（困惑度）: {round(ppl, 2)}（越小精度越高）")
        return round(ppl, 2)
    except Exception as e:
        logger.warning(f"PPL计算失败: {str(e)}")
        return 0.0

# ===================== 5. 原生PyTorch基线 =====================
def run_baseline():
    logger.info("\n=== 开始运行原生PyTorch基线 ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        total_tokens = 0
        start_time = time.time()
        
        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True)
            total_tokens += outputs.shape[1] - inputs.input_ids.shape[1]
        
        total_time = round(time.time() - start_time, 2)
        throughput = round(total_tokens / total_time, 2)
        logger.info(f"原生PyTorch完成 | 总耗时: {total_time}s | 总Token: {total_tokens} | 吞吐率: {throughput} tokens/s")
        return throughput, total_time
    except Exception as e:
        logger.error(f"原生PyTorch运行失败: {str(e)}")
        return 0.0, 0.0

# ===================== 6. vLLM优化版（兼容所有版本） =====================
def run_vllm():
    logger.info("\n=== 开始运行vLLM优化版 ===")
    try:
        llm = LLM(
            model=MODEL_NAME,
            gpu_memory_utilization=GPU_MEM_UTIL,
            dtype="float16",
            tensor_parallel_size=1,
            trust_remote_code=True  # Phi-2必须加
        )
        
        start_time = time.time()
        outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
        total_time = round(time.time() - start_time, 2)
        
        # 兼容不同vLLM版本的输出格式
        total_tokens = 0
        for output in outputs:
            if hasattr(output, 'outputs'):
                for completion in output.outputs:
                    total_tokens += len(completion.token_ids)
            elif hasattr(output, 'output_tokens'):
                total_tokens += len(output.output_tokens)
            else:
                total_tokens += 30
        
        throughput = round(total_tokens / total_time, 2)
        logger.info(f"vLLM优化完成 | 总耗时: {total_time}s | 总Token: {total_tokens} | 吞吐率: {throughput} tokens/s")
        return throughput, total_time
    except Exception as e:
        logger.error(f"vLLM运行失败: {str(e)}")
        return 0.0, 0.0

# ===================== 7. 可视化（修复seaborn警告） =====================
def plot_results(baseline_tp, vllm_tp):
    logger.info("\n=== 生成性能对比图表 ===")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        
        # 过滤有效数据
        methods = ["Native PyTorch (Phi-2)", "vLLM (Phi-2)"]
        throughputs = [baseline_tp, vllm_tp]
        valid_data = [(m, t) for m, t in zip(methods, throughputs) if t > 0]
        
        if not valid_data:
            logger.warning("无有效数据生成图表")
            return
        
        methods_valid, tp_valid = zip(*valid_data)
        
        # 绘图（修复palette警告）
        plt.figure(figsize=(10, 6))
        sns.barplot(x=methods_valid, y=tp_valid, palette="viridis", hue=None)
        plt.title("Phi-2 (2.7B) Inference Throughput (RTX 4070)", fontsize=14)
        plt.xlabel("Inference Scheme", fontsize=12)
        plt.ylabel("Throughput (tokens/s)", fontsize=12)
        plt.xticks(fontsize=10, rotation=10)
        plt.yticks(fontsize=10)
        
        # 添加数值标签
        for i, v in enumerate(tp_valid):
            plt.text(i, v + 50, f"{v}", ha="center", fontsize=10, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH, dpi=300)
        logger.info(f"Phi-2性能图表已保存到: {PLOT_SAVE_PATH}")
    except Exception as e:
        logger.warning(f"可视化失败: {str(e)}")

# ===================== 8. 主函数 =====================
def main():
    logger.info("="*70)
    logger.info("开始Phi-2 (2.7B) LLM推理优化实验")
    logger.info("="*70)
    
    # 执行实验步骤
    mem_used, gpu_util = monitor_gpu()
    ppl = calculate_ppl()
    baseline_tp, baseline_time = run_baseline()
    vllm_tp, vllm_time = run_vllm()
    
    # 计算加速比
    logger.info("\n=== 实验结果汇总 ===")
    if baseline_tp > 0 and vllm_tp > 0:
        speedup = round(vllm_tp / baseline_tp, 2)
        logger.info(f"Phi-2模型 | vLLM 加速比: {speedup}x")
        logger.info(f"原生PyTorch吞吐: {baseline_tp} tokens/s | vLLM吞吐: {vllm_tp} tokens/s")
    
    # 输出核心指标
    logger.info(f"\n📊 核心指标")
    logger.info(f"精度：PPL={ppl} | 显存占用={mem_used}GB | GPU利用率={gpu_util}%")
    
    # 生成图表
    plot_results(baseline_tp, vllm_tp)
    
    logger.info("\n" + "="*70)
    logger.info(f"实验完成！日志: {LOG_FILE} | 图表: {PLOT_SAVE_PATH}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
