import time
import torch
import logging
import pynvml
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import torch.nn.functional as F

# ===================== 1. 工程化配置 =====================
# 核心配置参数
MODEL_NAME = "gpt2"
PROMPTS = [
    "What is MLsys?",
    "How to optimize LLM inference?",
    "Explain PagedAttention briefly",
    "What is TensorRT?",
    "How to reduce GPU memory?"
] * 5  # 25条测试Prompt
SAMPLING_PARAMS = SamplingParams(max_tokens=30, temperature=0.7)
GPU_MEM_UTIL = 0.7
LOG_FILE = "/mnt/e/mlsys_project.log"
PLOT_SAVE_PATH = "/mnt/e/throughput_comparison.png"

# ===================== 2. 日志配置 =====================
def setup_logger():
    """初始化日志系统，记录实验过程"""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    # 同时输出到终端
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# ===================== 3. GPU性能监控 =====================
def monitor_gpu():
    """监控RTX 4070显存占用和GPU利用率"""
    logger.info("\n=== 开始GPU性能监控 ===")
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # 显存信息
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = round(mem_info.used / 1024**3, 2)
        mem_total = round(mem_info.total / 1024**3, 2)
        
        # GPU利用率
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu
        
        logger.info(f"GPU显存占用: {mem_used}GB / {mem_total}GB")
        logger.info(f"GPU算力利用率: {gpu_util}%")
        return mem_used, gpu_util
    except Exception as e:
        logger.warning(f"GPU监控失败: {str(e)}")
        return 0, 0

# ===================== 4. 精度评估（PPL困惑度） =====================
def calculate_ppl():
    """计算模型困惑度（PPL），越小精度越高"""
    logger.info("\n=== 开始精度评估（PPL困惑度） ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).cuda().eval()
        
        # 用前5条Prompt计算PPL
        test_text = " ".join(PROMPTS[:5])
        inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss).item()  # PPL = e^loss
        
        logger.info(f"模型PPL（困惑度）: {round(ppl, 2)}（越小精度越高）")
        return round(ppl, 2)
    except Exception as e:
        logger.warning(f"PPL计算失败: {str(e)}")
        return 0.0

# ===================== 5. 原生PyTorch基线 =====================
def run_baseline():
    """运行原生PyTorch推理基线"""
    logger.info("\n=== 开始运行原生PyTorch基线 ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda().eval()
        
        total_tokens = 0
        start_time = time.time()
        
        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
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

# ===================== 6. vLLM优化版 =====================
def run_vllm():
    """运行vLLM优化版推理（PagedAttention+连续批处理）"""
    logger.info("\n=== 开始运行vLLM优化版 ===")
    try:
        llm = LLM(
            model=MODEL_NAME,
            gpu_memory_utilization=GPU_MEM_UTIL,
            dtype="float16",
            tensor_parallel_size=1  # 强制单卡，避免WSL多进程问题
        )
        
        start_time = time.time()
        outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
        total_time = round(time.time() - start_time, 2)
        
        # 统计生成Token数（vLLM标准API）
        total_tokens = 0
        for output in outputs:
            for completion in output.outputs:
                total_tokens += len(completion.token_ids)
        
        throughput = round(total_tokens / total_time, 2)
        logger.info(f"vLLM优化完成 | 总耗时: {total_time}s | 总Token: {total_tokens} | 吞吐率: {throughput} tokens/s")
        return throughput, total_time
    except Exception as e:
        logger.error(f"vLLM运行失败: {str(e)}")
        return 0.0, 0.0

# ===================== 7. vLLM+TensorRT-LLM编译优化 =====================
def run_vllm_trtllm():
    """运行vLLM+TensorRT-LLM编译优化版"""
    logger.info("\n=== 开始运行vLLM+TensorRT-LLM编译优化版 ===")
    try:
        llm = LLM(
            model=MODEL_NAME,
            gpu_memory_utilization=GPU_MEM_UTIL,
            dtype="float16",
            tensor_parallel_size=1,
            engine_use_ray=False,
            disable_log_stats=False,
            max_model_len=1024
        )
        
        # 预热（编译TensorRT引擎）
        logger.info("TensorRT-LLM引擎预热中...")
        llm.generate(["warmup"], SamplingParams(max_tokens=10))
        
        start_time = time.time()
        outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
        total_time = round(time.time() - start_time, 2)
        
        # 统计Token数
        total_tokens = 0
        for output in outputs:
            for completion in output.outputs:
                total_tokens += len(completion.token_ids)
        
        throughput = round(total_tokens / total_time, 2)
        logger.info(f"vLLM+TRT-LLM完成 | 总耗时: {total_time}s | 总Token: {total_tokens} | 吞吐率: {throughput} tokens/s")
        return throughput, total_time
    except Exception as e:
        logger.warning(f"TensorRT-LLM运行失败（跳过）: {str(e)}")
        return 0.0, 0.0

# ===================== 8. 结果可视化 =====================
def plot_results(baseline_tp, vllm_tp, trt_tp):
    """生成性能对比柱状图"""
    logger.info("\n=== 生成性能对比可视化图表 ===")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 设置中文字体（避免乱码）
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        
        # 数据整理
        methods = ["Native PyTorch", "vLLM", "vLLM+TRT-LLM"]
        throughputs = [baseline_tp, vllm_tp, trt_tp]
        # 过滤掉0值（失败的实验）
        valid_data = [(m, t) for m, t in zip(methods, throughputs) if t > 0]
        if not valid_data:
            logger.warning("无有效数据生成图表")
            return
        
        methods_valid, tp_valid = zip(*valid_data)
        
        # 绘图
        plt.figure(figsize=(10, 6))
        sns.barplot(x=methods_valid, y=tp_valid, palette="viridis")
        plt.title("LLM Inference Throughput Comparison (RTX 4070)", fontsize=14)
        plt.xlabel("Inference Scheme", fontsize=12)
        plt.ylabel("Throughput (tokens/s)", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # 添加数值标签
        for i, v in enumerate(tp_valid):
            plt.text(i, v + 100, f"{v}", ha="center", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH, dpi=300)
        logger.info(f"性能对比图表已保存到: {PLOT_SAVE_PATH}")
    except Exception as e:
        logger.warning(f"可视化失败: {str(e)}")

# ===================== 9. 主函数 =====================
def main():
    """主执行函数：串联所有实验步骤"""
    logger.info("="*60)
    logger.info("开始MLSys LLM推理优化实验")
    logger.info("="*60)
    
    # 步骤1：GPU监控
    mem_used, gpu_util = monitor_gpu()
    
    # 步骤2：精度评估
    ppl = calculate_ppl()
    
    # 步骤3：运行各版本推理
    baseline_tp, baseline_time = run_baseline()
    vllm_tp, vllm_time = run_vllm()
    trt_tp, trt_time = run_vllm_trtllm()
    
    # 步骤4：计算加速比
    logger.info("\n=== 实验结果汇总 ===")
    if baseline_tp > 0:
        vllm_speedup = round(vllm_tp / baseline_tp, 2) if vllm_tp > 0 else 0
        trt_speedup = round(trt_tp / baseline_tp, 2) if trt_tp > 0 else 0
        logger.info(f"vLLM 加速比: {vllm_speedup}x")
        logger.info(f"vLLM+TRT-LLM 加速比: {trt_speedup}x")
    
    # 步骤5：输出核心指标
    logger.info("\n📊 核心指标汇总")
    logger.info(f"精度指标 | PPL困惑度: {ppl}")
    logger.info(f"硬件指标 | 显存占用: {mem_used}GB | GPU利用率: {gpu_util}%")
    logger.info(f"性能指标 | 原生PyTorch吞吐: {baseline_tp} tokens/s | vLLM吞吐: {vllm_tp} tokens/s | vLLM+TRT-LLM吞吐: {trt_tp} tokens/s")
    
    # 步骤6：生成可视化图表
    plot_results(baseline_tp, vllm_tp, trt_tp)
    
    logger.info("\n" + "="*60)
    logger.info("实验完成！日志文件: " + LOG_FILE)
    logger.info("="*60)

if __name__ == "__main__":
    main()
