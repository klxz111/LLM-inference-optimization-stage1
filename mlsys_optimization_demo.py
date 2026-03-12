import time
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== 1. 配置 =====================
MODEL_NAME = "gpt2"  # 用更小的模型，避免显存/版本问题
PROMPTS = [
    "What is MLsys?",
    "How to optimize LLM inference?",
    "Explain PagedAttention in vLLM"
] * 10  # 30条prompt，足够体现性能差异
SAMPLING_PARAMS = SamplingParams(max_tokens=50, temperature=0.7)

# ===================== 2. 原生 PyTorch Baseline =====================
def run_baseline():
    print("\n=== 原生 PyTorch Baseline ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda().eval()

    total_tokens = 0
    start_time = time.time()
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
        total_tokens += outputs.shape[1] - inputs.input_ids.shape[1]
    end_time = time.time()

    return {
        "throughput": total_tokens / (end_time - start_time),
        "latency": (end_time - start_time) / len(PROMPTS) * 1000,
        "memory": torch.cuda.max_memory_allocated() / 1024**3
    }

# ===================== 3. vLLM 优化（PagedAttention） =====================
def run_vllm():
    print("\n=== vLLM 优化（PagedAttention） ===")
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.7,
        dtype="float16"
    )

    start_time = time.time()
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    end_time = time.time()

    total_tokens = sum(len(output.output_tokens) for output in outputs)
    return {
        "throughput": total_tokens / (end_time - start_time),
        "latency": (end_time - start_time) / len(PROMPTS) * 1000,
        "memory": torch.cuda.max_memory_allocated() / 1024**3
    }

# ===================== 4. 主函数：运行对比 =====================
if __name__ == "__main__":
    results = {}
    results["Baseline"] = run_baseline()
    results["vLLM"] = run_vllm()

    # 打印对比表格
    print("\n" + "="*80)
    print("最终性能对比（RTX 4070 12GB）")
    print("="*80)
    print(f"{'方案':<20} {'吞吐率(tokens/s)':<20} {'首token延迟(ms)':<20} {'峰值显存(GB)':<20}")
    print("-"*80)
    for name, res in results.items():
        print(f"{name:<20} {res['throughput']:<20.2f} {res['latency']:<20.2f} {res['memory']:<20.2f}")
    print("="*80)
    print(f"vLLM 相比 Baseline 加速比: {results['vLLM']['throughput']/results['Baseline']['throughput']:.2f}x")
    print(f"vLLM 相比 Baseline 显存节省: {(1 - results['vLLM']['memory']/results['Baseline']['memory'])*100:.1f}%")
    print("🎉 项目完成！")
