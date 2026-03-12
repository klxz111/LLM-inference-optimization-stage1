import time
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ================= 配置 =================
MODEL_NAME = "gpt2"
# 用国内镜像（如果能访问）
# import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROMPTS = [
    "What is MLsys?",
    "How to optimize LLM inference?",
    "Explain PagedAttention briefly",
    "What is TensorRT?",
    "How to reduce GPU memory?"
] * 5  # 25条prompt，体现吞吐率

# ================= vLLM 推理 =================
def run_vllm():
    print("\n=== 运行 vLLM 优化版 ===")
    # 禁用分布式，避免WSL崩溃
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.7,
        dtype="float16",
        tensor_parallel_size=1  # 强制单卡，避免NCCL问题
    )
    sampling_params = SamplingParams(max_tokens=30, temperature=0.7)

    start = time.time()
    outputs = llm.generate(PROMPTS, sampling_params)
    total_time = time.time() - start

    # 正确统计token数（vLLM 0.5.0+ 标准API）
    total_tokens = 0
    for output in outputs:
        # RequestOutput -> List[CompletionOutput]
        for completion in output.outputs:
            total_tokens += len(completion.token_ids)
    
    throughput = total_tokens / total_time
    print(f"vLLM 完成 | 总耗时: {total_time:.2f}s | 总token: {total_tokens} | 吞吐: {throughput:.2f} tokens/s")
    return throughput, total_time

# ================= 原生 PyTorch Baseline =================
def run_baseline():
    print("\n=== 运行 原生 PyTorch Baseline ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda().eval()
    except Exception as e:
        print(f"模型下载失败，跳过Baseline: {e}")
        return 0, 0

    total_tokens = 0
    start = time.time()
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=True)
        total_tokens += out.shape[1] - inputs.input_ids.shape[1]
    total_time = time.time() - start
    throughput = total_tokens / total_time
    print(f"Baseline 完成 | 总耗时: {total_time:.2f}s | 总token: {total_tokens} | 吞吐: {throughput:.2f} tokens/s")
    return throughput, total_time

# ================= 主函数 =================
if __name__ == "__main__":
    vllm_throughput, vllm_time = run_vllm()
    baseline_throughput, baseline_time = run_baseline()

    if vllm_throughput > 0 and baseline_throughput > 0:
        print("\n" + "="*50)
        print("📊 最终性能对比")
        print("="*50)
        print(f"{'':<15} {'吞吐率(tokens/s)':<20} {'耗时(s)':<15}")
        print("-"*50)
        print(f"{'vLLM 优化':<15} {vllm_throughput:<20.2f} {vllm_time:<15.2f}")
        print(f"{'原生 PyTorch':<15} {baseline_throughput:<20.2f} {baseline_time:<15.2f}")
        print("="*50)
        print(f"🚀 vLLM 加速比: {vllm_throughput / baseline_throughput:.2f}x")
        print("🎉 项目完成！")
