import matplotlib.pyplot as plt
import seaborn as sns

# 你的实验数据
methods = ["Native PyTorch", "vLLM"]
throughputs = [143.13, 4411.76]

# 绘图（修复 seaborn 警告）
plt.figure(figsize=(10, 6))
sns.barplot(x=methods, y=throughputs, hue=methods, palette="viridis", legend=False)
plt.title("LLM Inference Throughput Comparison (RTX 4070)", fontsize=14)
plt.xlabel("Inference Scheme", fontsize=12)
plt.ylabel("Throughput (tokens/s)", fontsize=12)

# 添加数值标签
for i, v in enumerate(throughputs):
    plt.text(i, v + 100, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("/mnt/e/throughput_comparison_clean.png", dpi=300)
print("✅ 新图表已保存到 E 盘：throughput_comparison_clean.png")
