"""
Visualization Utilities

Generates comparison plots for model performance across different
tasks and metrics. Supports:
  - Classification metrics (accuracy, F1)
  - STS metrics (Pearson, Spearman correlation)
  - Inference time comparison
  - Computational complexity (parameter count vs performance)
"""

import matplotlib.pyplot as plt
import os


def plot_metrics(results, save_dir="plots"):
    """
    Plot accuracy, F1, and time bar charts for classification results.

    Args:
        results: dict of {model_name: {accuracy, f1, time, ...}}
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    models = list(results.keys())
    accuracy = [results[m]["accuracy"] for m in models]
    f1 = [results[m]["f1"] for m in models]
    time = [results[m]["time"] for m in models]

    plt.figure()
    plt.bar(models, accuracy)
    plt.title("Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy.png")

    plt.figure()
    plt.bar(models, f1)
    plt.title("F1 Score Comparison")
    plt.xlabel("Models")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/f1.png")

    plt.figure()
    plt.bar(models, time)
    plt.title("Time Comparison")
    plt.xlabel("Models")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/time.png")


def plot_sts_metrics(results, save_dir="plots"):
    """
    Plot Pearson and Spearman correlation bar charts for STS results.

    Args:
        results: dict of {model_name: {pearson, spearman, ...}}
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    models = list(results.keys())
    pearson = [results[m]["pearson"] for m in models]
    spearman = [results[m]["spearman"] for m in models]

    plt.figure(figsize=(10, 5))
    x = range(len(models))
    width = 0.35

    plt.bar([i - width/2 for i in x], pearson, width, label="Pearson r", color="#4A90D9")
    plt.bar([i + width/2 for i in x], spearman, width, label="Spearman ρ", color="#E8A838")
    plt.title("STS-B Correlation Comparison")
    plt.xlabel("Models")
    plt.ylabel("Correlation Coefficient")
    plt.xticks(x, models, rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sts_correlations.png")


def plot_complexity(results, save_dir="plots"):
    """
    Plot parameter count vs performance trade-off.

    Visualises the relationship between computational complexity
    (number of parameters) and model performance (accuracy or F1).

    Args:
        results: dict of {model_name: {accuracy, f1, total_params, ...}}
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    models = list(results.keys())

    # Only plot if parameter info is available
    if "total_params" not in results[models[0]]:
        print("Warning: No parameter count data available for complexity plot.")
        return

    params = [results[m]["total_params"] / 1e6 for m in models]  # in millions
    accuracy = [results[m]["accuracy"] for m in models]
    f1 = [results[m]["f1"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy vs Parameters
    ax1.scatter(params, accuracy, s=100, c="#4A90D9", zorder=5)
    for i, model in enumerate(models):
        ax1.annotate(model, (params[i], accuracy[i]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)
    ax1.set_xlabel("Parameters (millions)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs Model Complexity")
    ax1.grid(True, alpha=0.3)

    # F1 vs Parameters
    ax2.scatter(params, f1, s=100, c="#E8A838", zorder=5)
    for i, model in enumerate(models):
        ax2.annotate(model, (params[i], f1[i]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)
    ax2.set_xlabel("Parameters (millions)")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 Score vs Model Complexity")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/complexity_tradeoff.png")
    print(f"Complexity trade-off plot saved to {save_dir}/complexity_tradeoff.png")
