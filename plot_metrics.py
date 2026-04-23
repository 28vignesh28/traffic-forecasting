import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

log_dir = "logs"
models = ["CADGT", "CAMT", "AMC_DSTGNN", "ST_ACENet"]
colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]

metrics_data = {
    m: {
        5: {"MAE": 0, "MSE": 0, "RMSE": 0, "MAPE": 0},
        15: {"MAE": 0, "MSE": 0, "RMSE": 0, "MAPE": 0},
        30: {"MAE": 0, "MSE": 0, "RMSE": 0, "MAPE": 0},
        60: {"MAE": 0, "MSE": 0, "RMSE": 0, "MAPE": 0},
    }
    for m in models
}

for model in models:
    files = glob.glob(os.path.join(log_dir, f"{model}_test_*.log"))
    if not files:
        print(f"No test log found for {model}")
        continue
    latest_log = max(files, key=os.path.getctime)

    with open(latest_log, "r") as f:
        content = f.read()

    for horizon in [5, 15, 30, 60]:

        pattern = f"Performance \({horizon}-minute Horizon\):.*?MAE:\s+([\d\.]+).*?MSE:\s+([\d\.]+).*?RMSE:\s+([\d\.]+).*?MAPE:\s+([\d\.]+)%"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            metrics_data[model][horizon]["MAE"] = float(match.group(1))
            metrics_data[model][horizon]["MSE"] = float(match.group(2))
            metrics_data[model][horizon]["RMSE"] = float(match.group(3))
            metrics_data[model][horizon]["MAPE"] = float(match.group(4))


horizons = [5, 15, 30, 60]
x = np.arange(len(horizons))
width = 0.2

os.makedirs("visualizations", exist_ok=True)


def plot_metric(metric_name, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model in enumerate(models):
        values = [metrics_data[model][h][metric_name] for h in horizons]
        offset = (i - 1.5) * width
        ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.9)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Prediction Horizon")
    ax.set_title(f"{metric_name} Comparison Across Horizons")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} min" for h in horizons])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    out_path = f"visualizations/metric_comp_{metric_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


plot_metric("MAE", "Mean Absolute Error (mph)")
plot_metric("MSE", "Mean Squared Error (mph^2)")
plot_metric("RMSE", "Root Mean Square Error (mph)")
plot_metric("MAPE", "Mean Absolute Percentage Error (%)")
