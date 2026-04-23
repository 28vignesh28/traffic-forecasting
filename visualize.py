import torch
import torch.nn as nn
import yaml
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_dataloaders, load_traffic, load_static_adj
from src.utils import set_seed
import pandas as pd
import matplotlib.dates as mdates
from models.amc_dstgnn import AMC_DSTGNN
from models.cadgt import CADGT
from models.camt_gatformer import TrafficModel as CAMT
from models.st_acenet import ST_ACENet


def load_model(
    model_name,
    config,
    nodes,
    features,
    window,
    horizon,
    short_horizon,
    adj_static,
    device,
):
    if model_name == "CADGT":
        overrides = config.get("model_overrides", {}).get("CADGT", {})
        d_model = overrides.get("hidden_dim", config["model_defaults"]["hidden_dim"])
        model = CADGT(
            num_nodes=nodes,
            seq_len=window,
            future_len=horizon,
            ctx_dim=features - 1,
            d_model=d_model,
            static_adj=adj_static,
        ).to(device)
        path = "saved_models/cadgt_best.pth"

    elif model_name == "CAMT":
        overrides = config.get("model_overrides", {}).get("CAMT", {})
        shrt_hz = overrides.get("short_horizon", short_horizon)
        model = CAMT(
            nodes=nodes,
            nfeat=features,
            seq_len=window,
            short_horizon=shrt_hz,
            horizon=horizon,
        ).to(device)
        path = "saved_models/camt_best.pth"

    elif model_name == "AMC_DSTGNN":
        overrides = config.get("model_overrides", {}).get("AMC_DSTGNN", {})
        h_dim = overrides.get("hidden_dim", 128)
        model = AMC_DSTGNN(
            nfeat=features, N=nodes, hidden_dim=h_dim, horizon=horizon
        ).to(device)
        path = "saved_models/amc_dstgnn_best.pth"

    elif model_name == "ST_ACENet":
        overrides = config.get("model_overrides", {}).get("ST_ACENet", {})
        h_dim = overrides.get("hidden_dim", 64)
        model = ST_ACENet(
            nfeat=features,
            N=nodes,
            hidden_dim=h_dim,
            horizon=horizon,
            static_adj=adj_static,
        ).to(device)
        path = "saved_models/st_acenet_best.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Visualize Traffic Forecast Models")
    parser.add_argument(
        "--node", type=int, default=10, help="Node ID to plot (0 to 206)"
    )
    parser.add_argument(
        "--start_batch", type=int, default=0, help="Starting test batch"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=4,
        help="Number of continuous batches to plot",
    )
    parser.add_argument(
        "--horizon_idx",
        type=int,
        default=11,
        help="Horizon index (0 to 11, where 11 is 60 mins)",
    )
    args = parser.parse_args()

    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader, _, scaler, ds_info = get_dataloaders(config)

    _, timestamps = load_traffic(config["data"]["traffic_path"])

    total_len = len(timestamps)
    train_end = int(0.7 * total_len)
    val_end = int(0.8 * total_len)

    test_timestamps = timestamps[val_end:]

    nodes = ds_info["num_nodes"]
    features = ds_info["num_features"]
    window = config["training"]["window"]
    horizon = config["training"]["horizon"]
    short_horizon = 3

    adj_static = load_static_adj(config["data"]["adj_path"])
    adj_tensor = torch.FloatTensor(adj_static).to(device)

    models_to_test = ["CADGT", "CAMT", "AMC_DSTGNN", "ST_ACENet"]
    loaded_models = {}

    for name in models_to_test:
        model, ckpt = load_model(
            name,
            config,
            nodes,
            features,
            window,
            horizon,
            short_horizon,
            adj_static,
            device,
        )
        loaded_models[name] = model

    test_iter = iter(test_loader)

    for _ in range(args.start_batch):
        next(test_iter)

    all_y = []
    all_preds_list = {name: [] for name in models_to_test}

    with torch.no_grad():
        for _ in range(args.num_batches):
            try:
                x_b, y_b = next(test_iter)
            except StopIteration:
                break

            x_b, y_b = x_b.to(device), y_b.to(device)
            all_y.append(y_b)

            for name, model in loaded_models.items():
                if name == "CADGT":
                    preds = model(x_b)
                elif name == "CAMT":
                    _, preds = model(x_b, adj_tensor)
                elif name == "AMC_DSTGNN":
                    preds = model(x_b, adj_tensor, teacher_forcing_ratio=0.0)
                elif name == "ST_ACENet":
                    preds, _ = model(x_b)

                preds = scaler.inverse_transform(preds)
                all_preds_list[name].append(preds.cpu().numpy())

    if len(all_y) == 0:
        raise ValueError("No test data found for the selected batches.")

    y = torch.cat(all_y, dim=0).to(device)
    truth = scaler.inverse_transform(y).cpu().numpy()

    all_preds = {}
    for name in models_to_test:
        all_preds[name] = np.concatenate(all_preds_list[name], axis=0)

    node = args.node
    h_idx = args.horizon_idx

    batch_size = config["training"]["batch_size"]
    offset = (args.start_batch * batch_size) + window + h_idx
    times = test_timestamps[offset : offset + truth.shape[0]]

    plt.figure(figsize=(10, 4))

    plt.plot(
        times,
        truth[:, h_idx, node],
        color="#8cbce6",
        linewidth=1.0,
        alpha=0.9,
        label="Truth",
    )

    colors = {
        "CADGT": "#fcba79",
        "CAMT": "#81db8f",
        "AMC_DSTGNN": "#c49cce",
        "ST_ACENet": "#e88b8b",
    }

    for name, cmds in all_preds.items():
        plt.plot(
            times,
            cmds[:, h_idx, node],
            label=f"{name}",
            color=colors[name],
            linewidth=1.0,
            alpha=0.85,
        )

    plt.title(f"Node {node}", fontsize=12)
    plt.xlabel("Time", fontsize=11)
    plt.ylabel("Speed (mph)", fontsize=11)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

    plt.xticks(rotation=0)

    plt.legend(loc="lower left", fontsize=9, framealpha=1.0, edgecolor="lightgray")

    plt.tight_layout()

    os.makedirs("visualizations", exist_ok=True)
    out_path = f"visualizations/pred_node{node}_H{h_idx+1}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
