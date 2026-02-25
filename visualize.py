import torch
import torch.nn as nn
import yaml
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_dataloaders, load_static_adj
from src.utils import set_seed
from models.amc_dstgnn import AMC_DSTGNN
from models.cadgt import CADGT
from models.camt_gatformer import TrafficModel as CAMT
from models.st_acenet import ST_ACENet


def load_model(model_name, config, nodes, features, window, horizon, short_horizon, adj_static, device):
    if model_name == "CADGT":
        overrides = config.get('model_overrides', {}).get('CADGT', {})
        d_model = overrides.get('hidden_dim', config['model_defaults']['hidden_dim'])
        model = CADGT(
            num_nodes=nodes, seq_len=window, future_len=horizon,
            ctx_dim=features - 1, d_model=d_model, static_adj=adj_static
        ).to(device)
        path = "saved_models/cadgt_best.pth"

    elif model_name == "CAMT":
        overrides = config.get('model_overrides', {}).get('CAMT', {})
        shrt_hz = overrides.get('short_horizon', short_horizon)
        model = CAMT(
            nodes=nodes, nfeat=features,
            seq_len=window, short_horizon=shrt_hz, horizon=horizon
        ).to(device)
        path = "saved_models/camt_best.pth"

    elif model_name == "AMC_DSTGNN":
        overrides = config.get('model_overrides', {}).get('AMC_DSTGNN', {})
        h_dim = overrides.get('hidden_dim', 128)
        model = AMC_DSTGNN(
            nfeat=features, N=nodes, hidden_dim=h_dim, horizon=horizon
        ).to(device)
        path = "saved_models/amc_dstgnn_best.pth"

    elif model_name == "ST_ACENet":
        overrides = config.get('model_overrides', {}).get('ST_ACENet', {})
        h_dim = overrides.get('hidden_dim', 64)
        model = ST_ACENet(
            nfeat=features, N=nodes, hidden_dim=h_dim, horizon=horizon, static_adj=adj_static
        ).to(device)
        path = "saved_models/st_acenet_best.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Visualize Traffic Forecast Models")
    parser.add_argument('--node', type=int, default=10, help="Node ID to plot (0 to 206)")
    parser.add_argument('--batch_idx', type=int, default=1, help="Which test batch to use")
    parser.add_argument('--horizon_idx', type=int, default=11, help="Horizon index (0 to 11, where 11 is 60 mins)")
    args = parser.parse_args()

    # Load configuration
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['training'].get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    _, test_loader, _, scaler, ds_info = get_dataloaders(config)
    
    nodes    = ds_info['num_nodes']
    features = ds_info['num_features']
    window   = config['training']['window']
    horizon  = config['training']['horizon']
    short_horizon = 3

    adj_static = load_static_adj(config['data']['adj_path'])
    adj_tensor = torch.FloatTensor(adj_static).to(device)

    models_to_test = ["CADGT", "CAMT", "AMC_DSTGNN", "ST_ACENet"]
    loaded_models = {}

    for name in models_to_test:
        model, ckpt = load_model(name, config, nodes, features, window, horizon, short_horizon, adj_static, device)
        loaded_models[name] = model

    # Select a batch
    test_iter = iter(test_loader)
    for _ in range(args.batch_idx + 1):
        x, y = next(test_iter)
    
    x, y = x.to(device), y.to(device)

    # Dictionary to store predictions
    all_preds = {}

    with torch.no_grad():
        for name, model in loaded_models.items():
            if name == "CADGT":
                preds = model(x)
            elif name == "CAMT":
                _, preds = model(x, adj_tensor)
            elif name == "AMC_DSTGNN":
                preds = model(x, adj_tensor, teacher_forcing_ratio=0.0)
            elif name == "ST_ACENet":
                preds, _ = model(x)
            
            # Re-scale back to actual speeds
            preds = scaler.inverse_transform(preds)
            all_preds[name] = preds.cpu().numpy()
            
    # Ground truth
    truth = scaler.inverse_transform(y).cpu().numpy()

    # We want to plot a continuous sequence for a specific node at a specific horizon.
    # Shape of preds is (Batch, Horizon, Nodes)
    # We will plot across the batch dimension (time)
    
    node = args.node
    h_idx = args.horizon_idx
    times = np.arange(x.shape[0]) * 5  # Each batch step is 5 minutes progression

    plt.figure(figsize=(15, 7))
    plt.plot(times, truth[:, h_idx, node], 'k--', linewidth=3, label='Ground Truth')
    
    colors = {
        'CADGT':      '#ff7f0e', # orange
        'CAMT':       '#2ca02c', # green
        'AMC_DSTGNN': '#1f77b4', # blue
        'ST_ACENet':  '#d62728', # red
    }

    for name, cmds in all_preds.items():
        plt.plot(times, cmds[:, h_idx, node], label=f'{name}', color=colors[name], linewidth=2, alpha=0.8)

    plt.title(f"Model Predictions vs Ground Truth (Node {node}, { (h_idx+1)*5 }-min Horizon)")
    plt.xlabel("Time progression (minutes)")
    plt.ylabel("Traffic Speed (mph)")
    plt.legend(loc='lower right', bbox_to_anchor=(1.15, 0), fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('visualizations', exist_ok=True)
    out_path = f"visualizations/pred_node{node}_H{h_idx+1}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
