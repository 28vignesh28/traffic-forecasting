import os
import torch
import numpy as np
import yaml
import sys

# Ensure root directory is in path so 'models' and 'src' modules resolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_dataloaders, load_static_adj
from src.utils import evaluate_metrics, setup_logging

from models.cadgt import CADGT
from models.camt_gatformer import TrafficModel as CAMT_GATformer
from models.amc_dstgnn import AMC_DSTGNN
from models.st_acenet import ST_ACENet

def test_unified_model(model_name, config_path="src/config.yaml"):
    logger = setup_logging(model_name + "_test")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing {model_name} on {device}")
    
    # Load Data
    _, _, test_loader, scaler, _ = get_dataloaders(config)
    
    adj_static = None
    if model_name in ["CAMT", "AMC_DSTGNN", "ST_ACENet"]:
        adj_static = load_static_adj(config['data']['adj_path'])
        adj_tensor = torch.FloatTensor(adj_static).to(device)
    
    # Initialize Model (read config to match train.py exactly)
    nodes = 207
    features = 14
    hidden_dim = config.get('model_defaults', {}).get('hidden_dim', 64)
    window = config['training']['window']
    horizon = config['training']['horizon']
    
    if model_name == "CADGT":
        model = CADGT(num_nodes=nodes, seq_len=window, future_len=horizon, ctx_dim=features - 1, d_model=hidden_dim).to(device)
    elif model_name == "CAMT":
        model = CAMT_GATformer(nodes=nodes).to(device)
    elif model_name == "AMC_DSTGNN":
        model = AMC_DSTGNN(nfeat=features, N=nodes, hidden_dim=128, horizon=horizon).to(device)
    elif model_name == "ST_ACENet":
        model = ST_ACENet(nfeat=features, N=nodes, static_adj=adj_static).to(device)
    else:
        raise ValueError(f"Unknown model {model_name}")

    save_path = os.path.join("saved_models", f"{model_name.lower()}_best.pth")
    if not os.path.exists(save_path):
        logger.warning(f"Weights not found at {save_path}. Please train first.")
        return
        
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    
    logger.info("Starting Evaluation...")
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            if model_name == "CADGT":
                preds = model(x)
            elif model_name == "CAMT":
                _, preds = model(x, adj_tensor)
            elif model_name == "AMC_DSTGNN":
                preds = model(x, adj_tensor, teacher_forcing_ratio=0.0)
            elif model_name == "ST_ACENet":
                preds, _ = model(x)
                
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)      # [Samples, Horizon, Nodes]
    targets = np.concatenate(all_targets)  # [Samples, Horizon, Nodes]
    
    # Inverse Transform (Reshape to 2D [Samples*Horizon, Nodes] to match Scaler fit)
    S, H, N = preds.shape
    predictions_2d = scaler.inverse_transform(preds.reshape(-1, N))
    ground_truth_2d = scaler.inverse_transform(targets.reshape(-1, N))
    
    predictions = predictions_2d.reshape(S, H, N)
    ground_truth = ground_truth_2d.reshape(S, H, N)
    
    # Verify tensor shapes: [samples, horizon_steps, nodes]
    logger.info(f"Prediction shape: {predictions.shape}, Ground truth shape: {ground_truth.shape}")
    
    # Evaluate at multiple horizons explicitly capturing short and long-term
    # Dim 0 = samples, Dim 1 = horizon steps (12), Dim 2 = nodes (207)
    horizons = {
        "5-minute": 0,   # Step 1
        "15-minute": 2,  # Step 3
        "30-minute": 5,  # Step 6
        "60-minute": 11  # Step 12
    }
    
    logger.info("=== Multi-Horizon Evaluation ===")
    for h_name, h_idx in horizons.items():
        # Index dim 1 (horizon) to get all nodes at a specific forecast step
        mae, mse, rmse, mape = evaluate_metrics(ground_truth[:, h_idx, :], predictions[:, h_idx, :])
        
        logger.info(f"Performance ({h_name} Horizon):")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  MSE:  {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["CADGT", "CAMT", "AMC_DSTGNN", "ST_ACENet"])
    args = parser.parse_args()
    
    test_unified_model(args.model)
