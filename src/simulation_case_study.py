import os
import sys
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_dataloaders, load_static_adj, StandardScaler
from src.utils import setup_logging, set_seed
from models.cadgt import CADGT

def detect_congestion_window(true_speeds, node, length=100, threshold=40.0):
    """Finds a window of `length` steps where the specified node experiences a speed drop below `threshold`."""
    total_steps = true_speeds.shape[0]
    for start in range(total_steps - length):
        window = true_speeds[start:start+length, 0, node] # 0 is H1
        if np.min(window) < threshold:
            # Also ensure there's a recovery
            min_idx = np.argmin(window)
            if min_idx > 10 and min_idx < length - 10:
                if window[0] > threshold + 5 and window[-1] > threshold + 5:
                    return start
    return 0 # Fallback to start of test set

def run_simulation(config_path="src/config.yaml"):
    logger = setup_logging("Simulation_Case_Study")
    set_seed(42)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Case Study Simulation for CADGT on {device}")

    # Load Data
    _, _, test_loader, pip_scaler, dataset_info = get_dataloaders(config)
    adj_static = load_static_adj(config['data']['adj_path'])
    
    nodes = dataset_info['num_nodes']
    features = dataset_info['num_features']
    window = config['training']['window']
    horizon = config['training']['horizon']
    hidden_dim = config.get('model_defaults', {}).get('hidden_dim', 64)
    cadgt_overrides = config.get('model_overrides', {}).get('CADGT', {})

    model = CADGT(
        num_nodes=nodes, seq_len=window, future_len=horizon,
        ctx_dim=features - 1,
        d_model=cadgt_overrides.get('hidden_dim', hidden_dim),
        static_adj=adj_static
    ).to(device)

    save_path = os.path.join("saved_models", "cadgt_best.pth")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # Use scaler from checkpoint
    scaler = StandardScaler(mean=checkpoint['scaler_mean'], std=checkpoint['scaler_std'])
    logger.info(f"Loaded CADGT model. Scaler: mean={scaler.mean:.4f}, std={scaler.std:.4f}")
    model.eval()

    # Collect all test inputs
    logger.info("Extracting continuous test stream...")
    all_x, all_y = [], []
    for x, y in test_loader:
        all_x.append(x.numpy())
        all_y.append(y.numpy())
    stream_x = np.concatenate(all_x, axis=0)
    stream_y = np.concatenate(all_y, axis=0)

    # Inverse transform ground truth for analysis
    S, H, N = stream_y.shape
    true_speeds = scaler.inverse_transform(stream_y.reshape(-1, N)).reshape(S, H, N)

    # Simulation settings
    sim_length = 150
    target_nodes = [91, 196]
    horizons = {"H1 (5-min)": 0, "H3 (15-min)": 2, "H6 (30-min)": 5, "H12 (60-min)": 11}

    # Find a good start step with congestion for Node 91
    start_step = detect_congestion_window(true_speeds, target_nodes[0], length=sim_length, threshold=35.0)
    logger.info(f"Selected simulation window starting at step {start_step} for {sim_length} steps.")

    # Slice stream
    sim_x = torch.FloatTensor(stream_x[start_step : start_step + sim_length]).to(device)
    sim_y_true = true_speeds[start_step : start_step + sim_length]

    logger.info("Running CADGT predictions over simulation window...")
    with torch.no_grad():
        preds_scaled = model(sim_x).cpu().numpy()
        
    sim_preds = scaler.inverse_transform(preds_scaled.reshape(-1, N)).reshape(sim_length, H, N)

    # Create visual output dir
    out_dir = os.path.join("visualizations", "case_study")
    os.makedirs(out_dir, exist_ok=True)

    # Arrays to store behavioral metrics
    metrics = []

    for node in target_nodes:
        logger.info(f"Analyzing Node {node}")
        
        plt.figure(figsize=(14, 8))
        plt.title(f"CADGT Horizon-Wise Drift Comparison - Node {node}", fontsize=16, fontweight='bold')
        
        # Ground Truth H1
        base_true = sim_y_true[:, 0, node]
        plt.plot(base_true, label='Ground Truth (5-min actual)', color='black', linewidth=3)
        
        # We need to shift the predictions backward to align with the same "target time".
        # E.g., at step t, H1 predicts t+1. At step t-5, H6 predicts t+1. 
        # So to plot what CADGT predicted for time T across different horizons, we must align them.
        
        aligned_preds = {}
        for h_name, h_idx in horizons.items():
            # The model makes prediction `h_idx` periods into the future.
            # So `sim_preds[t, h_idx, node]` is a prediction for time `start_step + t + h_idx`.
            # To plot it at the x-axis position corresponding to target time, we shift it right by `h_idx`.
            
            # Actually, the x-axis of the plot represents `start_step + t`. 
            # `base_true[t]` is the true traffic at `start_step + t + 0` (H1).
            # If we want to align everything to target time `T = t`:
            
            shift = h_idx
            shifted_pred = np.full(sim_length, np.nan)
            if shift < sim_length:
                shifted_pred[shift:] = sim_preds[:-shift, h_idx, node] if shift > 0 else sim_preds[:, h_idx, node]
            aligned_preds[h_name] = shifted_pred
            
            # Plot
            styles = {"H1 (5-min)": ('#2ca02c', '-'), "H3 (15-min)": ('#1f77b4', '--'), 
                      "H6 (30-min)": ('#ff7f0e', '-.'), "H12 (60-min)": ('#d62728', ':')}
            
            plt.plot(shifted_pred, label=f'CADGT {h_name}', color=styles[h_name][0], 
                     linestyle=styles[h_name][1], linewidth=2, alpha=0.8)

        plt.xlabel("Simulation Steps (5-min intervals)")
        plt.ylabel("Speed (mph)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"CADGT_horizon_drift_node_{node}.png"), dpi=300)
        plt.close()

        # Calculate Behavioral Metrics
        # 1. Congestion Detection (Find the steepest drop in Ground Truth)
        min_speed_idx = np.argmin(base_true)
        min_speed = base_true[min_speed_idx]
        
        is_congestion = min_speed < 40.0
        
        if is_congestion:
            # We look around the min speed
            drop_start = max(0, min_speed_idx - 15)
            drop_end = min(sim_length, min_speed_idx + 15)
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Congestion Response - Node {node} Focus", fontsize=14, fontweight='bold')
            plt.plot(range(drop_start, drop_end), base_true[drop_start:drop_end], label="Ground Truth", color="black", linewidth=4)
            for h_name in horizons.keys():
                plt.plot(range(drop_start, drop_end), aligned_preds[h_name][drop_start:drop_end], 
                         label=f"CADGT {h_name}", linewidth=2, linestyle=styles[h_name][1], color=styles[h_name][0])
            plt.axvline(min_speed_idx, color='gray', linestyle='--', alpha=0.5, label='Max Congestion Point')
            plt.xlabel("Steps")
            plt.ylabel("Speed (mph)")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"CADGT_congestion_zoom_node_{node}.png"), dpi=300)
            plt.close()

            # Delay in detection (when did the prediction finally reach within 5 mph of the min congestion?)
            # Underestimation magnitude (Difference between true min and pred min in the window)
            for h_name in horizons.keys():
                pred_window = aligned_preds[h_name][drop_start:drop_end]
                true_window = base_true[drop_start:drop_end]
                
                valid_mask = ~np.isnan(pred_window)
                if not np.any(valid_mask):
                    continue
                    
                pred_min = np.min(pred_window[valid_mask])
                pred_min_idx = drop_start + np.argmin(pred_window[valid_mask])
                
                underestimation_mag = pred_min - min_speed
                delay_in_detection = max(0, pred_min_idx - min_speed_idx) # steps after actual min
                
                mae = np.mean(np.abs(true_window[valid_mask] - pred_window[valid_mask]))
                variance = np.var(pred_window[valid_mask])
                
                metrics.append({
                    "Node": node,
                    "Horizon": h_name,
                    "True Min Speed": min_speed,
                    "Pred Min Speed": pred_min,
                    "Underestimation (mph)": underestimation_mag,
                    "Delay (steps)": delay_in_detection,
                    "Local MAE": mae,
                    "Pred Variance": variance,
                    "True Obj Variance": np.var(true_window)
                })

    if metrics:
        df = pd.DataFrame(metrics)
        df = df.round(2)
        print("\n=== CADGT Behavioral Metrics during Congestion ===")
        print(df.to_string(index=False))
        df.to_csv(os.path.join(out_dir, "cadgt_behavioral_metrics.csv"), index=False)
        
        # Plot Error Growth Rate across horizons
        plt.figure(figsize=(8, 6))
        plt.title("CADGT Error Growth Across Horizons (Congestion Window)", fontsize=14, fontweight='bold')
        
        for node in target_nodes:
            node_df = df[df["Node"] == node]
            plt.plot(node_df["Horizon"], node_df["Local MAE"], marker='o', markersize=8, linewidth=2, label=f"Node {node}")
            
        plt.xlabel("Horizon")
        plt.ylabel("Mean Absolute Error (During Congestion)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "CADGT_error_growth.png"), dpi=300)
        plt.close()
        
    logger.info(f"Simulation completed. Outputs saved to {out_dir}")

if __name__ == "__main__":
    run_simulation()
