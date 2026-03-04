import os
import sys
import torch
import yaml
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_traffic, fetch_weather_api, fetch_holiday_api, add_time_features, StandardScaler, load_static_adj
from src.utils import evaluate_metrics, setup_logging, set_seed
from models.cadgt import CADGT

def analyze_model_performance(config_path="src/config.yaml"):
    logger = setup_logging("CADGT_Analysis")
    set_seed(42)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Analysis on {device}")

    # --- 1. Load Data Elements to create precise masks ---
    logger.info("Loading test data and computing conditions...")
    traffic, timestamps = load_traffic(config['data']['traffic_path'])
    
    total_len = len(traffic)
    train_end = int(0.7 * total_len)
    val_end = int(0.8 * total_len)
    
    # Get scaler from training data to match normalization
    traffic_train = traffic[:train_end]
    mean = float(np.mean(traffic_train))
    std = float(np.std(traffic_train))
    scaler = StandardScaler(mean, std)
    
    # We only care about the TEST period
    # Wait, the pipeline calls `get_dataloaders` which fetches weather and normalizes.
    # To keep exactly the exact same predictions, let's use get_dataloaders
    from src.data_loader import get_dataloaders
    _, _, test_loader, pip_scaler, dataset_info = get_dataloaders(config)
    
    # Let's load CADGT
    nodes = dataset_info['num_nodes']
    features = dataset_info['num_features']
    hidden_dim = config.get('model_defaults', {}).get('hidden_dim', 64)
    window = config['training']['window']
    horizon = config['training']['horizon']
    
    adj_static = load_static_adj(config['data']['adj_path'])
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
    model.eval()

    logger.info("Running baseline inference on test set...")
    all_preds, all_targets = [], []
    all_x = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_x.append(x.cpu().numpy())

    preds = np.concatenate(all_preds)    # [S, H, N]
    targets = np.concatenate(all_targets)  # [S, H, N]
    x_input = np.concatenate(all_x) # [S, W, N, F]

    # Inverse transform
    S, H, N = preds.shape
    predictions_2d = pip_scaler.inverse_transform(preds.reshape(-1, N))
    ground_truth_2d = pip_scaler.inverse_transform(targets.reshape(-1, N))
    predictions = predictions_2d.reshape(S, H, N)
    ground_truth = ground_truth_2d.reshape(S, H, N)

    logger.info("Defining condition masks...")
    
    # We can reconstruct the exact timestamps for the TARGETS
    # The dataset class returns y = X_merged[idx + window : idx + window + horizon]
    # For test set, idx=0 corresponds to val_end in the full array.
    # So for sample s, the first target timestep corresponds to:
    # index = val_end + s + window
    # Let's check the holiday feature from x_input
    # x_input shape: [S, W, N, F]
    # Feature 5 is Holiday. We can check the last timestep of the input window
    holiday_mask = (x_input[:, -1, 0, 5] == 1) 
    
    # For Peak hours, ToD is feature 6 and 7 (tod_sin, tod_cos) from x_input
    # It might be easier to use the exact timestamps
    test_target_start_indices = np.arange(S) + val_end + window
    
    # We will evaluate at 60-min horizon (index 11) for all of these
    target_horizon_idx = 11  # 60-min
    
    # Let's get the exact timestamps for the 60 min horizon:
    # target_index = val_end + s + window + 11
    target_timestamps_60m = timestamps[test_target_start_indices + target_horizon_idx]
    
    # Peak Hours: 7am-9am and 4pm-6pm (16:00-18:00)
    hours = target_timestamps_60m.hour
    peak_mask = ((hours >= 7) & (hours < 9)) | ((hours >= 16) & (hours < 18))
    
    # Abnormal Conditions: Let's use Open-Meteo weather
    # We need the weather for the target time, but let's just get the weather data unnormalized
    start_date = target_timestamps_60m.min().strftime("%Y-%m-%d")
    end_date = target_timestamps_60m.max().strftime("%Y-%m-%d")
    weather_df = fetch_weather_api(config['data']['lat'], config['data']['lon'], start_date, end_date)
    weather_aligned = weather_df.set_index("time").reindex(target_timestamps_60m).ffill().bfill()
    precip = pd.to_numeric(weather_aligned["precipitation"], errors='coerce').fillna(0.0).values
    visibility = pd.to_numeric(weather_aligned["visibility"], errors='coerce').fillna(10000.0).values
    
    # Abnormal: Precipitation > 0 or Visibility < 10000 meters (10km is standard good visibility)
    abnormal_mask = (precip > 0.0) | (visibility < 10000.0)
    
    # --- Performance Evaluation Functon ---
    def eval_mask(mask, name):
        if np.sum(mask) == 0:
            logger.info(f"{name}: No samples found.")
            return
        mae, mse, rmse, mape = evaluate_metrics(
            ground_truth[mask, target_horizon_idx, :], 
            predictions[mask, target_horizon_idx, :]
        )
        logger.info(f"--- {name} ({np.sum(mask)} samples) ---")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
    logger.info("\n========== CONDITION ANALYSIS (60-MIN HORIZON) ==========")
    logger.info("--- Baseline (All Data) ---")
    eval_mask(np.ones(S, dtype=bool), "All Data")
    
    eval_mask(peak_mask, "Peak Hours (7-9 AM, 4-6 PM)")
    eval_mask(~peak_mask, "Off-Peak Hours")
    
    eval_mask(holiday_mask, "Holidays")
    eval_mask(~holiday_mask, "Non-Holidays")
    
    eval_mask(abnormal_mask, "Abnormal Weather (Rain or Low Visibility)")
    eval_mask(~abnormal_mask, "Normal Weather")
    
    # --- NOISE SENSITIVITY ---
    logger.info("\n========== NOISE SENSITIVITY ANALYSIS ==========")
    # Add Gaussian noise to traffic speed feature only (Feature 0)
    # The traffic feature is normalized. We add noise proportional to standard deviation.
    noise_levels = [0.1, 0.2, 0.5] # 10%, 20%, 50% of std
    
    for noise_ratio in noise_levels:
        logger.info(f"Running inference with {noise_ratio*100}% Noise...")
        all_noisy_preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                # Traffic is x[:, :, :, 0]. It's already N(0,1) scaled roughly.
                noise = torch.randn_like(x[:, :, :, 0]) * noise_ratio
                noisy_x = x.clone()
                noisy_x[:, :, :, 0] += noise
                
                preds = model(noisy_x)
                all_noisy_preds.append(preds.cpu().numpy())
                
        noisy_preds = np.concatenate(all_noisy_preds)
        noisy_predictions_2d = pip_scaler.inverse_transform(noisy_preds.reshape(-1, N))
        noisy_predictions = noisy_predictions_2d.reshape(S, H, N)
        
        mae, mse, rmse, mape = evaluate_metrics(
            ground_truth[:, target_horizon_idx, :], 
            noisy_predictions[:, target_horizon_idx, :]
        )
        logger.info(f"--- Noise Level {noise_ratio*100}% ---")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")

if __name__ == "__main__":
    analyze_model_performance()
