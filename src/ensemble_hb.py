"""
Horizon-Based Voting Ensemble for Traffic Forecasting
========================================================
Loads all 4 trained model checkpoints, computes Inverse MAE^2 weights
for EACH of the 12 prediction horizons independently based on the
validation set, and produces a weighted ensemble prediction.

Weight Formula for horizon h:
    w_i[h] = (1 / MAE_i[h]^2) / sum(1 / MAE_j[h]^2)

Usage:
    python src/ensemble_hb.py
"""

import os
import sys
import torch
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader   import get_dataloaders, load_static_adj, StandardScaler
from src.utils         import evaluate_metrics, setup_logging, set_seed

from models.cadgt          import CADGT
from models.camt_gatformer import TrafficModel as CAMT_GATformer
from models.amc_dstgnn     import AMC_DSTGNN
from models.st_acenet      import ST_ACENet


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Helper: Load a single model from its checkpoint
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _load_model(model_name, config, device, adj_static):
    """Instantiate a model, load its checkpoint, and return (model, scaler, val_loss)."""

    nodes    = config['_dataset_info']['num_nodes']
    features = config['_dataset_info']['num_features']
    hidden   = config.get('model_defaults', {}).get('hidden_dim', 64)
    window   = config['training']['window']
    horizon  = config['training']['horizon']

    if model_name == "CADGT":
        ov = config.get('model_overrides', {}).get('CADGT', {})
        model = CADGT(
            num_nodes=nodes, seq_len=window, future_len=horizon,
            ctx_dim=features - 1,
            d_model=ov.get('hidden_dim', hidden),
            static_adj=adj_static
        )
    elif model_name == "CAMT":
        ov = config.get('model_overrides', {}).get('CAMT', {})
        model = CAMT_GATformer(
            nodes=nodes, nfeat=features,
            seq_len=window, short_horizon=ov.get('short_horizon', 3),
            horizon=horizon
        )
    elif model_name == "AMC_DSTGNN":
        ov = config.get('model_overrides', {}).get('AMC_DSTGNN', {})
        model = AMC_DSTGNN(
            nfeat=features, N=nodes,
            hidden_dim=ov.get('hidden_dim', 128),
            horizon=horizon
        )
    elif model_name == "ST_ACENet":
        ov = config.get('model_overrides', {}).get('ST_ACENet', {})
        model = ST_ACENet(
            nfeat=features, N=nodes,
            hidden_dim=ov.get('hidden_dim', 64),
            horizon=horizon, static_adj=adj_static
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load checkpoint
    ckpt_path = os.path.join("saved_models", f"{model_name.lower()}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()

    scaler   = StandardScaler(mean=ckpt['scaler_mean'], std=ckpt['scaler_std'])
    val_loss = ckpt.get('val_loss', None)

    return model, scaler, val_loss


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Helper: Run inference for a single model on the test set
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _run_inference(model, model_name, test_loader, adj_tensor, device):
    """Run inference and return raw (scaled-space) predictions [S, H, N]."""
    all_preds = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            if model_name == "CADGT":
                preds = model(x)
            elif model_name == "CAMT":
                _, preds = model(x, adj_tensor)
            elif model_name == "AMC_DSTGNN":
                preds = model(x, adj_tensor, teacher_forcing_ratio=0.0)
            elif model_name == "ST_ACENet":
                preds, _ = model(x)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds, axis=0)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Quadratic Voting weight computation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ──────────────────────────────────────────────────────────────────────────────
# Horizon-Based weight computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_horizon_weights(val_horizon_maes, logger):
    """
    Compute Inverse MAE^2 weights per horizon.
    
    val_horizon_maes: dict {model_name: np.array of shape (12,)}
    Returns: dict {model_name: np.array of shape (12,)} containing the weights
    """
    horizons = 12
    weights = {}
    
    logger.info("=" * 80)
    logger.info("Horizon-Based Weight Computation (Inverse MAE^2)")
    logger.info("=" * 80)
    
    # Initialize weight arrays
    for name in val_horizon_maes:
        weights[name] = np.zeros(horizons)
        
    for h in range(horizons):
        scores = {name: 1.0 / (val_horizon_maes[name][h] ** 2) for name in val_horizon_maes}
        total = sum(scores.values())
        
        for name in val_horizon_maes:
            weights[name][h] = scores[name] / total
            
    # Log weights for key horizons (5, 15, 30, 60 min)
    h_idx = [0, 2, 5, 11]
    h_labels = ["5-min", "15-min", "30-min", "60-min"]
    
    for i, idx in enumerate(h_idx):
        logger.info(f"\n--- {h_labels[i]} (Horizon {idx+1}) Weights ---")
        for name in val_horizon_maes:
            mae = val_horizon_maes[name][idx]
            w = weights[name][idx]
            logger.info(f"{name:<15} Val MAE: {mae:>6.4f} | Weight: {w:>6.4f}")
            
    logger.info("=" * 80)
    return weights


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main ensemble pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_hb_ensemble(config_path="src/config.yaml"):
    logger = setup_logging("HB_Ensemble")
    set_seed(42)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inverse MAE^2 Ensemble on {device}")

    # ── 1. Load data (val_loader and test_loader needed) ──
    _, val_loader, test_loader, _, dataset_info = get_dataloaders(config)
    config['_dataset_info'] = dataset_info

    # â”€â”€ 2. Load adjacency matrix â”€â”€
    adj_static = load_static_adj(config['data']['adj_path'])
    adj_tensor = torch.FloatTensor(adj_static).to(device)

    # â”€â”€ 3. Load the ensemble model list from config â”€â”€
    ensemble_cfg = config.get('ensemble', {})
    model_names  = ensemble_cfg.get('models', ["CADGT", "CAMT", "AMC_DSTGNN", "ST_ACENet"])

    # â”€â”€ 4. Load all models & collect val_losses â”€â”€
    models   = {}
    scalers  = {}
    val_losses = {}

    for name in model_names:
        logger.info(f"Loading {name}...")
        model, scaler, val_loss = _load_model(name, config, device, adj_static)
        if val_loss is None:
            raise RuntimeError(
                f"Checkpoint for {name} has no 'val_loss'. "
                f"Retrain with the latest train.py to include val_loss in the checkpoint."
            )
        models[name]     = model
        scalers[name]    = scaler
        val_losses[name] = val_loss
        logger.info(f"  {name}: val_MAE = {val_loss:.4f}, scaler(mean={scaler.mean:.4f}, std={scaler.std:.4f})")

    # ── 5. Run inference on VAL set to get per-horizon MAE ──
    val_horizon_maes = {}
    
    # Pre-fetch validation targets
    all_val_targets = []
    with torch.no_grad():
        for _, y in val_loader:
            all_val_targets.append(y.numpy())
    val_targets_scaled = np.concatenate(all_val_targets, axis=0)
    
    first_name = model_names[0]
    S_v, H_v, N_v = val_targets_scaled.shape
    val_ground_truth = scalers[first_name].inverse_transform(
        val_targets_scaled.reshape(-1, N_v)
    ).reshape(S_v, H_v, N_v)

    for name in model_names:
        logger.info(f"Getting validation MAEs per horizon for {name}...")
        val_pred_scaled = _run_inference(models[name], name, val_loader, adj_tensor, device)
        val_pred_orig = scalers[name].inverse_transform(
            val_pred_scaled.reshape(-1, N_v)
        ).reshape(S_v, H_v, N_v)
        
        # Calculate MAE for each horizon
        horizon_maes = np.zeros(H_v)
        for h in range(H_v):
            mae, _, _, _ = evaluate_metrics(val_ground_truth[:, h, :], val_pred_orig[:, h, :])
            horizon_maes[h] = mae
        val_horizon_maes[name] = horizon_maes

    # ── 6. Compute Horizon-Based weights ──
    weights = compute_horizon_weights(val_horizon_maes, logger)

    # ── 7. Run inference on TEST set for each model ──
    raw_preds   = {}   # scaled space
    orig_preds  = {}   # original speed space (inverse-transformed)

    for name in model_names:
        logger.info(f"Running inference for {name}...")
        pred_scaled = _run_inference(models[name], name, test_loader, adj_tensor, device)
        raw_preds[name] = pred_scaled

        # Inverse-transform to original speed space using the model's own scaler
        S, H, N = pred_scaled.shape
        pred_orig = scalers[name].inverse_transform(pred_scaled.reshape(-1, N)).reshape(S, H, N)
        orig_preds[name] = pred_orig

    # ── 8. Collect ground truth (use any model's scaler — targets are identical) ──
    all_targets = []
    with torch.no_grad():
        for _, y in test_loader:
            all_targets.append(y.numpy())
    targets_scaled = np.concatenate(all_targets, axis=0)

    # Use the first model's scaler for ground truth inverse-transform
    first_name = model_names[0]
    S, H, N = targets_scaled.shape
    ground_truth = scalers[first_name].inverse_transform(
        targets_scaled.reshape(-1, N)
    ).reshape(S, H, N)

    # ── 9. Weighted ensemble combination ──
    ensemble_pred = np.zeros_like(ground_truth)
    for name in model_names:
        for h in range(H):
            ensemble_pred[:, h, :] += weights[name][h] * orig_preds[name][:, h, :]

    # ── 10. Evaluate all models + ensemble ──
    horizons = {
        "5-minute":  0,
        "15-minute": 2,
        "30-minute": 5,
        "60-minute": 11
    }

    # Header
    logger.info("\n" + "=" * 90)
    logger.info("MULTI-HORIZON COMPARISON: Individual Models vs. Horizon-Based Ensemble")
    logger.info("=" * 90)

    all_results = {}

    # Individual models
    for name in model_names:
        results = {}
        for h_name, h_idx in horizons.items():
            mae, mse, rmse, mape = evaluate_metrics(
                ground_truth[:, h_idx, :], orig_preds[name][:, h_idx, :]
            )
            results[h_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}
        all_results[name] = results

    # Ensemble
    ensemble_results = {}
    for h_name, h_idx in horizons.items():
        mae, mse, rmse, mape = evaluate_metrics(
            ground_truth[:, h_idx, :], ensemble_pred[:, h_idx, :]
        )
        ensemble_results[h_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}
    all_results['HB_Ensemble'] = ensemble_results

    # ── 11. Pretty-print comparison table ──
    for h_name in horizons:
        logger.info(f"\n{'-' * 70}")
        logger.info(f"  {h_name.upper()} HORIZON")
        logger.info(f"{'-' * 70}")
        logger.info(f"  {'Model':<18} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>8}")
        logger.info(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8}")

        for name in model_names:
            r = all_results[name][h_name]
            logger.info(f"  {name:<18} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['MAPE']:>8.2f}")

        r = all_results['HB_Ensemble'][h_name]
        logger.info(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8}")
        logger.info(f"  {'* HB_Ensemble':<18} {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['MAPE']:>8.2f}")

    # ── 12. Summary verdict ──
    best_individual_60 = min(
        all_results[name]['60-minute']['MAE'] for name in model_names
    )
    ensemble_60 = all_results['HB_Ensemble']['60-minute']['MAE']

    logger.info(f"\n{'=' * 70}")
    logger.info("VERDICT")
    logger.info(f"{'=' * 70}")
    logger.info(f"  Best individual 60-min MAE : {best_individual_60:.4f}")
    logger.info(f"  HB Ensemble 60-min MAE     : {ensemble_60:.4f}")
    improvement = (best_individual_60 - ensemble_60) / best_individual_60 * 100
    if improvement > 0:
        logger.info(f"  [+] Ensemble IMPROVES by {improvement:.2f}%")
    else:
        logger.info(f"  [-] Ensemble is {-improvement:.2f}% worse (consider tuning weights or removing weak models)")
    logger.info(f"{'=' * 70}")

    return all_results


if __name__ == "__main__":
    run_hb_ensemble()
