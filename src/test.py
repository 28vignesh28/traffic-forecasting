import os
import torch
import numpy as np
import yaml
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader  import get_dataloaders, load_static_adj, StandardScaler
from src.utils        import evaluate_metrics, setup_logging, set_seed

from models.cadgt          import CADGT
from models.camt_gatformer import TrafficModel as CAMT_GATformer
from models.amc_dstgnn     import AMC_DSTGNN
from models.st_acenet      import ST_ACENet


def test_unified_model(model_name, config_path="src/config.yaml"):
    logger = setup_logging(model_name + "_test")

    # ==========================================================================
    # FIX #9: Seed RNG before any operations for reproducible evaluation.
    # ==========================================================================
    set_seed(42)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Testing {model_name} on {device}")

    # ---- Load Data (only needed for the test_loader) ----
    _, _, test_loader, _, dataset_info = get_dataloaders(config)

    adj_static = None
    if model_name in ["CADGT", "CAMT", "AMC_DSTGNN", "ST_ACENet"]:
        adj_static = load_static_adj(config['data']['adj_path'])
        adj_tensor = torch.FloatTensor(adj_static).to(device)

    # FIX #36: Use dynamic values from the dataset instead of hardcoded constants
    nodes    = dataset_info['num_nodes']
    features = dataset_info['num_features']
    hidden_dim = config.get('model_defaults', {}).get('hidden_dim', 64)
    window   = config['training']['window']
    horizon  = config['training']['horizon']

    if model_name == "CADGT":
        model = CADGT(
            num_nodes=nodes, seq_len=window, future_len=horizon,
            ctx_dim=features - 1, d_model=hidden_dim, static_adj=adj_static
        ).to(device)
    elif model_name == "CAMT":
        camt_overrides = config.get('model_overrides', {}).get('CAMT', {})
        short_horizon = camt_overrides.get('short_horizon', 3)
        model = CAMT_GATformer(
            nodes=nodes, nfeat=features,
            seq_len=window, short_horizon=short_horizon, horizon=horizon
        ).to(device)
    elif model_name == "AMC_DSTGNN":
        model = AMC_DSTGNN(
            nfeat=features, N=nodes, hidden_dim=128, horizon=horizon
        ).to(device)
    elif model_name == "ST_ACENet":
        model = ST_ACENet(
            nfeat=features, N=nodes, horizon=horizon, static_adj=adj_static
        ).to(device)
    else:
        raise ValueError(f"Unknown model {model_name}")

    save_path = os.path.join("saved_models", f"{model_name.lower()}_best.pth")
    if not os.path.exists(save_path):
        logger.warning(f"Weights not found at {save_path}. Please train first.")
        return

    # ==========================================================================
    # FIX #10: Load scaler from the checkpoint instead of recomputing it.
    # Previously, get_dataloaders() was called again to obtain the scaler,
    # which would re-run the weather API and re-derive mean/std — potentially
    # giving slightly different statistics and invalidating inverse-transform.
    # ==========================================================================
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    scaler = StandardScaler(
        mean=checkpoint['scaler_mean'],
        std=checkpoint['scaler_std']
    )
    # FIX #10: Safe formatting — avoid crash if val_loss key is missing from old checkpoints
    val_loss_str = f"{checkpoint['val_loss']:.4f}" if 'val_loss' in checkpoint else '?'
    logger.info(
        f"Loaded model (epoch {checkpoint.get('epoch', '?')}, "
        f"val_loss {val_loss_str}). "
        f"Scaler: mean={scaler.mean:.4f}, std={scaler.std:.4f}"
    )
    model.eval()

    logger.info("Starting Evaluation...")
    all_preds, all_targets = [], []
    # FIX #18: also collect sigma for ST_ACENet uncertainty evaluation
    all_sigmas = [] if model_name == "ST_ACENet" else None

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
                preds, sigma = model(x)
                all_sigmas.append(sigma.cpu().numpy())

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds)    # [S, H, N]
    targets = np.concatenate(all_targets)  # [S, H, N]

    # Inverse transform
    S, H, N        = preds.shape
    predictions_2d = scaler.inverse_transform(preds.reshape(-1, N))
    ground_truth_2d = scaler.inverse_transform(targets.reshape(-1, N))
    predictions    = predictions_2d.reshape(S, H, N)
    ground_truth   = ground_truth_2d.reshape(S, H, N)

    logger.info(f"Prediction shape: {predictions.shape}, Ground truth shape: {ground_truth.shape}")

    horizons = {
        "5-minute":  0,
        "15-minute": 2,
        "30-minute": 5,
        "60-minute": 11
    }

    logger.info("=== Multi-Horizon Evaluation ===")
    for h_name, h_idx in horizons.items():
        mae, mse, rmse, mape = evaluate_metrics(
            ground_truth[:, h_idx, :], predictions[:, h_idx, :]
        )
        logger.info(f"Performance ({h_name} Horizon):")
        logger.info(f"  MAE:  {mae:.4f}")
        logger.info(f"  MSE:  {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%\n")

    # ==========================================================================
    # FIX #18: Evaluate ST_ACENet uncertainty (sigma) output.
    # Previously sigma was discarded at test time, making the probabilistic
    # architecture completely unevaluated.  Now we report:
    #   • Average sigma per horizon (shows calibration spread)
    #   • Gaussian NLL on test set per horizon
    # ==========================================================================
    if model_name == "ST_ACENet" and all_sigmas is not None:
        sigmas = np.concatenate(all_sigmas)   # [S, H, N]  (still in scaled space)

        logger.info("=== ST_ACENet Uncertainty Evaluation ===")
        loss_fn = torch.nn.GaussianNLLLoss(reduction='mean')

        for h_name, h_idx in horizons.items():
            mu_h    = torch.from_numpy(predictions[:, h_idx, :]).float()
            gt_h    = torch.from_numpy(ground_truth[:, h_idx, :]).float()
            sigma_h = torch.from_numpy(sigmas[:, h_idx, :]).float()

            # Invert sigma to original scale (multiply by scaler std)
            sigma_h_orig = sigma_h * float(scaler.std)

            avg_sigma = sigma_h_orig.mean().item()
            nll       = loss_fn(mu_h, gt_h, sigma_h_orig ** 2).item()

            logger.info(f"Uncertainty ({h_name} Horizon):")
            logger.info(f"  Avg Sigma (orig scale): {avg_sigma:.4f}")
            logger.info(f"  Gaussian NLL:           {nll:.4f}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["CADGT", "CAMT", "AMC_DSTGNN", "ST_ACENet"]
    )
    args = parser.parse_args()
    test_unified_model(args.model)
