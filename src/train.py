import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import time
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.tensorboard import SummaryWriter

from src.data_loader  import get_dataloaders, load_static_adj
from src.utils        import evaluate_metrics, create_directories, setup_logging, set_seed

from models.cadgt          import CADGT
from models.camt_gatformer import TrafficModel as CAMT_GATformer
from models.amc_dstgnn     import AMC_DSTGNN
from models.st_acenet      import ST_ACENet


def train_unified_model(model_name, config_path="src/config.yaml"):
    create_directories()
    logger = setup_logging(model_name)
    writer = SummaryWriter(log_dir=os.path.join("logs", "tensorboard", model_name))

    logger.info(f"=== Starting unified runner for {model_name} ===")

    # ==========================================================================
    # FIX #9: Reproducibility — seed all RNG sources before any model or data
    # operations so that every run produces identical results.
    # ==========================================================================
    set_seed(42)

    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False   # required for determinism (FIX #9)
    logger.info(f"[{model_name}] Initializing Unified Training Pipeline on {device}")

    # 2. Data Loading
    logger.info(f"[{model_name}] Calling get_dataloaders...")
    train_loader, val_loader, _, scaler, dataset_info = get_dataloaders(config)
    nodes    = dataset_info['num_nodes']
    features = dataset_info['num_features']
    logger.info(f"[{model_name}] Data Loaded: {nodes} Nodes, {features} Features (Traffic + Context)")

    # 3. Model Initialization
    adj_static = None
    if model_name in ["CADGT", "CAMT", "AMC_DSTGNN", "ST_ACENet"]:
        logger.info(f"[{model_name}] Loading Physical adjacency matrix...")
        adj_static = load_static_adj(config['data']['adj_path'])
        adj_tensor = torch.FloatTensor(adj_static).to(device)

        # Validate adjacency matrix row-sums
        row_sums = adj_tensor.sum(dim=-1)
        max_dev = (row_sums - row_sums.mean()).abs().max().item()
        if max_dev > 0.5:
            logger.warning(
                f"Adjacency row-sums vary significantly (max deviation: {max_dev:.2f}). "
                f"This may indicate unnormalized or corrupted adjacency data."
            )

    logger.info(f"[{model_name}] Building Architecture...")
    if model_name == "CADGT":
        cadgt_overrides = config.get('model_overrides', {}).get('CADGT', {})
        model = CADGT(
            num_nodes=nodes,
            seq_len=config['training']['window'],
            future_len=config['training']['horizon'],
            ctx_dim=features - 1,
            d_model=cadgt_overrides.get('hidden_dim', config['model_defaults']['hidden_dim']),
            static_adj=adj_static
        ).to(device)
        loss_fn = nn.L1Loss()

    elif model_name == "CAMT":
        # FIX #9: Read model_overrides from config instead of hardcoding
        camt_overrides = config.get('model_overrides', {}).get('CAMT', {})
        short_horizon = camt_overrides.get('short_horizon', 3)
        model   = CAMT_GATformer(
            nodes=nodes, nfeat=features,
            seq_len=config['training']['window'],
            short_horizon=short_horizon,
            horizon=config['training']['horizon']
        ).to(device)
        loss_fn = nn.HuberLoss()

    elif model_name == "AMC_DSTGNN":
        amc_overrides = config.get('model_overrides', {}).get('AMC_DSTGNN', {})
        model = AMC_DSTGNN(
            nfeat=features, N=nodes,
            hidden_dim=amc_overrides.get('hidden_dim', 128),
            dropout=amc_overrides.get('dropout', 0.3),
            horizon=config['training']['horizon']
        ).to(device)
        loss_fn = nn.HuberLoss()

    elif model_name == "ST_ACENet":
        ace_overrides = config.get('model_overrides', {}).get('ST_ACENet', {})
        model   = ST_ACENet(
            nfeat=features, N=nodes,
            hidden_dim=ace_overrides.get('hidden_dim', 64),
            horizon=config['training']['horizon'],
            static_adj=adj_static
        ).to(device)
        loss_fn = nn.GaussianNLLLoss()

    else:
        raise ValueError(f"Unknown model {model_name}")

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # FIX #31: Use device.type for correct autocast on both CPU and CUDA
    device_type  = device.type
    use_amp      = device_type == 'cuda'
    grad_scaler  = torch.amp.GradScaler(device='cuda') if use_amp else None

    # 5. Training Loop
    epochs         = config['training']['epochs']
    patience       = config['training']['patience']
    max_grad_norm  = config['training']['max_grad_norm']
    save_path      = os.path.join("saved_models", f"{model_name.lower()}_best.pth")

    best_val_mae       = float('inf')
    early_stop_counter = 0

    logger.info(f"[{model_name}] Starting Training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # --- TRAIN ---
        model.train()
        total_train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type, enabled=use_amp):
                if model_name == "CADGT":
                    preds = model(x)
                    loss  = loss_fn(preds, y)

                elif model_name == "CAMT":
                    ps_short, pl_long = model(x, adj_tensor)
                    # ===========================================================
                    # FIX #11: Short-term loss multiplied by 0.25 (= 3/12) to
                    # equalise per-step gradient contribution between the 3-step
                    # short branch and 12-step long branch.  Previously the short
                    # branch received 4× the per-step gradient of the long branch,
                    # over-biasing the auxiliary objective.
                    # ===========================================================
                    # FIX #34: Use config-derived short_horizon, not hardcoded 3
                    loss = 0.25 * loss_fn(ps_short, y[:, :short_horizon, :]) + loss_fn(pl_long, y)

                elif model_name == "AMC_DSTGNN":
                    # ===========================================================
                    # FIX #13: Teacher-forcing decay rate slowed from 0.96 to
                    # 0.995 per epoch.  At 0.96, the ratio dropped to ~24% by
                    # epoch 18 — too fast for stable curriculum learning.
                    # At 0.995 the ratio stays above 40% for 50 epochs.
                    # ===========================================================
                    tf_ratio = 0.5 * (0.995 ** epoch)
                    preds    = model(x, adj_tensor, y=y, teacher_forcing_ratio=tf_ratio)
                    loss     = loss_fn(preds, y)

                elif model_name == "ST_ACENet":
                    mu, sigma = model(x)
                    nll_loss  = loss_fn(mu, y, sigma ** 2)
                    mae_loss  = nn.functional.l1_loss(mu, y)
                    loss      = nll_loss + 0.5 * mae_loss  # anchor point estimate

            if use_amp:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATE ---
        model.eval()
        total_val_loss = 0.0
        total_val_mae  = 0.0   # FIX #2: Uniform MAE for fair early stopping

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with torch.amp.autocast(device_type, enabled=use_amp):
                    if model_name == "CADGT":
                        preds = model(x)
                        loss  = loss_fn(preds, y)
                        preds_for_mae = preds
                    elif model_name == "CAMT":
                        ps_short, pl_long = model(x, adj_tensor)
                        loss = 0.25 * loss_fn(ps_short, y[:, :short_horizon, :]) + loss_fn(pl_long, y)
                        preds_for_mae = pl_long
                    elif model_name == "AMC_DSTGNN":
                        preds = model(x, adj_tensor, teacher_forcing_ratio=0.0)
                        loss  = loss_fn(preds, y)
                        preds_for_mae = preds
                    elif model_name == "ST_ACENet":
                        mu, sigma = model(x)
                        nll_loss  = loss_fn(mu, y, sigma ** 2)
                        mae_loss  = nn.functional.l1_loss(mu, y)
                        loss      = nll_loss + 0.5 * mae_loss
                        preds_for_mae = mu

                total_val_loss += loss.item()
                # FIX #2: Track uniform MAE so early stopping is comparable across models
                total_val_mae  += nn.functional.l1_loss(preds_for_mae, y).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mae  = total_val_mae  / len(val_loader)
        scheduler.step(avg_val_loss)

        epoch_dur = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val MAE: {avg_val_mae:.4f} | Time: {epoch_dur:.2f}s"
        )

        writer.add_scalar('Loss/Train',      avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss,   epoch)
        writer.add_scalar('LearningRate',    optimizer.param_groups[0]['lr'], epoch)

        # Checkpoint & Early Stopping
        # FIX #2: Early stopping on uniform MAE — same metric for all models
        if avg_val_mae < best_val_mae - 1e-4:
            best_val_mae       = avg_val_mae
            early_stop_counter = 0

            # =================================================================
            # FIX #10: Scaler statistics saved alongside model weights so that
            # test.py can load them from the checkpoint rather than recomputing
            # from a fresh get_dataloaders() call (which risks API variability).
            # =================================================================
            torch.save(
                {
                    'model':        model.state_dict(),
                    'scaler_mean':  scaler.mean,
                    'scaler_std':   scaler.std,
                    'epoch':        epoch + 1,
                    'val_loss':     avg_val_mae,
                },
                save_path
            )
            logger.info(f"  -> Saved Best Model (+ scaler) to {save_path}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            logger.info(f"[{model_name}] Early Stopping triggered at Epoch {epoch+1}")
            break

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["CADGT", "CAMT", "AMC_DSTGNN", "ST_ACENet"]
    )
    args = parser.parse_args()
    train_unified_model(args.model)
