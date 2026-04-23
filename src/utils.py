import torch
import numpy as np
import random
import os
import logging
from datetime import datetime


def set_seed(seed: int = 42):
    """Seed all RNG sources for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels > null_val

    if mask.sum() == 0:
        return torch.tensor(float("nan"))

    mask = mask.float()
    loss = torch.abs(preds - labels)
    loss = loss * mask
    return torch.sum(loss) / torch.sum(mask)


def masked_mape(preds, labels, null_val=0.0):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels > null_val

    if mask.sum() == 0:
        return torch.tensor(float("nan"))

    mask = mask.float()

    loss = torch.abs((preds - labels) / (torch.abs(labels) + 1e-5))
    loss = loss * mask
    return (torch.sum(loss) / torch.sum(mask)) * 100


def masked_rmse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels > null_val

    if mask.sum() == 0:
        return torch.tensor(float("nan"))

    mask = mask.float()
    loss = (preds - labels) ** 2
    loss = loss * mask
    return torch.sqrt(torch.sum(loss) / torch.sum(mask))


def masked_mse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels > null_val

    if mask.sum() == 0:
        return torch.tensor(float("nan"))

    mask = mask.float()
    loss = (preds - labels) ** 2
    loss = loss * mask
    return torch.sum(loss) / torch.sum(mask)


def evaluate_metrics(y_true, y_pred, null_val=0.0):
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()

    mae = masked_mae(y_pred, y_true, null_val).item()
    mse = masked_mse(y_pred, y_true, null_val).item()
    rmse = masked_rmse(y_pred, y_true, null_val).item()
    mape = masked_mape(y_pred, y_true, null_val).item()

    return mae, mse, rmse, mape


def create_directories():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)


def setup_logging(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"{model_name}_{timestamp}.log")

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
