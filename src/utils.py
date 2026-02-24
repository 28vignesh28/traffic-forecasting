import torch
import numpy as np

def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    
    if mask.sum() == 0:
        return torch.tensor(0.0)
        
    mask = mask.float()
    loss = torch.abs(preds - labels)
    loss = loss * mask
    # Safe mean: Sum / Count of valid pixels
    return torch.sum(loss) / torch.sum(mask)

def masked_mape(preds, labels, null_val=0.0):
    # Use a small threshold for MAPE to exclude absolute zero (or null) values
    # that cause percentage errors to explode (division by zero).
    mape_threshold = max(null_val, 1e-4)

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > mape_threshold)
    
    if mask.sum() == 0:
        return torch.tensor(0.0)
        
    mask = mask.float()
    loss = torch.abs((preds - labels) / (torch.abs(labels) + 1e-5))
    loss = loss * mask
    return (torch.sum(loss) / torch.sum(mask)) * 100

def masked_rmse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    
    if mask.sum() == 0:
        return torch.tensor(0.0)
        
    mask = mask.float()
    loss = (preds - labels) ** 2
    loss = loss * mask
    return torch.sqrt(torch.sum(loss) / torch.sum(mask))

def masked_mse(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    
    if mask.sum() == 0:
        return torch.tensor(0.0)
        
    mask = mask.float()
    loss = (preds - labels) ** 2
    loss = loss * mask
    return torch.sum(loss) / torch.sum(mask)

def evaluate_metrics(y_true, y_pred, null_val=0.0):
    # Convert numpy to torch if needed
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).float()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).float()
    
    mae = masked_mae(y_pred, y_true, null_val).item()
    mse = masked_mse(y_pred, y_true, null_val).item()
    rmse = masked_rmse(y_pred, y_true, null_val).item()
    mape = masked_mape(y_pred, y_true, null_val).item()
    
    return mae, mse, rmse, mape

import logging
import os
from datetime import datetime

def create_directories():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

def setup_logging(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"{model_name}_{timestamp}.log")

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
