import os
import pickle
import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger("DataLoader")

class StandardScaler:
    """Standardize data by removing the mean and scaling to unit variance."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-5)

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            return (data * self.std) + self.mean
        return (data * self.std) + self.mean


def load_traffic(file_path):
    """Loads traffic sensor data (h5 format)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Traffic file not found at {file_path}")
    
    logger.info(f"Loading Traffic Data from {file_path}...")
    df = pd.read_hdf(file_path)
    
    # Clean zeros and NaN
    df_clean = df.replace(0, np.nan)
    df_clean = df_clean.ffill().bfill()
    
    traffic_array = df_clean.values
    return traffic_array, df_clean.index


def fetch_weather_api(lat, lon, start_date, end_date):
    """Fetches weather from Open-Meteo API."""
    logger.info(f"Fetching Weather Data ({start_date} to {end_date})...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "visibility", "windspeed_10m"],
        "timezone": "UTC"
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        
        # [FIX] Resample from hourly to 5-minute intervals using linear interpolation
        # This provides sub-hourly weather variance across the 12 steps.
        df = df.set_index("time").resample("5min").interpolate(method="linear").bfill().ffill().reset_index()
        return df
    except Exception as e:
        logger.warning(f"  [Warning] Weather API failed ({e}). Using zeros.")
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        return pd.DataFrame({
            "time": pd.date_range(start_date, periods=days * 288, freq="5min"),
            "temperature_2m": 0, "precipitation": 0, "visibility": 10000, "windspeed_10m": 0
        })


def fetch_holiday_api(year=2012, country_code="US"):
    """Fetches public holidays from Nager.Date API."""
    logger.info(f"Fetching Holiday Data for {year}...")
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        holidays = response.json()
        return set(pd.to_datetime([h['date'] for h in holidays]).date)
    except Exception as e:
        logger.warning(f"  [Warning] Holiday API failed ({e}). Using empty set.")
        return set()


def add_time_features(timestamps, N):
    """Generates Cyclical Time Features (Sin/Cos)."""
    logger.info("Generating Cyclical Time Features...")
    T = len(timestamps)
    
    tod = ((timestamps.hour * 60 + timestamps.minute) // 5).values  # 0 to 287
    dow = timestamps.dayofweek.values  # 0 to 6
    day_of_month = timestamps.day.values # 1 to 31
    month = timestamps.month.values # 1 to 12
    
    tod_sin = np.sin(2 * np.pi * tod / 288.0).reshape(-1, 1)
    tod_cos = np.cos(2 * np.pi * tod / 288.0).reshape(-1, 1)
    
    dow_sin = np.sin(2 * np.pi * dow / 7.0).reshape(-1, 1)
    dow_cos = np.cos(2 * np.pi * dow / 7.0).reshape(-1, 1)
    
    dom_sin = np.sin(2 * np.pi * day_of_month / 31.0).reshape(-1, 1)
    dom_cos = np.cos(2 * np.pi * day_of_month / 31.0).reshape(-1, 1)
    
    mon_sin = np.sin(2 * np.pi * month / 12.0).reshape(-1, 1)
    mon_cos = np.cos(2 * np.pi * month / 12.0).reshape(-1, 1)
    
    # Shape: [T, 8] -> Broadcast to [T, N, 8]
    time_feat = np.concatenate([tod_sin, tod_cos, dow_sin, dow_cos, dom_sin, dom_cos, mon_sin, mon_cos], axis=1).astype(np.float32)
    time_feat = np.repeat(time_feat[:, None, :], N, axis=1)
    return time_feat




def merge_features(traffic, timestamps, config, train_end=None):
    """Merges all 14 features uniformly."""
    T, N = traffic.shape
    
    start_date = timestamps[0].strftime("%Y-%m-%d")
    end_date = timestamps[-1].strftime("%Y-%m-%d")
    
    # 1. Fetch Weather
    weather_df = fetch_weather_api(config['lat'], config['lon'], start_date, end_date)
    
    # EXACT ALIGNMENT: Reindex weather data to perfectly match the raw traffic timestamps
    # This prevents dimension mismatch if the Weather API is missing an hour at the end
    weather_aligned = weather_df.set_index("time").reindex(timestamps).ffill().bfill()
    weather_feat = weather_aligned[["temperature_2m", "precipitation", "visibility", "windspeed_10m"]].values
    
    # 2. Fetch Holidays (all years in dataset)
    years = set(timestamps.year)
    holiday_dates = set()
    for yr in years:
        holiday_dates |= fetch_holiday_api(yr)
    
    holiday_feat = np.zeros((T, 1))
    current_dates = timestamps.date
    for i, d in enumerate(current_dates):
        if d in holiday_dates:
            holiday_feat[i] = 1

    # 3. Time Features (8 features)
    time_feat = add_time_features(timestamps, N)
    
    # 4. Clean + Broadcast
    weather_feat = np.nan_to_num(np.array(weather_feat, dtype=np.float32), nan=0.0)
    holiday_feat = np.array(holiday_feat, dtype=np.float32)

    traffic_reshaped = traffic[:, :, None].astype(np.float32)
    
    weather_feat = np.repeat(weather_feat[:, None, :], N, axis=1)
    holiday_feat = np.repeat(holiday_feat[:, None, :], N, axis=1)

    # 5. Normalize Weather (Z-Score) — use ONLY training data stats to prevent leakage
    logger.info("Normalizing Weather Features...")
    if train_end is not None:
        mean_w = np.mean(weather_feat[:train_end], axis=(0, 1), keepdims=True)
        std_w = np.std(weather_feat[:train_end], axis=(0, 1), keepdims=True)
    else:
        mean_w = np.mean(weather_feat, axis=(0, 1), keepdims=True)
        std_w = np.std(weather_feat, axis=(0, 1), keepdims=True)
    weather_feat = (weather_feat - mean_w) / (std_w + 1e-5)

    # 6. Concatenate 14 Complete Features
    # [1 Traffic, 4 Weather, 1 Holiday, 8 Time]
    X_merged = np.concatenate([traffic_reshaped, weather_feat, holiday_feat, time_feat], axis=-1)
    logger.info(f"Data Merged. Final Shape: {X_merged.shape}")
    
    return X_merged


class TrafficSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X_merged, window=12, horizon=12, device="cpu"):
        # Move entire dataset to GPU VRAM immediately to bypass PCIe bottleneck
        self.X_merged = torch.FloatTensor(X_merged).to(device)
        self.window = window
        self.horizon = horizon
        self.length = len(X_merged) - window - horizon + 1
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        x = self.X_merged[idx : idx + self.window]
        # Target is only Feature 0 (Traffic Speed)
        y = self.X_merged[idx + self.window : idx + self.window + self.horizon, :, 0]
        return x, y

def get_dataloaders(config):
    """
    Main entry point for loading, merging, scaling, and splitting data.
    """
    traffic, timestamps = load_traffic(config['data']['traffic_path'])
    
    # Pre-split the data before merge to apply StandardScaler properly
    total_len = len(traffic)
    train_end = int(0.7 * total_len)
    val_end = int(0.8 * total_len)
    
    # Fit StandardScaler ONLY on Training Traffic Data
    logger.info("Scaling Traffic Feature...")
    traffic_train = traffic[:train_end]
    mean = float(np.mean(traffic_train))
    std = float(np.std(traffic_train))
    scaler = StandardScaler(mean, std)
    
    traffic_normalized = scaler.transform(traffic)
    
    # Merge Features (Using Normalized Traffic)
    X_merged = merge_features(traffic_normalized, timestamps, config['data'], train_end=train_end)
    
    # Split into sets
    train_data = X_merged[:train_end]
    val_data = X_merged[train_end:val_end]
    test_data = X_merged[val_end:]
    
    # Create Dynamic PyTorch Datasets (No memory explosion)
    window = config['training']['window']
    horizon = config['training']['horizon']
    
    # Detect GPU and load datasets directly into VRAM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading full datasets directly into {device} VRAM to bypass PCIe bottlenecks...")

    train_ds = TrafficSequenceDataset(train_data, window, horizon, device=device)
    val_ds = TrafficSequenceDataset(val_data, window, horizon, device=device)
    test_ds = TrafficSequenceDataset(test_data, window, horizon, device=device)
    
    batch_size = config['training']['batch_size']
    
    # Convert to PyTorch DataLoaders (Disabled pin_memory for Windows stability)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Extract structural dimensions
    num_nodes = X_merged.shape[1]
    num_features = X_merged.shape[2]
    dataset_info = {
        'num_nodes': num_nodes,
        'num_features': num_features
    }
    
    return train_loader, val_loader, test_loader, scaler, dataset_info


def load_static_adj(pkl_path):
    """Loads the physical adjacency matrix derived from the road network."""
    with open(pkl_path, 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
    return adj_mx
