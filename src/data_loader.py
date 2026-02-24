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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Traffic file not found at {file_path}")
    logger.info(f"Loading Traffic Data from {file_path}...")
    df = pd.read_hdf(file_path)
    df_clean = df.replace(0, np.nan).ffill().bfill()
    return df_clean.values, df_clean.index


def fetch_weather_api(lat, lon, start_date, end_date):
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

        # =================================================================
        # FIX #16: Linear interpolation replaces forward-fill.
        # Previously, ffill() caused all 12 five-minute slots inside one
        # hour to be identical (step function). Linear interpolation gives
        # smoothly varying sub-hourly weather values, improving feature
        # quality without inventing information.
        # =================================================================
        df = (df.set_index("time")
                .resample("5min")
                .interpolate(method="linear")
                .reset_index())
        return df
    except Exception as e:
        logger.warning(f"  [Warning] Weather API failed ({e}). Using zeros.")
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
        return pd.DataFrame({
            "time": pd.date_range(start_date, periods=days * 288, freq="5min"),
            "temperature_2m": 0, "precipitation": 0,
            "visibility": 10000, "windspeed_10m": 0
        })


def fetch_holiday_api(year=2012, country_code="US"):
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
    logger.info("Generating Cyclical Time Features...")
    tod = ((timestamps.hour * 60 + timestamps.minute) // 5).values
    dow = timestamps.dayofweek.values
    day_of_month = timestamps.day.values
    month = timestamps.month.values

    tod_sin = np.sin(2 * np.pi * tod / 288.0).reshape(-1, 1)
    tod_cos = np.cos(2 * np.pi * tod / 288.0).reshape(-1, 1)
    dow_sin = np.sin(2 * np.pi * dow / 7.0).reshape(-1, 1)
    dow_cos = np.cos(2 * np.pi * dow / 7.0).reshape(-1, 1)
    dom_sin = np.sin(2 * np.pi * day_of_month / 31.0).reshape(-1, 1)
    dom_cos = np.cos(2 * np.pi * day_of_month / 31.0).reshape(-1, 1)
    mon_sin = np.sin(2 * np.pi * month / 12.0).reshape(-1, 1)
    mon_cos = np.cos(2 * np.pi * month / 12.0).reshape(-1, 1)

    time_feat = np.concatenate(
        [tod_sin, tod_cos, dow_sin, dow_cos, dom_sin, dom_cos, mon_sin, mon_cos], axis=1
    ).astype(np.float32)
    time_feat = np.repeat(time_feat[:, None, :], N, axis=1)
    return time_feat


def merge_features(traffic, timestamps, config, train_end=None):
    T, N = traffic.shape
    start_date = timestamps[0].strftime("%Y-%m-%d")
    end_date   = timestamps[-1].strftime("%Y-%m-%d")

    weather_df = fetch_weather_api(config['lat'], config['lon'], start_date, end_date)
    weather_aligned = (weather_df.set_index("time")
                                 .reindex(timestamps)
                                 .ffill()
                                 .bfill())
    weather_feat = weather_aligned[
        ["temperature_2m", "precipitation", "visibility", "windspeed_10m"]
    ].values

    years = set(timestamps.year)
    holiday_dates = set()
    for year in years:
        holiday_dates |= fetch_holiday_api(year)

    holiday_feat = np.zeros((T, 1))
    current_dates = timestamps.date
    for i, d in enumerate(current_dates):
        if d in holiday_dates:
            holiday_feat[i] = 1

    time_feat = add_time_features(timestamps, N)

    weather_feat = np.nan_to_num(np.array(weather_feat, dtype=np.float32), nan=0.0)
    holiday_feat = np.array(holiday_feat, dtype=np.float32)

    traffic_reshaped = traffic[:, :, None].astype(np.float32)
    weather_feat = np.repeat(weather_feat[:, None, :], N, axis=1)
    holiday_feat = np.repeat(holiday_feat[:, None, :], N, axis=1)

    logger.info("Normalizing Weather Features...")
    if train_end is not None:
        mean_w = np.mean(weather_feat[:train_end], axis=(0, 1), keepdims=True)
        std_w  = np.std(weather_feat[:train_end],  axis=(0, 1), keepdims=True)
    else:
        mean_w = np.mean(weather_feat, axis=(0, 1), keepdims=True)
        std_w  = np.std(weather_feat,  axis=(0, 1), keepdims=True)
    weather_feat = (weather_feat - mean_w) / (std_w + 1e-5)

    X_merged = np.concatenate(
        [traffic_reshaped, weather_feat, holiday_feat, time_feat], axis=-1
    )
    logger.info(f"Data Merged. Final Shape: {X_merged.shape}")
    return X_merged


class TrafficSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X_merged, window=12, horizon=12):
        self.X_merged = torch.FloatTensor(X_merged)
        self.window   = window
        self.horizon  = horizon
        self.length   = len(X_merged) - window - horizon + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.X_merged[idx : idx + self.window]
        y = self.X_merged[idx + self.window : idx + self.window + self.horizon, :, 0]
        return x, y


def get_dataloaders(config):
    traffic, timestamps = load_traffic(config['data']['traffic_path'])

    total_len = len(traffic)
    train_end = int(0.7 * total_len)
    val_end   = int(0.8 * total_len)

    logger.info("Scaling Traffic Feature...")
    traffic_train = traffic[:train_end]
    mean = float(np.mean(traffic_train))
    std  = float(np.std(traffic_train))
    scaler = StandardScaler(mean, std)

    traffic_normalized = scaler.transform(traffic)

    X_merged = merge_features(
        traffic_normalized, timestamps, config['data'], train_end=train_end
    )

    train_data = X_merged[:train_end]
    val_data   = X_merged[train_end:val_end]
    test_data  = X_merged[val_end:]

    window  = config['training']['window']
    horizon = config['training']['horizon']

    train_ds = TrafficSequenceDataset(train_data, window, horizon)
    val_ds   = TrafficSequenceDataset(val_data,   window, horizon)
    test_ds  = TrafficSequenceDataset(test_data,  window, horizon)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    num_nodes    = X_merged.shape[1]
    num_features = X_merged.shape[2]
    dataset_info = {'num_nodes': num_nodes, 'num_features': num_features}

    return train_loader, val_loader, test_loader, scaler, dataset_info


def load_static_adj(pkl_path):
    with open(pkl_path, 'rb') as f:
        _, _, adj_mx = pickle.load(f, encoding='latin1')
    return adj_mx
