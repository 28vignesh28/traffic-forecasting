"""
Traffic Flow Forecasting & Congestion Behaviour Analysis Dashboard
AI-powered traffic prediction using CADGT (Context-Aware Dynamic Graph Transformer)
Merged: Teammate's premium dark theme + Live API future forecasting
"""

import os
import sys
import json
import time
import pickle
import datetime
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import requests

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so model imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cadgt import CADGT

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Flow Forecasting System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium smart-city theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.85);
    --border-card: rgba(99, 102, 241, 0.25);
    --accent-indigo: #6366f1;
    --accent-cyan: #22d3ee;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --glass: rgba(255,255,255,0.04);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #1a1a2e 40%, #16213e 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stTimeInput label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(17,24,39,0.9) 0%, rgba(30,41,59,0.8) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 24px 28px;
    backdrop-filter: blur(12px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 12px;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.18);
}
.metric-label {
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
    font-weight: 600;
}
.metric-value {
    color: #f1f5f9;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-sub {
    color: #64748b;
    font-size: 0.75rem;
    margin-top: 4px;
}

/* ── Congestion badges ── */
.badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
}
.badge-green {
    background: rgba(16,185,129,0.15);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.35);
}
.badge-yellow {
    background: rgba(245,158,11,0.15);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.35);
}
.badge-red {
    background: rgba(244,63,94,0.15);
    color: #fb7185;
    border: 1px solid rgba(244,63,94,0.35);
}

/* ── Header ── */
.dashboard-header {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(34,211,238,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 24px;
    text-align: center;
    backdrop-filter: blur(16px);
}
.dashboard-header h1 {
    background: linear-gradient(135deg, #818cf8, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
}
.dashboard-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 6px;
    font-weight: 400;
}

/* ── Section header ── */
.section-header {
    color: #cbd5e1;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(99,102,241,0.15);
}

/* ── AI insight card ── */
.insight-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(34,211,238,0.06) 100%);
    border-left: 3px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 8px 0;
    color: #cbd5e1;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* ── Weather card ── */
.weather-card {
    background: linear-gradient(135deg, rgba(34,211,238,0.08) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
}
.weather-icon { font-size: 2.2rem; margin-bottom: 4px; }
.weather-temp { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
.weather-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

/* ── Holiday pill ── */
.holiday-pill {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 24px;
    font-weight: 600;
    font-size: 0.85rem;
}
.holiday-yes {
    background: rgba(245,158,11,0.15);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.3);
}
.holiday-no {
    background: rgba(16,185,129,0.12);
    color: #6ee7b7;
    border: 1px solid rgba(16,185,129,0.25);
}

/* ── Animated pulse for live indicator ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #10b981;
    animation: pulse 2s infinite;
    margin-right: 6px;
    vertical-align: middle;
}

/* ── Change indicator arrows ── */
.change-up { color: #f43f5e; font-weight: 700; }
.change-down { color: #10b981; font-weight: 700; }
.change-neutral { color: #94a3b8; font-weight: 600; }

/* ── Hide default Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Plotly container rounding ── */
.stPlotlyChart {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(99,102,241,0.15);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cached data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading METR-LA traffic dataset…")
def load_traffic_data():
    """Load the METR-LA HDF5 dataset and return DataFrame + raw array."""
    h5_path = os.path.join(PROJECT_ROOT, "data", "METR-LA.h5")
    df = pd.read_hdf(h5_path)
    df_clean = df.replace(0, np.nan).ffill().bfill()
    return df_clean


@st.cache_resource(show_spinner="Loading CADGT model…")
def load_model():
    """Load the trained CADGT model + scaler from checkpoint."""
    adj_path = os.path.join(PROJECT_ROOT, "data", "adj_METR-LA.pkl")
    with open(adj_path, "rb") as f:
        _, _, adj_mx = pickle.load(f, encoding="latin1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(PROJECT_ROOT, "saved_models", "cadgt_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = CADGT(
        num_nodes=207, seq_len=12, future_len=12,
        ctx_dim=9, d_model=64, static_adj=adj_mx,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scaler_mean = checkpoint["scaler_mean"]
    scaler_std = checkpoint["scaler_std"]

    return model, device, scaler_mean, scaler_std


@st.cache_data(show_spinner="Loading weather data…")
def load_weather_cache():
    """Load the largest weather cache CSV."""
    data_dir = os.path.join(PROJECT_ROOT, "data")
    candidates = [f for f in os.listdir(data_dir) if f.startswith("weather_cache_") and f.endswith(".csv")]
    if not candidates:
        return None
    largest = max(candidates, key=lambda f: os.path.getsize(os.path.join(data_dir, f)))
    df = pd.read_csv(os.path.join(data_dir, largest), parse_dates=["time"])
    return df


@st.cache_data(show_spinner="Loading holiday data…")
def load_holidays():
    """Load holidays from the JSON cache."""
    cache_path = os.path.join(PROJECT_ROOT, "data", "holidays_cache.json")
    if not os.path.exists(cache_path):
        return set()
    with open(cache_path, "r") as f:
        data = json.load(f)
    all_dates = set()
    for year_key, dates in data.items():
        for d in dates:
            all_dates.add(pd.to_datetime(d).date())
    return all_dates


@st.cache_data(show_spinner=False)
def load_average_profile():
    """Load pre-computed average traffic profile for future forecasting."""
    path = os.path.join(PROJECT_ROOT, "data", "computed", "average_traffic_profile.npy")
    if os.path.exists(path):
        return np.load(path)
    return np.zeros((288, 7, 207))


# ──────────────────────────────────────────────────────────────────────────────
# Live API helpers (for Future Forecast tab)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_future_weather(lat, lon, target_date):
    """Fetches hourly weather forecast from Open-Meteo for a future date."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "visibility", "windspeed_10m"],
        "timezone": "America/Los_Angeles",
        "start_date": target_date.strftime("%Y-%m-%d"),
        "end_date": (target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        if "visibility" not in df.columns or df["visibility"].isnull().all():
            df["visibility"] = 10000.0
        return df
    except Exception as e:
        st.error(f"Failed to fetch live weather forecast: {e}")
        return None


@st.cache_data(ttl=86400)
def check_holiday_api(target_date):
    """Checks if the given date is a US holiday via Nager.Date API."""
    year = target_date.year
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/US"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            holidays = [datetime.datetime.strptime(h['date'], "%Y-%m-%d").date() for h in response.json()]
            return target_date in holidays
    except:
        pass
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_input_tensor(traffic_df, weather_df, holidays, target_ts, scaler_mean, scaler_std, num_nodes=207, window=12):
    """Build the [1, 12, 207, 10] input tensor for CADGT from raw data."""
    timestamps = traffic_df.index
    idx = timestamps.searchsorted(target_ts)
    idx = min(idx, len(timestamps) - 1)

    start = max(0, idx - window + 1)
    end = idx + 1
    if end - start < window:
        start = 0
        end = window

    actual_ts = timestamps[start:end]
    traffic_slice = traffic_df.iloc[start:end].values.astype(np.float32)

    found_in_dataset = abs((timestamps[idx] - target_ts).total_seconds()) < 600

    traffic_norm = (traffic_slice - scaler_mean) / (scaler_std + 1e-5)

    if weather_df is not None and len(weather_df) > 0:
        weather_indexed = weather_df.set_index("time")
        weather_aligned = weather_indexed.reindex(actual_ts, method="nearest")
        weather_feat = weather_aligned[["temperature_2m", "precipitation", "visibility", "windspeed_10m"]].values.astype(np.float32)
        weather_feat = np.nan_to_num(weather_feat, nan=0.0)
    else:
        weather_feat = np.zeros((window, 4), dtype=np.float32)

    w_mean = weather_feat.mean(axis=0, keepdims=True)
    w_std = weather_feat.std(axis=0, keepdims=True) + 1e-5
    weather_feat = (weather_feat - w_mean) / w_std

    holiday_feat = np.zeros((window, 1), dtype=np.float32)
    for i, ts in enumerate(actual_ts):
        if ts.date() in holidays:
            holiday_feat[i] = 1.0

    tod = ((actual_ts.hour * 60 + actual_ts.minute) // 5).values
    dow = actual_ts.dayofweek.values
    tod_sin = np.sin(2 * np.pi * tod / 288.0).reshape(-1, 1)
    tod_cos = np.cos(2 * np.pi * tod / 288.0).reshape(-1, 1)
    dow_sin = np.sin(2 * np.pi * dow / 7.0).reshape(-1, 1)
    dow_cos = np.cos(2 * np.pi * dow / 7.0).reshape(-1, 1)
    time_feat = np.concatenate([tod_sin, tod_cos, dow_sin, dow_cos], axis=1).astype(np.float32)

    traffic_3d = traffic_norm[:, :, None]
    weather_3d = np.repeat(weather_feat[:, None, :], num_nodes, axis=1)
    holiday_3d = np.repeat(holiday_feat[:, None, :], num_nodes, axis=1)
    time_3d = np.repeat(time_feat[:, None, :], num_nodes, axis=1)

    X = np.concatenate([traffic_3d, weather_3d, holiday_3d, time_3d], axis=-1)
    tensor = torch.FloatTensor(X).unsqueeze(0)

    return tensor, actual_ts, found_in_dataset


def get_congestion_info(speed_mph):
    """Classify congestion level from speed in mph."""
    if speed_mph > 50:
        return "Free Flow", "badge-green", "🟢"
    elif speed_mph >= 30:
        return "Moderate Traffic", "badge-yellow", "🟡"
    else:
        return "Heavy Congestion", "badge-red", "🔴"


def get_weather_display(weather_df, target_ts):
    """Get weather info nearest to the target timestamp."""
    if weather_df is None or len(weather_df) == 0:
        return {"temp": "N/A", "type": "Unknown", "icon": "❓", "wind": "N/A", "precip": "0.0 mm"}

    weather_indexed = weather_df.set_index("time")
    idx = weather_indexed.index.searchsorted(target_ts)
    idx = min(idx, len(weather_indexed) - 1)
    row = weather_indexed.iloc[idx]

    temp_c = row.get("temperature_2m", 0)
    precip = row.get("precipitation", 0)
    visibility = row.get("visibility", 10000)
    wind = row.get("windspeed_10m", 0)

    if precip > 2.0:
        wtype, icon = "Rainy", "🌧️"
    elif precip > 0.1:
        wtype, icon = "Light Rain", "🌦️"
    elif visibility < 2000:
        wtype, icon = "Foggy", "🌫️"
    elif wind > 30:
        wtype, icon = "Windy", "💨"
    elif temp_c > 30:
        wtype, icon = "Hot & Sunny", "☀️"
    elif temp_c > 20:
        wtype, icon = "Sunny", "🌤️"
    elif temp_c > 10:
        wtype, icon = "Partly Cloudy", "⛅"
    else:
        wtype, icon = "Cloudy", "☁️"

    temp_f = temp_c * 9 / 5 + 32
    wind_mph = wind * 0.621371

    return {
        "temp": f"{temp_f:.1f}°F",
        "type": wtype,
        "icon": icon,
        "wind": f"{wind_mph:.1f} mph",
        "precip": f"{precip:.1f} mm",
    }


def generate_ai_insights(pred_speed, congestion_label, horizon_label, actual_speed=None, trend_speeds=None):
    """Generate intelligent AI insights based on prediction results."""
    insights = []

    if trend_speeds is not None and len(trend_speeds) > 2:
        first_half = np.mean(trend_speeds[: len(trend_speeds) // 2])
        second_half = np.mean(trend_speeds[len(trend_speeds) // 2 :])
        diff = second_half - first_half
        if diff > 3:
            insights.append("📈 Traffic speed is expected to **increase** over the prediction window, suggesting improving conditions.")
        elif diff < -3:
            insights.append("📉 Traffic speed is projected to **decrease**, indicating potential congestion build-up ahead.")
        else:
            insights.append("📊 Traffic conditions remain **stable** throughout the forecast horizon.")

    if congestion_label == "Heavy Congestion":
        insights.append("🚨 **Heavy congestion** detected near this sensor. Consider alternative routes or delayed departure.")
    elif congestion_label == "Moderate Traffic":
        insights.append("⚠️ **Moderate congestion** likely near this sensor. Allow extra travel time for safety.")
    else:
        insights.append("✅ **Free flow** conditions expected. Optimal time for travel through this corridor.")

    if actual_speed is not None:
        change = pred_speed - actual_speed
        if abs(change) > 5:
            direction = "increase" if change > 0 else "decrease"
            insights.append(f"🔮 Model predicts a **{abs(change):.1f} mph {direction}** compared to the actual recorded speed at this timestamp.")

    if "60" in horizon_label:
        insights.append("🕐 Long-range 60-minute forecast — predictions at this range have somewhat higher uncertainty.")
    elif "5 " in horizon_label:
        insights.append("⚡ Short-range 5-minute forecast — highest confidence prediction window.")

    return insights


# ──────────────────────────────────────────────────────────────────────────────
# Load all data
# ──────────────────────────────────────────────────────────────────────────────
traffic_df = load_traffic_data()
model, device, scaler_mean, scaler_std = load_model()
weather_df = load_weather_cache()
holidays = load_holidays()
base_profile = load_average_profile()

dataset_start = traffic_df.index[0].date()
dataset_end = traffic_df.index[-1].date()
sensor_ids = list(range(traffic_df.shape[1]))

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <h1>🚦 Traffic Flow Forecasting System</h1>
    <p><span class="live-dot"></span> AI-powered traffic prediction using CADGT &nbsp;•&nbsp; METR-LA Dataset &nbsp;•&nbsp; 207 Sensors</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — Shared Controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 18px 0;">
        <span style="font-size:1.6rem;">🎛️</span>
        <div style="color:#cbd5e1; font-size:1.05rem; font-weight:700; letter-spacing:0.04em; margin-top:4px;">Control Panel</div>
        <div style="color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;">Smart Traffic Monitoring</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    sel_sensor = st.selectbox(
        "📡 Sensor ID",
        options=sensor_ids,
        index=100,
        help="METR-LA sensor index (0–206)",
    )

    horizon_map = {"5 minutes": 0, "15 minutes": 2, "30 minutes": 5, "60 minutes": 11}
    sel_horizon_label = st.selectbox(
        "🎯 Prediction Horizon",
        options=list(horizon_map.keys()),
        index=1,
    )
    sel_horizon_idx = horizon_map[sel_horizon_label]

    st.markdown("---")
    st.markdown(f"""
    <div style="color:#475569; font-size:0.7rem; text-align:center; line-height:1.6;">
        <strong style="color:#64748b;">Model:</strong> CADGT v1.0<br>
        <strong style="color:#64748b;">Dataset:</strong> METR-LA<br>
        <strong style="color:#64748b;">Sensors:</strong> 207 nodes<br>
        <strong style="color:#64748b;">Device:</strong> {'CUDA' if device.type == 'cuda' else 'CPU'}
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Future Forecast Simulator", "📊 Sensor Prediction View"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: FUTURE FORECAST SIMULATOR (Live APIs)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">🔮 Live Future Traffic Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
        Pick any <strong>future date and time</strong> — the system will fetch live weather forecasts from
        <strong>Open-Meteo API</strong> and check US holidays via <strong>Nager.Date API</strong> to generate a prediction.
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2 = st.columns(2)
    today = datetime.datetime.now()
    target_d = fc1.date_input("📅 Target Date", value=today.date(), key="date_tab1")
    m = (today.minute // 5) * 5
    target_t = fc2.time_input("🕐 Target Time", value=datetime.time(today.hour, m), key="time_tab1")

    if st.button("⚡ Generate Future Forecast", type="primary", use_container_width=True):
        target_datetime = datetime.datetime.combine(target_d, target_t)

        with st.spinner("Fetching Live Weather & Running CADGT..."):
            # Coordinates for LA (METR-LA dataset location)
            lat, lon = 34.0522, -118.2437
            future_weather = fetch_future_weather(lat, lon, target_datetime)
            is_holiday = check_holiday_api(target_d)

            # Build synthetic input from average traffic profile
            history_times = [target_datetime - datetime.timedelta(minutes=5 * i) for i in range(11, -1, -1)]
            custom_x = np.zeros((1, 12, 207, 10), dtype=np.float32)

            for i, t_step in enumerate(history_times):
                tod_idx = (t_step.hour * 60 + t_step.minute) // 5
                dow_idx = t_step.weekday()

                profiled_traffic_mph = base_profile[tod_idx, dow_idx, :]
                custom_x[0, i, :, 0] = (profiled_traffic_mph - scaler_mean) / (scaler_std + 1e-5)

                if future_weather is not None:
                    nearest = future_weather.iloc[(future_weather['time'] - t_step).abs().argsort()[:1]]
                    temp = nearest["temperature_2m"].values[0]
                    precip = nearest["precipitation"].values[0]
                    vis = nearest.get("visibility", pd.Series([10000.0])).values[0]
                    wind = nearest["windspeed_10m"].values[0]
                else:
                    temp, precip, vis, wind = 20.0, 0.0, 10000.0, 5.0

                # Simple z-score (profile-level)
                custom_x[0, i, :, 1] = temp / 20.0
                custom_x[0, i, :, 2] = precip / 5.0
                custom_x[0, i, :, 3] = vis / 10000.0
                custom_x[0, i, :, 4] = wind / 15.0

                custom_x[0, i, :, 5] = float(is_holiday)

                tod_norm = tod_idx / 288.0
                dow_norm = dow_idx / 7.0
                custom_x[0, i, :, 6] = np.sin(2 * np.pi * tod_norm)
                custom_x[0, i, :, 7] = np.cos(2 * np.pi * tod_norm)
                custom_x[0, i, :, 8] = np.sin(2 * np.pi * dow_norm)
                custom_x[0, i, :, 9] = np.cos(2 * np.pi * dow_norm)

            t_input = torch.FloatTensor(custom_x).to(device)
            start_infer = time.time()
            with torch.no_grad():
                preds_future = model(t_input)
            latency = (time.time() - start_infer) * 1000

            p_future_np = preds_future.cpu().numpy()[0]
            future_speeds_mph = p_future_np * scaler_std + scaler_mean

        # Results
        sensor_future = future_speeds_mph[:, sel_sensor]
        future_pred = float(sensor_future[sel_horizon_idx])
        congestion_label, congestion_badge, congestion_icon = get_congestion_info(future_pred)
        h_str = "🎉 Holiday" if is_holiday else "📋 Regular Day"
        target_display = target_datetime.strftime("%A, %b %d %Y at %I:%M %p")
        temp_f = temp * 9 / 5 + 32

        # Insights
        future_insights = generate_ai_insights(future_pred, congestion_label, sel_horizon_label, trend_speeds=sensor_future)

        # Metrics row
        st.markdown('<div class="section-header">📊 Forecast Results</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Speed</div>
                <div class="metric-value">{future_pred:.1f} <span style="font-size:0.9rem;color:#64748b;">mph</span></div>
                <div class="metric-sub">@ {sel_horizon_label} horizon</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Forecast For</div>
                <div class="metric-value" style="font-size:1.3rem;">{target_datetime.strftime('%H:%M')}</div>
                <div class="metric-sub">{target_display}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Weather</div>
                <div class="metric-value" style="font-size:1.3rem;">🌡️ {temp_f:.0f}°F &nbsp; 🌧️ {precip:.1f}mm</div>
                <div class="metric-sub">{h_str}</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Congestion Level</div>
                <div style="margin-top:8px;">
                    <span style="font-size:1.5rem;">{congestion_icon}</span>
                    <span class="badge {congestion_badge}" style="margin-left:6px;">{congestion_label}</span>
                </div>
                <div class="metric-sub">Inference: {latency:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)

        # Future chart
        st.markdown('<div class="section-header">📈 Future Speed Timeline</div>', unsafe_allow_html=True)
        future_minutes = [(i + 1) * 5 for i in range(12)]
        future_labels = [f"+{m}m" for m in future_minutes]

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=future_minutes, y=sensor_future,
            mode="lines+markers", name="CADGT Forecast",
            line=dict(color="#a855f7", width=3, shape="spline"),
            marker=dict(size=7, color="#a855f7", line=dict(width=1, color="#fff")),
            hovertemplate="<b>+%{x} min</b><br>Predicted: %{y:.1f} mph<extra></extra>",
        ))
        fig_f.add_vline(x=future_minutes[sel_horizon_idx], line_dash="dash", line_color="#6366f1", opacity=0.6,
                        annotation_text=f"Selected: {sel_horizon_label}", annotation_font_color="#a5b4fc", annotation_font_size=11)
        fig_f.add_hrect(y0=50, y1=80, fillcolor="rgba(16,185,129,0.06)", line_width=0,
                        annotation_text="Free Flow", annotation_position="top right", annotation_font=dict(color="#6ee7b7", size=10))
        fig_f.add_hrect(y0=30, y1=50, fillcolor="rgba(245,158,11,0.06)", line_width=0,
                        annotation_text="Moderate", annotation_position="top right", annotation_font=dict(color="#fbbf24", size=10))
        fig_f.add_hrect(y0=0, y1=30, fillcolor="rgba(244,63,94,0.06)", line_width=0,
                        annotation_text="Congested", annotation_position="top right", annotation_font=dict(color="#fb7185", size=10))
        fig_f.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            title=dict(text=f"Future Forecast — Sensor #{sel_sensor}", font=dict(size=16, color="#e2e8f0")),
            xaxis=dict(title="Minutes from Selected Time", gridcolor="rgba(148,163,184,0.1)", tickvals=future_minutes, ticktext=future_labels),
            yaxis=dict(title="Speed (mph)", gridcolor="rgba(148,163,184,0.1)", range=[0, max(80, max(sensor_future) + 10)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
            margin=dict(l=50, r=30, t=60, b=50), height=420, hovermode="x unified",
        )
        st.plotly_chart(fig_f, width='stretch', key="future_chart")

        # AI Insights
        st.markdown('<div class="section-header">🤖 AI Traffic Insights</div>', unsafe_allow_html=True)
        for insight in future_insights:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SENSOR PREDICTION VIEW (Historical Replay)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📅 Historical Data Explorer</div>', unsafe_allow_html=True)

    h1, h2 = st.columns(2)
    sel_date = h1.date_input(
        "📅 Select Date",
        value=datetime.date(2012, 4, 15),
        min_value=dataset_start,
        max_value=dataset_end,
        help=f"METR-LA range: {dataset_start} to {dataset_end}",
        key="date_tab2"
    )
    sel_time = h2.time_input(
        "🕐 Select Time",
        value=datetime.time(8, 45),
        step=datetime.timedelta(minutes=5),
        help="24-hour format, 5-minute intervals",
        key="time_tab2"
    )

    # Run prediction
    target_datetime = datetime.datetime.combine(sel_date, sel_time)
    target_ts = pd.Timestamp(target_datetime)

    input_tensor, actual_timestamps, found_in_dataset = build_input_tensor(
        traffic_df, weather_df, holidays, target_ts, scaler_mean, scaler_std
    )

    with torch.no_grad():
        input_tensor_dev = input_tensor.to(device)
        preds_raw = model(input_tensor_dev)
        preds_np = preds_raw.cpu().numpy()[0]

    # Inverse transform to mph (raw METR-LA units)
    preds_mph = preds_np * scaler_std + scaler_mean
    sensor_preds_mph = preds_mph[:, sel_sensor]
    pred_speed = float(sensor_preds_mph[sel_horizon_idx])

    # Actual speed
    actual_speed = None
    if found_in_dataset:
        horizon_offset = (sel_horizon_idx + 1) * 5
        future_ts = target_ts + pd.Timedelta(minutes=horizon_offset)
        if future_ts in traffic_df.index:
            actual_speed = float(traffic_df.loc[future_ts].iloc[sel_sensor])

    congestion_label, congestion_badge, congestion_icon = get_congestion_info(pred_speed)
    weather_info = get_weather_display(weather_df, target_ts)
    is_holiday = sel_date in holidays
    insights = generate_ai_insights(pred_speed, congestion_label, sel_horizon_label, actual_speed=actual_speed, trend_speeds=sensor_preds_mph)

    # ── Metrics ──
    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Speed</div>
            <div class="metric-value">{pred_speed:.1f} <span style="font-size:0.9rem;color:#64748b;">mph</span></div>
            <div class="metric-sub">@ {sel_horizon_label} horizon</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prediction Time</div>
            <div class="metric-value">{sel_time.strftime('%H:%M')}</div>
            <div class="metric-sub">{sel_date.strftime('%B %d, %Y')}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sensor Station</div>
            <div class="metric-value">#{sel_sensor}</div>
            <div class="metric-sub">METR-LA network node</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Congestion Level</div>
            <div style="margin-top:8px;">
                <span style="font-size:1.5rem;">{congestion_icon}</span>
                <span class="badge {congestion_badge}" style="margin-left:6px;">{congestion_label}</span>
            </div>
            <div class="metric-sub">Based on predicted speed</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chart + Info ──
    st.markdown('<div class="section-header">📈 Traffic Visualization</div>', unsafe_allow_html=True)
    chart_col, info_col = st.columns([3, 1.3])

    with chart_col:
        future_minutes = [(i + 1) * 5 for i in range(12)]
        future_labels = [f"+{m}m" for m in future_minutes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_minutes, y=sensor_preds_mph,
            mode="lines+markers", name="Predicted Speed",
            line=dict(color="#f43f5e", width=3, shape="spline"),
            marker=dict(size=7, color="#f43f5e", line=dict(width=1, color="#fff")),
            hovertemplate="<b>+%{x} min</b><br>Predicted: %{y:.1f} mph<extra></extra>",
        ))

        if found_in_dataset:
            actual_future_speeds = []
            actual_future_valid = []
            for i in range(12):
                ft = target_ts + pd.Timedelta(minutes=(i + 1) * 5)
                if ft in traffic_df.index:
                    actual_future_speeds.append(float(traffic_df.loc[ft].iloc[sel_sensor]))
                    actual_future_valid.append(future_minutes[i])
                else:
                    actual_future_speeds.append(None)
                    actual_future_valid.append(future_minutes[i])

            actual_filtered = [(x, y) for x, y in zip(actual_future_valid, actual_future_speeds) if y is not None]
            if actual_filtered:
                ax, ay = zip(*actual_filtered)
                fig.add_trace(go.Scatter(
                    x=list(ax), y=list(ay),
                    mode="lines+markers", name="Actual Speed",
                    line=dict(color="#3b82f6", width=3, shape="spline"),
                    marker=dict(size=7, color="#3b82f6", symbol="diamond", line=dict(width=1, color="#fff")),
                    hovertemplate="<b>+%{x} min</b><br>Actual: %{y:.1f} mph<extra></extra>",
                ))

        fig.add_vline(x=future_minutes[sel_horizon_idx], line_dash="dash", line_color="#6366f1", opacity=0.6,
                      annotation_text=f"Selected: {sel_horizon_label}", annotation_font_color="#a5b4fc", annotation_font_size=11)
        fig.add_hrect(y0=50, y1=max(80, max(sensor_preds_mph) + 10), fillcolor="rgba(16,185,129,0.06)", line_width=0,
                      annotation_text="Free Flow", annotation_position="top right", annotation_font=dict(color="#6ee7b7", size=10))
        fig.add_hrect(y0=30, y1=50, fillcolor="rgba(245,158,11,0.06)", line_width=0,
                      annotation_text="Moderate", annotation_position="top right", annotation_font=dict(color="#fbbf24", size=10))
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(244,63,94,0.06)", line_width=0,
                      annotation_text="Congested", annotation_position="top right", annotation_font=dict(color="#fb7185", size=10))

        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            title=dict(text=f"Traffic Speed Forecast — Sensor #{sel_sensor}", font=dict(size=16, color="#e2e8f0")),
            xaxis=dict(title="Minutes from Selected Time", gridcolor="rgba(148,163,184,0.1)", tickvals=future_minutes, ticktext=future_labels),
            yaxis=dict(title="Speed (mph)", gridcolor="rgba(148,163,184,0.1)", range=[0, max(80, max(sensor_preds_mph) + 15)]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
            margin=dict(l=50, r=30, t=60, b=50), height=420, hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch', key="main_chart")

    with info_col:
        st.markdown('<div class="section-header">🔄 Traffic Change</div>', unsafe_allow_html=True)
        if actual_speed is not None:
            change = pred_speed - actual_speed
            if change > 0:
                change_html = f'<span class="change-up">+{change:.1f} mph ↑</span>'
            elif change < 0:
                change_html = f'<span class="change-down">{change:.1f} mph ↓</span>'
            else:
                change_html = '<span class="change-neutral">0.0 mph →</span>'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Actual Speed</div>
                <div class="metric-value" style="font-size:1.4rem;">{actual_speed:.1f} <span style="font-size:0.8rem;color:#64748b;">mph</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Predicted Speed</div>
                <div class="metric-value" style="font-size:1.4rem;">{pred_speed:.1f} <span style="font-size:0.8rem;color:#64748b;">mph</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Traffic Change</div>
                <div style="margin-top:6px;">{change_html}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Speed</div>
                <div class="metric-value" style="font-size:1.4rem;">{pred_speed:.1f} <span style="font-size:0.8rem;color:#64748b;">mph</span></div>
                <div class="metric-sub" style="margin-top:8px;color:#f59e0b;">
                    ⚠️ Actual data not available for this timestamp.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">🤖 AI Insights</div>', unsafe_allow_html=True)
        for insight in insights:
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    # ── Weather & Holiday ──
    st.markdown('<div class="section-header">🌍 Context Information</div>', unsafe_allow_html=True)
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        st.markdown(f"""<div class="weather-card"><div class="weather-icon">{weather_info['icon']}</div><div class="weather-temp">{weather_info['temp']}</div><div class="weather-label">Temperature</div></div>""", unsafe_allow_html=True)
    with w2:
        st.markdown(f"""<div class="weather-card"><div class="weather-icon">🌤️</div><div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;margin-top:4px;">{weather_info['type']}</div><div class="weather-label">Weather Type</div></div>""", unsafe_allow_html=True)
    with w3:
        st.markdown(f"""<div class="weather-card"><div class="weather-icon">💧</div><div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;margin-top:4px;">{weather_info['precip']}</div><div class="weather-label">Precipitation</div></div>""", unsafe_allow_html=True)
    with w4:
        st.markdown(f"""<div class="weather-card"><div class="weather-icon">💨</div><div style="font-size:1.1rem;font-weight:600;color:#f1f5f9;margin-top:4px;">{weather_info['wind']}</div><div class="weather-label">Wind Speed</div></div>""", unsafe_allow_html=True)
    with w5:
        pill = '<span class="holiday-pill holiday-yes">🎉 Holiday</span>' if is_holiday else '<span class="holiday-pill holiday-no">📋 Regular Day</span>'
        st.markdown(f"""<div class="weather-card"><div class="weather-icon">📅</div><div style="margin-top:8px;">{pill}</div><div class="weather-label" style="margin-top:6px;">Holiday Status</div></div>""", unsafe_allow_html=True)


# ── Footer ──
st.markdown("""
<div style="text-align:center; padding:30px 0 10px 0; color:#475569; font-size:0.72rem; letter-spacing:0.04em;">
    Traffic Flow Forecasting &amp; Congestion Behaviour Analysis &nbsp;•&nbsp; CADGT &nbsp;•&nbsp; METR-LA Dataset<br>
    Context-Aware Dynamic Graph Transformer &nbsp;|&nbsp; Intelligent Transportation System
</div>
""", unsafe_allow_html=True)
