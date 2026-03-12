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
    """Fetches hourly weather forecast from Open-Meteo for a future date, or historical archive for past dates."""
    today = datetime.datetime.now().date()
    
    # Open-Meteo requires different endpoints for past vs future
    if target_date.date() < today:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ["temperature_2m", "precipitation", "visibility", "windspeed_10m"],
        "timezone": "America/Los_Angeles",
        "start_date": target_date.strftime("%Y-%m-%d"),
        "end_date": (target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    }
    
    # Archive API doesn't have visibility, provide default if needed
    if "archive" in url:
        params["hourly"] = ["temperature_2m", "precipitation", "windspeed_10m"]
        
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
        st.error(f"Failed to fetch weather data: {e}")
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: FUTURE FORECAST SIMULATOR (Live APIs)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔮 Live Future Traffic Prediction</div>', unsafe_allow_html=True)
st.markdown("""
<div class="insight-card">
    Pick any <strong>future date and time</strong> — the system will fetch live weather forecasts from
    <strong>Open-Meteo API</strong> and check US holidays via <strong>Nager.Date API</strong> to generate a prediction.
</div>
""", unsafe_allow_html=True)

fc1, fc2 = st.columns(2)
today = datetime.datetime.now()
default_sim_date = datetime.date(2012, 7, 1) if dataset_end < today.date() else today.date()
target_d = fc1.date_input("📅 Target Date", value=default_sim_date, key="date_tab1")
m = (today.minute // 5) * 5
target_t = fc2.time_input("🕐 Target Time", value=datetime.time(today.hour, m), key="time_tab1")

if st.button("⚡ Generate Future Forecast", type="primary", use_container_width=True):
    target_datetime = datetime.datetime.combine(target_d, target_t)
    target_ts = pd.Timestamp(target_datetime)

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
    
    # We will show weather/time in a separate row, then the 4 horizons
    h_str = "🎉 Holiday" if is_holiday else "📋 Regular Day"
    target_display = target_datetime.strftime("%A, %b %d %Y at %I:%M %p")
    temp_f = temp * 9 / 5 + 32


    # 4 Prediction Horizons
    st.markdown('<div class="section-header">📊 All Prediction Horizons</div>', unsafe_allow_html=True)
    h_cols = st.columns(4)
    horizons_to_show = [(0, "5 min"), (2, "15 min"), (5, "30 min"), (11, "60 min")]
    
    for col, (h_idx, h_label) in zip(h_cols, horizons_to_show):
        h_pred = float(sensor_future[h_idx])
        h_cong_lbl, h_cong_bdg, h_cong_icn = get_congestion_info(h_pred)
        with col:
            st.markdown(f"""
            <div class="metric-card" style="padding:16px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="metric-label" style="font-size:0.9rem; margin:0;">{h_label} Forecast</span>
                    <span class="badge {h_cong_bdg}" style="font-size:0.65rem; padding:2px 8px;">{h_cong_lbl}</span>
                </div>
                <div class="metric-value" style="font-size:2rem; margin-top:12px;">{h_pred:.1f} <span style="font-size:1rem;color:#64748b;">mph</span></div>
            </div>
            """, unsafe_allow_html=True)

    # AI Insights
    st.markdown('<div class="section-header">🤖 AI Traffic Insights</div>', unsafe_allow_html=True)
    # Generate overall insights based on the 60-min horizon
    future_pred_overall = float(sensor_future[11])
    cong_lbl_overall, _, _ = get_congestion_info(future_pred_overall)
    future_insights = generate_ai_insights(future_pred_overall, cong_lbl_overall, "60 minutes", trend_speeds=sensor_future)
    for insight in future_insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── Footer ──
st.markdown("""
<div style="text-align:center; padding:30px 0 10px 0; color:#475569; font-size:0.72rem; letter-spacing:0.04em;">
    Traffic Flow Forecasting &amp; Congestion Behaviour Analysis &nbsp;•&nbsp; CADGT &nbsp;•&nbsp; METR-LA Dataset<br>
    Context-Aware Dynamic Graph Transformer &nbsp;|&nbsp; Intelligent Transportation System
</div>
""", unsafe_allow_html=True)
