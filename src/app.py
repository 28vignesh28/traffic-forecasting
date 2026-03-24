"""
Traffic Flow Forecasting & Congestion Behaviour Analysis Dashboard
AI-powered traffic prediction using CADGT (Context-Aware Dynamic Graph Transformer)
"""

import os
import sys
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import torch
import streamlit as st

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

/* ── Root variables — Light / white theme ── */
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-card: #ffffff;
    --border-card: #e2e8f0;
    --accent-indigo: #4f46e5;
    --accent-cyan: #0891b2;
    --accent-emerald: #059669;
    --accent-amber: #d97706;
    --accent-rose: #e11d48;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
}

/* ── Global — larger base font ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 18px !important;
    color: #1e293b !important;
}
.stApp {
    background: #ffffff !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #f8fafc !important;
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stTimeInput label,
section[data-testid="stSidebar"] .stNumberInput label {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px 32px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 14px;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
}
.metric-label {
    color: #64748b;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    font-weight: 700;
}
.metric-value {
    color: #1e293b;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.2;
}
.metric-sub {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 6px;
}

/* ── Congestion badges ── */
.badge {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.03em;
}
.badge-green {
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}
.badge-yellow {
    background: #fffbeb;
    color: #d97706;
    border: 1px solid #fde68a;
}
.badge-red {
    background: #fff1f2;
    color: #e11d48;
    border: 1px solid #fecdd3;
}

/* ── Header ── */
.dashboard-header {
    background: linear-gradient(135deg, #eef2ff 0%, #ecfeff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 36px 44px;
    margin-bottom: 28px;
    text-align: center;
}
.dashboard-header h1 {
    background: linear-gradient(135deg, #4f46e5, #0891b2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
}
.dashboard-header p {
    color: #64748b;
    font-size: 1.15rem;
    margin-top: 8px;
    font-weight: 400;
}

/* ── Section header ── */
.section-header {
    color: #334155;
    font-size: 1.05rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}

/* ── AI insight card ── */
.insight-card {
    background: #f8fafc;
    border-left: 4px solid #4f46e5;
    border-radius: 0 12px 12px 0;
    padding: 16px 22px;
    margin: 10px 0;
    color: #334155;
    font-size: 1.05rem;
    line-height: 1.6;
}

/* ── Weather card ── */
.weather-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 28px;
    text-align: center;
}
.weather-icon { font-size: 2.6rem; margin-bottom: 6px; }
.weather-temp { font-size: 2rem; font-weight: 700; color: #1e293b; }
.weather-label { font-size: 0.9rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; font-weight: 600; }

/* ── Holiday pill ── */
.holiday-pill {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 24px;
    font-weight: 700;
    font-size: 1rem;
}
.holiday-yes {
    background: #fffbeb;
    color: #d97706;
    border: 1px solid #fde68a;
}
.holiday-no {
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}

/* ── Animated pulse for live indicator ── */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.live-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #059669;
    animation: pulse 2s infinite;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Change indicator arrows ── */
.change-up { color: #e11d48; font-weight: 700; font-size: 1.1rem; }
.change-down { color: #059669; font-weight: 700; font-size: 1.1rem; }
.change-neutral { color: #64748b; font-weight: 600; font-size: 1.1rem; }

/* ── Hide default Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Plotly container rounding ── */
.stPlotlyChart {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
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
    # Replace sensor-malfunction zeros with forward fill
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
        num_nodes=207,
        seq_len=12,
        future_len=12,
        ctx_dim=9,   # 10 features - 1 (traffic)
        d_model=64,
        static_adj=adj_mx,
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
    # Pick the larger CSV (full dataset range)
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


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers (mirrors data_loader.py logic)
# ──────────────────────────────────────────────────────────────────────────────

def build_input_tensor(traffic_df, weather_df, holidays, target_ts, scaler_mean, scaler_std, num_nodes=207, window=12):
    """
    Build the [1, 12, 207, 10] input tensor for CADGT from raw data.
    Returns (tensor, actual_timestamps, found_in_dataset).
    """
    timestamps = traffic_df.index

    # Find closest timestamp in dataset
    idx = timestamps.searchsorted(target_ts)
    idx = min(idx, len(timestamps) - 1)

    # We need `window` steps ending at idx
    start = max(0, idx - window + 1)
    end = idx + 1
    if end - start < window:
        start = 0
        end = window

    actual_ts = timestamps[start:end]
    traffic_slice = traffic_df.iloc[start:end].values.astype(np.float32)  # [W, N]

    found_in_dataset = abs((timestamps[idx] - target_ts).total_seconds()) < 600  # within 10 min

    # Normalize traffic
    traffic_norm = (traffic_slice - scaler_mean) / (scaler_std + 1e-5)

    # --- Weather features [W, 4] ---
    if weather_df is not None and len(weather_df) > 0:
        weather_indexed = weather_df.set_index("time")
        weather_aligned = weather_indexed.reindex(actual_ts, method="nearest")
        weather_feat = weather_aligned[["temperature_2m", "precipitation", "visibility", "windspeed_10m"]].values.astype(np.float32)
        weather_feat = np.nan_to_num(weather_feat, nan=0.0)
    else:
        weather_feat = np.zeros((window, 4), dtype=np.float32)

    # Z-score normalize weather (use dataset-level rough stats)
    w_mean = weather_feat.mean(axis=0, keepdims=True)
    w_std = weather_feat.std(axis=0, keepdims=True) + 1e-5
    weather_feat = (weather_feat - w_mean) / w_std

    # --- Holiday feature [W, 1] ---
    holiday_feat = np.zeros((window, 1), dtype=np.float32)
    for i, ts in enumerate(actual_ts):
        if ts.date() in holidays:
            holiday_feat[i] = 1.0

    # --- Time features [W, 4] ---
    tod = ((actual_ts.hour * 60 + actual_ts.minute) // 5).values
    dow = actual_ts.dayofweek.values
    tod_sin = np.sin(2 * np.pi * tod / 288.0).reshape(-1, 1)
    tod_cos = np.cos(2 * np.pi * tod / 288.0).reshape(-1, 1)
    dow_sin = np.sin(2 * np.pi * dow / 7.0).reshape(-1, 1)
    dow_cos = np.cos(2 * np.pi * dow / 7.0).reshape(-1, 1)
    time_feat = np.concatenate([tod_sin, tod_cos, dow_sin, dow_cos], axis=1).astype(np.float32)

    # Broadcast to [W, N, feat]
    traffic_3d = traffic_norm[:, :, None]  # [W, N, 1]
    weather_3d = np.repeat(weather_feat[:, None, :], num_nodes, axis=1)  # [W, N, 4]
    holiday_3d = np.repeat(holiday_feat[:, None, :], num_nodes, axis=1)  # [W, N, 1]
    time_3d = np.repeat(time_feat[:, None, :], num_nodes, axis=1)  # [W, N, 4]

    # Concatenate: [W, N, 10] = 1 traffic + 4 weather + 1 holiday + 4 time
    X = np.concatenate([traffic_3d, weather_3d, holiday_3d, time_3d], axis=-1)
    tensor = torch.FloatTensor(X).unsqueeze(0)  # [1, W, N, 10]

    return tensor, actual_ts, found_in_dataset


def get_congestion_info(speed_mph):
    """Classify congestion level from speed."""
    if speed_mph > 37:
        return "Free Flow", "badge-green", "🟢"
    elif speed_mph >= 25:
        return "Moderate Traffic", "badge-yellow", "🟡"
    else:
        return "Heavy Congestion", "badge-red", "🔴"


def get_weather_display(weather_df, target_ts):
    """Get weather info nearest to the target timestamp."""
    if weather_df is None or len(weather_df) == 0:
        return {"temp": "N/A", "type": "Unknown", "icon": "❓", "humidity": "N/A", "wind": "N/A", "precip": 0}

    weather_indexed = weather_df.set_index("time")
    idx = weather_indexed.index.searchsorted(target_ts)
    idx = min(idx, len(weather_indexed) - 1)
    row = weather_indexed.iloc[idx]

    temp_c = row.get("temperature_2m", 0)
    precip = row.get("precipitation", 0)
    visibility = row.get("visibility", 10000)
    wind = row.get("windspeed_10m", 0)

    # Determine weather type
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

    return {
        "temp": f"{temp_c:.1f}°C",
        "type": wtype,
        "icon": icon,
        "wind": f"{wind:.1f} mph",
        "precip": f"{precip:.1f} mm",
    }


def generate_ai_insights(pred_speed, congestion_label, horizon_label, actual_speed=None, trend_speeds=None):
    """Generate intelligent AI insights based on prediction results."""
    insights = []

    # Trend analysis
    if trend_speeds is not None and len(trend_speeds) > 2:
        first_half = np.mean(trend_speeds[: len(trend_speeds) // 2])
        second_half = np.mean(trend_speeds[len(trend_speeds) // 2 :])
        diff = second_half - first_half
        if diff > 2:
            insights.append("📈 Traffic speed is expected to **increase** over the prediction window, suggesting improving conditions.")
        elif diff < -2:
            insights.append("📉 Traffic speed is projected to **decrease**, indicating potential congestion build-up ahead.")
        else:
            insights.append("📊 Traffic conditions remain **stable** throughout the forecast horizon.")

    # Congestion-based
    if congestion_label == "Heavy Congestion":
        insights.append("🚨 **Heavy congestion** detected near this sensor. Consider alternative routes or delayed departure.")
    elif congestion_label == "Moderate Traffic":
        insights.append("⚠️ **Moderate congestion** likely near this sensor. Allow extra travel time for safety.")
    else:
        insights.append("✅ **Free flow** conditions expected. Optimal time for travel through this corridor.")

    # Comparison insight
    if actual_speed is not None:
        change = pred_speed - actual_speed
        if abs(change) > 3:
            direction = "increase" if change > 0 else "decrease"
            insights.append(f"🔮 Model predicts a **{abs(change):.1f} mph {direction}** compared to the actual recorded speed at this timestamp.")

    # Horizon-based
    if "60" in horizon_label:
        insights.append("🕐 Long-range 60-minute forecast — predictions at this range have somewhat higher uncertainty.")
    elif "5" in horizon_label:
        insights.append("⚡ Short-range 5-minute forecast — highest confidence prediction window.")

    return insights


# ──────────────────────────────────────────────────────────────────────────────
# Load all data
# ──────────────────────────────────────────────────────────────────────────────
traffic_df = load_traffic_data()
model, device, scaler_mean, scaler_std = load_model()
weather_df = load_weather_cache()
holidays = load_holidays()

# Dataset date range
dataset_start = traffic_df.index[0].date()
dataset_end = traffic_df.index[-1].date()
sensor_ids = list(range(traffic_df.shape[1]))  # 0 to 206

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
# Sidebar — Input Controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 14px 0 20px 0;">
        <span style="font-size:1.8rem;">🎛️</span>
        <div style="color:#1e293b; font-size:1.2rem; font-weight:700; letter-spacing:0.04em; margin-top:4px;">Control Panel</div>
        <div style="color:#64748b; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.1em;">Smart Traffic Monitoring</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    sel_date = st.date_input(
        "📅 Select Date",
        value=datetime.date(2012, 4, 15),
        min_value=dataset_start,
        max_value=dataset_end,
        help=f"METR-LA range: {dataset_start} to {dataset_end}",
    )

    sel_time = st.time_input(
        "🕐 Select Time",
        value=datetime.time(8, 00),
        step=datetime.timedelta(minutes=5),
        help="24-hour format, 5-minute intervals",
    )

    sel_sensor = st.selectbox(
        "📡 Sensor ID",
        options=sensor_ids,
        index=91,
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
    <div style="color:#475569; font-size:0.9rem; text-align:center; line-height:1.8;">
        <strong style="color:#334155;">Model:</strong> CADGT v1.0<br>
        <strong style="color:#334155;">Dataset:</strong> METR-LA<br>
        <strong style="color:#334155;">Sensors:</strong> 207 nodes<br>
        <strong style="color:#334155;">Device:</strong> {'CUDA' if device.type == 'cuda' else 'CPU'}
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Run prediction (auto-triggers on any input change)
# ──────────────────────────────────────────────────────────────────────────────
target_datetime = datetime.datetime.combine(sel_date, sel_time)
target_ts = pd.Timestamp(target_datetime)

# Build input and run model
input_tensor, actual_timestamps, found_in_dataset = build_input_tensor(
    traffic_df, weather_df, holidays, target_ts, scaler_mean, scaler_std
)

with torch.no_grad():
    input_tensor_dev = input_tensor.to(device)
    preds_raw = model(input_tensor_dev)  # [1, 12, 207]
    preds_np = preds_raw.cpu().numpy()[0]  # [12, 207]

# Inverse transform to mph then convert to mph
preds_mph = preds_np * scaler_std + scaler_mean
preds_mph_array = preds_mph  # kept as mph

# Target sensor predicted speeds (all horizons)
sensor_preds_mph_array = preds_mph_array[:, sel_sensor]  # [12]
pred_speed = float(sensor_preds_mph_array[sel_horizon_idx])

# Actual speed (if available)
actual_speed = None
if found_in_dataset:
    # Get the actual speed at the target timestamp + horizon offset
    horizon_offset = (sel_horizon_idx + 1) * 5  # minutes
    future_ts = target_ts + pd.Timedelta(minutes=horizon_offset)
    if future_ts in traffic_df.index:
        actual_mph = float(traffic_df.loc[future_ts].iloc[sel_sensor])
        actual_speed = actual_mph  # kept as mph

# Also get actual historical speeds for the past window
actual_history_mph = None
if found_in_dataset:
    history = traffic_df.iloc[
        traffic_df.index.searchsorted(actual_timestamps[0]): traffic_df.index.searchsorted(actual_timestamps[-1]) + 1
    ].iloc[:, sel_sensor].values # kept as mph
    actual_history_mph = history

# Congestion info
congestion_label, congestion_badge, congestion_icon = get_congestion_info(pred_speed)

# Weather info
weather_info = get_weather_display(weather_df, target_ts)

# Holiday check
is_holiday = sel_date in holidays

# AI insights
insights = generate_ai_insights(
    pred_speed, congestion_label, sel_horizon_label,
    actual_speed=actual_speed, trend_speeds=sensor_preds_mph_array
)

# ──────────────────────────────────────────────────────────────────────────────
# LAYOUT — Main dashboard panels
# ──────────────────────────────────────────────────────────────────────────────

# ── Row 1: Key Metrics ──
st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Speed</div>
        <div class="metric-value">{pred_speed:.1f} <span style="font-size:1.1rem;color:#64748b;">mph</span></div>
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
            <span style="font-size:1.8rem;">{congestion_icon}</span>
            <span class="badge {congestion_badge}" style="margin-left:6px;">{congestion_label}</span>
        </div>
        <div class="metric-sub">Based on predicted speed</div>
    </div>
    """, unsafe_allow_html=True)


# ── Row 2: Chart + Traffic Change / AI Insights ──
info_col = st.container()


with info_col:
    # Traffic Change Analysis
    st.markdown('<div class="section-header">🔄 Traffic Change Analysis</div>', unsafe_allow_html=True)

    if actual_speed is not None:
        change = pred_speed - actual_speed
        if change > 0:
            change_html = f'<span class="change-up">Increase (+{change:.1f} mph) ↑</span>'
        elif change < 0:
            change_html = f'<span class="change-down">Decrease ({change:.1f} mph) ↓</span>'
        else:
            change_html = '<span class="change-neutral">No Change (0.0 mph) →</span>'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Actual Speed</div>
            <div class="metric-value" style="font-size:1.7rem;">{actual_speed:.1f} <span style="font-size:1rem;color:#64748b;">mph</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Predicted Speed</div>
            <div class="metric-value" style="font-size:1.7rem;">{pred_speed:.1f} <span style="font-size:1rem;color:#64748b;">mph</span></div>
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
            <div class="metric-value" style="font-size:1.7rem;">{pred_speed:.1f} <span style="font-size:1rem;color:#64748b;">mph</span></div>
            <div class="metric-sub" style="margin-top:8px;color:#d97706;">
                ⚠️ Actual data not available for this timestamp.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # AI Insights
    st.markdown('<div class="section-header">🤖 AI Traffic Insights</div>', unsafe_allow_html=True)
    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)


# ── Row 3: Weather & Holiday ──
st.markdown('<div class="section-header">🌍 Context Information</div>', unsafe_allow_html=True)
w1, w2, w3, w4, w5 = st.columns(5)

with w1:
    st.markdown(f"""
    <div class="weather-card">
        <div class="weather-icon">{weather_info['icon']}</div>
        <div class="weather-temp">{weather_info['temp']}</div>
        <div class="weather-label">Temperature</div>
    </div>
    """, unsafe_allow_html=True)

with w2:
    st.markdown(f"""
    <div class="weather-card">
        <div class="weather-icon">🌤️</div>
        <div style="font-size:1.2rem;font-weight:600;color:#1e293b;margin-top:4px;">{weather_info['type']}</div>
        <div class="weather-label">Weather Type</div>
    </div>
    """, unsafe_allow_html=True)

with w3:
    st.markdown(f"""
    <div class="weather-card">
        <div class="weather-icon">💧</div>
        <div style="font-size:1.2rem;font-weight:600;color:#1e293b;margin-top:4px;">{weather_info['precip']}</div>
        <div class="weather-label">Precipitation</div>
    </div>
    """, unsafe_allow_html=True)

with w4:
    st.markdown(f"""
    <div class="weather-card">
        <div class="weather-icon">💨</div>
        <div style="font-size:1.2rem;font-weight:600;color:#1e293b;margin-top:4px;">{weather_info['wind']}</div>
        <div class="weather-label">Wind Speed</div>
    </div>
    """, unsafe_allow_html=True)

with w5:
    if is_holiday:
        pill = '<span class="holiday-pill holiday-yes">🎉 Holiday</span>'
    else:
        pill = '<span class="holiday-pill holiday-no">📋 Regular Day</span>'
    st.markdown(f"""
    <div class="weather-card">
        <div class="weather-icon">📅</div>
        <div style="margin-top:8px;">{pill}</div>
        <div class="weather-label" style="margin-top:6px;">Holiday Status</div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──
st.markdown("""
<div style="text-align:center; padding:30px 0 10px 0; color:#64748b; font-size:0.9rem; letter-spacing:0.04em;">
    Traffic Flow Forecasting &amp; Congestion Behaviour Analysis &nbsp;•&nbsp; CADGT &nbsp;•&nbsp; METR-LA Dataset<br>
    Context-Aware Dynamic Graph Transformer &nbsp;|&nbsp; Intelligent Transportation System
</div>
""", unsafe_allow_html=True)
