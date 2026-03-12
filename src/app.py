import os
import sys
import time
import datetime
import torch
import numpy as np
import yaml
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_traffic, StandardScaler, load_static_adj
from src.utils import set_seed
from models.cadgt import CADGT

# --- Configure Streamlit Page ---
st.set_page_config(
    page_title="METR-LA Traffic Speed Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Clean Card-Based Design ---
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    /* Prediction Card */
    .pred-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8edf5 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #dde3ed;
    }
    .pred-card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 1.2rem;
    }

    /* Speed Display */
    .speed-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 1.5rem;
    }
    .speed-block {
        flex: 1;
        min-width: 150px;
    }
    .speed-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    .speed-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.1;
    }
    .speed-unit {
        font-size: 1rem;
        color: #888;
        font-weight: 400;
    }
    .speed-delta {
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .delta-positive { color: #e74c3c; }
    .delta-negative { color: #27ae60; }
    .delta-neutral  { color: #888; }

    /* Status Badge */
    .status-block {
        text-align: center;
        min-width: 120px;
    }
    .status-dot {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        display: inline-block;
        margin-bottom: 0.4rem;
    }
    .status-dot-green  { background-color: #2ecc71; box-shadow: 0 0 8px rgba(46,204,113,0.5); }
    .status-dot-yellow { background-color: #f1c40f; box-shadow: 0 0 8px rgba(241,196,15,0.5); }
    .status-dot-orange { background-color: #e67e22; box-shadow: 0 0 8px rgba(230,126,34,0.5); }
    .status-dot-red    { background-color: #e74c3c; box-shadow: 0 0 8px rgba(231,76,60,0.5); }
    .status-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-text {
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .status-free    { color: #2ecc71; }
    .status-slow    { color: #f1c40f; }
    .status-heavy   { color: #e67e22; }
    .status-stopped { color: #e74c3c; }

    /* Chart card */
    .chart-card {
        background: #fff;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #eee;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* Force light background */
    .stApp {
        background-color: #f7f9fc;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper: Traffic Status ---
def get_traffic_status(speed_mph):
    """Returns (dot_class, text_class, label) based on speed thresholds for LA freeways."""
    if speed_mph >= 50:
        return "status-dot-green", "status-free", "FREE FLOW"
    elif speed_mph >= 30:
        return "status-dot-yellow", "status-slow", "SLOW"
    elif speed_mph >= 15:
        return "status-dot-orange", "status-heavy", "HEAVY"
    else:
        return "status-dot-red", "status-stopped", "CONGESTED"


# --- APIs for Future Forecasting ---
@st.cache_data(ttl=3600)
def fetch_future_weather(lat, lon, target_date):
    """Fetches hourly weather forecast for a future date using Open-Meteo."""
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
def check_holiday(target_date):
    """Checks if the given date is a US holiday."""
    year = target_date.year
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/US"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            holidays = [datetime.datetime.strptime(h['date'], "%Y-%m-%d").date() for h in response.json()]
            return target_date.date() in holidays
    except:
        pass
    return False


# --- Caching Data & Model Loading ---
@st.cache_resource
def load_forecast_environment():
    """Loads config, test datasets, scaler, and the trained CADGT model."""
    config_path = "src/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.data_loader import get_dataloaders
    _, _, test_loader, pip_scaler, dataset_info = get_dataloaders(config)

    adj_static = load_static_adj(config['data']['adj_path'])

    hidden_dim = config.get('model_defaults', {}).get('hidden_dim', 64)
    cadgt_overrides = config.get('model_overrides', {}).get('CADGT', {})

    model = CADGT(
        num_nodes=dataset_info['num_nodes'],
        seq_len=config['training']['window'],
        future_len=config['training']['horizon'],
        ctx_dim=dataset_info['num_features'] - 1,
        d_model=cadgt_overrides.get('hidden_dim', hidden_dim),
        static_adj=adj_static
    ).to(device)

    ckpt_path = os.path.join("saved_models", "cadgt_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    stream_x, stream_y = [], []
    for x, y in test_loader:
        stream_x.append(x.numpy())
        stream_y.append(y.numpy())

    stream_x = np.concatenate(stream_x, axis=0)
    stream_y = np.concatenate(stream_y, axis=0)

    base_profile_path = "data/computed/average_traffic_profile.npy"
    base_profile = np.load(base_profile_path) if os.path.exists(base_profile_path) else np.zeros((288, 7, 207))

    weather_test = stream_x[:, :, :, 1:5]
    w_mean = np.mean(weather_test, axis=(0, 1, 2))
    w_std = np.std(weather_test, axis=(0, 1, 2)) + 1e-5

    return model, stream_x, stream_y, pip_scaler, dataset_info, device, config, base_profile, (w_mean, w_std)


# --- Load resources ---
with st.spinner("Loading Model and Data..."):
    model, stream_x, stream_y, pip_scaler, dataset_info, device, config, base_profile, weather_stats = load_forecast_environment()

total_steps = len(stream_x)

# ======================= HEADER =======================
st.markdown('<div class="main-title">🚦 METR-LA Traffic Speed Forecasting</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Powered by the Context-Aware Dynamic Graph Transformer (CADGT)</div>', unsafe_allow_html=True)

# ======================= SIDEBAR =======================
st.sidebar.header("Controls")

sensor_id = st.sidebar.selectbox(
    f"Select Sensor (all {dataset_info['num_nodes']})",
    options=range(dataset_info['num_nodes']),
    index=50
)

st.sidebar.markdown("---")

# ======================= TABS =======================
tab1, tab2 = st.tabs(["📊 Sensor Prediction View", "🔮 Future Forecast Simulator"])

# ==================== TAB 1: REPLAY ====================
with tab1:
    step_index = st.sidebar.slider(
        "Select Date & Time",
        min_value=0,
        max_value=total_steps - 1,
        value=0,
        step=1,
        help="Scroll through the METR-LA test set (each step = 5 minutes)."
    )

    # --- Inference ---
    live_x = stream_x[step_index:step_index + 1]
    true_y = stream_y[step_index:step_index + 1]

    x_tensor = torch.FloatTensor(live_x).to(device)
    start_time = time.time()
    with torch.no_grad():
        preds = model(x_tensor)
    inference_time_ms = (time.time() - start_time) * 1000

    p_np = preds.cpu().numpy()[0]
    pred_speeds = pip_scaler.inverse_transform(p_np)
    past_scaled = live_x[0, :, :, 0]
    past_speeds = pip_scaler.inverse_transform(past_scaled)
    true_scaled = true_y[0]
    true_speeds = pip_scaler.inverse_transform(true_scaled)

    # Pick the 60-minute (last horizon step) prediction
    predicted_speed = pred_speeds[-1, sensor_id]
    actual_speed = true_speeds[-1, sensor_id]
    delta = predicted_speed - actual_speed

    # Delta display
    if abs(delta) < 0.5:
        delta_class = "delta-neutral"
        delta_sign = ""
    elif delta > 0:
        delta_class = "delta-positive"
        delta_sign = "+"
    else:
        delta_class = "delta-negative"
        delta_sign = ""

    # Traffic status
    dot_class, text_class, status_label = get_traffic_status(predicted_speed)

    # --- PREDICTION CARD ---
    st.markdown(f"""
    <div class="pred-card">
        <div class="pred-card-title">Prediction — Sensor {sensor_id}</div>
        <div class="speed-container">
            <div class="speed-block">
                <div class="speed-label">Predicted Speed (60 min)</div>
                <div class="speed-value">{predicted_speed:.1f} <span class="speed-unit">mph</span></div>
                <div class="speed-delta {delta_class}">{delta_sign}{delta:.1f} vs actual</div>
            </div>
            <div class="speed-block">
                <div class="speed-label">Actual Speed (60 min)</div>
                <div class="speed-value">{actual_speed:.1f} <span class="speed-unit">mph</span></div>
            </div>
            <div class="status-block">
                <div class="speed-label">Status</div>
                <div class="status-dot {dot_class}"></div>
                <div class="status-text {text_class}">{status_label}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- CHART ---
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.subheader(f"Speed Forecast Timeline — Sensor {sensor_id}")

    past_times = np.arange(-55, 5, 5)
    future_times = np.arange(5, 65, 5)
    sensor_past = past_speeds[:, sensor_id]
    sensor_pred = pred_speeds[:, sensor_id]
    sensor_true = true_speeds[:, sensor_id]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.append(past_times, 5), y=np.append(sensor_past, sensor_pred[0]),
        mode='lines', name='Historical Context',
        line=dict(color='#3498db', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=future_times, y=sensor_pred,
        mode='lines+markers', name='CADGT Forecast',
        line=dict(color='#2ecc71', width=3, dash='dash'),
        marker=dict(size=7)
    ))
    fig.add_trace(go.Scatter(
        x=future_times, y=sensor_true,
        mode='lines', name='True Future',
        line=dict(color='#e74c3c', width=2),
        opacity=0.6
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#aaa", annotation_text="Now")
    fig.update_layout(
        xaxis_title="Minutes relative to current time",
        yaxis_title="Speed (mph)",
        yaxis_range=[0, 75],
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Info bar
    c1, c2, c3 = st.columns(3)
    c1.metric("Inference Time", f"{inference_time_ms:.1f} ms")
    c2.metric("Features Ingested", "10")
    c3.metric("Model", "CADGT")


# ==================== TAB 2: FUTURE FORECAST ====================
with tab2:
    st.markdown("""
    **Simulate traffic in the future.** Pick a date and time — the app will automatically 
    fetch the live weather forecast from Open-Meteo and check for US holidays.
    """)

    col1, col2 = st.columns(2)
    today = datetime.datetime.now()
    target_d = col1.date_input("Date", value=today.date())
    m = (today.minute // 5) * 5
    target_t = col2.time_input("Select Date & Time", value=datetime.time(today.hour, m))

    if st.button("Generate Future Forecast", type="primary"):
        target_datetime = datetime.datetime.combine(target_d, target_t)

        with st.spinner("Fetching Live Weather & Preparing Context..."):
            weather_df = fetch_future_weather(config['data']['lat'], config['data']['lon'], target_datetime)
            is_holiday = float(check_holiday(target_datetime))

            history_times = [target_datetime - datetime.timedelta(minutes=5 * i) for i in range(11, -1, -1)]

            custom_x = np.zeros((1, 12, dataset_info['num_nodes'], 10), dtype=np.float32)

            for i, t_step in enumerate(history_times):
                tod_idx = (t_step.hour * 60 + t_step.minute) // 5
                dow_idx = t_step.weekday()

                profiled_traffic_mph = base_profile[tod_idx, dow_idx, :]
                custom_x[0, i, :, 0] = pip_scaler.transform(profiled_traffic_mph)

                if weather_df is not None:
                    nearest_weather = weather_df.iloc[(weather_df['time'] - t_step).abs().argsort()[:1]]
                    temp = nearest_weather["temperature_2m"].values[0]
                    precip = nearest_weather["precipitation"].values[0]
                    vis = nearest_weather["visibility"].values[0]
                    wind = nearest_weather["windspeed_10m"].values[0]
                else:
                    temp, precip, vis, wind = 20.0, 0.0, 10000.0, 5.0

                w_features = np.array([temp, precip, vis, wind])
                w_scaled = (w_features - weather_stats[0]) / weather_stats[1]

                custom_x[0, i, :, 1] = w_scaled[0]
                custom_x[0, i, :, 2] = w_scaled[1]
                custom_x[0, i, :, 3] = w_scaled[2]
                custom_x[0, i, :, 4] = w_scaled[3]

                custom_x[0, i, :, 5] = is_holiday

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
            future_speeds = pip_scaler.inverse_transform(p_future_np)

        # Pick 60-min prediction for main card
        future_pred_60 = future_speeds[-1, sensor_id]
        dot_cls, txt_cls, stat_lbl = get_traffic_status(future_pred_60)
        h_str = "Yes" if is_holiday else "No"
        target_display = target_datetime.strftime("%A, %b %d %Y at %I:%M %p")

        st.success(f"Forecast Generated! Inference: {latency:.1f} ms")

        # Future Prediction Card
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-card-title">Future Prediction — Sensor {sensor_id}</div>
            <div style="font-size: 0.9rem; color: #666; margin-bottom: 1rem;">{target_display}</div>
            <div class="speed-container">
                <div class="speed-block">
                    <div class="speed-label">Predicted Speed (60 min)</div>
                    <div class="speed-value">{future_pred_60:.1f} <span class="speed-unit">mph</span></div>
                </div>
                <div class="speed-block">
                    <div class="speed-label">Weather</div>
                    <div style="font-size:1.1rem; color:#1a1a2e;">🌡️ {temp:.1f}°C &nbsp; 🌧️ {precip:.1f}mm</div>
                    <div style="font-size:0.9rem; color:#888; margin-top:0.3rem;">Holiday: {h_str}</div>
                </div>
                <div class="status-block">
                    <div class="speed-label">Status</div>
                    <div class="status-dot {dot_cls}"></div>
                    <div class="status-text {txt_cls}">{stat_lbl}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Future Chart
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.subheader(f"Future Speed Timeline — Sensor {sensor_id}")

        f_times = [target_datetime + datetime.timedelta(minutes=5 * (i + 1)) for i in range(12)]
        sensor_f_pred = future_speeds[:, sensor_id]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=f_times, y=sensor_f_pred,
            mode='lines+markers', name='CADGT Future Forecast',
            line=dict(color='#9b59b6', width=3),
            marker=dict(size=8)
        ))
        fig2.update_layout(
            xaxis_title="Time",
            yaxis_title="Predicted Speed (mph)",
            yaxis_range=[0, 75],
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(gridcolor="#f0f0f0"),
            yaxis=dict(gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
