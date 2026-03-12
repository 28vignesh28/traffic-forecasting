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
    page_title="CADGT Live Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- APIs and Helpers for Future Forecasting ---
@st.cache_data(ttl=3600)
def fetch_future_weather(lat, lon, target_date):
    """Fetches hourly weather forecast for a future date using Open-Meteo"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
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
        # Set missing visibility to a reasonable default (10000m = clear) if API drops it
        if "visibility" not in df.columns or df["visibility"].isnull().all():
            df["visibility"] = 10000.0
        return df
    except Exception as e:
        st.error(f"Failed to fetch live weather forecast: {e}")
        return None

@st.cache_data(ttl=86400)
def check_holiday(target_date):
    """Checks if the given date is a US holiday"""
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
    """Loads config, test datasets, scaler, and the trained CADGT model. Cached so it only runs once."""
    config_path = "src/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Needs to grab internal dataloaders to simulate a live stream
    from src.data_loader import get_dataloaders
    _, _, test_loader, pip_scaler, dataset_info = get_dataloaders(config)
    
    adj_static = load_static_adj(config['data']['adj_path'])
    
    hidden_dim = config.get('model_defaults', {}).get('hidden_dim', 64)
    cadgt_overrides = config.get('model_overrides', {}).get('CADGT', {})
    
    # Initialize Model
    model = CADGT(
        num_nodes=dataset_info['num_nodes'],
        seq_len=config['training']['window'],
        future_len=config['training']['horizon'],
        ctx_dim=dataset_info['num_features'] - 1,
        d_model=cadgt_overrides.get('hidden_dim', hidden_dim),
        static_adj=adj_static
    ).to(device)

    # Load Weights
    ckpt_path = os.path.join("saved_models", "cadgt_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Extract all test batches to act as a "stream" buffer
    stream_x = []
    stream_y = []
    for x, y in test_loader:
        stream_x.append(x.numpy())
        stream_y.append(y.numpy())
        
    stream_x = np.concatenate(stream_x, axis=0)
    stream_y = np.concatenate(stream_y, axis=0)

    # Load historical baseline profile for future forecasts
    base_profile_path = "data/computed/average_traffic_profile.npy"
    if os.path.exists(base_profile_path):
        base_profile = np.load(base_profile_path)
    else:
        base_profile = np.zeros((288, 7, 207))

    # Calculate global weather standardization parameters based on test stream for future scaling
    # We grab weather features [1, 2, 3, 4] from the stream_x tensor [Samples, T, N, D]
    weather_test = stream_x[:, :, :, 1:5]
    w_mean = np.mean(weather_test, axis=(0, 1, 2))
    w_std = np.std(weather_test, axis=(0, 1, 2)) + 1e-5

    return model, stream_x, stream_y, pip_scaler, dataset_info, device, config, base_profile, (w_mean, w_std)

# --- Load resources ---
with st.spinner("Loading Model and Data..."):
    model, stream_x, stream_y, pip_scaler, dataset_info, device, config, base_profile, weather_stats = load_forecast_environment()

total_steps = len(stream_x)

# --- Sidebar Controls ---
st.sidebar.header("Simulation Controls")

# Node selector
selected_node = st.sidebar.selectbox(
    "Select Sensor Node", 
    options=range(dataset_info['num_nodes']), 
    index=50,
    help="Choose a specific traffic sensor (out of 207) to visualize."
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset Info**\n- Nodes: {dataset_info['num_nodes']}")

# --- Main App Execution ---
st.title("🚦 CADGT: Traffic Flow Forecasting Dashboard")

# Tab interface
tab1, tab2 = st.tabs(["📊 Replay Historical Tests", "🔮 Future Forecast Simulator (Interactive)"])

with tab1:
    st.markdown("""
    Welcome to the interactive interface for the **Context-Aware Dynamic Graph Transformer (CADGT)** model. 
    This tab replays the model's performance on real-world historical test data. Adjust the slider to step forward in time.
    """)
    
    # Use session state to remember if we are "playing" automatically
    if 'autostep' not in st.session_state:
        st.session_state.autostep = False

    def toggle_autostep():
        st.session_state.autostep = not st.session_state.autostep

    # Target Time Step 
    step_index = st.slider(
        "Current Time Step (T=0) for Test Replay", 
        min_value=0, 
        max_value=total_steps - 1, 
        value=0, 
        step=1
    )

    if st.button("Toggle Auto-Play (Simulates Real-time)" , on_click=toggle_autostep):
        pass

    if st.session_state.autostep:
        st.warning("Auto-Play is ON. Moving forward 5 minutes every second...")
        time.sleep(1)
        st.rerun()

    # --- Perform Inference (Replay Mode) ---
    live_x = stream_x[step_index:step_index+1] 
    true_y = stream_y[step_index:step_index+1]

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

    # Metrics Display
    c1, c2 = st.columns(2)
    c1.metric("Inference Time", f"{inference_time_ms:.1f} ms", delta="Fast", delta_color="normal")
    c2.metric("Features Ingested", "10", delta="Weather + Time included", delta_color="normal")

    st.subheader(f"Test Set Forecast for Sensor {selected_node}")
    past_times = np.arange(-55, 5, 5) 
    future_times = np.arange(5, 65, 5) 
    sensor_past = past_speeds[:, selected_node]
    sensor_pred = pred_speeds[:, selected_node]
    sensor_true = true_speeds[:, selected_node]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.append(past_times, 5), y=np.append(sensor_past, sensor_pred[0]), mode='lines', name='1-Hour Context', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=future_times, y=sensor_pred, mode='lines+markers', name='CADGT Forecast', line=dict(color='green', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=future_times, y=sensor_true, mode='lines', name='True Future', line=dict(color='gray', width=2), opacity=0.8))
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="T=0")
    fig.update_layout(xaxis_title="Minutes relative to selected step", yaxis_title="Speed (mph)", yaxis_range=[0, 75], hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.autostep and step_index < total_steps - 1:
        st.session_state.step_index = step_index + 1


with tab2:
    st.markdown("""
    **Simulate the future!** Pick a future date and time. This tool automatically fetches the live weather prediction from `Open-Meteo` for Los Angeles and computes whether it's a holiday to deliver a realistic prediction.
    """)

    # Interactive Inputs
    col1, col2 = st.columns(2)
    today = datetime.datetime.now()
    target_d = col1.date_input("Target Date", value=today.date())
    
    # Needs to align to nearest 5 minutes
    m = (today.minute // 5) * 5
    target_t = col2.time_input("Target Time", value=datetime.time(today.hour, m))

    if st.button("Generate Future Forecast", type="primary"):
        target_datetime = datetime.datetime.combine(target_d, target_t)
        
        with st.spinner("Fetching Live Weather & Preparing Context..."):
            # 1. Fetch Weather
            weather_df = fetch_future_weather(config['data']['lat'], config['data']['lon'], target_datetime)
            
            # 2. Check Holiday
            is_holiday = float(check_holiday(target_datetime))
            
            # 3. Build 12-step historical context (Past 60 minutes ending at target_time)
            # The model needs past data to predict the future. We use the historical average for this specific ToD/DoW.
            history_times = [target_datetime - datetime.timedelta(minutes=5 * i) for i in range(11, -1, -1)]
            
            # [W=12, N=207, F=10]
            custom_x = np.zeros((1, 12, dataset_info['num_nodes'], 10), dtype=np.float32)

            for i, t_step in enumerate(history_times):
                tod_idx = (t_step.hour * 60 + t_step.minute) // 5
                dow_idx = t_step.weekday()
                
                # Baseline traffic (Feature 0)
                profiled_traffic_mph = base_profile[tod_idx, dow_idx, :]
                # Scale traffic using the pip_scaler
                custom_x[0, i, :, 0] = pip_scaler.transform(profiled_traffic_mph)
                
                # Weather Features (Features 1-4)
                if weather_df is not None:
                    # Find nearest hourly weather reading
                    nearest_weather = weather_df.iloc[(weather_df['time'] - t_step).abs().argsort()[:1]]
                    temp = nearest_weather["temperature_2m"].values[0]
                    precip = nearest_weather["precipitation"].values[0]
                    vis = nearest_weather["visibility"].values[0]
                    wind = nearest_weather["windspeed_10m"].values[0]
                else:
                    # Fallback to defaults if API fails
                    temp, precip, vis, wind = 20.0, 0.0, 10000.0, 5.0

                w_features = np.array([temp, precip, vis, wind])
                w_scaled = (w_features - weather_stats[0]) / weather_stats[1]
                
                custom_x[0, i, :, 1] = w_scaled[0]
                custom_x[0, i, :, 2] = w_scaled[1]
                custom_x[0, i, :, 3] = w_scaled[2]
                custom_x[0, i, :, 4] = w_scaled[3]
                
                # Holiday (Feature 5)
                custom_x[0, i, :, 5] = is_holiday
                
                # Time Features (Features 6-9)
                tod_norm = tod_idx / 288.0
                dow_norm = dow_idx / 7.0
                custom_x[0, i, :, 6] = np.sin(2 * np.pi * tod_norm)
                custom_x[0, i, :, 7] = np.cos(2 * np.pi * tod_norm)
                custom_x[0, i, :, 8] = np.sin(2 * np.pi * dow_norm)
                custom_x[0, i, :, 9] = np.cos(2 * np.pi * dow_norm)


            # --- Run Model ---
            t_input = torch.FloatTensor(custom_x).to(device)
            start_infer = time.time()
            with torch.no_grad():
                preds_future = model(t_input)
            latency = (time.time() - start_infer) * 1000

            # Decode
            p_future_np = preds_future.cpu().numpy()[0]
            future_speeds = pip_scaler.inverse_transform(p_future_np)
            past_ctx_mph = base_profile[tod_idx, dow_idx, :] # Used the final recorded step for context

        st.success(f"Forecast Generated! Inference Time: {latency:.1f}ms")

        # Visuals
        target_display = target_datetime.strftime("%A, %b %d %Y at %I:%M %p")
        
        w_col1, w_col2, w_col3 = st.columns(3)
        w_col1.info(f"🌡️ Forecast Temp: {temp:.1f} °C")
        h_str = "Yes" if is_holiday else "No"
        w_col2.info(f"🎉 Public Holiday: {h_str}")
        w_col3.info(f"🌧️ Precipitation: {precip:.1f} mm")

        st.subheader(f"Future CADGT Prediction for Sensor {selected_node}")
        st.write(f"Starting precisely at **{target_display}**")

        f_times = [target_datetime + datetime.timedelta(minutes=5*(i+1)) for i in range(12)]
        
        sensor_f_pred = future_speeds[:, selected_node]

        fig2 = go.Figure()

        # Just plot the forecasted line
        fig2.add_trace(go.Scatter(
            x=f_times, 
            y=sensor_f_pred, 
            mode='lines+markers',
            name='CADGT Future Forecast',
            line=dict(color='purple', width=3),
            marker=dict(size=8)
        ))

        fig2.update_layout(
            xaxis_title="Time", 
            yaxis_title="Predicted Speed (mph)", 
            yaxis_range=[0, 75], 
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)
