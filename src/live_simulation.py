import os
import sys
import time
import datetime
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_traffic, StandardScaler, load_static_adj
from src.utils import set_seed, setup_logging
from models.cadgt import CADGT

class RealTimeTrafficSimulator:
    def __init__(self, config_path="src/config.yaml"):
        self.logger = setup_logging("Live_Simulation")
        set_seed(42)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initializing Live Simulator on {self.device}")

        # Load Network
        self.adj_static = load_static_adj(self.config['data']['adj_path'])
        
        # We need the full test dataset to simulate an incoming "live" stream
        from src.data_loader import get_dataloaders
        _, _, self.test_loader, self.pip_scaler, dataset_info = get_dataloaders(self.config)
        
        self.nodes = dataset_info['num_nodes']
        self.features = dataset_info['num_features']
        self.window = self.config['training']['window']
        self.horizon = self.config['training']['horizon']
        
        hidden_dim = self.config.get('model_defaults', {}).get('hidden_dim', 64)
        cadgt_overrides = self.config.get('model_overrides', {}).get('CADGT', {})
        
        self.model = CADGT(
            num_nodes=self.nodes, seq_len=self.window, future_len=self.horizon,
            ctx_dim=self.features - 1,
            d_model=cadgt_overrides.get('hidden_dim', hidden_dim),
            static_adj=self.adj_static
        ).to(self.device)

        ckpt_path = os.path.join("saved_models", "cadgt_best.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"CADGT checkpoint not found at {ckpt_path}. Please train first.")
            
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        # To simulate a stream, we extract all test set sequences
        self.logger.info("Connecting to 'Live' sensor streams (Loading test set)...")
        self.stream_x = []
        self.stream_y = []
        for x, y in self.test_loader:
            self.stream_x.append(x.numpy())
            self.stream_y.append(y.numpy())
            
        self.stream_x = np.concatenate(self.stream_x, axis=0) # [Total_Samples, Window, Nodes, Features]
        self.stream_y = np.concatenate(self.stream_y, axis=0) # [Total_Samples, Horizon, Nodes]
        self.total_steps = len(self.stream_x)
        self.current_step = 0
        
        # Let's pick 3 random sensors to track continuously on the dashboard
        # Instead of random, let's just track node 0, 50, and 100
        self.tracked_nodes = [0, 50, 100]
        
    def start_dashboard(self):
        """Launches a live matplotlib dashboard simulating real-world UI."""
        
        # Turn on interactive mode
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 10))
        self.fig.suptitle("Live Traffic Forecasting Dashboard (CADGT)", fontsize=16, fontweight='bold')
        self.fig.canvas.manager.set_window_title('Live Simulation')
        
        # Initialize lines
        self.lines_past = []
        self.lines_pred = []
        self.lines_actual = []
        
        for i, ax in enumerate(self.axes):
            ax.set_title(f"Sensor Node {self.tracked_nodes[i]}")
            ax.set_ylabel("Speed (mph)")
            ax.set_ylim(0, 75)
            ax.grid(True, alpha=0.3)
            
            # X-axis will represent relative time steps (minutes offset from "now")
            # Past limits: -60 to 0. Future limits: 0 to +60
            ax.set_xlim(-60, +65)
            ax.axvline(x=0, color='r', linestyle='--', label='Current Time (T-0)')
            
            lp_line, = ax.plot([], [], 'b-', linewidth=2, label='1-Hour Historical Context') # 12 past
            lpred_line, = ax.plot([], [], 'g--', linewidth=2, marker='o', markersize=4, label='CADGT 1-Hour Forecast') # 12 future
            lact_line, = ax.plot([], [], 'k-', alpha=0.3, linewidth=2, label='True Future Outcome (Revealed Later)') # 12 future true
            
            self.lines_past.append(lp_line)
            self.lines_pred.append(lpred_line)
            self.lines_actual.append(lact_line)
            
            if i == 0:
                ax.legend(loc='lower right', framealpha=0.9)
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Simulate time moving forward
        print("\n\n" + "="*80)
        print("STARTING LIVE SIMULATION DASHBOARD")
        print("="*80)
        print("This simulates retrieving 60 minutes of real-world historical data every 5 minutes")
        print("and instantly feeding it to the PyTorch CADGT model to predict the next 60 minutes.")
        print("Watch the Matplotlib window for live updates.\n")

        try:
            # We skip ahead 12 steps every loop so the timeline moves visibly
            for step in range(0, self.total_steps, 12):
                if not plt.fignum_exists(self.fig.number):
                    # User closed the window
                    break 
                    
                self._update_dashboard(step)
                
                # Pause to simulate the wait for real-world sensor data
                # Actually wait just 1-2 seconds visually
                plt.pause(1.5) 
                
        except KeyboardInterrupt:
            print("\nSimulation aborted by user.")
        finally:
            plt.ioff()
            plt.show() # Keep open at the end

    def _update_dashboard(self, step):
        # 1. "Receive" live data from sensors (The X context tensor)
        # Shape: [1, Window=12, Nodes=207, Features=10]
        live_x = self.stream_x[step:step+1] 
        true_y = self.stream_y[step:step+1]
        
        # Convert to tensor and run prediction
        x_tensor = torch.FloatTensor(live_x).to(self.device)
        start_time = time.time()
        with torch.no_grad():
            preds = self.model(x_tensor)
        inference_time = (time.time() - start_time) * 1000 # ms
        
        # Inverse transform to get speeds in mph
        p_np = preds.cpu().numpy()[0] # [H=12, N]
        pred_speeds = self.pip_scaler.inverse_transform(p_np)
        
        past_scaled = live_x[0, :, :, 0] # [W=12, N]
        past_speeds = self.pip_scaler.inverse_transform(past_scaled)
        
        true_scaled = true_y[0] # [H=12, N]
        true_speeds = self.pip_scaler.inverse_transform(true_scaled)
        
        # Update plots
        past_times = np.arange(-55, 5, 5) # -55, -50, ... 0 (12 steps)
        future_times = np.arange(5, 65, 5) # 5, 10, ... 60 (12 steps)
        
        # We also want to connect the past line to the first future point visually
        past_times_connected = np.append(past_times, 5)
        
        for i, n in enumerate(self.tracked_nodes):
            sensor_past = past_speeds[:, n]
            sensor_pred = pred_speeds[:, n]
            sensor_true = true_speeds[:, n]
            
            # Connect the ending of the past to the start of the prediction
            connected_past = np.append(sensor_past, sensor_pred[0])
            
            self.lines_past[i].set_xdata(past_times_connected)
            self.lines_past[i].set_ydata(connected_past)
            self.lines_pred[i].set_xdata(future_times)
            self.lines_pred[i].set_ydata(sensor_pred)
            self.lines_actual[i].set_xdata(future_times)
            self.lines_actual[i].set_ydata(sensor_true)
            
        sys.stdout.write(f"\r[Live Sensor Loop] TimeStep: {step:>5} | "
                         f"Ingested 10 Features (Weather/Time) | "
                         f"CADGT Inference: {inference_time:.1f}ms")
        sys.stdout.flush()

if __name__ == "__main__":
    sim = RealTimeTrafficSimulator()
    sim.start_dashboard()
