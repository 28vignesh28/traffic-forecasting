"""
CAMT-GATformer Diagnostic Script
Diagnoses why CAMT performs 2× worse than other models (MAE 8.44 vs ~3.5).
Checks: gradient norms, prediction statistics, and collapse detection.
"""
import torch
import torch.nn as nn
import yaml
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_dataloaders, load_static_adj
from models.camt_gatformer import TrafficModel as CAMT


def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_loader, _, _, scaler, dataset_info = get_dataloaders(config)
    adj_static = load_static_adj(config['data']['adj_path'])
    adj_tensor = torch.FloatTensor(adj_static).to(device)

    camt_overrides = config.get('model_overrides', {}).get('CAMT', {})
    short_horizon = camt_overrides.get('short_horizon', 3)

    model = CAMT(
        nodes=dataset_info['num_nodes'],
        nfeat=dataset_info['num_features'],
        seq_len=config['training']['window'],
        short_horizon=short_horizon,
        horizon=config['training']['horizon']
    ).to(device)

    # === Gradient Norm Analysis ===
    grad_norms = {}

    def save_grad_norm(name):
        def hook(grad):
            grad_norms[name] = grad.norm().item()
        return hook

    # Register hooks on key layers
    model.graph.linear.weight.register_hook(save_grad_norm('graph_conv'))
    model.short_conv.weight.register_hook(save_grad_norm('short_conv'))
    model.long.layers[0].self_attn.in_proj_weight.register_hook(save_grad_norm('transformer'))
    model.short_head.weight.register_hook(save_grad_norm('short_head'))
    model.long_head.weight.register_hook(save_grad_norm('long_head'))
    model.context.weight.register_hook(save_grad_norm('context_fc'))
    model.gate.weight.register_hook(save_grad_norm('gate'))

    # Train one batch
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    ps_short, pl_long = model(x, adj_tensor)
    loss_fn = nn.HuberLoss()
    loss = 0.25 * loss_fn(ps_short, y[:, :short_horizon, :]) + loss_fn(pl_long, y)
    loss.backward()

    print(f"\nTraining Loss: {loss.item():.6f}")
    print("\n{'='*50}")
    print("=== Gradient Norms (healthy range: 0.01 - 1.0) ===")
    print(f"{'='*50}")
    for name, norm in sorted(grad_norms.items()):
        if norm < 0.001:
            status = "X VANISHING"
        elif norm > 10.0:
            status = "X EXPLODING"
        else:
            status = "OK"
        print(f"  {name:20s}: {norm:10.6f}  {status}")

    # === Prediction Statistics ===
    model.eval()
    with torch.no_grad():
        ps_short, pl_long = model(x, adj_tensor)

        print(f"\n{'='*50}")
        print("=== Prediction Statistics ===")
        print(f"{'='*50}")
        print(f"  Short-term (raw):   mean={ps_short.mean():.4f}, std={ps_short.std():.4f}, "
              f"min={ps_short.min():.4f}, max={ps_short.max():.4f}")
        print(f"  Long-term (raw):    mean={pl_long.mean():.4f}, std={pl_long.std():.4f}, "
              f"min={pl_long.min():.4f}, max={pl_long.max():.4f}")
        print(f"  Ground truth (raw): mean={y.mean():.4f}, std={y.std():.4f}, "
              f"min={y.min():.4f}, max={y.max():.4f}")

        # Check for collapse
        if pl_long.std() < 0.01:
            print("\n  X CRITICAL: Model collapsed to near-constant predictions!")
        elif abs(pl_long.mean() - y.mean()) > 0.5:
            print(f"\n  X WARNING: Predictions severely offset from ground truth "
                  f"(delta={abs(pl_long.mean() - y.mean()):.4f})")
        else:
            print("\n  OK: Prediction distribution looks reasonable")

    # === Adjacency Analysis ===
    print(f"\n{'='*50}")
    print("=== Adjacency Matrix Analysis ===")
    print(f"{'='*50}")
    with torch.no_grad():
        adj_adapt = model.adapt_graph()
        x_traffic = x[:, :, :, 0]
        adj_dyn = model.dynamic_graph(x_traffic)

        adj_static_norm = adj_tensor / (adj_tensor.sum(dim=-1, keepdim=True) + 1e-8)
        adj_base = adj_static_norm + adj_adapt
        adj_total = adj_base.unsqueeze(0) + adj_dyn

        print(f"  Static adj:   mean={adj_tensor.mean():.4f}, max={adj_tensor.max():.4f}")
        print(f"  Adaptive adj: mean={adj_adapt.mean():.4f}, max={adj_adapt.max():.4f}")
        print(f"  Dynamic adj:  mean={adj_dyn.mean():.4f}, max={adj_dyn.max():.4f}")
        print(f"  Combined (pre-softmax): mean={adj_total.mean():.4f}, max={adj_total.max():.4f}")

        row_sums = adj_total.sum(dim=-1)
        print(f"  Row sums (pre-softmax): mean={row_sums.mean():.4f}, "
              f"min={row_sums.min():.4f}, max={row_sums.max():.4f}")

    print(f"\n{'='*50}")
    print("=== Diagnosis Complete ===")
    print(f"{'='*50}")
    print("\nWhat to look for:")
    print("  1. Vanishing gradients (<0.001): Layer is dead, not learning")
    print("  2. Exploding gradients (>10):    Training instability")
    print("  3. Constant predictions (std<0.01): Model collapsed")
    print("  4. Mean offset (>0.5):             Bias initialization problem")


if __name__ == "__main__":
    main()
