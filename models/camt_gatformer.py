import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    """
    Per-node Graph Convolution Layer.

    FIX #5: Previous implementation used nn.Linear(nodes, out_features),
    mapping the full 207-node vector to 128 — a globally-mixed spatial
    operation, not a GCN.  After graph aggregation each node's embedding
    is an independent [F]-dimensional vector; the linear projection must
    act on that *feature* dimension (in_features→out_features), not the
    node dimension.

    Corrected:
      x      : [B, T, N, in_features]
      adj    : [B, N, N]
      x_agg  = Σ_m adj[b,n,m] * x[b,t,m,f]   → [B, T, N, in_features]
      output = Linear(in_features → out_features) applied per-node
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)

        self.skip = nn.Linear(in_features, out_features)

    def forward(self, x, adj):

        x_agg = torch.einsum("bnm,btmf->btnf", adj, x)
        out = self.linear(x_agg)

        return out + self.skip(x)


class AdaptiveGraph(nn.Module):
    def __init__(self, nodes, embed_dim=16):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(nodes, embed_dim))

    def forward(self):
        A = torch.relu(torch.matmul(self.E1, self.E2.T))
        A = torch.softmax(A, dim=1)
        return A


class DynamicGraph(nn.Module):
    def __init__(self, nodes, hidden_dim=32):
        super().__init__()

        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):

        B, T, N = x.shape

        x_flat = x.permute(0, 2, 1).reshape(B * N, 1, T)

        h1 = self.conv1(x_flat).mean(dim=-1).view(B, N, -1)
        h2 = self.conv2(x_flat).mean(dim=-1).view(B, N, -1)

        A = torch.matmul(h1, h2.transpose(-1, -2))
        A = torch.softmax(torch.relu(A), dim=-1)
        return A


class TrafficModel(nn.Module):
    """
    Context-Aware Multi-scale Traffic model (CAMT GATformer).

    Fixes applied:
      #5  — GraphLayer now projects feature dim (1→128), not node dim (207→128).
      #11 — Short-term auxiliary loss weighted by 0.25 in train.py (not here).
      #14 — ctx_dim, seq_len, horizon are now constructor parameters.
      #15 — adj_total is re-normalised with softmax after the three matrices
            are summed, preventing row-sums > 1 from causing over-smoothing.
      #17 — Context is no longer averaged across nodes (was a no-op);
            the full per-node, per-timestep context is used.
      #18 — Conv1d uses causal (left-only) padding to prevent future leakage.
    """

    def __init__(self, nodes, nfeat=10, seq_len=12, short_horizon=3, horizon=12):
        super().__init__()
        self.seq_len = seq_len
        self.short_horizon = short_horizon
        self.horizon = horizon

        self.adapt_graph = AdaptiveGraph(nodes)
        self.dynamic_graph = DynamicGraph(nodes)

        self.alpha_static = nn.Parameter(torch.tensor(0.0))
        self.alpha_adapt = nn.Parameter(torch.tensor(0.0))

        self.graph = GraphLayer(in_features=1, out_features=128)

        self.short_pad = 2
        self.short_conv = nn.Conv1d(128, 128, 3, padding=0)

        encoder = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.long = nn.TransformerEncoder(encoder, 2)

        ctx_dim = nfeat - 1
        self.context = nn.Linear(ctx_dim, 128)
        self.gate = nn.Linear(128, 128)

        self.feature_attn = nn.Sequential(nn.Linear(nfeat, 1), nn.Sigmoid())

        self.short_temporal = nn.Linear(seq_len, short_horizon)
        self.long_temporal = nn.Linear(seq_len, horizon)

        self.short_head = nn.Linear(128, 1)
        self.long_head = nn.Linear(128, 1)

    def forward(self, x_full, adj_static):
        x_raw = x_full[:, :, :, 0]
        c = x_full[:, :, :, 1:]
        B, T, N, _ = x_full.shape

        alpha = self.feature_attn(x_full).squeeze(-1)
        x = x_raw * alpha

        adj_adapt = self.adapt_graph()

        adj_static_norm = adj_static / (adj_static.sum(dim=-1, keepdim=True) + 1e-8)

        adj_dyn = self.dynamic_graph(x)

        w_static = torch.sigmoid(self.alpha_static)
        w_adapt = torch.sigmoid(self.alpha_adapt)

        w_dyn = 1.0 - w_static - w_adapt

        w_dyn = w_dyn.clamp(min=0.05)

        adj_total = (
            w_static * adj_static_norm.unsqueeze(0)
            + w_adapt * adj_adapt.unsqueeze(0)
            + w_dyn * adj_dyn
        )

        adj_total = adj_total / (adj_total.sum(dim=-1, keepdim=True) + 1e-8)

        x_feat = x.unsqueeze(-1)
        x_feat = self.graph(x_feat, adj_total)

        x_flat = x_feat.permute(0, 2, 1, 3).reshape(B * N, T, 128)

        s = x_flat.permute(0, 2, 1)
        s = F.pad(s, (self.short_pad, 0))
        s = self.short_conv(s)
        s = s.permute(0, 2, 1)

        l = self.long(x_flat)

        fused = s + l

        ctx = self.context(c)
        ctx_flat = ctx.permute(0, 2, 1, 3).reshape(B * N, T, 128)
        fused = fused * torch.sigmoid(self.gate(ctx_flat))

        fused_T = fused.transpose(1, 2)
        short_feat = self.short_temporal(fused_T).transpose(1, 2)
        long_feat = self.long_temporal(fused_T).transpose(1, 2)

        short_out = self.short_head(short_feat).squeeze(-1)
        short_out = short_out.view(B, N, self.short_horizon).transpose(1, 2)

        long_out = self.long_head(long_feat).squeeze(-1)
        long_out = long_out.view(B, N, self.horizon).transpose(1, 2)

        return short_out, long_out
