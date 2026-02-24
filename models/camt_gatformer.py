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
        # FIX #5: project the *feature* dimension, not the node dimension
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x   : [B, T, N, in_features]
        # adj : [B, N, N]
        # Aggregate neighbours per node (batched, per-timestep)
        x_agg = torch.einsum('bnm,btmf->btnf', adj, x)  # [B, T, N, in_features]
        return self.linear(x_agg)                          # [B, T, N, out_features]


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
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(1, hidden_dim)

    def forward(self, x):
        # x : [B, T, N]
        summary = x.mean(dim=1).unsqueeze(-1)   # [B, N, 1]
        h1 = self.fc1(summary)                   # [B, N, hidden_dim]
        h2 = self.fc2(summary)                   # [B, N, hidden_dim]
        A  = torch.matmul(h1, h2.transpose(-1, -2))
        A  = torch.softmax(torch.relu(A), dim=-1)
        return A   # [B, N, N]


class TrafficModel(nn.Module):
    """
    Context-Aware Multi-scale Traffic model (CAMT GATformer).

    Fixes applied:
      #5  — GraphLayer now projects feature dim (1→128), not node dim (207→128).
      #11 — Short-term auxiliary loss weighted by 0.25 in train.py (not here).
      #15 — adj_total is re-normalised with softmax after the three matrices
            are summed, preventing row-sums > 1 from causing over-smoothing.
      #17 — Context is no longer averaged across nodes (was a no-op);
            the full per-node, per-timestep context is used.
    """
    def __init__(self, nodes):
        super().__init__()
        self.adapt_graph  = AdaptiveGraph(nodes)
        self.dynamic_graph = DynamicGraph(nodes)

        # FIX #5: in_features=1 (traffic feature dim), out_features=128
        self.graph = GraphLayer(in_features=1, out_features=128)

        self.short = nn.Conv1d(128, 128, 3, padding=1)

        encoder = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, batch_first=True
        )
        self.long = nn.TransformerEncoder(encoder, 2)

        # FIX #17: context maps from 13 → 128 without squashing node dim
        self.context = nn.Linear(13, 128)
        self.gate    = nn.Linear(128, 128)

        # Feature-level attention over all 14 features
        self.feature_attn = nn.Sequential(
            nn.Linear(14, 1),
            nn.Sigmoid()
        )

        # Temporal projection: Past [T=12] → Future horizons [3, 12]
        self.short_temporal = nn.Linear(12, 3)
        self.long_temporal  = nn.Linear(12, 12)

        # FIX #5: output heads now produce per-node predictions (1 value/node)
        self.short_head = nn.Linear(128, 1)
        self.long_head  = nn.Linear(128, 1)

    def forward(self, x_full, adj_static):
        x_raw = x_full[:, :, :, 0]    # [B, T, N]
        c     = x_full[:, :, :, 1:]   # [B, T, N, 13]
        B, T, N, _ = x_full.shape

        # Adaptive feature gating over all 14 features
        alpha = self.feature_attn(x_full).squeeze(-1)   # [B, T, N]
        x     = x_raw * alpha                            # [B, T, N]

        # ---- Graph Construction ----
        adj_adapt = self.adapt_graph()                   # [N, N]
        # Row-normalize static adj so it has same scale as softmax-normalized adaptive graph
        adj_static_norm = adj_static / (adj_static.sum(dim=-1, keepdim=True) + 1e-8)
        adj_base  = adj_static_norm + adj_adapt          # [N, N]

        adj_dyn   = self.dynamic_graph(x)                # [B, N, N]
        adj_total = adj_base.unsqueeze(0) + adj_dyn      # [B, N, N]

        # =================================================================
        # FIX #15: Re-normalise the summed adjacency so row-sums = 1.
        # Previously three softmax-normalised matrices were added without
        # renormalisation, letting row-sums exceed 2 and causing
        # over-smoothing in the graph aggregation.
        # =================================================================
        adj_total = F.softmax(adj_total, dim=-1)         # [B, N, N]

        # ---- Graph Convolution (FIX #5) ----
        # Expand traffic to [B, T, N, 1] so GraphLayer works on feature dim
        x_feat = x.unsqueeze(-1)                         # [B, T, N, 1]
        x_feat = self.graph(x_feat, adj_total)           # [B, T, N, 128]

        # ---- Temporal Branches (node-independent) ----
        # Reshape: [B, T, N, 128] → [B*N, T, 128]
        x_flat = x_feat.permute(0, 2, 1, 3).reshape(B * N, T, 128)

        # Short-term: Conv1d over time
        s      = x_flat.permute(0, 2, 1)                # [B*N, 128, T]
        s      = self.short(s)                           # [B*N, 128, T]
        s      = s.permute(0, 2, 1)                      # [B*N, T, 128]

        # Long-term: Transformer over time
        l      = self.long(x_flat)                       # [B*N, T, 128]

        fused  = s + l                                   # [B*N, T, 128]

        # ---- Context Gating (FIX #17: per-node, per-timestep) ----
        # c: [B, T, N, 13] → ctx: [B, T, N, 128] → [B*N, T, 128]
        ctx      = self.context(c)                                    # [B, T, N, 128]
        ctx_flat = ctx.permute(0, 2, 1, 3).reshape(B * N, T, 128)   # [B*N, T, 128]
        fused    = fused * torch.sigmoid(self.gate(ctx_flat))         # [B*N, T, 128]

        # ---- Temporal Projection to Future Horizons ----
        # [B*N, 128, T] → Linear(T, H) → [B*N, 128, H] → [B*N, H, 128]
        fused_T    = fused.transpose(1, 2)                          # [B*N, 128, T]
        short_feat = self.short_temporal(fused_T).transpose(1, 2)  # [B*N, 3,  128]
        long_feat  = self.long_temporal(fused_T).transpose(1, 2)   # [B*N, 12, 128]

        # ---- Per-node Output (FIX #5: head maps 128→1, not 128→207) ----
        # short_out: [B*N, 3, 1] → [B, N, 3] → [B, 3, N]
        short_out = self.short_head(short_feat).squeeze(-1)         # [B*N, 3]
        short_out = short_out.view(B, N, 3).transpose(1, 2)        # [B, 3, N]

        # long_out : [B*N, 12, 1] → [B, N, 12] → [B, 12, N]
        long_out  = self.long_head(long_feat).squeeze(-1)           # [B*N, 12]
        long_out  = long_out.view(B, N, 12).transpose(1, 2)        # [B, 12, N]

        return short_out, long_out
