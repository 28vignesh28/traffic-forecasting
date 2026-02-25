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
        # Residual projection to preserve individual node identity
        self.skip   = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x   : [B, T, N, in_features]
        # adj : [B, N, N]
        # Aggregate neighbours per node (batched, per-timestep)
        x_agg = torch.einsum('bnm,btmf->btnf', adj, x)  # [B, T, N, in_features]
        out   = self.linear(x_agg)                         # [B, T, N, out_features]
        # Residual: project input to match out_features and add identity
        # This prevents over-smoothing by preserving individual node signal
        return out + self.skip(x)                           # [B, T, N, out_features]


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
        # Use Conv1d to extract multi-dimensional temporal features per node
        # This avoids the rank-1 problem where Linear(1→D) maps scalars to collinear embeddings
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x : [B, T, N]
        B, T, N = x.shape
        # Reshape to [B*N, 1, T] — each node's time series as a 1D signal
        x_flat = x.permute(0, 2, 1).reshape(B * N, 1, T)  # [B*N, 1, T]
        
        # Extract multi-dimensional temporal features via Conv1d
        h1 = self.conv1(x_flat).mean(dim=-1).view(B, N, -1)  # [B, N, hidden_dim]
        h2 = self.conv2(x_flat).mean(dim=-1).view(B, N, -1)  # [B, N, hidden_dim]
        
        A  = torch.matmul(h1, h2.transpose(-1, -2))
        A  = torch.softmax(torch.relu(A), dim=-1)
        return A   # [B, N, N]


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
        self.seq_len       = seq_len
        self.short_horizon = short_horizon
        self.horizon       = horizon

        self.adapt_graph   = AdaptiveGraph(nodes)
        self.dynamic_graph = DynamicGraph(nodes)

        # =================================================================
        # FIX: Learnable blend weights for the three adjacency components.
        # Previously the three matrices were simply added (raw sum),
        # producing uniform row-sums of 3.0 and near-uniform softmax
        # attention — causing severe over-smoothing that collapsed
        # prediction variance to 7× below ground truth (MAE 8.44).
        # Now: α₁·static + α₂·adaptive + α₃·dynamic with sigmoid gates.
        # =================================================================
        self.alpha_static  = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        self.alpha_adapt   = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5

        # FIX #5: in_features=1 (traffic feature dim), out_features=128
        self.graph = GraphLayer(in_features=1, out_features=128)

        # FIX #18: Causal Conv1d — left-pad with kernel_size-1, no right-pad
        self.short_pad  = 2  # kernel_size - 1
        self.short_conv = nn.Conv1d(128, 128, 3, padding=0)  # no built-in padding

        encoder = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, batch_first=True
        )
        self.long = nn.TransformerEncoder(encoder, 2)

        # FIX #14: ctx_dim derived from nfeat (total features minus traffic)
        ctx_dim = nfeat - 1
        self.context = nn.Linear(ctx_dim, 128)
        self.gate    = nn.Linear(128, 128)

        # Feature-level attention over all features
        self.feature_attn = nn.Sequential(
            nn.Linear(nfeat, 1),
            nn.Sigmoid()
        )

        # FIX #19: Temporal projection uses parameterized dimensions
        self.short_temporal = nn.Linear(seq_len, short_horizon)
        self.long_temporal  = nn.Linear(seq_len, horizon)

        # FIX #5: output heads now produce per-node predictions (1 value/node)
        self.short_head = nn.Linear(128, 1)
        self.long_head  = nn.Linear(128, 1)

    def forward(self, x_full, adj_static):
        x_raw = x_full[:, :, :, 0]    # [B, T, N]
        c     = x_full[:, :, :, 1:]   # [B, T, N, 9]
        B, T, N, _ = x_full.shape

        # Adaptive feature gating over all 10 features
        alpha = self.feature_attn(x_full).squeeze(-1)   # [B, T, N]
        x     = x_raw * alpha                            # [B, T, N]

        # ---- Graph Construction (FIX: learnable blend) ----
        adj_adapt = self.adapt_graph()                   # [N, N]
        # Row-normalize static adj to match softmax-normalized graphs
        adj_static_norm = adj_static / (adj_static.sum(dim=-1, keepdim=True) + 1e-8)

        adj_dyn = self.dynamic_graph(x)                  # [B, N, N]

        # Learnable weighted blend instead of raw sum
        w_static = torch.sigmoid(self.alpha_static)
        w_adapt  = torch.sigmoid(self.alpha_adapt)
        # Dynamic weight = remainder to ensure weights sum to 1
        w_dyn    = 1.0 - w_static - w_adapt
        # Clamp dynamic weight to be non-negative
        w_dyn    = w_dyn.clamp(min=0.05)

        adj_total = (w_static * adj_static_norm.unsqueeze(0)
                     + w_adapt * adj_adapt.unsqueeze(0)
                     + w_dyn * adj_dyn)                   # [B, N, N]

        # Row-normalize to prevent magnitude drift
        adj_total = adj_total / (adj_total.sum(dim=-1, keepdim=True) + 1e-8)

        # ---- Graph Convolution (FIX #5) ----
        # Expand traffic to [B, T, N, 1] so GraphLayer works on feature dim
        x_feat = x.unsqueeze(-1)                         # [B, T, N, 1]
        x_feat = self.graph(x_feat, adj_total)           # [B, T, N, 128]

        # ---- Temporal Branches (node-independent) ----
        # Reshape: [B, T, N, 128] → [B*N, T, 128]
        x_flat = x_feat.permute(0, 2, 1, 3).reshape(B * N, T, 128)

        # Short-term: Causal Conv1d over time (FIX #18: left-pad only)
        s      = x_flat.permute(0, 2, 1)                           # [B*N, 128, T]
        s      = F.pad(s, (self.short_pad, 0))                     # left-pad only
        s      = self.short_conv(s)                                # [B*N, 128, T]
        s      = s.permute(0, 2, 1)                                # [B*N, T, 128]

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
        short_out = self.short_head(short_feat).squeeze(-1)                           # [B*N, short_horizon]
        short_out = short_out.view(B, N, self.short_horizon).transpose(1, 2)          # [B, short_horizon, N]

        long_out  = self.long_head(long_feat).squeeze(-1)                             # [B*N, horizon]
        long_out  = long_out.view(B, N, self.horizon).transpose(1, 2)                 # [B, horizon, N]

        return short_out, long_out
