import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ContextEncoder(nn.Module):
    def __init__(self, in_dim=9, d_model=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, c):
        return self.net(c)


class DynamicGraph(nn.Module):
    """
    Traffic-conditioned dynamic graph generator.

    FIX #7: Previously the graph was computed from context features ONLY
    (weather, time-of-day), completely ignoring actual traffic state.
    The most informative signal for graph adaptation is current traffic
    congestion, so traffic projections (h) are now concatenated with
    context embeddings before computing Q and K.

    FIX #8: Previously A_t [B, T, N, N] was averaged over time into
    [B, N, N], losing within-sequence temporal dynamics.  The full
    temporal graph [B, T, N, N] is now returned so each STBlock timestep
    can use a distinct adjacency.
    """

    def __init__(self, d_model=64):
        super().__init__()

        self.query = nn.Linear(2 * d_model, d_model)
        self.key = nn.Linear(2 * d_model, d_model)
        self.d_model = d_model

    def forward(self, h, ctx_embed):
        """
        h         : traffic projection  [B, T, N, D]
        ctx_embed : context embedding   [B, T, N, D]
        returns A : temporal graph      [B, T, N, N]  (FIX #8: no time-mean)
        """

        combined = torch.cat([h, ctx_embed], dim=-1)

        Q = self.query(combined)
        K = self.key(combined)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_model**0.5)
        A_t = torch.softmax(scores, dim=-1)

        return A_t


class TrafficProjection(nn.Module):
    def __init__(self, num_nodes=207, seq_len=12, d_model=64):
        super().__init__()
        self.linear = nn.Linear(1, d_model)

        self.temporal_emb = nn.Parameter(
            torch.randn(1, seq_len, 1, d_model) * (d_model**-0.5)
        )
        self.spatial_emb = nn.Parameter(
            torch.randn(1, 1, num_nodes, d_model) * (d_model**-0.5)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.linear(x)
        return x + self.temporal_emb + self.spatial_emb


class STBlock(nn.Module):
    """
    Spatio-Temporal Block.

    FIX #8: Accepts per-timestep adjacency A [B, T, N, N] and applies
    spatial aggregation independently at each timestep via einsum.

    FIX #12: Double residual removed.
    TransformerEncoderLayer already applies its own residual internally:
        output = LayerNorm(x + MultiHeadAttn(x))
    The original code added 'h' a second time in the outer residual,
    resulting in h appearing twice in the computation.
    Corrected to: norm(h_temp + h_spatial) only.
    """

    def __init__(self, d_model=64, heads=4):
        super().__init__()
        self.temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.spatial_linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, A):
        """
        h : [B, T, N, D]
        A : [B, T, N, N]  — per-timestep adjacency (FIX #8)
        """
        B, T, N, D = h.shape

        h_temp_flat = h.permute(0, 2, 1, 3).reshape(B * N, T, D)
        h_temp_flat = self.temporal_layer(h_temp_flat)
        h_temp = h_temp_flat.view(B, N, T, D).permute(0, 2, 1, 3)

        h_spatial = torch.matmul(A, h_temp)
        h_spatial = F.relu(self.spatial_linear(h_spatial))

        return self.norm(h_temp + h_spatial)


class CADGT(nn.Module):
    """
    Context-Aware Dynamic Graph Transformer.

    Fixes applied in this version:
      #7  — Dynamic graph conditioned on traffic state + context.
      #8  — Full temporal graph [B, T, N, N] returned & applied per-step.
      #12 — Double residual in STBlock removed.
    """

    def __init__(
        self,
        num_nodes=207,
        seq_len=12,
        future_len=12,
        ctx_dim=13,
        d_model=64,
        static_adj=None,
    ):
        super().__init__()
        self.ctx_encoder = ContextEncoder(ctx_dim, d_model)

        self.graph1 = DynamicGraph(d_model)
        self.graph2 = DynamicGraph(d_model)
        self.proj = TrafficProjection(num_nodes, seq_len, d_model)

        if static_adj is not None:
            adj_t = torch.FloatTensor(static_adj)
            adj_t = adj_t / (adj_t.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer("static_adj", adj_t)
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.static_adj = None

        self.feature_gate = nn.Sequential(
            nn.Linear(d_model + d_model, d_model), nn.Sigmoid()
        )

        self.st1 = STBlock(d_model)
        self.st2 = STBlock(d_model)

        self.output_head = nn.Sequential(
            nn.Linear(seq_len * d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, future_len),
        )

    def _blend_static(self, A_dyn):
        """FIX #9: Blend dynamic adjacency with physical road graph if available."""
        if self.static_adj is not None:
            alpha = torch.sigmoid(self.alpha)
            A_static = self.static_adj.unsqueeze(0).unsqueeze(0)
            return alpha * A_static + (1 - alpha) * A_dyn
        return A_dyn

    def forward(self, x_full):
        x = x_full[:, :, :, 0]
        c = x_full[:, :, :, 1:]

        c_embed = self.ctx_encoder(c)

        h = self.proj(x)

        A1 = self.graph1(h, c_embed)
        A1 = self._blend_static(A1)

        gate_input = torch.cat([h, c_embed], dim=-1)
        gate_alpha = self.feature_gate(gate_input)
        h = h * gate_alpha

        if self.training:
            h = checkpoint(self.st1, h, A1, use_reentrant=False)
        else:
            h = self.st1(h, A1)

        A2 = self.graph2(h, c_embed)
        A2 = self._blend_static(A2)

        if self.training:
            h = checkpoint(self.st2, h, A2, use_reentrant=False)
        else:
            h = self.st2(h, A2)

        B, T, N, D = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, N, T * D)
        pred = self.output_head(h).transpose(1, 2)
        return pred
