import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphGenerator(nn.Module):
    """
    Traffic-conditioned dynamic adjacency matrix.
    Blends the physical road graph with a data-driven graph.

    FIX #14: alpha initialised at 0.0 so sigmoid(0.0) = 0.5,
    giving a true 50/50 physical/dynamic blend at initialisation.
    """
    def __init__(self, N, embed_dim=16):
        super().__init__()
        self.fc_embed = nn.Linear(1, embed_dim)
        # FIX #14: was torch.tensor(0.5) → sigmoid(0.5) ≈ 0.622 (not 0.5)
        self.alpha    = nn.Parameter(torch.tensor(0.0))

    def forward(self, traffic, static_adj):
        # traffic    : [B, T, N, 1]
        # static_adj : [N, N]
        state     = traffic[:, -1, :, :]                # [B, N, 1]
        state_emb = torch.tanh(self.fc_embed(state))   # [B, N, embed_dim]

        A_dyn = torch.bmm(state_emb, state_emb.transpose(1, 2))   # [B, N, N]
        A_dyn = F.softmax(F.relu(A_dyn), dim=-1)

        alpha           = torch.sigmoid(self.alpha)
        A_static_batch  = static_adj.unsqueeze(0).expand_as(A_dyn)
        A_final         = alpha * A_static_batch + (1 - alpha) * A_dyn
        return A_final   # [B, N, N]


class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super(GCNLayer, self).__init__()
        self.W = nn.Linear(in_f, out_f)

    def forward(self, x, A):
        # A : [B, N, N],  x : [B, T, N, F]
        out = torch.einsum("bnm,btmf->btnf", A, x)
        return self.W(out)


class ST_ACENet(nn.Module):
    """
    Spatial-Temporal Adaptive Context-Enhanced Network.

    Fixes applied:
      #1  — Sigma floor raised to 0.1 to prevent GaussianNLLLoss instability
             when combined with AMP.  Previously sigma ≥ 1e-3 caused
             log(var) → -13.8 which collapsed training and produced NaN
             loss spikes (confirmed in ST_ACENet logs from 16:10 run).
      #14 — Alpha init corrected: 0.0 instead of 0.5.
      #18 — sigma is now returned to test.py where it is evaluated as
             a calibration/NLL metric alongside MAE/MSE/RMSE/MAPE.
    """
    def __init__(self, nfeat=14, N=207, hidden_dim=64, static_adj=None):
        super(ST_ACENet, self).__init__()

        if static_adj is not None:
            adj_t = torch.FloatTensor(static_adj)
            # Row-normalize so static adj has same scale as softmax-normalized dynamic graph
            adj_t = adj_t / (adj_t.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer('static_adj', adj_t)
        else:
            self.static_adj = None

        self.context_fc  = nn.Linear(nfeat - 1, 32)
        self.graph_gen   = DynamicGraphGenerator(N, embed_dim=16)

        self.gcn1 = GCNLayer(1, 32)
        self.gcn2 = GCNLayer(32, 32)

        fusion_dim       = 1 + 32 + 32 + 32   # traffic + hop1 + hop2 + ctx
        self.alpha_layer = nn.Linear(fusion_dim, 1)
        self.fusion_fc   = nn.Linear(fusion_dim, hidden_dim)

        self.short_branch  = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.long_branch   = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.horizon  = 12
        self.fc_mu    = nn.Linear(hidden_dim, self.horizon)
        self.fc_sigma = nn.Linear(hidden_dim, self.horizon)

    def forward(self, x):
        B, T, N, num_feat = x.shape

        traffic = x[:, :, :, 0:1]
        context = x[:, :, :, 1:]

        # Dynamic Graph
        if self.static_adj is not None:
            A_dynamic = self.graph_gen(traffic, self.static_adj)
        else:
            I         = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
            A_dynamic = self.graph_gen(traffic, I)

        # Context Encoding
        c_emb  = torch.relu(self.context_fc(context))

        # 2-hop Spatial GCN
        x_hop1 = F.relu(self.gcn1(traffic, A_dynamic))
        x_hop2 = F.relu(self.gcn2(x_hop1,  A_dynamic))

        # Adaptive Feature Fusion
        fusion_input = torch.cat([traffic, x_hop1, x_hop2, c_emb], dim=-1)
        alpha        = torch.sigmoid(self.alpha_layer(fusion_input))
        z            = self.fusion_fc(fusion_input) * alpha

        # Multi-Scale Temporal Modelling
        z_flat  = z.view(B * N, T, -1)

        # Short-term (Conv1d)
        z_short = z_flat.permute(0, 2, 1)
        z_short = F.relu(self.short_branch(z_short))
        z_short = z_short[:, :, -1]

        # Long-term (GRU)
        gru_out, _ = self.long_branch(z_flat)
        z_long     = gru_out[:, -1, :]

        # Gated fusion
        z_combined = torch.cat([z_short, z_long], dim=-1)
        gate       = torch.sigmoid(self.temporal_gate(z_combined))
        h          = gate * z_short + (1 - gate) * z_long

        # Point estimate: residual from last observed speed
        delta_mu = self.fc_mu(h).view(B, N, self.horizon).transpose(1, 2)
        last_speed = traffic[:, -1, :, 0].unsqueeze(1)
        mu = last_speed + delta_mu   # [B, horizon, N]

        # =================================================================
        # FIX #1: Sigma floor raised from 1e-3 to 0.1.
        # With sigma→1e-3, var→1e-6, log(var)≈-13.8 and GaussianNLLLoss
        # collapsed: the model chased log(var)→-∞ while the error²/var
        # term periodically exploded.  AMP fp16 can also underflow 1e-3
        # to zero, giving NaN immediately.
        # A floor of 0.1 keeps the loss well-behaved across all epochs.
        # =================================================================
        raw_sigma = self.fc_sigma(h).view(B, N, self.horizon).transpose(1, 2)
        sigma     = F.softplus(raw_sigma) + 0.1   # FIX #1: floor 0.001 → 0.1

        return mu, sigma
