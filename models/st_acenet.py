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

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, traffic, static_adj):

        state = traffic.mean(dim=1)
        state_emb = torch.tanh(self.fc_embed(state))

        A_dyn = torch.bmm(state_emb, state_emb.transpose(1, 2))
        A_dyn = F.softmax(F.relu(A_dyn), dim=-1)

        alpha = torch.sigmoid(self.alpha)
        A_static_batch = static_adj.unsqueeze(0).expand_as(A_dyn)
        A_final = alpha * A_static_batch + (1 - alpha) * A_dyn
        return A_final


class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super(GCNLayer, self).__init__()
        self.W = nn.Linear(in_f, out_f)

    def forward(self, x, A):

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

    def __init__(self, nfeat=10, N=207, hidden_dim=64, horizon=12, static_adj=None):
        super(ST_ACENet, self).__init__()

        if static_adj is not None:
            adj_t = torch.FloatTensor(static_adj)

            adj_t = adj_t / (adj_t.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer("static_adj", adj_t)
        else:
            self.static_adj = None

        self.context_fc = nn.Linear(nfeat - 1, 32)
        self.graph_gen = DynamicGraphGenerator(N, embed_dim=16)

        self.gcn1 = GCNLayer(1, 32)
        self.gcn2 = GCNLayer(32, 32)

        fusion_dim = 1 + 32 + 32 + 32
        self.alpha_layer = nn.Linear(fusion_dim, 1)
        self.fusion_fc = nn.Linear(fusion_dim, hidden_dim)

        self.short_pad = 2
        self.short_branch = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=0)
        self.long_branch = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.horizon = horizon
        self.fc_mu = nn.Linear(hidden_dim, self.horizon)
        self.fc_sigma = nn.Linear(hidden_dim, self.horizon)

    def forward(self, x):
        B, T, N, num_feat = x.shape

        traffic = x[:, :, :, 0:1]
        context = x[:, :, :, 1:]

        if self.static_adj is not None:
            A_dynamic = self.graph_gen(traffic, self.static_adj)
        else:
            I = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)
            A_dynamic = self.graph_gen(traffic, I)

        c_emb = torch.relu(self.context_fc(context))

        x_hop1 = F.relu(self.gcn1(traffic, A_dynamic))
        x_hop2 = F.relu(self.gcn2(x_hop1, A_dynamic))

        fusion_input = torch.cat([traffic, x_hop1, x_hop2, c_emb], dim=-1)
        alpha = torch.sigmoid(self.alpha_layer(fusion_input))
        z = self.fusion_fc(fusion_input) * alpha

        z_flat = z.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, -1)

        z_short = z_flat.permute(0, 2, 1)
        z_short = F.pad(z_short, (self.short_pad, 0))
        z_short = F.relu(self.short_branch(z_short))
        z_short = z_short[:, :, -1]

        gru_out, _ = self.long_branch(z_flat)
        z_long = gru_out[:, -1, :]

        z_combined = torch.cat([z_short, z_long], dim=-1)
        gate = torch.sigmoid(self.temporal_gate(z_combined))
        h = gate * z_short + (1 - gate) * z_long

        delta_mu = self.fc_mu(h).view(B, N, self.horizon).transpose(1, 2)
        last_speed = traffic[:, -1, :, 0].unsqueeze(1)
        mu = last_speed + delta_mu

        raw_sigma = self.fc_sigma(h).view(B, N, self.horizon).transpose(1, 2)

        sigma_increments = F.softplus(raw_sigma)
        sigma = torch.cumsum(sigma_increments, dim=1) + 0.1

        return mu, sigma
