import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphGenerator(nn.Module):
    """
    Generates a traffic-conditioned dynamic adjacency matrix per batch.
    Blends the physical road graph with a data-driven graph using a learnable alpha.
    """
    def __init__(self, N, embed_dim=16):
        super().__init__()
        self.fc_embed = nn.Linear(1, embed_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, traffic, static_adj):
        # traffic: [B, T, N, 1]
        state = traffic[:, -1, :, :]          # [B, N, 1] — latest traffic snapshot
        state_emb = torch.tanh(self.fc_embed(state))  # [B, N, embed_dim]

        # Compute input-dependent dynamic adjacency via outer product
        A_dyn = torch.bmm(state_emb, state_emb.transpose(1, 2))  # [B, N, N]
        A_dyn = F.softmax(F.relu(A_dyn), dim=-1)

        # Blend static road geometry with dynamic traffic-conditioned graph
        alpha = torch.sigmoid(self.alpha)
        A_static_batch = static_adj.unsqueeze(0).expand_as(A_dyn)
        A_final = alpha * A_static_batch + (1 - alpha) * A_dyn

        return A_final  # [B, N, N]

class GCNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super(GCNLayer, self).__init__()
        self.W = nn.Linear(in_f, out_f)

    def forward(self, x, A):
        # A: [B, N, N] (batch-wise dynamic), x: [B, T, N, F]
        # Use einsum to handle batched adjacency
        out = torch.einsum("bnm,btmf->btnf", A, x)
        return self.W(out)

class ST_ACENet(nn.Module):
    def __init__(self, nfeat=14, N=207, hidden_dim=64, static_adj=None):
        super(ST_ACENet, self).__init__()

        if static_adj is not None:
            self.register_buffer('static_adj', torch.FloatTensor(static_adj))
        else:
            self.static_adj = None

        # --- Context Integration ---
        self.context_fc = nn.Linear(nfeat - 1, 32)

        # --- Dynamic Graph Generator (Claim 1) ---
        self.graph_gen = DynamicGraphGenerator(N, embed_dim=16)

        # --- Spatial: 2-hop GCN ---
        self.gcn1 = GCNLayer(1, 32)
        self.gcn2 = GCNLayer(32, 32)

        # --- Adaptive Feature Fusion ---
        fusion_dim = 1 + 32 + 32 + 32  # traffic(1) + x_hop1(32) + x_hop2(32) + c_emb(32)
        self.alpha_layer = nn.Linear(fusion_dim, 1)
        self.fusion_fc = nn.Linear(fusion_dim, hidden_dim)

        # --- Multi-Scale Temporal Modeling (Claim 3) ---
        # Short-term branch: Conv1d captures local patterns
        self.short_branch = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        # Long-term branch: GRU captures sequential dependencies
        self.long_branch = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # Temporal fusion gate
        self.temporal_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # --- Unified Multi-Horizon Output ---
        self.horizon = 12
        self.fc_mu = nn.Linear(hidden_dim, self.horizon)
        self.fc_sigma = nn.Linear(hidden_dim, self.horizon)

    def forward(self, x):
        B, T, N, num_feat = x.shape

        traffic = x[:, :, :, 0:1]
        context = x[:, :, :, 1:]

        # === Dynamic Graph (Claim 1) ===
        if self.static_adj is not None:
            A_dynamic = self.graph_gen(traffic, self.static_adj)
        else:
            # Fallback: identity + self-loops if no static adj provided
            I = torch.eye(N).to(x.device).unsqueeze(0).expand(B, -1, -1)
            A_dynamic = self.graph_gen(traffic, I)

        # === Context Encoding ===
        c_emb = torch.relu(self.context_fc(context))

        # === Spatial: 2-hop GCN with dynamic graph ===
        x_hop1 = F.relu(self.gcn1(traffic, A_dynamic))
        x_hop2 = F.relu(self.gcn2(x_hop1, A_dynamic))

        # === Adaptive Feature Fusion ===
        fusion_input = torch.cat([traffic, x_hop1, x_hop2, c_emb], dim=-1)
        alpha = torch.sigmoid(self.alpha_layer(fusion_input))
        z = self.fusion_fc(fusion_input)
        z = z * alpha  # Feature-level gating

        # === Multi-Scale Temporal Modeling (Claim 3) ===
        z_flat = z.view(B * N, T, -1)  # [B*N, T, hidden_dim]

        # Short-term branch (Conv1d over time)
        z_short = z_flat.permute(0, 2, 1)       # [B*N, hidden_dim, T]
        z_short = F.relu(self.short_branch(z_short))
        z_short = z_short[:, :, -1]              # [B*N, hidden_dim] — last timestep

        # Long-term branch (GRU over time)
        gru_out, _ = self.long_branch(z_flat)
        z_long = gru_out[:, -1, :]               # [B*N, hidden_dim] — last hidden

        # Fuse short + long via gated combination
        z_combined = torch.cat([z_short, z_long], dim=-1)  # [B*N, hidden_dim*2]
        gate = torch.sigmoid(self.temporal_gate(z_combined))
        h = gate * z_short + (1 - gate) * z_long

        # === Multi-Horizon Output ===
        delta_mu = self.fc_mu(h)
        delta_mu = delta_mu.view(B, N, self.horizon).transpose(1, 2)

        last_speed = traffic[:, -1, :, 0].unsqueeze(1)
        mu = last_speed + delta_mu

        raw_sigma = self.fc_sigma(h).view(B, N, self.horizon).transpose(1, 2)
        sigma = F.softplus(raw_sigma) + 1e-3

        return mu, sigma