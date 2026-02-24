import torch
import torch.nn as nn

class GraphLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: [B, T, N], adj: [B, N, N]
        # We need to apply adj over the N dimension.
        # Einsum is the safest way: B=batch, N=nodes, M=neighbors, T=time
        # x is conceptually features here, we want out[B, T, N] = sum_M adj[B, N, M] * x[B, T, M]
        x = torch.einsum('bnm,btm->btn', adj, x)
        return self.linear(x)

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
        # Per-node feature projection instead of node-mixing
        self.fc1 = nn.Linear(1, hidden_dim)  # Process per-node traffic features
        self.fc2 = nn.Linear(1, hidden_dim)

    def forward(self, x):
        # x: [B, T, N] — traffic data
        summary = x.mean(dim=1).unsqueeze(-1)  # [B, N, 1] — temporal average per node
        h1 = self.fc1(summary)  # [B, N, hidden_dim]
        h2 = self.fc2(summary)  # [B, N, hidden_dim]
        # Outer product in embedding space → [B, N, N]
        A = torch.matmul(h1, h2.transpose(-1, -2))
        A = torch.softmax(torch.relu(A), dim=-1)
        return A

class TrafficModel(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.adapt_graph = AdaptiveGraph(nodes)
        self.dynamic_graph = DynamicGraph(nodes)
        self.graph = GraphLayer(nodes, 128)
        self.short = nn.Conv1d(128,128,3,padding=1)

        encoder = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, batch_first=True
        )
        self.long = nn.TransformerEncoder(encoder,2)
        self.context = nn.Linear(13,128)
        self.gate = nn.Linear(128,128)

        # UPGRADE: Feature-level attention weights (Novelty 3)
        self.feature_attn = nn.Sequential(
            nn.Linear(14, 1), 
            nn.Sigmoid()
        )

        # Map sequence length [T=12] -> [Horizon=3 (short) or Horizon=12 (long)]
        self.short_temporal = nn.Linear(12, 3)
        self.long_temporal = nn.Linear(12, 12)
        
        self.short_head = nn.Linear(128, nodes)
        self.long_head = nn.Linear(128, nodes)

    def forward(self, x_full, adj_static):
        x_raw = x_full[:, :, :, 0] 
        c = x_full[:, :, :, 1:] 

        # UPGRADE: Adaptive Feature Fusion
        alpha = self.feature_attn(x_full).squeeze(-1) 
        x = x_raw * alpha 

        adj_adapt = self.adapt_graph()
        adj_base = adj_static + adj_adapt
        
        adj_dyn = self.dynamic_graph(x)
        adj_total = adj_base.unsqueeze(0) + adj_dyn
        
        x = self.graph(x, adj_total)

        s = x.permute(0,2,1)
        s = self.short(s)
        s = s.permute(0,2,1)

        l = self.long(x)

        ctx = self.context(c.mean(dim=2))

        fused = s + l
        fused = fused * torch.sigmoid(self.gate(ctx)) # [B, T=12, 128]
        
        # Temporal projection: map Past window [12] -> Future horizons [3, 12]
        # Transpose so time is last dim: [B, 128, T] -> Linear(T, H) -> [B, 128, H] -> Transpose back
        fused_T = fused.transpose(1, 2)
        short_feat = self.short_temporal(fused_T).transpose(1, 2) # [B, 3, 128]
        long_feat = self.long_temporal(fused_T).transpose(1, 2)   # [B, 12, 128]

        return self.short_head(short_feat), self.long_head(long_feat)
