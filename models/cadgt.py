import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEncoder(nn.Module):
    def __init__(self, in_dim=13, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, c):
        return self.net(c)

class DynamicGraph(nn.Module):
    def __init__(self, ctx_dim=13, d_model=64):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, ctx_embed):
        # Generate A_t dynamically per batch using Query-Key attention
        Q = self.query(ctx_embed) # [B, T, N, D]
        K = self.key(ctx_embed)   # [B, T, N, D]
        
        # QK^T -> [B, T, N, N]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_model ** 0.5)
        A_t = torch.softmax(scores, dim=-1)
        
        # Pool across time to provide a batch-specific dynamic graph to ST blocks
        return A_t.mean(dim=1) # [B, N, N]

class TrafficProjection(nn.Module):
    def __init__(self, num_nodes=207, seq_len=12, d_model=64):
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.temporal_emb = nn.Parameter(torch.randn(1, seq_len, 1, d_model))
        self.spatial_emb = nn.Parameter(torch.randn(1, 1, num_nodes, d_model))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.linear(x)
        return x + self.temporal_emb + self.spatial_emb

class STBlock(nn.Module):
    def __init__(self, d_model=64, heads=4):
        super().__init__()
        self.temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model*4, batch_first=True, dropout=0.1
        )
        self.spatial_linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, A):
        B, T, N, D = h.shape

        h_temp = h.permute(0, 2, 1, 3).reshape(B*N, T, D)
        h_temp = self.temporal_layer(h_temp)
        h_temp = h_temp.view(B, N, T, D).permute(0, 2, 1, 3)

        h_spatial = torch.matmul(A.unsqueeze(1), h_temp)
        h_spatial = F.relu(self.spatial_linear(h_spatial))

        return self.norm(h + h_temp + h_spatial)

class CADGT(nn.Module):
    def __init__(self, num_nodes=207, seq_len=12, future_len=12, ctx_dim=13, d_model=64):
        super().__init__()
        self.ctx_encoder = ContextEncoder(ctx_dim, d_model)
        self.graph = DynamicGraph(ctx_dim, d_model)
        self.proj = TrafficProjection(num_nodes, seq_len, d_model)
        
        # UPGRADE: Feature-level attention gate (operates in projected space)
        self.feature_gate = nn.Sequential(
            nn.Linear(d_model + d_model, d_model),  # traffic_proj + context_embed
            nn.Sigmoid()
        )
        
        self.st1 = STBlock(d_model)
        self.st2 = STBlock(d_model)
        
        self.output_head = nn.Sequential(
            nn.Linear(seq_len * d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, future_len)
        )

    def forward(self, x_full):
        x = x_full[:, :, :, 0] # Traffic [B, T, N]
        c = x_full[:, :, :, 1:] # Context [B, T, N, 13]
        
        c_embed = self.ctx_encoder(c)
        A = self.graph(c_embed)
        
        h = self.proj(x)
        
        # UPGRADE: Adaptive Feature Fusion (using projected features + context)
        gate_input = torch.cat([h, c_embed], dim=-1)  # [B, T, N, 2*D]
        alpha = self.feature_gate(gate_input)  # [B, T, N, D]
        h = h * alpha  # Feature-level gating in projected space
        
        h = self.st1(h, A)
        h = self.st2(h, A)
        
        B, T, N, D = h.shape
        h = h.permute(0, 2, 1, 3).reshape(B, N, T * D) 
        
        pred = self.output_head(h).transpose(1, 2) 
        return pred
