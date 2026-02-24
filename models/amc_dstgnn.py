import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DynamicGraphGenerator(nn.Module):
    """
    Generates a dynamic Adjacency Matrix blended with the physical road graph.
    - Physical Graph: Ground-truth road distances (from adj_METR-LA.pkl)
    - Dynamic Graph: Learned from current traffic state, Top-K sparsified
    - Alpha: Learnable blend weight between physical and dynamic
    """
    def __init__(self, N, hidden_dim):
        super(DynamicGraphGenerator, self).__init__()
        self.N = N
        self.fc_start = nn.Linear(1, 16)
        
        # Learnable weight to balance Physical vs. Dynamic graph
        self.alpha = nn.Parameter(torch.tensor(0.5)) 

    def forward(self, x, A_physical):
        # x: [B, T, N, 1]
        # A_physical: [N, N] (from adj_METR-LA.pkl)
        B, T, N, _ = x.shape
        
        state = x[:, -1, :, :]  # [B, N, 1]
        state_emb = torch.tanh(self.fc_start(state))  # [B, N, 16]
        
        # 1. Compute Dynamic Adjustments
        A_dyn = torch.bmm(state_emb, state_emb.transpose(1, 2))  # [B, N, N]
        A_dyn = F.relu(A_dyn)
        
        # 2. Top-K Sparsification (Keep only the 10 strongest connections per node)
        k = 10
        topk_values, topk_indices = torch.topk(A_dyn, k, dim=2)
        sparse_A_dyn = torch.zeros_like(A_dyn).scatter_(2, topk_indices, topk_values)
        A_dyn = F.softmax(sparse_A_dyn, dim=2)
        
        # 3. Process Physical Graph (expand to batch)
        A_phys_batch = A_physical.unsqueeze(0).expand(B, -1, -1)
        
        # 4. Blend Physical (Real Roads) with Dynamic (Current Traffic)
        alpha_clamped = torch.sigmoid(self.alpha) 
        A_final = (alpha_clamped * A_phys_batch) + ((1 - alpha_clamped) * A_dyn)
        
        return A_final

class ChebConvLayer(nn.Module):
    """Chebyshev Graph Convolution (K-hop spatial)"""
    def __init__(self, K, in_f, out_f):
        super(ChebConvLayer, self).__init__()
        self.K = K
        self.out_f = out_f
        self.Theta = nn.Parameter(torch.FloatTensor(K, in_f, out_f))
        nn.init.xavier_uniform_(self.Theta)

    def forward(self, x, A):
        B, T, N, Fin = x.shape
        
        # Rescale: L = 2*A/λ_max - I (range [-1, 1] for Chebyshev stability)
        I = torch.eye(N).to(x.device).unsqueeze(0).expand(B, -1, -1)
        # Compute λ_max per batch for proper normalization
        lambda_max = torch.max(torch.sum(A, dim=-1), dim=-1)[0].view(B, 1, 1).clamp(min=1.0)
        L = 2 * A / lambda_max - I

        outputs = torch.zeros(B, T, N, self.out_f).to(x.device)
        T_k_prev = I
        T_k_prev2 = None
        
        for k in range(self.K):
            if k == 0: T_k = I
            elif k == 1: T_k = L
            else: T_k = 2 * torch.bmm(L, T_k_prev) - T_k_prev2
            
            T_k_prev2, T_k_prev = T_k_prev, T_k
            
            theta_k = self.Theta[k]
            rhs = torch.einsum("btnf,fo->btno", x, theta_k)
            outputs += torch.einsum("bnm,btmo->btno", T_k, rhs)
            
        return F.relu(outputs)

class TemporalConvNet(nn.Module):
    """Gated Temporal Convolution (GLU) for local trend extraction"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvNet, self).__init__()
        self.pad = (kernel_size - 1) 
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (1, kernel_size))

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # [B, F, N, T]
        x = F.pad(x, (self.pad, 0, 0, 0))
        x_conv = self.conv(x)
        P, Q = torch.chunk(x_conv, 2, dim=1)
        out = P * torch.sigmoid(Q)
        return out.permute(0, 3, 2, 1)

class AMC_DSTGNN(nn.Module):
    """
    Adaptive Multi-Context Dynamic Spatio-Temporal Graph Neural Network
    
    Seq2Seq with Scheduled Sampling (Teacher Forcing):
    - Encoder: DynGraph -> GLU -> ChebNet -> Attention Fusion -> GRU
    - Decoder: Autoregressive GRUCell with optional teacher forcing
    """
    def __init__(self, nfeat=14, N=207, hidden_dim=128, dropout=0.3, horizon=12):
        super(AMC_DSTGNN, self).__init__()
        self.dropout_p = dropout
        self.N = N
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # --- ENCODER ---
        self.graph_gen = DynamicGraphGenerator(N, hidden_dim)
        self.context_fc = nn.Linear(nfeat - 1, 32)
        self.tcn = TemporalConvNet(1, 32, kernel_size=3)
        self.cheb_conv = ChebConvLayer(K=3, in_f=32, out_f=32)
        self.attn_fc = nn.Linear(32 + 32 + 1, 1)
        self.encoder_gru = nn.GRU(32 + 32 + 1, hidden_dim, batch_first=True)
        
        # --- DECODER ---
        self.decoder_cell = nn.GRUCell(1 + 32, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, A_physical, y=None, teacher_forcing_ratio=0.5):
        """
        x: Input [B, T, N, F]
        y: Ground truth target [B, N, Horizon] (optional, for teacher forcing)
        teacher_forcing_ratio: probability of using true value as next input (0.0-1.0)
        """
        B, T, N, _ = x.shape
        
        traffic = x[:, :, :, 0:1]  # [B, T, N, 1]
        context = x[:, :, :, 1:]   # [B, T, N, F-1]
        
        # === ENCODER ===
        A_dynamic = self.graph_gen(traffic, A_physical)
        t_feat = self.tcn(traffic)
        s_feat = self.cheb_conv(t_feat, A_dynamic)
        s_feat = F.dropout(s_feat, p=self.dropout_p, training=self.training)
        c_emb = torch.relu(self.context_fc(context))
        
        fusion_input = torch.cat([traffic, s_feat, c_emb], dim=-1)
        attn_scores = torch.sigmoid(self.attn_fc(fusion_input))
        z = fusion_input * attn_scores
        
        z_flat = z.view(B * N, T, -1)
        # encoder_gru is batch_first=True, output h_n is [num_layers, B*N, hidden_dim]
        _, h_n = self.encoder_gru(z_flat)
        hidden_state = h_n[0]  # [B*N, hidden_dim]
        
        # === DECODER (with Teacher Forcing) ===
        outputs = []
        current_input = traffic[:, -1, :, 0].reshape(B * N, 1)  # Last observed traffic
        # We use the final summarized context embedding as static context for the decoder
        static_context = c_emb[:, -1, :, :].reshape(B * N, 32)
        
        for step in range(self.horizon):
            decoder_input = torch.cat([current_input, static_context], dim=1)
            hidden_state = self.decoder_cell(decoder_input, hidden_state)
            prediction = self.fc_out(hidden_state)  # [B*N, 1]
            outputs.append(prediction)
            
            # Teacher Forcing: use true value or own prediction for next step
            if self.training and y is not None and random.random() < teacher_forcing_ratio:
                # y is [B, Horizon, N] -> we need [B*N, 1] for current step
                current_input = y[:, step, :].reshape(B * N, 1)
            else:
                current_input = prediction
        
        outputs = torch.cat(outputs, dim=1)  # [B*N, horizon]
        # Reshape to [B, N, Horizon], then transpose to [B, Horizon, N] to match standard targets
        return outputs.view(B, N, self.horizon).transpose(1, 2)
