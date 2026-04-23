import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphGenerator(nn.Module):
    """
    Generates a dynamic Adjacency Matrix blended with the physical road graph.
    - Physical Graph : Ground-truth road distances (from adj_METR-LA.pkl)
    - Dynamic Graph  : Learned from current traffic state, Top-K sparsified
    - Alpha          : Learnable blend weight — initialised so sigmoid(0)=0.5
    """

    def __init__(self, N, hidden_dim, k=10):
        super(DynamicGraphGenerator, self).__init__()
        self.N = N
        self.k = k
        self.fc_start = nn.Linear(1, 16)

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, A_physical):

        B, T, N, _ = x.shape

        state = x.mean(dim=1)
        state_emb = torch.tanh(self.fc_start(state))

        A_dyn = torch.bmm(state_emb, state_emb.transpose(1, 2))
        A_dyn = F.relu(A_dyn)

        k = min(self.k, N)
        topk_values, topk_indices = torch.topk(A_dyn, k, dim=2)
        mask = torch.full_like(A_dyn, float("-inf"))
        sparse_A_dyn = mask.scatter_(2, topk_indices, topk_values)
        A_dyn = F.softmax(sparse_A_dyn, dim=2)

        A_phys = A_physical.unsqueeze(0).expand(B, -1, -1)
        A_phys_batch = A_phys / (A_phys.sum(dim=-1, keepdim=True) + 1e-8)

        alpha_clamped = torch.sigmoid(self.alpha)
        A_final = (alpha_clamped * A_phys_batch) + ((1 - alpha_clamped) * A_dyn)
        return A_final


class ChebConvLayer(nn.Module):
    """
    Chebyshev Graph Convolution (K-hop spatial).

    FIX #3: Now uses the proper normalised Graph Laplacian.
    Previously, the raw adjacency matrix A was used as if it were the
    Laplacian — mathematically invalid for spectral graph convolution.
    Correct procedure (Defferrard et al., 2016):
        D     = diag(row-sums of A)   [degree matrix]
        L     = D - A                 [combinatorial Laplacian]
        L̃    = 2*L / λ_max - I       [scaled to eigenvalue range [-1, 1]]
    Chebyshev polynomials are then computed on L̃.
    """

    def __init__(self, K, in_f, out_f, N=207):
        super(ChebConvLayer, self).__init__()
        self.K = K
        self.out_f = out_f
        self.Theta = nn.Parameter(torch.FloatTensor(K, in_f, out_f))
        nn.init.xavier_uniform_(self.Theta)

        self.register_buffer("I", torch.eye(N))

    def forward(self, x, A):
        B, T, N, Fin = x.shape

        A_sym = (A + A.transpose(-1, -2)) / 2.0

        degree = A_sym.sum(dim=-1).clamp(min=1e-8)
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(degree))

        I = self.I.unsqueeze(0).expand(B, -1, -1)
        L_sym = I - D_inv_sqrt @ A_sym @ D_inv_sqrt
        L_scaled = L_sym - I

        outputs = torch.zeros(B, T, N, self.out_f, device=x.device)
        T_k_prev2 = None
        T_k_prev = I

        for k in range(self.K):
            if k == 0:
                T_k = I
            elif k == 1:
                T_k = L_scaled
            else:
                T_k = 2.0 * torch.bmm(L_scaled, T_k_prev) - T_k_prev2

            T_k_prev2, T_k_prev = T_k_prev, T_k

            theta_k = self.Theta[k]
            rhs = torch.einsum("btnf,fo->btno", x, theta_k)
            outputs += torch.einsum("bnm,btmo->btno", T_k, rhs)

        return F.relu(outputs)


class TemporalConvNet(nn.Module):
    """Gated Temporal Convolution (GLU) for local trend extraction."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvNet, self).__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (1, kernel_size))

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, (self.pad, 0, 0, 0))
        x_conv = self.conv(x)
        P, Q = torch.chunk(x_conv, 2, dim=1)
        out = P * torch.sigmoid(Q)
        return out.permute(0, 3, 2, 1)


class AMC_DSTGNN(nn.Module):
    """
    Adaptive Multi-Context Dynamic Spatio-Temporal Graph Neural Network.

    Seq2Seq with Scheduled Sampling (Teacher Forcing):
      Encoder : DynGraph → GLU → ChebNet (proper Laplacian) → Attention Fusion → GRU
      Decoder : Autoregressive GRUCell  [traffic-only input, no stale context]

    Fixes applied:
      #3  — ChebConv now uses proper normalised graph Laplacian.
      #4  — Top-K sparsification uses -inf masking (not zero masking).
      #6  — Decoder no longer injects stale last-step context for all horizons.
      #13 — Teacher forcing decay slowed to 0.995^epoch.
      #14 — Alpha initialised at 0.0 (sigmoid → 0.5).
    """

    def __init__(self, nfeat=10, N=207, hidden_dim=128, dropout=0.3, horizon=12):
        super(AMC_DSTGNN, self).__init__()
        self.dropout_p = dropout
        self.N = N
        self.hidden_dim = hidden_dim
        self.horizon = horizon

        self.graph_gen = DynamicGraphGenerator(N, hidden_dim)
        self.context_fc = nn.Linear(nfeat - 1, 32)
        self.tcn = TemporalConvNet(1, 32, kernel_size=3)
        self.cheb_conv = ChebConvLayer(K=3, in_f=32, out_f=32, N=N)
        self.attn_fc = nn.Linear(32 + 32 + 1, 1)
        self.encoder_gru = nn.GRU(32 + 32 + 1, hidden_dim, batch_first=True)

        self.decoder_cell = nn.GRUCell(1, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, A_physical, y=None, teacher_forcing_ratio=0.5):
        """
        x  : [B, T, N, F]
        y  : [B, Horizon, N]  ground-truth (optional, for teacher forcing)
        teacher_forcing_ratio : probability of using true value as next input
        """
        B, T, N, _ = x.shape

        traffic = x[:, :, :, 0:1]
        context = x[:, :, :, 1:]

        A_dynamic = self.graph_gen(traffic, A_physical)
        t_feat = self.tcn(traffic)
        s_feat = self.cheb_conv(t_feat, A_dynamic)
        s_feat = F.dropout(s_feat, p=self.dropout_p, training=self.training)
        c_emb = torch.relu(self.context_fc(context))

        fusion_input = torch.cat([traffic, s_feat, c_emb], dim=-1)
        attn_scores = torch.sigmoid(self.attn_fc(fusion_input))
        z = fusion_input * attn_scores

        z_flat = z.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, -1)
        _, h_n = self.encoder_gru(z_flat)
        hidden_state = h_n[0]

        outputs = []
        current_input = traffic[:, -1, :, 0].reshape(B * N, 1)

        for step in range(self.horizon):

            hidden_state = self.decoder_cell(current_input, hidden_state)
            prediction = self.fc_out(hidden_state)
            outputs.append(prediction)

            if self.training and y is not None and teacher_forcing_ratio > 0:
                use_teacher = torch.bernoulli(
                    torch.full(
                        (B * N, 1), teacher_forcing_ratio, device=prediction.device
                    )
                )
                gt_input = y[:, step, :].reshape(B * N, 1)
                current_input = use_teacher * gt_input + (1 - use_teacher) * prediction
            else:
                current_input = prediction

        outputs = torch.cat(outputs, dim=1)
        return outputs.view(B, N, self.horizon).transpose(1, 2)
