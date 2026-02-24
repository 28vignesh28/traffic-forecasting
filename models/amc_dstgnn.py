import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DynamicGraphGenerator(nn.Module):
    """
    Generates a dynamic Adjacency Matrix blended with the physical road graph.
    - Physical Graph : Ground-truth road distances (from adj_METR-LA.pkl)
    - Dynamic Graph  : Learned from current traffic state, Top-K sparsified
    - Alpha          : Learnable blend weight — initialised so sigmoid(0)=0.5
    """
    def __init__(self, N, hidden_dim):
        super(DynamicGraphGenerator, self).__init__()
        self.N = N
        self.fc_start = nn.Linear(1, 16)

        # =================================================================
        # FIX #14: Alpha initialised at 0.0 so sigmoid(0.0) = 0.5, giving
        #          a true 50/50 physical/dynamic blend at the start.
        #          Previous value of 0.5 gave sigmoid(0.5) ≈ 0.622.
        # =================================================================
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, A_physical):
        # x: [B, T, N, 1]  A_physical: [N, N]
        B, T, N, _ = x.shape

        state     = x[:, -1, :, :]              # [B, N, 1]
        state_emb = torch.tanh(self.fc_start(state))  # [B, N, 16]

        # Compute raw dynamic scores
        A_dyn = torch.bmm(state_emb, state_emb.transpose(1, 2))  # [B, N, N]
        A_dyn = F.relu(A_dyn)

        # =================================================================
        # FIX #4: Top-K Sparsification done CORRECTLY.
        # Previously, non-top-K entries were set to 0 and then softmax was
        # applied over the whole row — exp(0) gave all zeros a non-zero
        # weight, destroying the sparsification entirely.
        # Fix: fill non-selected positions with -inf so exp(-inf) = 0,
        # making softmax truly zero those connections out.
        # =================================================================
        k = 10
        topk_values, topk_indices = torch.topk(A_dyn, k, dim=2)
        mask = torch.full_like(A_dyn, float('-inf'))
        sparse_A_dyn = mask.scatter_(2, topk_indices, topk_values)
        A_dyn = F.softmax(sparse_A_dyn, dim=2)   # Now properly sparse

        # Expand and row-normalize physical graph so it has consistent scale
        # with the softmax-normalized dynamic graph (both have row sums ≈ 1)
        A_phys = A_physical.unsqueeze(0).expand(B, -1, -1)
        A_phys_batch = A_phys / (A_phys.sum(dim=-1, keepdim=True) + 1e-8)

        # Blend
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
    def __init__(self, K, in_f, out_f):
        super(ChebConvLayer, self).__init__()
        self.K     = K
        self.out_f = out_f
        self.Theta = nn.Parameter(torch.FloatTensor(K, in_f, out_f))
        nn.init.xavier_uniform_(self.Theta)

    def forward(self, x, A):
        B, T, N, Fin = x.shape

        # --- FIX #3: Proper Graph Laplacian ---
        # Degree matrix
        degree = A.sum(dim=-1)                    # [B, N]
        D      = torch.diag_embed(degree)         # [B, N, N]
        L      = D - A                            # [B, N, N]  Laplacian

        I = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)

        # λ_max approximated by largest row-sum (largest degree)
        lambda_max = degree.max(dim=-1)[0].view(B, 1, 1).clamp(min=1.0)
        L_scaled   = 2.0 * L / lambda_max - I    # Scaled Laplacian ∈ [-1, 1]

        outputs    = torch.zeros(B, T, N, self.out_f, device=x.device)
        T_k_prev2  = None
        T_k_prev   = I  # T_0 = I

        for k in range(self.K):
            if   k == 0: T_k = I
            elif k == 1: T_k = L_scaled
            else:        T_k = 2.0 * torch.bmm(L_scaled, T_k_prev) - T_k_prev2

            T_k_prev2, T_k_prev = T_k_prev, T_k

            theta_k = self.Theta[k]
            rhs     = torch.einsum("btnf,fo->btno", x, theta_k)
            outputs += torch.einsum("bnm,btmo->btno", T_k, rhs)

        return F.relu(outputs)


class TemporalConvNet(nn.Module):
    """Gated Temporal Convolution (GLU) for local trend extraction."""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvNet, self).__init__()
        self.pad  = (kernel_size - 1)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (1, kernel_size))

    def forward(self, x):
        x      = x.permute(0, 3, 2, 1)           # [B, F, N, T]
        x      = F.pad(x, (self.pad, 0, 0, 0))
        x_conv = self.conv(x)
        P, Q   = torch.chunk(x_conv, 2, dim=1)
        out    = P * torch.sigmoid(Q)
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
    def __init__(self, nfeat=14, N=207, hidden_dim=128, dropout=0.3, horizon=12):
        super(AMC_DSTGNN, self).__init__()
        self.dropout_p  = dropout
        self.N          = N
        self.hidden_dim = hidden_dim
        self.horizon    = horizon

        # --- ENCODER ---
        self.graph_gen  = DynamicGraphGenerator(N, hidden_dim)
        self.context_fc = nn.Linear(nfeat - 1, 32)
        self.tcn        = TemporalConvNet(1, 32, kernel_size=3)
        self.cheb_conv  = ChebConvLayer(K=3, in_f=32, out_f=32)
        self.attn_fc    = nn.Linear(32 + 32 + 1, 1)
        self.encoder_gru = nn.GRU(32 + 32 + 1, hidden_dim, batch_first=True)

        # --- DECODER ---
        # =================================================================
        # FIX #6: Decoder input is now only current traffic (dim=1).
        # Previously, the static context from the *last observed timestep*
        # was fed at every decoding step — this injected stale past context
        # into all future predictions.  The correct design is to either
        # inject *future* context (known at inference) per step, or to
        # remove the stale context entirely.  Here we remove it cleanly.
        # =================================================================
        self.decoder_cell = nn.GRUCell(1, hidden_dim)   # input = traffic only
        self.fc_out       = nn.Linear(hidden_dim, 1)

    def forward(self, x, A_physical, y=None, teacher_forcing_ratio=0.5):
        """
        x  : [B, T, N, F]
        y  : [B, Horizon, N]  ground-truth (optional, for teacher forcing)
        teacher_forcing_ratio : probability of using true value as next input
        """
        B, T, N, _ = x.shape

        traffic = x[:, :, :, 0:1]   # [B, T, N, 1]
        context = x[:, :, :, 1:]    # [B, T, N, F-1]

        # === ENCODER ===
        A_dynamic = self.graph_gen(traffic, A_physical)
        t_feat    = self.tcn(traffic)
        s_feat    = self.cheb_conv(t_feat, A_dynamic)
        s_feat    = F.dropout(s_feat, p=self.dropout_p, training=self.training)
        c_emb     = torch.relu(self.context_fc(context))

        fusion_input = torch.cat([traffic, s_feat, c_emb], dim=-1)
        attn_scores  = torch.sigmoid(self.attn_fc(fusion_input))
        z = fusion_input * attn_scores

        z_flat    = z.view(B * N, T, -1)
        _, h_n    = self.encoder_gru(z_flat)
        hidden_state = h_n[0]   # [B*N, hidden_dim]

        # === DECODER ===
        outputs      = []
        current_input = traffic[:, -1, :, 0].reshape(B * N, 1)  # last obs traffic

        for step in range(self.horizon):
            # FIX #6: decoder input = traffic only (no stale context)
            hidden_state = self.decoder_cell(current_input, hidden_state)
            prediction   = self.fc_out(hidden_state)   # [B*N, 1]
            outputs.append(prediction)

            # Teacher Forcing
            if self.training and y is not None and random.random() < teacher_forcing_ratio:
                current_input = y[:, step, :].reshape(B * N, 1)
            else:
                current_input = prediction

        outputs = torch.cat(outputs, dim=1)   # [B*N, horizon]
        return outputs.view(B, N, self.horizon).transpose(1, 2)
