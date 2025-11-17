"""
SE(3)-Equivariant CNF Model (EGNN-style).

This model uses E(n)-Equivariant Graph Neural Networks to respect
rotational and translational symmetries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DataScaler, MLP


class EGNNLayer(nn.Module):
    """
    EGNN-style SE(3)-equivariant layer with distance-based sparsity.

    Input:
        s: (B, N, D_s)  scalar features per DOM
        x: (B, N, 3)    vector features per DOM (coords or learned)
        mask: (B, N) or None
        max_distance: maximum distance for attention (default 150m)

    Output:
        s_out: (B, N, D_s)
        x_out: (B, N, 3)
    """
    
    def __init__(self, scalar_dim, edge_dim=32, max_distance=150.0):
        super().__init__()
        self.max_distance = max_distance

        # Edge MLP: concat(s_i, s_j, dist_ij^2) -> edge feature e_ij
        self.edge_mlp = MLP(2 * scalar_dim + 1, edge_dim, edge_dim)

        # Convert edge feature to scalar attention logit β_ij
        self.att_mlp = MLP(edge_dim, edge_dim, 1)

        # Scalar update MLP: concat(s_i, aggregated_edge_features) -> Δs_i
        self.scalar_mlp = MLP(scalar_dim + edge_dim, scalar_dim, scalar_dim)

        # Coordinate MLP: edge feature -> scalar weight for vector update
        self.coord_mlp = MLP(edge_dim, edge_dim, 1)

    def forward(self, s, x, mask=None):
        B, N, D_s = s.shape

        # Pairwise geometric quantities
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B, N, N, 3)
        dist2 = torch.sum(diff ** 2, dim=-1, keepdim=True)  # (B, N, N, 1)
        dist = torch.sqrt(dist2.squeeze(-1))  # (B, N, N)
        
        # Distance-based sparsity mask: only attend to DOMs within max_distance
        distance_mask = dist <= self.max_distance  # (B, N, N)
        
        # Expand scalar features to pairwise
        s_i = s.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D_s)
        s_j = s.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D_s)

        # Edge input: concat(s_i, s_j, dist^2)
        edge_input = torch.cat([s_i, s_j, dist2], dim=-1)  # (B, N, N, 2*D_s+1)

        # Edge feature e_ij
        edge_feat = self.edge_mlp(edge_input)  # (B, N, N, edge_dim)

        # Attention weights α_ij from edge features
        att_logits = self.att_mlp(edge_feat).squeeze(-1)  # (B, N, N)
        
        # Apply distance mask
        att_logits = att_logits.masked_fill(~distance_mask, float("-inf"))

        # Apply optional node mask
        if mask is not None:
            # Build pairwise mask
            m_i = mask.unsqueeze(2)
            m_j = mask.unsqueeze(1)
            pair_mask = m_i * m_j
            att_logits = att_logits.masked_fill(pair_mask == 0, float("-inf"))

        # Softmax over j
        att = torch.softmax(att_logits, dim=-1)

        # Scalar update
        m_s = torch.sum(att.unsqueeze(-1) * edge_feat, dim=2)
        s_update = self.scalar_mlp(torch.cat([s, m_s], dim=-1))
        s_out = s + s_update

        # Vector (coordinate) update
        coord_weight = self.coord_mlp(edge_feat).squeeze(-1)
        m_x = torch.sum(att.unsqueeze(-1) * coord_weight.unsqueeze(-1) * diff, dim=2)
        x_out = x + m_x

        return s_out, x_out


class EquivariantCNF(nn.Module):
    """
    SE(3)-Equivariant CNF (EGNN-style).

    Input:
        dom_positions: (B, N, 3)  fixed DOM positions r_i
        dom_signals:   (B, N, 2)  (t_i, q_i) at flow time τ
        cond:          (B, C)     conditioning c
        t_scalar:      (B,)       CNF time τ
        mask:          (B, N) or None
        scale_inputs:  bool       whether to scale inputs

    Output:
        velocity:      (B, N, 2)  d(t_i, q_i)/dτ per DOM
    """
    
    def __init__(
        self,
        signal_dim=2,
        cond_dim=8,
        time_embed_dim=64,
        sig_embed_dim=64,
        cond_embed_dim=64,
        scalar_dim=128,
        num_layers=8,
        max_distance=150.0,  # Maximum attention distance in meters
        scaling_stats=None,
    ):
        super().__init__()
        
        # Initialize data scaler
        if scaling_stats is not None:
            self.scaler = DataScaler(**scaling_stats)
        else:
            self.scaler = DataScaler()

        # Embed DOM signals x_i = (t_i, q_i): R^2 -> R^{sig_embed_dim}
        self.sig_mlp = MLP(signal_dim, sig_embed_dim, sig_embed_dim)

        # Embed conditioning c: R^{cond_dim} -> R^{cond_embed_dim}
        self.cond_mlp = MLP(cond_dim, cond_embed_dim, cond_embed_dim)

        # Embed flow time τ: R -> R^{time_embed_dim}
        self.time_mlp = MLP(1, time_embed_dim, time_embed_dim)

        # Scalar input dimension = sig_emb + cond_emb + time_emb
        in_scalar_dim = sig_embed_dim + cond_embed_dim + time_embed_dim

        # Project to scalar_dim (D_s)
        self.scalar_proj = nn.Linear(in_scalar_dim, scalar_dim)

        # Stack EGNN layers with distance-based sparsity
        self.layers = nn.ModuleList([
            EGNNLayer(scalar_dim=scalar_dim, edge_dim=32, max_distance=max_distance)
            for _ in range(num_layers)
        ])

        # Output head: scalar_dim -> 2 (dt/dτ, dq/dτ)
        self.out_proj = nn.Linear(scalar_dim, signal_dim)

    def forward(self, dom_positions, dom_signals, cond, t_scalar, mask=None, scale_inputs=True):
        """
        Args:
            dom_positions: (B, N, 3) raw DOM positions
            dom_signals: (B, N, 2) raw DOM signals [time, charge]
            cond: (B, 6) raw conditioning [energy, theta, phi, vx, vy, vz] OR
                  (B, 8) scaled conditioning (if scale_inputs=False)
            t_scalar: (B,) CNF time
            mask: (B, N) optional mask
            scale_inputs: whether to apply scaling (True during training/inference)
        """
        B, N, _ = dom_positions.shape

        # Apply scaling if requested
        if scale_inputs:
            dom_positions = self.scaler.scale_positions(dom_positions)
            dom_signals = self.scaler.scale_signals(dom_signals)
            if cond.shape[-1] == 6:  # Raw labels
                cond = self.scaler.scale_labels(cond)

        # Build initial scalar features s_i^(0)
        sig_emb = self.sig_mlp(dom_signals)

        # Conditioning embedding
        cond_emb = self.cond_mlp(cond)
        cond_emb = cond_emb.unsqueeze(1).expand(-1, N, -1)

        # Time embedding
        t_in = t_scalar.unsqueeze(-1)
        t_emb = self.time_mlp(t_in)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)

        # Concatenate [sig_emb, cond_emb, t_emb]
        s0 = torch.cat([sig_emb, cond_emb, t_emb], dim=-1)

        # Project to scalar_dim
        s = self.scalar_proj(s0)

        # Initial vector features v_i^(0) = r_i
        x = dom_positions

        # Apply L equivariant layers
        for layer in self.layers:
            s, x = layer(s, x, mask=mask)

        # Output per-DOM velocity from scalar channel
        vel = self.out_proj(s)
        return vel

