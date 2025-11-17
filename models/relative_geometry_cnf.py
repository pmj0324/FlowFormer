"""
Relative-Geometry Transformer CNF Model.

This model uses geometry-aware multi-head attention to process DOM signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DataScaler, MLP, RBFEmbed


class GeometryMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with geometry-aware bias.

    - Inputs:
        h:         (B, N, D)   per-DOM hidden features
        positions: (B, N, 3)   fixed DOM positions r_i
        mask:      (B, N) or None, 1=valid, 0=padding

    - Output:
        out:       (B, N, D)   updated hidden features
    """
    
    def __init__(self, dim, num_heads, rbf_dim):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Linear maps for Q, K, V, and output
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # RBF distance embedding -> per-head scalar bias
        self.rbf_embed = RBFEmbed(num_centers=rbf_dim)
        self.rbf_to_bias = nn.Linear(rbf_dim, num_heads)

    def forward(self, h, positions, mask=None):
        B, N, D = h.shape

        # Compute Q, K, V projections
        Q = self.q_proj(h)
        K = self.k_proj(h)
        V = self.v_proj(h)

        # Reshape for multi-head: (B, H, N, d_h)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Geometry-based bias: distance between DOMs
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dists = torch.norm(diff, dim=-1)

        # RBF embedding of distances
        rbf = self.rbf_embed(dists)
        bias_per_head = self.rbf_to_bias(rbf)
        bias_per_head = bias_per_head.permute(0, 3, 1, 2)

        # Attention logits with geometric bias
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_logits = attn_logits + bias_per_head

        # Optional mask
        if mask is not None:
            m = mask[:, None, None, :].to(dtype=attn_logits.dtype)
            attn_logits = attn_logits.masked_fill(m == 0, float("-inf"))

        # Normalized attention weights
        attn = torch.softmax(attn_logits, dim=-1)

        # Aggregate values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return out


class GeometryTransformerBlock(nn.Module):
    """
    One Transformer block with:
    - LayerNorm
    - Geometry-aware multihead self-attention
    - Residual connection
    - Feed-forward MLP
    """
    
    def __init__(self, dim, num_heads, rbf_dim, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GeometryMultiHeadAttention(dim, num_heads, rbf_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.SiLU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, h, positions, mask=None):
        # Self-attention + residual
        h_norm = self.norm1(h)
        attn_out = self.attn(h_norm, positions, mask)
        h = h + self.dropout1(attn_out)

        # Feed-forward + residual
        h_norm = self.norm2(h)
        ff_out = self.ff(h_norm)
        h = h + self.dropout2(ff_out)

        return h


class RelativeGeometryCNF(nn.Module):
    """
    Relative-Geometry Transformer CNF.

    Input:
        dom_positions: (B, N, 3)  fixed DOM positions r_i
        dom_signals:   (B, N, 2)  (t_i, q_i) at flow time τ
        cond:          (B, C)     conditioning c (E, θ, φ, x_v, y_v, z_v)
        t_scalar:      (B,)       CNF time τ ∈ [0,1]
        mask:          (B, N) or None
        scale_inputs:  bool       whether to scale inputs (default True)

    Output:
        velocity:      (B, N, 2)  d(t_i, q_i)/dτ per DOM
    """
    
    def __init__(
        self,
        signal_dim=2,
        cond_dim=8,
        time_embed_dim=64,
        pos_embed_dim=32,
        sig_embed_dim=32,
        cond_embed_dim=64,
        model_dim=256,
        num_layers=8,
        num_heads=8,
        rbf_dim=16,
        ff_mult=4,
        dropout=0.0,
        scaling_stats=None,
    ):
        super().__init__()
        
        # Initialize data scaler
        if scaling_stats is not None:
            self.scaler = DataScaler(**scaling_stats)
        else:
            self.scaler = DataScaler()

        # DOM position embedding φ_r : R^3 -> R^{pos_embed_dim}
        self.pos_mlp = MLP(3, pos_embed_dim, pos_embed_dim)

        # DOM signal embedding φ_x : R^2 -> R^{sig_embed_dim}
        self.sig_mlp = MLP(signal_dim, sig_embed_dim, sig_embed_dim)

        # Conditioning embedding φ_c : R^{cond_dim} -> R^{cond_embed_dim}
        self.cond_mlp = MLP(cond_dim, cond_embed_dim, cond_embed_dim)

        # Time embedding φ_τ : R -> R^{time_embed_dim}
        self.time_mlp = MLP(1, time_embed_dim, time_embed_dim)

        # Total concatenated dim
        in_dim = pos_embed_dim + sig_embed_dim + cond_embed_dim + time_embed_dim

        # Linear projection to Transformer hidden dim
        self.in_proj = nn.Linear(in_dim, model_dim)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            GeometryTransformerBlock(
                dim=model_dim,
                num_heads=num_heads,
                rbf_dim=rbf_dim,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output head: model_dim -> 2 (dt/dτ, dq/dτ)
        self.out_proj = nn.Linear(model_dim, signal_dim)

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

        # Per-DOM embeddings
        pos_emb = self.pos_mlp(dom_positions)
        sig_emb = self.sig_mlp(dom_signals)

        # Conditioning embedding
        cond_emb = self.cond_mlp(cond)
        cond_emb = cond_emb.unsqueeze(1).expand(-1, N, -1)

        # Time embedding
        t_in = t_scalar.unsqueeze(-1)
        t_emb = self.time_mlp(t_in)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)

        # Concatenate all embeddings per DOM
        h0 = torch.cat([pos_emb, sig_emb, cond_emb, t_emb], dim=-1)

        # Project to Transformer hidden space
        h = self.in_proj(h0)

        # Transformer layers with geometry-aware attention
        for blk in self.blocks:
            h = blk(h, dom_positions, mask)

        # Output head: per-DOM velocity
        vel = self.out_proj(h)
        return vel

