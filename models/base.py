"""
Base utilities and components for FlowFormer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataScaler(nn.Module):
    """
    Data scaling utilities for positions, signals, and labels.
    All scaling parameters are registered as buffers for easy device transfer.
    """
    
    def __init__(
        self,
        # Position scaling (detector bounds)
        pos_x_max=576.37,
        pos_y_max=521.08,
        pos_z_max=524.56,
        # Signal scaling (time)
        time_99percentile=117.00,
        time_mean=14.3358,
        time_std=24.6273,
        # Signal scaling (charge)
        charge_mean=7.5155,
        charge_std=0.8190,
        # Label scaling (energy)
        energy_log10_mean=7.0168,
        energy_log10_std=0.5761,
        # Label scaling (vertex) - same as position
        vertex_x_max=576.37,
        vertex_y_max=521.08,
        vertex_z_max=524.56,
    ):
        super().__init__()
        
        # Register all as buffers (will move with model to GPU/CPU)
        # Position
        self.register_buffer('pos_x_max', torch.tensor(pos_x_max))
        self.register_buffer('pos_y_max', torch.tensor(pos_y_max))
        self.register_buffer('pos_z_max', torch.tensor(pos_z_max))
        
        # Signal - Time
        self.register_buffer('time_99percentile', torch.tensor(time_99percentile))
        self.register_buffer('time_mean', torch.tensor(time_mean))
        self.register_buffer('time_std', torch.tensor(time_std))
        
        # Signal - Charge
        self.register_buffer('charge_mean', torch.tensor(charge_mean))
        self.register_buffer('charge_std', torch.tensor(charge_std))
        
        # Label - Energy
        self.register_buffer('energy_log10_mean', torch.tensor(energy_log10_mean))
        self.register_buffer('energy_log10_std', torch.tensor(energy_log10_std))
        
        # Label - Vertex
        self.register_buffer('vertex_x_max', torch.tensor(vertex_x_max))
        self.register_buffer('vertex_y_max', torch.tensor(vertex_y_max))
        self.register_buffer('vertex_z_max', torch.tensor(vertex_z_max))
    
    def scale_positions(self, positions):
        """
        Scale DOM positions: (B, N, 3) -> (B, N, 3)
        Normalize by detector max dimensions.
        """
        x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
        x_scaled = x / self.pos_x_max
        y_scaled = y / self.pos_y_max
        z_scaled = z / self.pos_z_max
        return torch.stack([x_scaled, y_scaled, z_scaled], dim=-1)
    
    def scale_signals(self, signals):
        """
        Scale DOM signals: (B, N, 2) -> (B, N, 2)
        signals[..., 0] = time, signals[..., 1] = charge
        
        Time: clip at 99 percentile, then standardize (only for non-zero)
        Charge: ln(1+x), then standardize (only for non-zero)
        """
        time = signals[..., 0]
        charge = signals[..., 1]
        
        # Time: clip and standardize
        time_clipped = torch.clamp(time, 0, self.time_99percentile)
        # Only standardize non-zero values
        mask = charge > 0
        time_scaled = time_clipped.clone()
        time_scaled[mask] = (time_clipped[mask] - self.time_mean) / self.time_std
        
        # Charge: ln(1+x) and standardize
        charge_transformed = torch.log(1 + charge)
        charge_scaled = charge_transformed.clone()
        charge_scaled[mask] = (charge_transformed[mask] - self.charge_mean) / self.charge_std
        
        return torch.stack([time_scaled, charge_scaled], dim=-1)
    
    def scale_labels(self, labels):
        """
        Scale labels: (B, 6) -> (B, 8)
        Input: [energy, theta, phi, vx, vy, vz]
        Output: [energy_scaled, sin(theta), cos(theta), sin(phi), cos(phi), vx_scaled, vy_scaled, vz_scaled]
        
        Energy: log10 and standardize
        Angles: convert to unit circle (sin/cos)
        Vertex: normalize by detector max
        """
        energy = labels[..., 0]
        theta = labels[..., 1]
        phi = labels[..., 2]
        vx = labels[..., 3]
        vy = labels[..., 4]
        vz = labels[..., 5]
        
        # Energy: log10 and standardize
        energy_log10 = torch.log10(energy)
        energy_scaled = (energy_log10 - self.energy_log10_mean) / self.energy_log10_std
        
        # Angles: to unit circle
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        
        # Vertex: normalize
        vx_scaled = vx / self.vertex_x_max
        vy_scaled = vy / self.vertex_y_max
        vz_scaled = vz / self.vertex_z_max
        
        return torch.stack([
            energy_scaled, sin_theta, cos_theta, sin_phi, cos_phi,
            vx_scaled, vy_scaled, vz_scaled
        ], dim=-1)
    
    def unscale_labels(self, labels_scaled):
        """
        Unscale labels: (B, 8) -> (B, 6)
        Input: [energy_scaled, sin(theta), cos(theta), sin(phi), cos(phi), vx_scaled, vy_scaled, vz_scaled]
        Output: [energy, theta, phi, vx, vy, vz]
        """
        energy_scaled = labels_scaled[..., 0]
        sin_theta = labels_scaled[..., 1]
        cos_theta = labels_scaled[..., 2]
        sin_phi = labels_scaled[..., 3]
        cos_phi = labels_scaled[..., 4]
        vx_scaled = labels_scaled[..., 5]
        vy_scaled = labels_scaled[..., 6]
        vz_scaled = labels_scaled[..., 7]
        
        # Energy: unscale and un-log10
        energy_log10 = energy_scaled * self.energy_log10_std + self.energy_log10_mean
        energy = 10 ** energy_log10
        
        # Angles: atan2
        theta = torch.atan2(sin_theta, cos_theta)
        # Make sure theta is in [0, pi]
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)
        
        phi = torch.atan2(sin_phi, cos_phi)
        # Make sure phi is in [0, 2pi]
        phi = torch.where(phi < 0, phi + 2 * torch.pi, phi)
        
        # Vertex: unscale
        vx = vx_scaled * self.vertex_x_max
        vy = vy_scaled * self.vertex_y_max
        vz = vz_scaled * self.vertex_z_max
        
        return torch.stack([energy, theta, phi, vx, vy, vz], dim=-1)
    
    def unscale_signals(self, signals_scaled, original_charge_mask=None):
        """
        Unscale signals: (B, N, 2) -> (B, N, 2)
        
        Time: unstandardize and clip back
        Charge: unstandardize and inverse ln(1+x)
        
        Args:
            signals_scaled: scaled signals
            original_charge_mask: optional mask of originally non-zero charges
        """
        time_scaled = signals_scaled[..., 0]
        charge_scaled = signals_scaled[..., 1]
        
        # Time: unstandardize
        time = time_scaled * self.time_std + self.time_mean
        time = torch.clamp(time, 0, None)  # Ensure non-negative
        
        # Charge: unstandardize and inverse ln(1+x)
        charge_transformed = charge_scaled * self.charge_std + self.charge_mean
        charge = torch.exp(charge_transformed) - 1
        charge = torch.clamp(charge, 0, None)  # Ensure non-negative
        
        # If mask provided, zero out originally zero charges
        if original_charge_mask is not None:
            time = time * original_charge_mask
            charge = charge * original_charge_mask
        
        return torch.stack([time, charge], dim=-1)


class MLP(nn.Module):
    """
    Generic MLP: R^in_dim -> R^out_dim
    Used for embeddings and small sub-networks.
    """
    
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, activation=nn.SiLU):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (..., in_dim)
        return self.net(x)  # (..., out_dim)


class RBFEmbed(nn.Module):
    """
    Radial Basis Function (RBF) embedding for scalar distances.

    Input:
        dists: (...,)  scalar distances
    Output:
        rbf:   (..., num_centers)

    Each channel is:
        rbf_k(d) = exp( -gamma * (d - center_k)^2 )
    """
    
    def __init__(self, num_centers=16, r_min=0.0, r_max=300.0):
        super().__init__()
        centers = torch.linspace(r_min, r_max, num_centers)
        self.register_buffer("centers", centers)   # shape: (num_centers,)
        self.gamma = nn.Parameter(torch.tensor(10.0))

    def forward(self, dists):
        # dists: (...,)
        x = dists.unsqueeze(-1)  # (..., 1)
        # centers: (num_centers,)
        return torch.exp(-self.gamma * (x - self.centers) ** 2)  # (..., num_centers)

