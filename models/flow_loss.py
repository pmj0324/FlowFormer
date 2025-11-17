"""
Flow-matching loss for training CNF models.
"""

import torch
import torch.nn.functional as F


def flow_matching_loss(model, dom_positions, dom_signals_data, cond, mask=None):
    """
    Compute one minibatch of flow-matching loss.

    Inputs:
        model:           RelativeGeometryCNF or EquivariantCNF
        dom_positions:   (B, N, 3)   fixed DOM positions
        dom_signals_data:(B, N, 2)   ground-truth event x_1 (normalized)
        cond:            (B, C)      conditioning c
        mask:            (B, N) or None

    Steps:
        1) Sample base noise z ~ N(0, I)
        2) Sample τ ~ Uniform(0,1)
        3) x_τ = (1-τ) z + τ x_1
        4) v_target = x_1 - z
        5) v_pred = model(dom_positions, x_τ, cond, τ)
        6) Loss = MSE(v_pred, v_target)
    """
    B, N, _ = dom_signals_data.shape

    # Base noise z: (B,N,2)
    z = torch.randn_like(dom_signals_data)

    # CNF time τ: (B,)
    tau = torch.rand(B, device=dom_signals_data.device)

    # x_τ = (1-τ) z + τ x_1, broadcast τ over N,2
    tau_ = tau.view(B, 1, 1)
    x_tau = (1.0 - tau_) * z + tau_ * dom_signals_data

    # Target velocity: v* = x_1 - z
    v_target = dom_signals_data - z

    # Model prediction
    v_pred = model(dom_positions, x_tau, cond, tau, mask)

    # Mean squared error over batch, DOMs, and feature dims
    loss = F.mse_loss(v_pred, v_target)
    return loss

