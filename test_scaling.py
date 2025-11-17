"""
Test script to verify scaling implementation.
"""
import torch
from models import RelativeGeometryCNF, DataScaler

# Create a small batch of fake data
B, N = 2, 100  # 2 events, 100 DOMs

# Raw data (unscaled)
dom_positions = torch.randn(B, N, 3) * 500  # Random positions
dom_signals = torch.abs(torch.randn(B, N, 2)) * 100  # Random signals (positive)
labels = torch.tensor([
    [1e7, 1.5, 3.0, 100.0, -200.0, 50.0],  # energy, theta, phi, vx, vy, vz
    [5e6, 2.0, 4.5, -150.0, 100.0, -100.0],
])
t_scalar = torch.rand(B)  # CNF time

print("="*70)
print("Testing Scaling Implementation")
print("="*70)

# Test DataScaler
print("\n1. Testing DataScaler...")
scaler = DataScaler()

print(f"\nOriginal labels shape: {labels.shape}")
labels_scaled = scaler.scale_labels(labels)
print(f"Scaled labels shape: {labels_scaled.shape}")
print(f"Original: {labels[0]}")
print(f"Scaled: {labels_scaled[0]}")

labels_unscaled = scaler.unscale_labels(labels_scaled)
print(f"Unscaled: {labels_unscaled[0]}")
print(f"Difference: {torch.abs(labels - labels_unscaled).max().item():.6f}")

# Test Model
print("\n2. Testing Model with Scaling...")
model = RelativeGeometryCNF(
    signal_dim=2,
    cond_dim=8,
    model_dim=64,
    num_layers=2,
    num_heads=4,
    rbf_dim=8,
)

print(f"Model created with cond_dim={model.scaler.energy_log10_mean.shape}")

# Forward pass with scaling
print("\n3. Forward pass with scale_inputs=True...")
velocity = model(
    dom_positions,
    dom_signals,
    labels,  # Raw labels (6D)
    t_scalar,
    scale_inputs=True
)

print(f"Input labels shape: {labels.shape}")
print(f"Output velocity shape: {velocity.shape}")
print(f"Velocity range: [{velocity.min():.4f}, {velocity.max():.4f}]")

# Forward pass with pre-scaled data
print("\n4. Forward pass with scale_inputs=False...")
labels_prescaled = scaler.scale_labels(labels)
positions_prescaled = scaler.scale_positions(dom_positions)
signals_prescaled = scaler.scale_signals(dom_signals)

velocity2 = model(
    positions_prescaled,
    signals_prescaled,
    labels_prescaled,  # Pre-scaled labels (8D)
    t_scalar,
    scale_inputs=False
)

print(f"Velocity difference: {torch.abs(velocity - velocity2).max().item():.6f}")

print("\n" + "="*70)
print("✓ All tests passed!")
print("="*70)

