import argparse
import h5py
import numpy as np
import json
from pathlib import Path


def compute_scaling_statistics(h5_path, output_path="scaling_stats.json"):
    """
    Compute scaling statistics for labels, signals, and positions from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        output_path: Path to save JSON file with statistics
    """
    print("\n" + "="*70)
    print("Computing Scaling Statistics")
    print("="*70)
    print(f"Input file: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Load data
        print("\nLoading data...")
        input_data = f['input'][:]  # (N_events, 2, N_DOMs)
        labels = f['label'][:]      # (N_events, 6)
        xpmt = f['xpmt'][:]
        ypmt = f['ypmt'][:]
        zpmt = f['zpmt'][:]
    
    n_events = input_data.shape[0]
    print(f"Loaded {n_events:,} events")
    
    # ========================================
    # 1. Position Statistics (Detector Bounds)
    # ========================================
    print("\n" + "="*70)
    print("1. Position Statistics (Detector Bounds)")
    print("="*70)
    
    x_max = np.abs(xpmt).max()
    y_max = np.abs(ypmt).max()
    z_max = np.abs(zpmt).max()
    
    position_stats = {
        'x_max': float(x_max),
        'y_max': float(y_max),
        'z_max': float(z_max),
        'x_range': [float(xpmt.min()), float(xpmt.max())],
        'y_range': [float(ypmt.min()), float(ypmt.max())],
        'z_range': [float(zpmt.min()), float(zpmt.max())],
    }
    
    print(f"X range: [{xpmt.min():.2f}, {xpmt.max():.2f}] → max_abs: {x_max:.2f}")
    print(f"Y range: [{ypmt.min():.2f}, {ypmt.max():.2f}] → max_abs: {y_max:.2f}")
    print(f"Z range: [{zpmt.min():.2f}, {zpmt.max():.2f}] → max_abs: {z_max:.2f}")
    
    # ========================================
    # 2. Label Statistics
    # ========================================
    print("\n" + "="*70)
    print("2. Label Statistics")
    print("="*70)
    
    energy = labels[:, 0]
    theta = labels[:, 1]
    phi = labels[:, 2]
    vertex_x = labels[:, 3]
    vertex_y = labels[:, 4]
    vertex_z = labels[:, 5]
    
    # Energy: log10 scaling
    energy_log10 = np.log10(energy)
    energy_log10_mean = float(np.mean(energy_log10))
    energy_log10_std = float(np.std(energy_log10))
    
    print("\nEnergy (log10 scale):")
    print(f"  Original range: [{energy.min():.2e}, {energy.max():.2e}]")
    print(f"  Log10 range: [{energy_log10.min():.4f}, {energy_log10.max():.4f}]")
    print(f"  Log10 mean: {energy_log10_mean:.4f}")
    print(f"  Log10 std: {energy_log10_std:.4f}")
    
    # Angles: convert to unit circle (no statistics needed, just formula)
    print("\nAngles (theta, phi):")
    print(f"  Theta range: [{theta.min():.4f}, {theta.max():.4f}] rad")
    print(f"  Phi range: [{phi.min():.4f}, {phi.max():.4f}] rad")
    print("  → Will be converted to sin/cos (unit circle)")
    
    # Vertex: normalize by detector max
    vertex_x_norm = vertex_x / x_max
    vertex_y_norm = vertex_y / y_max
    vertex_z_norm = vertex_z / z_max
    
    print("\nVertex (normalized by detector max):")
    print(f"  X: [{vertex_x.min():.2f}, {vertex_x.max():.2f}] → [{vertex_x_norm.min():.4f}, {vertex_x_norm.max():.4f}]")
    print(f"  Y: [{vertex_y.min():.2f}, {vertex_y.max():.2f}] → [{vertex_y_norm.min():.4f}, {vertex_y_norm.max():.4f}]")
    print(f"  Z: [{vertex_z.min():.2f}, {vertex_z.max():.2f}] → [{vertex_z_norm.min():.4f}, {vertex_z_norm.max():.4f}]")
    
    label_stats = {
        'energy_log10_mean': energy_log10_mean,
        'energy_log10_std': energy_log10_std,
        'energy_range': [float(energy.min()), float(energy.max())],
        'theta_range': [float(theta.min()), float(theta.max())],
        'phi_range': [float(phi.min()), float(phi.max())],
        'vertex_x_max': float(x_max),
        'vertex_y_max': float(y_max),
        'vertex_z_max': float(z_max),
    }
    
    # ========================================
    # 3. Signal Statistics
    # ========================================
    print("\n" + "="*70)
    print("3. Signal Statistics")
    print("="*70)
    
    time_data = input_data[:, 0, :]   # (N_events, N_DOMs)
    charge_data = input_data[:, 1, :]  # (N_events, N_DOMs)
    
    # Flatten
    time_flat = time_data.flatten()
    charge_flat = charge_data.flatten()
    
    # Non-zero masks
    time_nonzero_mask = charge_flat > 0  # Only consider time for hit DOMs
    charge_nonzero_mask = charge_flat > 0
    
    time_nonzero = time_flat[time_nonzero_mask]
    charge_nonzero = charge_flat[charge_nonzero_mask]
    
    # --- TIME ---
    # 99 percentile clipping
    time_99percentile = float(np.percentile(time_nonzero, 99))
    time_clipped = np.clip(time_nonzero, 0, time_99percentile)
    time_mean = float(np.mean(time_clipped))
    time_std = float(np.std(time_clipped))
    
    print("\nTime:")
    print(f"  Original range (hit DOMs): [{time_nonzero.min():.2f}, {time_nonzero.max():.2f}]")
    print(f"  99 percentile: {time_99percentile:.2f}")
    print(f"  After clipping: [{time_clipped.min():.2f}, {time_clipped.max():.2f}]")
    print(f"  Mean (clipped): {time_mean:.4f}")
    print(f"  Std (clipped): {time_std:.4f}")
    
    # --- CHARGE ---
    # ln(1 + x) transformation
    charge_transformed = np.log(1 + charge_nonzero)
    charge_mean = float(np.mean(charge_transformed))
    charge_std = float(np.std(charge_transformed))
    
    print("\nCharge:")
    print(f"  Original range (non-zero): [{charge_nonzero.min():.2f}, {charge_nonzero.max():.2f}]")
    print(f"  After ln(1+x): [{charge_transformed.min():.4f}, {charge_transformed.max():.4f}]")
    print(f"  Mean (ln(1+x)): {charge_mean:.4f}")
    print(f"  Std (ln(1+x)): {charge_std:.4f}")
    
    signal_stats = {
        'time_99percentile': time_99percentile,
        'time_mean': time_mean,
        'time_std': time_std,
        'charge_mean': charge_mean,
        'charge_std': charge_std,
        'hit_rate': float(len(charge_nonzero) / len(charge_flat)),
    }
    
    # ========================================
    # Combine all statistics
    # ========================================
    all_stats = {
        'dataset': str(Path(h5_path).name),
        'n_events': int(n_events),
        'n_doms': int(input_data.shape[2]),
        'position': position_stats,
        'label': label_stats,
        'signal': signal_stats,
    }
    
    # ========================================
    # Save to JSON
    # ========================================
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✓ Scaling statistics saved to: {output_path}")
    print("="*70)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY - Copy these values to model config:")
    print("="*70)
    print("\nPosition scaling:")
    print(f"  x_max = {x_max:.2f}")
    print(f"  y_max = {y_max:.2f}")
    print(f"  z_max = {z_max:.2f}")
    
    print("\nLabel scaling:")
    print(f"  energy_log10_mean = {energy_log10_mean:.4f}")
    print(f"  energy_log10_std = {energy_log10_std:.4f}")
    print(f"  vertex_x_max = {x_max:.2f}")
    print(f"  vertex_y_max = {y_max:.2f}")
    print(f"  vertex_z_max = {z_max:.2f}")
    
    print("\nSignal scaling:")
    print(f"  time_99percentile = {time_99percentile:.2f}")
    print(f"  time_mean = {time_mean:.4f}")
    print(f"  time_std = {time_std:.4f}")
    print(f"  charge_mean = {charge_mean:.4f}")
    print(f"  charge_std = {charge_std:.4f}")
    
    print("\n" + "="*70)
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Compute scaling statistics for model preprocessing'
    )
    parser.add_argument('-p', '--path', type=str, required=True,
                       help='Path to HDF5 file')
    parser.add_argument('-o', '--output', type=str, default='scaling_stats.json',
                       help='Output JSON file (default: scaling_stats.json)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.path).exists():
        print(f"Error: File not found: {args.path}")
        return
    
    # Compute statistics
    compute_scaling_statistics(
        h5_path=args.path,
        output_path=args.output
    )


if __name__ == '__main__':
    main()

