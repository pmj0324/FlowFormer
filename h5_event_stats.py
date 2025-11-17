import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def get_unique_filename(base_path):
    """
    Generate unique filename by appending numbers if file exists.
    
    Args:
        base_path: Path object or string
    
    Returns:
        Path object with unique filename
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        return base_path
    
    # Split name and extension
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    
    # Find unique number
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = parent / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def plot_event_statistics(h5_path, output_dir="./statistics"):
    """
    Plot histograms for event input data (time and charge) from HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        output_dir: Directory to save output plots
    """
    # Load data
    print("\n" + "="*60)
    print(f"Loading data from {Path(h5_path).name}...")
    print("="*60)
    
    with h5py.File(h5_path, 'r') as f:
        input_data = f['input'][:]  # (N_events, 2, N_DOMs)
    
    n_events, n_channels, n_doms = input_data.shape
    
    # Extract time and charge
    time_data = input_data[:, 0, :]  # (N_events, N_DOMs)
    charge_data = input_data[:, 1, :]  # (N_events, N_DOMs)
    
    # Flatten to get all values across all events and DOMs
    time_flat = time_data.flatten()
    charge_flat = charge_data.flatten()
    
    # Filter non-zero charges (for hit DOMs)
    charge_nonzero = charge_flat[charge_flat > 0]
    time_nonzero = time_flat[charge_flat > 0]  # Time values for hit DOMs
    
    print(f"\nData loaded:")
    print(f"  Total events: {n_events:,}")
    print(f"  DOMs per event: {n_doms:,}")
    print(f"  Total measurements: {n_events * n_doms:,}")
    
    print("\n" + "="*60)
    print("Time Statistics (all DOMs)")
    print("="*60)
    print(f"  Mean:   {np.mean(time_flat):.4f}")
    print(f"  Std:    {np.std(time_flat):.4f}")
    print(f"  Min:    {np.min(time_flat):.4f}")
    print(f"  Max:    {np.max(time_flat):.4f}")
    print(f"  Median: {np.median(time_flat):.4f}")
    
    print("\n" + "="*60)
    print("Charge Statistics (all DOMs)")
    print("="*60)
    print(f"  Mean:   {np.mean(charge_flat):.4f}")
    print(f"  Std:    {np.std(charge_flat):.4f}")
    print(f"  Min:    {np.min(charge_flat):.4f}")
    print(f"  Max:    {np.max(charge_flat):.4f}")
    print(f"  Median: {np.median(charge_flat):.4f}")
    print(f"  Non-zero: {len(charge_nonzero):,} ({100*len(charge_nonzero)/len(charge_flat):.2f}%)")
    
    print("\n" + "="*60)
    print("Charge Statistics (non-zero only)")
    print("="*60)
    print(f"  Mean:   {np.mean(charge_nonzero):.4f}")
    print(f"  Std:    {np.std(charge_nonzero):.4f}")
    print(f"  Min:    {np.min(charge_nonzero):.4f}")
    print(f"  Max:    {np.max(charge_nonzero):.4f}")
    print(f"  Median: {np.median(charge_nonzero):.4f}")
    
    print("\n" + "="*60)
    print("Time Statistics (hit DOMs only)")
    print("="*60)
    print(f"  Mean:   {np.mean(time_nonzero):.4f}")
    print(f"  Std:    {np.std(time_nonzero):.4f}")
    print(f"  Min:    {np.min(time_nonzero):.4f}")
    print(f"  Max:    {np.max(time_nonzero):.4f}")
    print(f"  Median: {np.median(time_nonzero):.4f}")
    
    print("\n" + "="*60)
    print(f"Generating plots...")
    print("="*60 + "\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== Time histogram (all DOMs) ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts, bins, patches = ax.hist(time_flat, bins=200, alpha=0.7, color='steelblue', 
                                    edgecolor='black', linewidth=0.5)
    
    mean_val = np.mean(time_flat)
    median_val = np.median(time_flat)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
              label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Time Distribution (All DOMs)\n{len(time_flat):,} measurements from {n_events:,} events', 
                fontsize=14, fontweight='bold')
    
    stats_text = f'Min: {np.min(time_flat):.2f}\n'
    stats_text += f'Max: {np.max(time_flat):.2f}\n'
    stats_text += f'Std: {np.std(time_flat):.2f}'
    
    ax.text(0.98, 0.97, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10,
           family='monospace')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    base_filename = "event_hist_time_all.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # ========== Charge histogram (all DOMs) ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts, bins, patches = ax.hist(charge_flat, bins=200, alpha=0.7, color='coral', 
                                    edgecolor='black', linewidth=0.5)
    
    mean_val = np.mean(charge_flat)
    median_val = np.median(charge_flat)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
              label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel('Charge', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Charge Distribution (All DOMs)\n{len(charge_flat):,} measurements from {n_events:,} events', 
                fontsize=14, fontweight='bold')
    
    stats_text = f'Min: {np.min(charge_flat):.2f}\n'
    stats_text += f'Max: {np.max(charge_flat):.2f}\n'
    stats_text += f'Non-zero: {len(charge_nonzero):,}\n'
    stats_text += f'({100*len(charge_nonzero)/len(charge_flat):.2f}%)'
    
    ax.text(0.98, 0.97, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10,
           family='monospace')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    base_filename = "event_hist_charge_all.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # ========== Charge histogram (non-zero only) ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts, bins, patches = ax.hist(charge_nonzero, bins=200, alpha=0.7, color='orange', 
                                    edgecolor='black', linewidth=0.5)
    
    mean_val = np.mean(charge_nonzero)
    median_val = np.median(charge_nonzero)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
              label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel('Charge', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Charge Distribution (Hit DOMs Only)\n{len(charge_nonzero):,} hits from {n_events:,} events', 
                fontsize=14, fontweight='bold')
    
    stats_text = f'Min: {np.min(charge_nonzero):.2f}\n'
    stats_text += f'Max: {np.max(charge_nonzero):.2f}\n'
    stats_text += f'Std: {np.std(charge_nonzero):.2f}'
    
    ax.text(0.98, 0.97, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10,
           family='monospace')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    base_filename = "event_hist_charge_nonzero.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # ========== Time histogram (hit DOMs only) ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts, bins, patches = ax.hist(time_nonzero, bins=200, alpha=0.7, color='teal', 
                                    edgecolor='black', linewidth=0.5)
    
    mean_val = np.mean(time_nonzero)
    median_val = np.median(time_nonzero)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
              label=f'Median: {median_val:.2f}')
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Time Distribution (Hit DOMs Only)\n{len(time_nonzero):,} hits from {n_events:,} events', 
                fontsize=14, fontweight='bold')
    
    stats_text = f'Min: {np.min(time_nonzero):.2f}\n'
    stats_text += f'Max: {np.max(time_nonzero):.2f}\n'
    stats_text += f'Std: {np.std(time_nonzero):.2f}'
    
    ax.text(0.98, 0.97, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10,
           family='monospace')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    base_filename = "event_hist_time_hit.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # ========== Combined plot (2x2) ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Time - All
    ax = axes[0, 0]
    ax.hist(time_flat, bins=150, alpha=0.7, color='steelblue', 
           edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(time_flat), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(time_flat), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Time Distribution (All DOMs)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Time - Hit only
    ax = axes[0, 1]
    ax.hist(time_nonzero, bins=150, alpha=0.7, color='teal', 
           edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(time_nonzero), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(time_nonzero), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Time Distribution (Hit DOMs Only)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Charge - All
    ax = axes[1, 0]
    ax.hist(charge_flat, bins=150, alpha=0.7, color='coral', 
           edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(charge_flat), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(charge_flat), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('Charge', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Charge Distribution (All DOMs)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Charge - Non-zero
    ax = axes[1, 1]
    ax.hist(charge_nonzero, bins=150, alpha=0.7, color='orange', 
           edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(charge_nonzero), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(charge_nonzero), color='green', linestyle='--', linewidth=2, label='Median')
    ax.set_xlabel('Charge', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Charge Distribution (Hit DOMs Only)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    fig.suptitle(f'Event Input Distributions - {n_events:,} events from {Path(h5_path).name}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    base_filename = "event_hist_all.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # ========== 2D histogram: Time vs Charge (hit DOMs only) ==========
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sample data if too many points
    max_points = 100000
    if len(time_nonzero) > max_points:
        indices = np.random.choice(len(time_nonzero), max_points, replace=False)
        time_sample = time_nonzero[indices]
        charge_sample = charge_nonzero[indices]
        title_suffix = f" (sampled {max_points:,} from {len(time_nonzero):,})"
    else:
        time_sample = time_nonzero
        charge_sample = charge_nonzero
        title_suffix = f" ({len(time_nonzero):,} hits)"
    
    h = ax.hist2d(time_sample, charge_sample, bins=100, cmap='viridis', 
                 norm=matplotlib.colors.LogNorm())
    
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Charge', fontsize=12, fontweight='bold')
    ax.set_title(f'Time vs Charge (Hit DOMs){title_suffix}', 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Count (log scale)', fontsize=11)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    base_filename = "event_hist_time_vs_charge.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # ========== Hits per event histogram ==========
    hits_per_event = np.sum(charge_data > 0, axis=1)  # Count non-zero charges per event
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts, bins, patches = ax.hist(hits_per_event, bins=100, alpha=0.7, color='purple', 
                                    edgecolor='black', linewidth=0.5)
    
    mean_val = np.mean(hits_per_event)
    median_val = np.median(hits_per_event)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
              label=f'Median: {median_val:.1f}')
    
    ax.set_xlabel('Number of Hit DOMs per Event', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Hit DOMs per Event Distribution\n{n_events:,} events', 
                fontsize=14, fontweight='bold')
    
    stats_text = f'Min: {np.min(hits_per_event):.0f}\n'
    stats_text += f'Max: {np.max(hits_per_event):.0f}\n'
    stats_text += f'Std: {np.std(hits_per_event):.2f}\n'
    stats_text += f'Total DOMs: {n_doms:,}'
    
    ax.text(0.98, 0.97, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10,
           family='monospace')
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    base_filename = "event_hist_hits_per_event.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    print("\n" + "="*60)
    print("All plots saved successfully!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Plot histogram statistics for event input data (time/charge) in HDF5 file'
    )
    parser.add_argument('-p', '--path', type=str, required=True,
                       help='Path to HDF5 file')
    parser.add_argument('-o', '--output', type=str, default='./statistics',
                       help='Output directory for plots (default: ./statistics)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.path).exists():
        print(f"Error: File not found: {args.path}")
        return
    
    # Plot statistics
    plot_event_statistics(
        h5_path=args.path,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()

