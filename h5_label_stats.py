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


def plot_label_statistics(h5_path, output_dir="./statistics"):
    """
    Plot histograms for each label variable in the HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        output_dir: Directory to save output plots
    """
    # Load data
    with h5py.File(h5_path, 'r') as f:
        labels = f['label'][:]  # (N_events, 6)
    
    n_events = labels.shape[0]
    
    # Label names and their corresponding columns
    label_names = ['Energy', 'Theta', 'Phi', 'Vertex_X', 'Vertex_Y', 'Vertex_Z']
    label_units = ['', 'rad', 'rad', 'm', 'm', 'm']
    
    print("\n" + "="*60)
    print(f"Label Statistics from {Path(h5_path).name}")
    print("="*60)
    print(f"Total number of events: {n_events:,}\n")
    
    # Print statistics for each label
    for i, (name, unit) in enumerate(zip(label_names, label_units)):
        data = labels[:, i]
        unit_str = f" ({unit})" if unit else ""
        
        print(f"{name}{unit_str}:")
        print(f"  Mean:   {np.mean(data):.4e}")
        print(f"  Std:    {np.std(data):.4e}")
        print(f"  Min:    {np.min(data):.4e}")
        print(f"  Max:    {np.max(data):.4e}")
        print(f"  Median: {np.median(data):.4e}")
        print()
    
    print("="*60 + "\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create individual histogram for each label
    for i, (name, unit) in enumerate(zip(label_names, label_units)):
        data = labels[:, i]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        counts, bins, patches = ax.hist(data, bins=100, alpha=0.7, color='steelblue', 
                                       edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for mean and median
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.4e}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                  label=f'Median: {median_val:.4e}')
        
        # Labels and title
        unit_str = f" ({unit})" if unit else ""
        ax.set_xlabel(f'{name}{unit_str}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution of {name}\n{n_events:,} events', 
                    fontsize=14, fontweight='bold')
        
        # Add statistics text box
        stats_text = f'Min: {np.min(data):.4e}\n'
        stats_text += f'Max: {np.max(data):.4e}\n'
        stats_text += f'Std: {np.std(data):.4e}'
        
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
        
        # Save figure
        safe_name = name.lower().replace(' ', '_')
        base_filename = f"label_hist_{safe_name}.png"
        output_path = get_unique_filename(output_dir / base_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")
    
    # Create combined plot with all 6 histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (name, unit) in enumerate(zip(label_names, label_units)):
        data = labels[:, i]
        ax = axes[i]
        
        # Plot histogram
        ax.hist(data, bins=50, alpha=0.7, color='steelblue', 
               edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for mean and median
        mean_val = np.mean(data)
        median_val = np.median(data)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                  label=f'Mean')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, 
                  label=f'Median')
        
        # Labels and title
        unit_str = f" ({unit})" if unit else ""
        ax.set_xlabel(f'{name}{unit_str}', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        
        # Add compact statistics
        stats_text = f'μ={mean_val:.2e}\nσ={np.std(data):.2e}'
        ax.text(0.98, 0.97, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=9,
               family='monospace')
        
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Label Distributions - {n_events:,} events from {Path(h5_path).name}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save combined figure
    base_filename = "label_hist_all.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # Create special plots for angles (with degree conversion)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Theta (polar angle)
    theta_data = labels[:, 1]
    theta_degrees = np.degrees(theta_data)
    
    ax = axes[0]
    ax.hist(theta_degrees, bins=50, alpha=0.7, color='orange', 
           edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(theta_degrees), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(theta_degrees):.2f}°')
    ax.axvline(np.median(theta_degrees), color='green', linestyle='--', 
              linewidth=2, label=f'Median: {np.median(theta_degrees):.2f}°')
    ax.set_xlabel('Theta (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Polar Angle Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Phi (azimuthal angle)
    phi_data = labels[:, 2]
    phi_degrees = np.degrees(phi_data)
    
    ax = axes[1]
    ax.hist(phi_degrees, bins=50, alpha=0.7, color='purple', 
           edgecolor='black', linewidth=0.5)
    ax.axvline(np.mean(phi_degrees), color='red', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(phi_degrees):.2f}°')
    ax.axvline(np.median(phi_degrees), color='green', linestyle='--', 
              linewidth=2, label=f'Median: {np.median(phi_degrees):.2f}°')
    ax.set_xlabel('Phi (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Azimuthal Angle Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Angular Distributions - {n_events:,} events', 
                fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    # Save angle figure
    base_filename = "label_hist_angles_degrees.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # 3D scatter plot of vertex positions
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    vx = labels[:, 3]
    vy = labels[:, 4]
    vz = labels[:, 5]
    energy = labels[:, 0]
    
    # Color by energy (log scale)
    scatter = ax.scatter(vx, vy, vz, c=np.log10(energy), s=1, alpha=0.3, 
                        cmap='viridis', marker='.')
    
    ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Vertex Position Distribution (colored by log10(Energy))\n{n_events:,} events', 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('log10(Energy)', fontsize=11)
    
    # Save vertex plot
    base_filename = "label_vertex_3d.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    print("\n" + "="*60)
    print("All plots saved successfully!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Plot histogram statistics for labels in HDF5 file'
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
    plot_label_statistics(
        h5_path=args.path,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()

