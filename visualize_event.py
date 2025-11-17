import argparse
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def visualize_event(h5_path, event_idx, output_dir="./visualizations", show_stats=True):
    """
    Visualize a specific event from the HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        event_idx: Event index to visualize
        output_dir: Directory to save output images
        show_stats: Whether to print event statistics
    """
    # Load data
    with h5py.File(h5_path, 'r') as f:
        # Check if event index is valid
        n_events = f['input'].shape[0]
        if event_idx < 0 or event_idx >= n_events:
            print(f"Error: Event index {event_idx} out of range [0, {n_events-1}]")
            return
        
        # Load event data
        signals = f['input'][event_idx]  # (2, N_DOMs)
        labels = f['label'][event_idx]   # (6,)
        
        # Load DOM positions
        xpmt = f['xpmt'][:]
        ypmt = f['ypmt'][:]
        zpmt = f['zpmt'][:]
    
    # Transpose signals to (N_DOMs, 2)
    time_signals = signals[0]    # (N_DOMs,)
    charge_signals = signals[1]  # (N_DOMs,)
    
    # Event parameters
    energy, theta, phi, vx, vy, vz = labels
    
    if show_stats:
        print("\n" + "="*60)
        print(f"Event #{event_idx} Statistics")
        print("="*60)
        print(f"Energy: {energy:.2e}")
        print(f"Theta: {theta:.4f} rad ({np.degrees(theta):.2f}°)")
        print(f"Phi: {phi:.4f} rad ({np.degrees(phi):.2f}°)")
        print(f"Vertex: ({vx:.2f}, {vy:.2f}, {vz:.2f})")
        print(f"\nNumber of DOMs: {len(xpmt)}")
        print(f"Time range: [{time_signals.min():.2f}, {time_signals.max():.2f}]")
        print(f"Charge range: [{charge_signals.min():.2f}, {charge_signals.max():.2f}]")
        print(f"Non-zero charges: {(charge_signals > 0).sum()}")
        print("="*60 + "\n")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter DOMs with charge > 0 for better visualization
    mask = charge_signals > 0
    x_active = xpmt[mask]
    y_active = ypmt[mask]
    z_active = zpmt[mask]
    t_active = time_signals[mask]
    q_active = charge_signals[mask]
    
    # Calculate direction vector from theta and phi (spherical coordinates)
    # In IceCube convention: theta is polar angle, phi is azimuthal angle
    dir_x = np.sin(theta) * np.cos(phi)
    dir_y = np.sin(theta) * np.sin(phi)
    dir_z = np.cos(theta)
    
    # Calculate trajectory length limited to detector bounds (both directions)
    # Find intersection with detector boundaries
    x_min, x_max = xpmt.min(), xpmt.max()
    y_min, y_max = ypmt.min(), ypmt.max()
    z_min, z_max = zpmt.min(), zpmt.max()
    
    # Calculate max length in positive direction (forward)
    max_lengths_forward = []
    if dir_x > 0:
        max_lengths_forward.append((x_max - vx) / dir_x)
    elif dir_x < 0:
        max_lengths_forward.append((x_min - vx) / dir_x)
    if dir_y > 0:
        max_lengths_forward.append((y_max - vy) / dir_y)
    elif dir_y < 0:
        max_lengths_forward.append((y_min - vy) / dir_y)
    if dir_z > 0:
        max_lengths_forward.append((z_max - vz) / dir_z)
    elif dir_z < 0:
        max_lengths_forward.append((z_min - vz) / dir_z)
    
    max_lengths_forward = [l for l in max_lengths_forward if l > 0]
    length_forward = min(max_lengths_forward) * 0.9 if max_lengths_forward else 500
    
    # Calculate max length in negative direction (backward)
    max_lengths_backward = []
    if dir_x > 0:
        max_lengths_backward.append((vx - x_min) / dir_x)
    elif dir_x < 0:
        max_lengths_backward.append((vx - x_max) / (-dir_x))
    else:
        max_lengths_backward.append(1e6)
    
    if dir_y > 0:
        max_lengths_backward.append((vy - y_min) / dir_y)
    elif dir_y < 0:
        max_lengths_backward.append((vy - y_max) / (-dir_y))
    else:
        max_lengths_backward.append(1e6)
    
    if dir_z > 0:
        max_lengths_backward.append((vz - z_min) / dir_z)
    elif dir_z < 0:
        max_lengths_backward.append((vz - z_max) / (-dir_z))
    else:
        max_lengths_backward.append(1e6)
    
    max_lengths_backward = [l for l in max_lengths_backward if l > 0]
    length_backward = min(max_lengths_backward) * 0.9 if max_lengths_backward else 500
    
    # Trajectory endpoints (from backward to forward through vertex)
    traj_x = [vx - dir_x * length_backward, vx, vx + dir_x * length_forward]
    traj_y = [vy - dir_y * length_backward, vy, vy + dir_y * length_forward]
    traj_z = [vz - dir_z * length_backward, vz, vz + dir_z * length_forward]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ========== 3D Scatter Plot (Charge) ==========
    ax1 = fig.add_subplot(231, projection='3d')
    scatter1 = ax1.scatter(x_active, y_active, z_active, 
                          c=q_active, s=20, alpha=0.6, 
                          cmap='hot', vmin=0, vmax=np.percentile(q_active, 95))
    ax1.set_xlabel('X (m)', fontsize=10)
    ax1.set_ylabel('Y (m)', fontsize=10)
    ax1.set_zlabel('Z (m)', fontsize=10)
    ax1.set_title(f'Event {event_idx}: Charge Distribution', fontsize=12, fontweight='bold')
    cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.8)
    cbar1.set_label('Charge', fontsize=10)
    
    # Add vertex point
    ax1.scatter([vx], [vy], [vz], c='red', s=200, marker='*', 
               edgecolors='white', linewidths=2, label='Vertex', zorder=100)
    # Add trajectory line
    ax1.plot(traj_x, traj_y, traj_z, 'cyan', linewidth=3, alpha=1.0, 
            linestyle='-', label='Trajectory', zorder=95)
    ax1.legend(fontsize=8)
    
    # ========== 3D Scatter Plot (Time) ==========
    ax2 = fig.add_subplot(232, projection='3d')
    scatter2 = ax2.scatter(x_active, y_active, z_active, 
                          c=t_active, s=20, alpha=0.6, 
                          cmap='viridis', vmin=np.percentile(t_active, 5), 
                          vmax=np.percentile(t_active, 95))
    ax2.set_xlabel('X (m)', fontsize=10)
    ax2.set_ylabel('Y (m)', fontsize=10)
    ax2.set_zlabel('Z (m)', fontsize=10)
    ax2.set_title(f'Event {event_idx}: Time Distribution', fontsize=12, fontweight='bold')
    cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.8)
    cbar2.set_label('Time', fontsize=10)
    
    # Add vertex point
    ax2.scatter([vx], [vy], [vz], c='red', s=200, marker='*', 
               edgecolors='white', linewidths=2, label='Vertex', zorder=100)
    # Add trajectory line
    ax2.plot(traj_x, traj_y, traj_z, 'cyan', linewidth=3, alpha=1.0, 
            linestyle='-', label='Trajectory', zorder=95)
    ax2.legend(fontsize=8)
    
    # ========== XY Projection ==========
    ax3 = fig.add_subplot(233)
    scatter3 = ax3.scatter(x_active, y_active, c=q_active, s=30, 
                          alpha=0.6, cmap='hot', vmin=0, 
                          vmax=np.percentile(q_active, 95))
    ax3.scatter([vx], [vy], c='red', s=300, marker='*', 
               edgecolors='white', linewidths=2, label='Vertex', zorder=100)
    # Add trajectory line (XY projection)
    ax3.plot(traj_x, traj_y, 'cyan', linewidth=3, alpha=1.0, 
            linestyle='-', label='Trajectory', zorder=50)
    ax3.set_xlabel('X (m)', fontsize=10)
    ax3.set_ylabel('Y (m)', fontsize=10)
    ax3.set_title('XY Projection (Charge)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Charge', fontsize=10)
    ax3.set_aspect('equal', adjustable='box')
    
    # ========== XZ Projection ==========
    ax4 = fig.add_subplot(234)
    scatter4 = ax4.scatter(x_active, z_active, c=q_active, s=30, 
                          alpha=0.6, cmap='hot', vmin=0, 
                          vmax=np.percentile(q_active, 95))
    ax4.scatter([vx], [vz], c='red', s=300, marker='*', 
               edgecolors='white', linewidths=2, label='Vertex', zorder=100)
    # Add trajectory line (XZ projection)
    ax4.plot(traj_x, traj_z, 'cyan', linewidth=3, alpha=1.0, 
            linestyle='-', label='Trajectory', zorder=50)
    ax4.set_xlabel('X (m)', fontsize=10)
    ax4.set_ylabel('Z (m)', fontsize=10)
    ax4.set_title('XZ Projection (Charge)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label('Charge', fontsize=10)
    ax4.set_aspect('equal', adjustable='box')
    
    # ========== YZ Projection ==========
    ax5 = fig.add_subplot(235)
    scatter5 = ax5.scatter(y_active, z_active, c=q_active, s=30, 
                          alpha=0.6, cmap='hot', vmin=0, 
                          vmax=np.percentile(q_active, 95))
    ax5.scatter([vy], [vz], c='red', s=300, marker='*', 
               edgecolors='white', linewidths=2, label='Vertex', zorder=100)
    # Add trajectory line (YZ projection)
    ax5.plot(traj_y, traj_z, 'cyan', linewidth=3, alpha=1.0, 
            linestyle='-', label='Trajectory', zorder=50)
    ax5.set_xlabel('Y (m)', fontsize=10)
    ax5.set_ylabel('Z (m)', fontsize=10)
    ax5.set_title('YZ Projection (Charge)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)
    cbar5 = plt.colorbar(scatter5, ax=ax5)
    cbar5.set_label('Charge', fontsize=10)
    ax5.set_aspect('equal', adjustable='box')
    
    # ========== Charge vs Time Scatter ==========
    ax6 = fig.add_subplot(236)
    ax6.scatter(t_active, q_active, alpha=0.5, s=20)
    ax6.set_xlabel('Time', fontsize=10)
    ax6.set_ylabel('Charge', fontsize=10)
    ax6.set_title('Charge vs Time', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    # Add overall title with event info
    fig.suptitle(
        f'Event {event_idx} | E={energy:.2e} | θ={np.degrees(theta):.1f}° | φ={np.degrees(phi):.1f}° | Active DOMs: {mask.sum()}/{len(mask)}',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure with unique filename
    base_filename = f"event_{event_idx}.png"
    output_path = get_unique_filename(output_dir / base_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {output_path}")
    
    # Also create a simple 3D-only view
    fig2 = plt.figure(figsize=(12, 10))
    ax = fig2.add_subplot(111, projection='3d')
    
    # Plot all DOMs (inactive in gray)
    ax.scatter(xpmt, ypmt, zpmt, c='lightgray', s=5, alpha=0.2, label='Inactive DOMs')
    
    # Plot active DOMs colored by charge
    scatter = ax.scatter(x_active, y_active, z_active, 
                        c=q_active, s=50, alpha=0.8, 
                        cmap='hot', vmin=0, vmax=np.percentile(q_active, 95),
                        edgecolors='black', linewidths=0.5)
    
    # Plot vertex
    ax.scatter([vx], [vy], [vz], c='red', s=500, marker='*', 
              edgecolors='white', linewidths=3, label='Vertex', zorder=100)
    
    # Plot trajectory line
    ax.plot(traj_x, traj_y, traj_z, 'cyan', linewidth=4, alpha=1.0, 
           linestyle='-', label='Trajectory', zorder=90)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'Event {event_idx}: 3D Detector View\nE={energy:.2e}, Active DOMs: {mask.sum()}', 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Charge', fontsize=12)
    ax.legend(fontsize=10)
    
    # Set axis limits to actual detector bounds (no extra space)
    margin = 50  # Small margin in meters
    ax.set_xlim(xpmt.min() - margin, xpmt.max() + margin)
    ax.set_ylim(ypmt.min() - margin, ypmt.max() + margin)
    ax.set_zlim(zpmt.min() - margin, zpmt.max() + margin)
    
    # Save 3D view
    base_filename_3d = f"event_{event_idx}_3d.png"
    output_path_3d = get_unique_filename(output_dir / base_filename_3d)
    plt.savefig(output_path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 3D view saved: {output_path_3d}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize IceCube neutrino events from HDF5 file'
    )
    parser.add_argument('-p', '--path', type=str, required=True,
                       help='Path to HDF5 file')
    parser.add_argument('-e', '--event', type=int, required=True,
                       help='Event index to visualize')
    parser.add_argument('-o', '--output', type=str, default='./visualizations',
                       help='Output directory for visualizations (default: ./visualizations)')
    parser.add_argument('--no-stats', action='store_true',
                       help='Do not print event statistics')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.path).exists():
        print(f"Error: File not found: {args.path}")
        return
    
    # Visualize event
    visualize_event(
        h5_path=args.path,
        event_idx=args.event,
        output_dir=args.output,
        show_stats=not args.no_stats
    )


if __name__ == '__main__':
    main()

