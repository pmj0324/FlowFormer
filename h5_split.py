"""
Split HDF5 file into train/val/test sets or create a smaller subset.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def split_h5_file(
    input_path,
    output_dir="./",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    subset_size=None,
    seed=42
):
    """
    Split HDF5 file into train/val/test sets.
    
    Args:
        input_path: Path to input HDF5 file
        output_dir: Directory to save output files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        subset_size: If specified, only use first N events (for testing)
        seed: Random seed
    """
    np.random.seed(seed)
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("HDF5 File Splitting")
    print("="*70)
    print(f"Input file: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Load original file
    print("\nLoading original HDF5 file...")
    with h5py.File(input_path, 'r') as f:
        n_events_total = f['input'].shape[0]
        
        print(f"Total events in file: {n_events_total:,}")
        
        # Determine how many events to use
        if subset_size is not None and subset_size < n_events_total:
            n_events = subset_size
            print(f"Using subset: {n_events:,} events")
        else:
            n_events = n_events_total
            print(f"Using all events: {n_events:,}")
        
        # Random permutation
        indices = np.random.permutation(n_events)
        
        # Calculate split sizes
        n_train = int(n_events * train_ratio)
        n_val = int(n_events * val_ratio)
        n_test = n_events - n_train - n_val
        
        print(f"\nSplit sizes:")
        print(f"  Train: {n_train:,} ({train_ratio*100:.1f}%)")
        print(f"  Val:   {n_val:,} ({val_ratio*100:.1f}%)")
        print(f"  Test:  {n_test:,} ({test_ratio*100:.1f}%)")
        
        # Split indices
        train_indices = sorted(indices[:n_train])
        val_indices = sorted(indices[n_train:n_train+n_val])
        test_indices = sorted(indices[n_train+n_val:])
        
        # Load data
        print("\nLoading data into memory...")
        input_data = f['input'][:]
        label_data = f['label'][:]
        xpmt = f['xpmt'][:]
        ypmt = f['ypmt'][:]
        zpmt = f['zpmt'][:]
        
        # Optional: load info if exists
        info_data = None
        if 'info' in f:
            info_data = f['info'][:]
    
    # Save splits
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }
    
    for split_name, split_indices in splits.items():
        if len(split_indices) == 0:
            continue
        
        output_path = output_dir / f"{input_path.stem}_{split_name}.h5"
        print(f"\nSaving {split_name} set to {output_path}...")
        
        with h5py.File(output_path, 'w') as f_out:
            # Save input and label data
            f_out.create_dataset('input', data=input_data[split_indices])
            f_out.create_dataset('label', data=label_data[split_indices])
            
            # Save DOM positions (same for all events)
            f_out.create_dataset('xpmt', data=xpmt)
            f_out.create_dataset('ypmt', data=ypmt)
            f_out.create_dataset('zpmt', data=zpmt)
            
            # Save info if exists
            if info_data is not None:
                f_out.create_dataset('info', data=info_data[split_indices])
        
        print(f"  ✓ Saved {len(split_indices):,} events")
        
        # Print statistics
        split_input = input_data[split_indices]
        split_label = label_data[split_indices]
        
        print(f"  Shape: input={split_input.shape}, label={split_label.shape}")
        print(f"  Size: {output_path.stat().st_size / 1024**2:.1f} MB")
    
    print("\n" + "="*70)
    print("✓ Splitting completed successfully!")
    print("="*70)


def create_subset(input_path, output_path, n_events, seed=42):
    """
    Create a smaller subset of the HDF5 file.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        n_events: Number of events to include
        seed: Random seed
    """
    np.random.seed(seed)
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Creating HDF5 Subset")
    print("="*70)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Subset size: {n_events:,} events")
    print(f"Random seed: {seed}")
    
    with h5py.File(input_path, 'r') as f_in:
        n_total = f_in['input'].shape[0]
        
        if n_events > n_total:
            print(f"Warning: Requested {n_events} events but file only has {n_total}")
            n_events = n_total
        
        # Random selection
        indices = np.random.choice(n_total, n_events, replace=False)
        indices = sorted(indices)
        
        print(f"\nSelected {n_events:,} random events from {n_total:,} total")
        
        # Load and save
        print("Loading and saving data...")
        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('input', data=f_in['input'][indices])
            f_out.create_dataset('label', data=f_in['label'][indices])
            f_out.create_dataset('xpmt', data=f_in['xpmt'][:])
            f_out.create_dataset('ypmt', data=f_in['ypmt'][:])
            f_out.create_dataset('zpmt', data=f_in['zpmt'][:])
            
            if 'info' in f_in:
                f_out.create_dataset('info', data=f_in['info'][indices])
    
    size_mb = output_path.stat().st_size / 1024**2
    print(f"\n✓ Subset created: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Split HDF5 file into train/val/test or create subset'
    )
    parser.add_argument('-p', '--path', type=str, required=True,
                       help='Path to input HDF5 file')
    parser.add_argument('-o', '--output', type=str, default='./',
                       help='Output directory (default: ./)')
    
    # Splitting options
    parser.add_argument('--train', type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    
    # Subset options
    parser.add_argument('-n', '--subset', type=int, default=None,
                       help='Create subset with N events instead of splitting')
    parser.add_argument('--subset-output', type=str, default=None,
                       help='Output path for subset file')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.path).exists():
        print(f"Error: File not found: {args.path}")
        return
    
    # Subset mode
    if args.subset is not None:
        if args.subset_output is None:
            input_name = Path(args.path).stem
            args.subset_output = Path(args.output) / f"{input_name}_subset_{args.subset}.h5"
        
        create_subset(
            input_path=args.path,
            output_path=args.subset_output,
            n_events=args.subset,
            seed=args.seed
        )
    
    # Split mode
    else:
        split_h5_file(
            input_path=args.path,
            output_dir=args.output,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed
        )


if __name__ == '__main__':
    main()

