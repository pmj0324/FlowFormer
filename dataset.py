import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class IceCubeDataset(Dataset):
    """
    Dataset for IceCube neutrino events.
    
    HDF5 structure:
        - input: (N_events, 2, N_DOMs) - DOM signals (time, charge)
        - label: (N_events, 6) - event parameters (E, θ, φ, x, y, z)
        - xpmt, ypmt, zpmt: (N_DOMs,) - DOM positions
    
    Note: Scaling is now handled by the model, not the dataset.
    """
    
    def __init__(self, h5_path):
        """
        Args:
            h5_path: Path to HDF5 file
        """
        self.h5_path = h5_path
        
        # Load data into memory for faster training
        with h5py.File(h5_path, 'r') as f:
            # Signals: (N_events, 2, N_DOMs) -> transpose to (N_events, N_DOMs, 2)
            self.signals = torch.from_numpy(f['input'][:]).float()
            self.signals = self.signals.transpose(1, 2)  # (N, N_DOMs, 2)
            
            # Labels (conditions): (N_events, 6)
            self.labels = torch.from_numpy(f['label'][:]).float()
            
            # DOM positions: (N_DOMs, 3)
            xpmt = f['xpmt'][:]
            ypmt = f['ypmt'][:]
            zpmt = f['zpmt'][:]
            self.dom_positions = torch.from_numpy(
                np.stack([xpmt, ypmt, zpmt], axis=1)
            ).float()
        
        self.n_events = self.signals.shape[0]
        self.n_doms = self.signals.shape[1]
        
        print(f"Loaded {self.n_events} events with {self.n_doms} DOMs")
        print(f"Signals shape: {self.signals.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"DOM positions shape: {self.dom_positions.shape}")
        print("Note: Scaling will be handled by the model.")
    
    def __len__(self):
        return self.n_events
    
    def __getitem__(self, idx):
        """
        Returns:
            dom_positions: (N_DOMs, 3) - DOM positions
            dom_signals: (N_DOMs, 2) - DOM signals (time, charge)
            condition: (6,) - Event parameters (E, θ, φ, x, y, z)
        """
        return {
            'dom_positions': self.dom_positions,  # Same for all events
            'dom_signals': self.signals[idx],      # (N_DOMs, 2)
            'condition': self.labels[idx],          # (6,)
        }
    


def collate_fn(batch):
    """
    Custom collate function for batching.
    
    Since dom_positions are the same for all events, we can just use one copy.
    """
    # Use repeat instead of expand to avoid memory sharing issues with pin_memory
    dom_positions = batch[0]['dom_positions'].unsqueeze(0).repeat(len(batch), 1, 1)
    dom_signals = torch.stack([item['dom_signals'] for item in batch])
    conditions = torch.stack([item['condition'] for item in batch])
    
    return {
        'dom_positions': dom_positions,  # (B, N_DOMs, 3)
        'dom_signals': dom_signals,      # (B, N_DOMs, 2)
        'condition': conditions,          # (B, 6)
    }

