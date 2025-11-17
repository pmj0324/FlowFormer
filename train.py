import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import RelativeGeometryCNF, EquivariantCNF, flow_matching_loss
from dataset import IceCubeDataset, collate_fn


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create model based on configuration."""
    model_type = config['model']['type']
    scaling_stats = config['model'].get('scaling_stats', None)
    
    if model_type == 'RelativeGeometryCNF':
        model = RelativeGeometryCNF(
            signal_dim=config['model']['signal_dim'],
            cond_dim=config['model']['cond_dim'],
            time_embed_dim=config['model']['time_embed_dim'],
            pos_embed_dim=config['model']['pos_embed_dim'],
            sig_embed_dim=config['model']['sig_embed_dim'],
            cond_embed_dim=config['model']['cond_embed_dim'],
            model_dim=config['model']['model_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            rbf_dim=config['model']['rbf_dim'],
            ff_mult=config['model']['ff_mult'],
            dropout=config['model']['dropout'],
            scaling_stats=scaling_stats,
        )
    elif model_type == 'EquivariantCNF':
        model = EquivariantCNF(
            signal_dim=config['model']['signal_dim'],
            cond_dim=config['model']['cond_dim'],
            time_embed_dim=config['model']['time_embed_dim'],
            sig_embed_dim=config['model']['sig_embed_dim'],
            cond_embed_dim=config['model']['cond_embed_dim'],
            scalar_dim=config['model']['scalar_dim'],
            num_layers=config['model']['num_layers'],
            max_distance=config['model'].get('max_distance', 150.0),
            scaling_stats=scaling_stats,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler."""
    scheduler_type = config['training']['scheduler']['type']
    
    if scheduler_type == 'cosine':
        total_steps = config['training']['num_epochs'] * steps_per_epoch
        warmup_steps = config['training']['scheduler']['warmup_epochs'] * steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                min_lr = config['training']['scheduler']['min_lr']
                max_lr = config['training']['learning_rate']
                return min_lr / max_lr + 0.5 * (1 - min_lr / max_lr) * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler, True  # True = step every iteration
    
    elif scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return scheduler, False  # False = step every epoch
    
    else:  # 'none'
        return None, False


def train_epoch(model, dataloader, optimizer, scheduler, config, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        dom_positions = batch['dom_positions'].to(device)
        dom_signals = batch['dom_signals'].to(device)
        condition = batch['condition'].to(device)
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)  # set_to_none=True saves memory
        loss = flow_matching_loss(
            model=model,
            dom_positions=dom_positions,
            dom_signals_data=dom_signals,
            cond=condition,
            mask=None,
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training']['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip']
            )
        
        optimizer.step()
        
        # Explicitly delete to free memory
        del dom_positions, dom_signals, condition
        
        # Clear GPU cache periodically
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
        
        # Step scheduler if per-iteration
        if scheduler is not None and config.get('step_per_iter', False):
            scheduler.step()
        
        # Update stats
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        
        # Log to wandb if enabled
        if config['logging']['use_wandb'] and batch_idx % config['logging']['log_interval'] == 0:
            import wandb
            wandb.log({
                'train/loss': loss.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
                'train/epoch': epoch,
            })
    
    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device, epoch, config):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f"Validation")
    for batch in pbar:
        dom_positions = batch['dom_positions'].to(device)
        dom_signals = batch['dom_signals'].to(device)
        condition = batch['condition'].to(device)
        
        loss = flow_matching_loss(
            model=model,
            dom_positions=dom_positions,
            dom_signals_data=dom_signals,
            cond=condition,
            mask=None,
        )
        
        total_loss += loss.item()
        pbar.set_postfix({'val_loss': f'{total_loss / (pbar.n + 1):.4f}'})
        
        # Free memory
        del dom_positions, dom_signals, condition, loss
    
    avg_loss = total_loss / num_batches
    
    # Log to wandb if enabled
    if config['logging']['use_wandb']:
        import wandb
        wandb.log({
            'val/loss': avg_loss,
            'val/epoch': epoch,
        })
    
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, config, save_path, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': config,
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']


@torch.no_grad()
def generate_samples(model, dom_positions, cond, device, num_steps=100):
    """
    Generate samples using ODE solver (Euler method).
    
    Args:
        model: CNF model
        dom_positions: (B, N, 3) DOM positions
        cond: (B, 6) or (B, 8) conditioning
        device: device to use
        num_steps: number of ODE integration steps
    
    Returns:
        generated_signals: (B, N, 2) generated DOM signals
    """
    B, N, _ = dom_positions.shape
    
    # Start from noise
    x = torch.randn(B, N, 2, device=device)
    
    # ODE integration: dx/dt = v(x, t)
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        t = torch.ones(B, device=device) * (step * dt)
        v = model(dom_positions, x, cond, t, scale_inputs=True)
        x = x + v * dt
    
    return x


def visualize_epoch(model, dataloader, device, epoch, save_dir, num_samples=4):
    """
    Visualize generated events at the end of an epoch.
    
    Args:
        model: CNF model
        dataloader: validation dataloader
        device: device to use
        epoch: current epoch number
        save_dir: directory to save visualizations
        num_samples: number of samples to generate
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a batch of real data
    batch = next(iter(dataloader))
    dom_positions = batch['dom_positions'][:num_samples].to(device)
    real_signals = batch['dom_signals'][:num_samples].to(device)
    conditions = batch['condition'][:num_samples].to(device)
    
    # Generate samples
    print(f"\n  Generating {num_samples} samples for visualization...")
    generated_signals = generate_samples(model, dom_positions, conditions, device)
    
    # Move to CPU for plotting
    dom_positions_cpu = dom_positions.cpu().numpy()
    real_signals_cpu = real_signals.cpu().numpy()
    generated_signals_cpu = generated_signals.cpu().numpy()
    conditions_cpu = conditions.cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Extract data for this sample
        pos = dom_positions_cpu[i]  # (N, 3)
        real_time = real_signals_cpu[i, :, 0]
        real_charge = real_signals_cpu[i, :, 1]
        gen_time = generated_signals_cpu[i, :, 0]
        gen_charge = generated_signals_cpu[i, :, 1]
        
        # Filter active DOMs (non-zero charge)
        real_mask = real_charge > real_charge.mean()
        gen_mask = gen_charge > gen_charge.mean()
        
        # Real Time Distribution
        ax = axes[i, 0]
        if real_mask.sum() > 0:
            ax.scatter(pos[real_mask, 0], pos[real_mask, 2], 
                      c=real_time[real_mask], s=30, cmap='viridis', alpha=0.6)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Z (m)', fontsize=10)
        ax.set_title(f'Real Time (Sample {i+1})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Generated Time Distribution
        ax = axes[i, 1]
        if gen_mask.sum() > 0:
            ax.scatter(pos[gen_mask, 0], pos[gen_mask, 2], 
                      c=gen_time[gen_mask], s=30, cmap='viridis', alpha=0.6)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Z (m)', fontsize=10)
        ax.set_title(f'Generated Time (Sample {i+1})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Real Charge Distribution
        ax = axes[i, 2]
        if real_mask.sum() > 0:
            scatter = ax.scatter(pos[real_mask, 0], pos[real_mask, 2], 
                                c=real_charge[real_mask], s=30, cmap='hot', alpha=0.6)
            plt.colorbar(scatter, ax=ax)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Z (m)', fontsize=10)
        ax.set_title(f'Real Charge (Sample {i+1})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Generated Charge Distribution
        ax = axes[i, 3]
        if gen_mask.sum() > 0:
            scatter = ax.scatter(pos[gen_mask, 0], pos[gen_mask, 2], 
                                c=gen_charge[gen_mask], s=30, cmap='hot', alpha=0.6)
            plt.colorbar(scatter, ax=ax)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Z (m)', fontsize=10)
        ax.set_title(f'Generated Charge (Sample {i+1})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Epoch {epoch}: Real vs Generated Events', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    save_path = save_dir / f'epoch_{epoch:04d}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Visualization saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train FlowFormer model')
    parser.add_argument('-c', '--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set random seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create save directory
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Initialize wandb if enabled
    if config['logging']['use_wandb']:
        import wandb
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging']['wandb_entity'],
            config=config,
        )
    
    # Load dataset
    print("\n" + "="*50)
    print("Loading dataset...")
    print("="*50)
    full_dataset = IceCubeDataset(
        h5_path=config['data']['train_path'],
    )
    
    # Split into train and validation
    val_size = int(len(full_dataset) * config['data']['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )
    
    print(f"\nTrain size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=False,  # Disable to avoid memory issues
        persistent_workers=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=False,  # Disable to avoid memory issues
        persistent_workers=False,
    )
    
    # Create model
    print("\n" + "="*50)
    print("Creating model...")
    print("="*50)
    model = create_model(config)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']['type']}")
    print(f"Number of parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler, step_per_iter = create_scheduler(optimizer, config, len(train_loader))
    config['step_per_iter'] = step_per_iter
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if config['training']['resume'] is not None:
        print(f"\nResuming from checkpoint: {config['training']['resume']}")
        start_epoch = load_checkpoint(model, optimizer, scheduler, config['training']['resume'])
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, device, epoch)
        
        # Step scheduler if per-epoch
        if scheduler is not None and not step_per_iter:
            scheduler.step()
        
        # Validate
        val_loss = validate(model, val_loader, device, epoch, config)
        
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Visualize generated samples
        if config.get('visualization', {}).get('enabled', True):
            visualize_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                epoch=epoch + 1,
                save_dir=config.get('visualization', {}).get('save_dir', './training_visualizations'),
                num_samples=config.get('visualization', {}).get('num_samples', 4)
            )
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}")
        
        if (epoch + 1) % config['training']['save_interval'] == 0 or is_best:
            save_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch + 1, config, str(save_path), is_best)
            print(f"Checkpoint saved: {save_path}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*50)
    
    if config['logging']['use_wandb']:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()

