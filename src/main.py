import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import random_split
import wandb
from config import load_config
from data.dataset import load_dataset
from models.transformer import TransformerModel
from train import train, evaluate

# Load configuration
config = load_config('configs/config.yaml')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
dataset_name = config['data']['dataset_name']
tokenizer_name = config['data']['tokenizer_name']
max_length = config['data']['max_length']
val_split = config['data']['val_split']

dataset = load_dataset(dataset_name, tokenizer_name, max_length)
train_size = int((1 - val_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Initialize model
model = TransformerModel(
    vocab_size=config['model']['vocab_size'],
    dim=config['model']['dim'],
    max_len=config['model']['max_len'],
    num_heads=config['model']['num_heads'],
    num_layers=config['model']['num_layers'],
    dropout=config['model']['dropout']
).to(device)

# Initialize optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
lr_decay_strategy = config['training']['lr_decay_strategy']
lr_decay_steps = config['training']['lr_decay_steps']
min_lr = config['training']['min_lr']

if lr_decay_strategy == 'cosine':
    lr_scheduler = CosineAnnealingLR(optimizer, lr_decay_steps, eta_min=min_lr)
elif lr_decay_strategy == 'linear':
    lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=min_lr / config['training']['learning_rate'], total_iters=lr_decay_steps)
else:  # Constant learning rate
    lr_scheduler = None

# Initialize Weights & Biases
wandb_project = config['training']['wandb_project']
wandb_run_name = config['training']['wandb_run_name']
wandb.init(project=wandb_project, name=wandb_run_name)

# Create directory for saving model checkpoints
save_dir = config['training']['save_dir']
os.makedirs(save_dir, exist_ok=True)

# Training loop
global_step = 0
best_val_loss = float('inf')
patience = config['training']['patience']
early_stop_counter = 0

for epoch in range(config['training']['max_epochs']):
    for batch in train_loader:
        # ... (training loop implementation)
        global_step += 1

        if global_step % config['training']['log_interval'] == 0:
            # Log training loss to WandB
            wandb.log({"train_loss": total_loss / config['training']['log_interval']})
            total_loss = 0

        if global_step % config['training']['eval_interval'] == 0:
            # Evaluate model on validation set
            val_loss = evaluate(model, val_loader, config, device)

            # Early stopping and checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                checkpoint_path = os.path.join(save_dir, 'best_model.pt')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model checkpoint at step {global_step}.")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping after {global_step} steps.")
                    break

        if lr_scheduler:
            lr_scheduler.step()

    # Save model checkpoint after each epoch
    checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pt')
    torch.save(model.state_dict(), checkpoint_path)

    if early_stop_counter >= patience:
        break