import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

def train(model, data_loader, optimizer, config, device):
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Training")

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        # Log gradients and model weights
        wandb.watch(model, log_freq=config['training']['log_interval'])

    wandb.log({"train_loss": total_loss / len(data_loader)})