import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import math

def evaluate(model, data_loader, config, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    pbar = tqdm(data_loader, desc="Evaluation")

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), input_ids.view(-1), ignore_index=model.tokenizer.pad_token_id)

            total_loss += loss.item() * input_ids.size(0)
            total_tokens += torch.sum(attention_mask).item()

            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

    val_loss = total_loss / total_tokens
    val_ppl = math.exp(val_loss)
    wandb.log({"val_loss": val_loss, "val_ppl": val_ppl})
    print(f"Validation Loss: {val_loss:.4f}, Validation Perplexity: {val_ppl:.4f}")