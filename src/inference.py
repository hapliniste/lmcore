import torch

def generate_text(model, prompt, max_length=100, top_k=50, top_p=0.95, num_beams=5, early_stopping=True, device='cpu'):
    model.eval()
    with torch.no_grad():
        encoded_prompt = model.tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = model.generate(
            encoded_prompt,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1
        )
        generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text