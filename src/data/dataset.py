import torch
from transformers import AutoTokenizer
from datasets import load_dataset

class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoded = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors='pt')
        return encoded

def load_dataset(dataset_name, tokenizer_name, max_length, custom_data=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if dataset_name in ['openwebtext', 'bookcorpus', ...]:  # Add popular dataset names here
        dataset = load_dataset(dataset_name)
        dataset = dataset.map(lambda x: tokenizer(x['text'], max_length=max_length, truncation=True, return_tensors='pt'), batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    else:
        # Assume custom_data is a list of text strings
        dataset = CustomTextDataset(custom_data, tokenizer, max_length)

    return dataset