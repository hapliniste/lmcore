# Using Datasets

This project supports loading both popular datasets from the Hugging Face Datasets library and custom datasets. The data loading module provides a unified interface for working with different types of datasets.

## Loading Popular Datasets

To load a popular dataset from the Hugging Face Datasets library, you can use the `load_dataset` function from the `data.dataset` module. Here's an example:

```python
from data.dataset import load_dataset

# Load OpenWebText dataset
dataset_name = 'openwebtext'
tokenizer_name = 'gpt2'
max_length = 512

dataset = load_dataset(dataset_name, tokenizer_name, max_length)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

The `load_dataset` function automatically handles tokenization, preprocessing, and caching for popular datasets. Currently supported popular datasets include:

- OpenWebText
- BookCorpus
- TinyStories
- ... (add more dataset names as needed)

## Loading Custom Datasets

If you want to load a custom dataset, you can provide a list of text strings to the `load_dataset` function. It will create an instance of the `CustomTextDataset` class and handle tokenization and preprocessing for you. Here's an example:

```python
from data.dataset import load_dataset

# Load custom dataset
tokenizer_name = 'gpt2'
max_length = 512
custom_data = [
    "This is a sample text.",
    "Another example sentence.",
    # Add your custom text data here
]

dataset = load_dataset('custom', tokenizer_name, max_length, custom_data=custom_data)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

You can load your custom data from various sources (e.g., text files, databases, APIs) and pass it as a list of strings to the `load_dataset` function.

## Using the Data Loader

After loading the dataset (either a popular dataset or a custom dataset), you can create a `DataLoader` instance and use it for training or evaluation:

```python
for batch in data_loader:
    # Process the batch data
    print(batch)
```

The `batch` variable will contain the tokenized and preprocessed input data, ready for use in your model.

## Extending the Data Loading Module

If you need to add support for additional popular datasets or implement custom preprocessing steps, you can modify the `data/dataset.py` file accordingly. The `load_dataset` function and the `CustomTextDataset` class are designed to be extensible and flexible.