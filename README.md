# LMCore

This project is a modular and extensible codebase for experimenting with and comparing different language model architectures and training techniques. It incorporates best practices and techniques from state-of-the-art models like Mistral and Mixtral.

## Features

- Efficient data loading and preprocessing
- Rotary Positional Embedding (RoPE)
- Modular attention mechanisms (Multi-Head Attention, Sparse Attention, etc.)
- Depth-wise convolutions in feed-forward layers
- Modular and extensible model architecture
- Flexible and modular training loop
- Integration with Weights & Biases (WandB) for logging and experiment tracking
- Evaluation metrics and inference module
- Parallelism and efficient computation techniques
- Reproducibility and documentation

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/hapliniste/lmcore.git
    ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Follow the instructions in the project documentation to set up the data, train models, and run experiments.

## Contributing

Contributions are welcome! Please follow the guidelines in the `CONTRIBUTING.md` file.

## License

This project is licensed under the [MIT License](LICENSE).