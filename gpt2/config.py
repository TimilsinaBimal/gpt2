from dataclasses import dataclass


@dataclass
class Config:
    block_size: int = 1024  # max_sequence length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768  # equivalent to d_model
    seq_length: int = 1024
    eps: float = 1e-5
    learning_rate: float = 3e-4
    num_epochs: int = 1
    batch_size: int = 4
    device: str = "mps"
    data_path: str = "data/input.txt"
    model_path: str = "models/tiny_shakespeare_model.pth"
