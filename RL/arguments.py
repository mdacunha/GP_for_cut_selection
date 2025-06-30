import torch

args = {
    'lr': 1e-4,
    'epochs': 15,
    'batch_size': 10,
    'cuda': torch.cuda.is_available(),
    'embed_dim': 64,
    'lstm_hidden_dim': 128
}