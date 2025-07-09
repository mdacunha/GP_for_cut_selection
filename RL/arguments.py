import torch

args = {
    'lr': 1e-5,
    'epochs': 10,
    'batch_size': 10,
    'cuda': torch.cuda.is_available(),
    'embed_dim': 64,
    'lstm_hidden_dim': 128
}