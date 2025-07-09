import torch

args = {
    'lr': 1e-5,
    'epochs': 20,
    'batch_size': 10,
    'alpha': 0.3,
    'cuda': torch.cuda.is_available(),
    'embed_dim': 64,
    'lstm_hidden_dim': 128
}