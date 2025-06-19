import torch

args = {
    'lr': 0.001,
    'epochs': 15,
    'batch_size': 15,
    'cuda': torch.cuda.is_available(),
    'embed_dim': 64,
    'lstm_hidden_dim': 128
}