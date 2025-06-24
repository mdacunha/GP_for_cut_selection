import torch

args = {
    'lr': 0.001,
    'epochs': 1,
    'batch_size': 10,
    'cuda': torch.cuda.is_available(),
    'embed_dim': 64,
    'lstm_hidden_dim': 128
}