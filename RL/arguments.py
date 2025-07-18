import torch

args = {
    'lr': 1e-2,
    'epochs': 100,
    'batch_size': 16,
    'alpha': 0.3,
    'cuda': torch.cuda.is_available(),
    'embed_dim': 64,
    'lstm_hidden_dim': 256,
    'mlp1_dim': 256,
    'mlp2_dim': 128,
    'dropout': 0.3,
    'weight_decay': 1e-5,
}