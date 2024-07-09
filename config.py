import torch

config_dict = {
    'batch_size': 32,
    'learning_rate': 0.01,
    'epochs': 5,
    'gamma': 0.5,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_classes': 10,
    'in_channels': 1,
    'delta': 0.1
}