from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import config_dict


def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config_dict['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config_dict['batch_size'], shuffle=False)

    return train_loader, test_loader