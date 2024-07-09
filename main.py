import os
import torch
import torch.optim as optim
from config import config_dict
from dataset import get_data_loaders
from stackelberg import StackelbergGame
from train import train, test
from model import LeNet5
from utils import generate_P


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    print("Number of parameters in the model:", model.num_params())
    optimizer = optim.SGD(model.parameters(), lr=config_dict['learning_rate'])

    train_loader, test_loader = get_data_loaders()

    num_params = sum(p.numel() for p in model.parameters())
    P = generate_P(num_params, config_dict['delta'], device)

    game = StackelbergGame(model, config_dict['gamma'], P, device)

    for epoch in range(1, config_dict['epochs'] + 1):
        train(model, train_loader, optimizer, game)
        accuracy, loss, orig_enc_diff, orig_dec_diff, enc_dec_diff = test(model, test_loader, game, device)
        print(f'Epoch {epoch}/{config_dict["epochs"]}:')
        print(f'Test set: Average loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
        print(f'Gradient differences:')
        print(f'  Original vs Encoded: {orig_enc_diff:.4f}')
        print(f'  Original vs Decoded: {orig_dec_diff:.4f}')
        print(f'  Encoded vs Decoded: {enc_dec_diff:.4f}')


if __name__ == '__main__':
    main()
