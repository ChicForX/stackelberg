import torch
from utils import compute_loss, compute_gradient_diff
from config import config_dict
import numpy as np


def train(model, train_loader, optimizer, game, accumulation_steps=4):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(config_dict['device']), target.to(config_dict['device'])
        output = model(data)
        loss = compute_loss(output, target)
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            original_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters() if p.grad is not None])
            encoded_grad = game.encode_gradient(original_grad)
            decoded_grad = game.decode_gradient(encoded_grad)

            idx = 0
            for p in model.parameters():
                numel = p.numel()
                p.grad.data = decoded_grad[idx:idx + numel].view_as(p.data)
                idx += numel

            optimizer.step()
            optimizer.zero_grad()
            game.update_L()


def test(model, test_loader, game):
    model.eval()
    test_loss = 0
    correct = 0
    original_grads, encoded_grads, decoded_grads = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config_dict['device']), target.to(config_dict['device'])
            output = model(data)
            test_loss += compute_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss = compute_loss(output, target)
            loss.backward()

            original_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])
            original_grads.append(original_grad.cpu().numpy())

            encoded_grad = game.encode_gradient(original_grad)
            encoded_grads.append(encoded_grad.cpu().numpy())

            decoded_grad = game.decode_gradient(encoded_grad)
            decoded_grads.append(decoded_grad.cpu().numpy())

            model.zero_grad()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    original_grads = np.mean(original_grads, axis=0)
    encoded_grads = np.mean(encoded_grads, axis=0)
    decoded_grads = np.mean(decoded_grads, axis=0)

    orig_enc_diff = compute_gradient_diff(original_grads, encoded_grads)
    orig_dec_diff = compute_gradient_diff(original_grads, decoded_grads)
    enc_dec_diff = compute_gradient_diff(encoded_grads, decoded_grads)

    return accuracy, test_loss, orig_enc_diff, orig_dec_diff, enc_dec_diff
