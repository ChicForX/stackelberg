import torch
import torch.nn.functional as F
import numpy as np


def compute_loss(output, target):
    return F.nll_loss(output, target)


def compute_gradient_diff(grad1, grad2):
    return np.linalg.norm(grad1 - grad2)


def generate_P(num_params, delta, device):
    diag_elements = torch.empty(num_params, device=device).uniform_(1 - delta, 1 + delta)

    P = torch.diag(diag_elements)

    return P