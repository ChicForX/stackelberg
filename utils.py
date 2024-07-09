import torch
import torch.nn.functional as F
import numpy as np

def compute_loss(output, target):
    return F.nll_loss(output, target)

def solve_nonlinear_equation(P, gamma):
    # simplified
    return torch.inverse(P + gamma * torch.eye(P.shape[0]))

def compute_gradient_diff(grad1, grad2):
    return np.linalg.norm(grad1 - grad2)