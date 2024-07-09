import torch
from utils import solve_nonlinear_equation
from config import config_dict


class StackelbergGame:
    def __init__(self, model, gamma, P):
        self.model = model
        self.gamma = gamma
        self.P = P
        self.L = None

    def encode_gradient(self, g):
        g = g.to(config_dict['device'])
        if self.L is None:
            self.L = torch.eye(g.numel(), dtype=g.dtype, device=g.device)
        L_T = self.L.t()
        I = torch.eye(g.numel(), dtype=g.dtype, device=g.device)
        temp = L_T @ self.L + self.gamma * I
        return torch.linalg.solve(temp, (L_T @ self.P + self.gamma * I) @ g)

    def decode_gradient(self, g_tilde):
        return self.L @ g_tilde

    def update_L(self):
        self.L = solve_nonlinear_equation(self.P, self.gamma)

