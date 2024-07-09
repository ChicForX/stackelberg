import torch


class StackelbergGame:
    def __init__(self, model, gamma, P, device):
        self.model = model
        self.gamma = gamma
        self.P = P
        self.device = device
        self.L = self.initialize_L()

    def initialize_L(self):
        I = torch.eye(self.P.shape[0], device=self.device)
        return torch.inverse(self.P + self.gamma * I)

    def encode_gradient(self, g):
        g_bar = self.design_deceptive_gradient(g)
        L_T = self.L.t()
        I = torch.eye(g.shape[0], device=self.device)
        temp = L_T @ self.L + self.gamma * I

        encoded = torch.linalg.solve(temp, (L_T @ g_bar + self.gamma * g))

        # regularizer -> minimize obf cost
        distortion_cost = self.gamma * torch.norm(encoded - g) ** 2

        return encoded, distortion_cost

    def decode_gradient(self, g_tilde):
        return self.L @ g_tilde

    def design_deceptive_gradient(self, g):
        return self.P.diag() * g

    def update_L(self, max_iterations=100, tolerance=1e-6):
        I = torch.eye(self.P.shape[0], device=self.device)
        for _ in range(max_iterations):
            # (L^T L + γI)^(-1)
            temp = torch.inverse(self.L.T @ self.L + self.gamma * I)

            # (L^T P + γI)
            right_term = self.L.T @ self.P + self.gamma * I

            L_new = I @ right_term @ temp

            # check if converge
            if torch.norm(L_new - self.L) < tolerance:
                break

            self.L = L_new