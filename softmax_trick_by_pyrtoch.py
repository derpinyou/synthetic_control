import torch
import numpy as np


class MyModel(torch.nn.Module):
    def __init__(self, X_pre: np.ndarray, seed: int):
        super().__init__()
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.w = torch.nn.Parameter(
            torch.normal(
                0,
                0.1,
                (X_pre.shape[1],),
                device="cpu",
                generator=self.generator,
                dtype=torch.float32,
                requires_grad=True,
            )
        )

    def forward(self, x):
        return torch.matmul(x, torch.softmax(self.w, dim=-1))

