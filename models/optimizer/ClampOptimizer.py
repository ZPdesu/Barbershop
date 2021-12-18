import math
import torch
from torch.optim import Optimizer
import numpy as np

class ClampOptimizer(Optimizer):
    def __init__(self, optimizer, params, **kwargs):
        self.opt = optimizer(params, **kwargs)
        self.params = params




    @torch.no_grad()
    def step(self, closure=None):
        loss = self.opt.step(closure)


        for param in self.params:
            tmp_latent_norm = torch.clamp(param.data, 0, 1)
            param.data.add_(tmp_latent_norm - param.data)


        return loss


    def zero_grad(self):
        self.opt.zero_grad()


