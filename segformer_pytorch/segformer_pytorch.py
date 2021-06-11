import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

# helpers

def exists(val):
    return val is not None

# classes

class MiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64
    ):
        super().__init__()

    def forward(self, x):
        return x

class Segformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        num_classes = 4,
        dim_head = 64
    ):
        super().__init__()

    def forward(self, x):
        return x
