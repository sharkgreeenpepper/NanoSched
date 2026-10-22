import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile(options={"emulate_precision_casts": True})
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
