import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_tokens = logits.argmax(dim=-1)
        # A safe divisor keeps mixed greedy/stochastic batches finite. Greedy rows
        # return the argmax of the unscaled logits, with the lowest index on ties.
        greedy = temperatures == 0
        scales = torch.where(greedy, 1.0, temperatures)
        logits = logits.float() / scales.unsqueeze(dim=1)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(greedy, greedy_tokens, sample_tokens)
