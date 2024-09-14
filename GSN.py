import torch
from torch import nn, Tensor
from torch.nn import init

def inner(v: Tensor, u: Tensor) -> Tensor:
    return torch.trace(v.transpose(-1, -2) @ u)

def proj(u: Tensor, v: Tensor) -> Tensor:
    return u * (inner(v, u) / inner(u, u))

def norm(v: Tensor) -> Tensor:
    return torch.sqrt(inner(v, v))

def eff_gram_schmidt(prior: Tensor, next: Tensor) -> Tensor:
    next -= proj(prior, next)
    return next

class GramSchmidtLayer(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            num_linears: int
    ) -> None:
        super().__init__()
        self.num_linears = num_linears
        self.weight = torch.empty(in_size, out_size, num_linears)
        self._init_weight()

    def _init_weight(self) -> None:
        init.xavier_normal_(self.weight)

    def gram_schmidt(self, weight) -> Tensor:
        for t in range(self.num_linears):
            curr_weight = weight[:, :, t]
            if t > 0:
                prev_weight: Tensor = weight[:, :, :t-1]
                for i in range(t):
                    curr_weight -= proj(prev_weight[:, :, i], curr_weight)
            norm_val = norm(curr_weight)
            if norm_val != 0:
                weight[:, :, t] /= norm_val
        return weight

    def forward(self, input: Tensor) -> Tensor:
        weight = self.gram_schmidt(self.weight)
        out = torch.einsum('bi,ioj->boj', input, weight)
        return out.sum(-1)
