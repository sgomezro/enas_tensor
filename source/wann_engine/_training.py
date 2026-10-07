import torch
import torch.nn.functional as F

from ._forward import wann_forward
from ._population import Compiled

# ----------------------------------------------------------------------------
# Gradient-based weight training
#
# Topology, activations and depth stay discrete (evolved); gradients tune a
# per-edge weight tensor theta (P,N,N) applied as adj_f * theta. Off-edge entries
# receive zero gradient, so the topology mask is preserved automatically.
# The population's losses are SUMMED and Adam is element-wise, so one backward
# pass trains every genome independently and in parallel.
# ----------------------------------------------------------------------------
LOGIT_CLAMP = 50.0  # keeps cross-entropy finite for exploding linear chains


def logits_shared(comp: Compiled, X, weights):
    """Weight-agnostic evaluation. X: (S,d) -> logits (P,W,S,C)."""
    P, (S, d), Wn = comp.adj_f.shape[0], X.shape, weights.numel()
    obs = X[None, None].expand(P, Wn, S, d).reshape(P, Wn * S, d)
    wcol = weights.to(X.dtype).repeat_interleave(S)[None, :, None]
    out = wann_forward(obs, comp.adj_f, comp.act_masks, comp.depth_masks, wcol, comp.out_idx)
    return out.view(P, Wn, S, -1).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)


def logits_edges(comp: Compiled, theta, X):
    """Per-edge weights theta (P,N,N). X: (S,d) -> logits (P,S,C)."""
    P, (S, d) = comp.adj_f.shape[0], X.shape
    obs = X[None].expand(P, S, d)
    one = torch.ones(1, 1, 1, dtype=X.dtype, device=X.device)
    out = wann_forward(obs, comp.adj_f * theta, comp.act_masks, comp.depth_masks,
                       one, comp.out_idx)
    return out.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)


def ce_acc(logits, y):
    """logits (...,S,C), y (S,) -> per-genome mean cross-entropy and accuracy (...)."""
    C = logits.shape[-1]
    yy = y.expand(logits.shape[:-1])
    ce = F.cross_entropy(logits.reshape(-1, C), yy.reshape(-1), reduction="none")
    ce = torch.nan_to_num(ce.view(logits.shape[:-1]), nan=1e3).mean(-1)
    acc = (logits.argmax(-1) == yy).float().mean(-1)
    return ce, acc


def train_edges(comp: Compiled, theta0, X, y, steps, lr, batch, gen):
    """Minibatch Adam on per-edge weights for the whole population at once."""
    theta = theta0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (batch,), device=X.device, generator=gen)
        ce, _ = ce_acc(logits_edges(comp, theta, X[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        ce.sum().backward()          # sum => independent per-genome gradients
        opt.step()
    return (theta * comp.adj_f).detach()


@torch.no_grad()
def test_acc_edges(comp, theta, X, y):
    return float(ce_acc(logits_edges(comp, theta, X), y)[1][0])
