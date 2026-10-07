from __future__ import annotations

import torch
import torch.nn.functional as F

from ._activations import SCALAR_ACTS, apply_activations
from ._population import Population


def wann_forward(obs: torch.Tensor, adj_f, act_masks, depth_masks, wcol, n_out_slice):
    """
    obs:  (P,R,n_obs) observations; R = shared-weight values x episodes
    wcol: (1,R,1) shared weight for each row
    Returns (P,R,n_out).
    """
    P, R, _ = obs.shape
    N = adj_f.shape[-1]
    ones = torch.ones(P, R, 1, dtype=obs.dtype, device=obs.device)
    h = F.pad(torch.cat([ones, obs], -1), (0, N - obs.shape[-1] - 1))
    for d in range(depth_masks.shape[0]):
        pre = wcol * torch.bmm(h, adj_f)        # shared weight factored out of the sum
        h = torch.where(depth_masks[d], apply_activations(pre, act_masks), h)
    return h[..., n_out_slice]


def reference_forward(pop: Population, p: int, obs: list[float], w: float) -> list[float]:
    """Scalar, node-by-node evaluation of genome p (for testing)."""
    adj, act, rank, active = (pop.adj[p].cpu(), pop.act[p].cpu(),
                              pop.rank[p].cpu(), pop.active[p].cpu())
    N, n_in = adj.shape[0], pop.n_in
    val = [0.0] * N
    val[0] = 1.0
    for i, o in enumerate(obs):
        val[1 + i] = o
    order = sorted([j for j in range(n_in, N) if active[j]], key=lambda j: float(rank[j]))
    for j in order:
        s = sum(val[i] for i in range(N) if adj[i, j])
        val[j] = SCALAR_ACTS[int(act[j])](w * s)
    return val[n_in:n_in + pop.n_out]
