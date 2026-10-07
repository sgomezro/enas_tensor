import torch

from ._activations import K_ACT
from ._population import Population


# ----------------------------------------------------------------------------
# Mutation (vectorized, no host syncs)
# ----------------------------------------------------------------------------
def _sample_flat(mask: torch.Tensor, gen) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample one True entry per row of a (C, M) mask."""
    score = torch.rand(mask.shape, device=mask.device, generator=gen)
    score = torch.where(mask, score, torch.full_like(score, -1.0))
    best, idx = score.max(1)
    return idx, best >= 0


def mutate(pop: Population, gen, p_add_node=0.25, p_add_conn=0.25) -> Population:
    """Apply exactly one topological/activation mutation to every genome in place."""
    C, N = pop.size, pop.max_nodes
    dev = pop.adj.device
    rows = torch.arange(C, device=dev)
    u = torch.rand(C, device=dev, generator=gen)
    do_node = u < p_add_node
    do_conn = (u >= p_add_node) & (u < p_add_node + p_add_conn)
    do_act = u >= p_add_node + p_add_conn
    n_io = pop.n_in + pop.n_out
    flat = pop.adj.view(C, N * N)

    # --- add connection: i -> j with rank_i < rank_j, not already present
    valid = (pop.active[:, :, None] & pop.active[:, None, :]
             & (pop.rank[:, :, None] < pop.rank[:, None, :]) & ~pop.adj)
    idx, ok = _sample_flat(valid.view(C, -1), gen)
    ok = ok & do_conn
    cur = flat.gather(1, idx[:, None])
    flat.scatter_(1, idx[:, None], cur | ok[:, None])
    wflat = None if pop.weight is None else pop.weight.view(C, N * N)
    if wflat is not None:  # new edge gets a N(0,1) weight
        w_new = torch.randn(C, 1, device=dev, generator=gen, dtype=wflat.dtype)
        wflat.scatter_(1, idx[:, None], torch.where(ok[:, None], w_new,
                                                    wflat.gather(1, idx[:, None])))

    # --- add node: split an existing edge i -> j into i -> k -> j
    free = ~pop.active
    free[:, :n_io] = False
    k = free.to(torch.uint8).argmax(1)
    has_free = free.any(1)
    e_idx, has_edge = _sample_flat(flat, gen)
    ok = do_node & has_free & has_edge
    i, j = e_idx // N, e_idx % N
    flat.scatter_(1, e_idx[:, None], flat.gather(1, e_idx[:, None]) & ~ok[:, None])
    ik, kj = (i * N + k)[:, None], (k * N + j)[:, None]
    flat.scatter_(1, ik, flat.gather(1, ik) | ok[:, None])
    flat.scatter_(1, kj, flat.gather(1, kj) | ok[:, None])
    if wflat is not None:  # NEAT convention: i->k gets 1, k->j inherits w_ij
        w_old = wflat.gather(1, e_idx[:, None])
        okc = ok[:, None]
        wflat.scatter_(1, ik, torch.where(okc, torch.ones_like(w_old), wflat.gather(1, ik)))
        wflat.scatter_(1, kj, torch.where(okc, w_old, wflat.gather(1, kj)))
        wflat.scatter_(1, e_idx[:, None], torch.where(okc, torch.zeros_like(w_old), w_old))
    pop.active[rows, k] |= ok
    mid = 0.5 * (pop.rank[rows, i] + pop.rank[rows, j])
    pop.rank[rows, k] = torch.where(ok, mid, pop.rank[rows, k])
    new_act = torch.randint(0, K_ACT, (C,), device=dev, generator=gen)
    pop.act[rows, k] = torch.where(ok, new_act, pop.act[rows, k])

    # --- change activation of a hidden or output node
    cand = pop.active.clone()
    cand[:, :pop.n_in] = False
    n_idx, ok = _sample_flat(cand, gen)
    ok = ok & do_act
    old = pop.act[rows, n_idx]
    shift = torch.randint(1, K_ACT, (C,), device=dev, generator=gen)
    pop.act[rows, n_idx] = torch.where(ok, (old + shift) % K_ACT, old)
    return pop
