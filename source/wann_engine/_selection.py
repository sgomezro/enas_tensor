import torch

from ._population import Population
from ._variation import mutate


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------
def pareto_fronts(objs: torch.Tensor) -> torch.Tensor:
    """objs: (P,M), maximize all. Returns front index per individual (0 = best)."""
    P = objs.shape[0]
    ge = (objs[:, None, :] >= objs[None, :, :]).all(-1)
    gt = (objs[:, None, :] > objs[None, :, :]).any(-1)
    dom = ge & gt                                   # dom[i,j]: i dominates j
    front = torch.full((P,), -1, dtype=torch.long, device=objs.device)
    remaining = torch.ones(P, dtype=torch.bool, device=objs.device)
    f = 0
    while bool(remaining.any()):
        dominated = (dom & remaining[:, None]).any(0)
        cur = remaining & ~dominated
        front[cur] = f
        remaining &= ~cur
        f += 1
    return front


def next_generation(pop, mean_f, alt_f, gen, n_elite, tourn_size, p_complexity,
                    p_add_node=0.25, p_add_conn=0.25):
    """WANN-style ranking + tournament selection + mutation.
    mean_f: primary objective (P,).  alt_f: secondary objective (P,), replaced by
    -connections with probability p_complexity (the WANN alternation)."""
    P, dev = pop.size, pop.adj.device
    use_complexity = torch.rand((), device=dev, generator=gen) < p_complexity
    second = torch.where(use_complexity, -pop.n_connections().to(mean_f.dtype),
                         alt_f.to(mean_f.dtype))
    front = pareto_fronts(torch.stack([mean_f, second], 1))
    # order: front ascending, ties broken by mean fitness descending
    order = torch.argsort(-mean_f, stable=True)
    order = order[torch.argsort(front[order], stable=True)]
    position = torch.empty_like(order)
    position[order] = torch.arange(P, device=dev)
    # tournament
    n_child = P - n_elite
    cand = torch.randint(0, P, (n_child, tourn_size), device=dev, generator=gen)
    parents = cand.gather(1, position[cand].argmin(1, keepdim=True)).squeeze(1)
    children = mutate(pop.index(parents), gen, p_add_node, p_add_conn)
    return Population.cat(pop.index(order[:n_elite]), children), order
