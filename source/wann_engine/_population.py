from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._activations import ACT_NAMES, K_ACT


# ----------------------------------------------------------------------------
# Population
# ----------------------------------------------------------------------------
@dataclass
class Population:
    adj: torch.Tensor     # (P,N,N) bool, adj[p,i,j] = edge i -> j
    act: torch.Tensor     # (P,N) long
    rank: torch.Tensor    # (P,N) float64, topological key
    active: torch.Tensor  # (P,N) bool
    n_in: int             # bias + observations
    n_out: int
    weight: torch.Tensor | None = None  # (P,N,N) float, per-edge weights (Lamarckian only)

    @property
    def size(self) -> int:
        return self.adj.shape[0]

    @property
    def max_nodes(self) -> int:
        return self.adj.shape[1]

    def index(self, idx: torch.Tensor) -> "Population":
        w = None if self.weight is None else self.weight[idx].clone()
        return Population(self.adj[idx].clone(), self.act[idx].clone(),
                          self.rank[idx].clone(), self.active[idx].clone(),
                          self.n_in, self.n_out, w)

    @staticmethod
    def cat(a: "Population", b: "Population") -> "Population":
        w = None if a.weight is None else torch.cat([a.weight, b.weight])
        return Population(torch.cat([a.adj, b.adj]), torch.cat([a.act, b.act]),
                          torch.cat([a.rank, b.rank]), torch.cat([a.active, b.active]),
                          a.n_in, a.n_out, w)

    def n_connections(self) -> torch.Tensor:
        return self.adj.sum((1, 2))

    def n_hidden(self) -> torch.Tensor:
        return self.active[:, self.n_in + self.n_out:].sum(1)


def init_population(P, n_in, n_out, max_nodes, p_conn, device, gen,
                    with_weights=False) -> Population:
    """Minimal networks: sparse input->output links, no hidden nodes (as in WANN)."""
    N, n_io = max_nodes, n_in + n_out
    adj = torch.zeros(P, N, N, dtype=torch.bool, device=device)
    block = torch.rand(P, n_in, n_out, device=device, generator=gen) < p_conn
    # guarantee at least one connection per genome
    forced = torch.randint(0, n_in * n_out, (P,), device=device, generator=gen)
    empty = ~block.flatten(1).any(1)
    block.view(P, -1)[torch.arange(P, device=device), forced] |= empty
    adj[:, :n_in, n_in:n_io] = block
    act = torch.randint(0, K_ACT, (P, N), device=device, generator=gen)
    act[:, :n_in] = 0
    rank = torch.full((P, N), 0.5, dtype=torch.float64, device=device)
    rank[:, :n_in] = 0.0
    rank[:, n_in:n_io] = 1.0
    active = torch.zeros(P, N, dtype=torch.bool, device=device)
    active[:, :n_io] = True
    weight = None
    if with_weights:  # Lamarckian genomes carry per-edge weights, N(0,1) init
        weight = torch.randn(P, N, N, device=device, generator=gen) * adj
    return Population(adj, act, rank, active, n_in, n_out, weight)


def compute_depth(pop: Population) -> torch.Tensor:
    """Longest-path depth per node. Inputs/inactive = 0; others >= 1.
    Iterates to a fixed point (one host sync per iteration, once per generation)."""
    adj, n_in = pop.adj, pop.n_in
    P, N, _ = adj.shape
    depth = torch.zeros(P, N, dtype=torch.long, device=adj.device)
    compute = pop.active.clone()
    compute[:, :n_in] = False
    for _ in range(N):
        pred = torch.where(adj, depth[:, :, None], torch.full_like(depth[:, :, None], -1))
        new = (pred.max(1).values.clamp(min=0) + 1) * compute
        if torch.equal(new, depth):
            break
        depth = new
    return depth


@dataclass
class Compiled:
    """Per-generation tensors consumed by the forward pass."""
    adj_f: torch.Tensor        # (P,N,N) float
    act_masks: torch.Tensor    # (K,P,1,N) bool
    depth_masks: torch.Tensor  # (D,P,1,N) bool
    out_idx: slice


def compile_population(pop: Population, dtype=torch.float32, depth_bucket=4) -> Compiled:
    depth = compute_depth(pop)
    D = int(depth.max())
    D = max(depth_bucket, math.ceil(D / depth_bucket) * depth_bucket)  # fewer recompiles
    levels = torch.arange(1, D + 1, device=depth.device)
    depth_masks = (depth[None] == levels[:, None, None])[:, :, None, :]
    acts = torch.arange(K_ACT, device=depth.device)
    act_masks = (pop.act[None] == acts[:, None, None])[:, :, None, :]
    return Compiled(pop.adj.to(dtype), act_masks, depth_masks,
                    slice(pop.n_in, pop.n_in + pop.n_out))


def describe(pop: Population, p: int) -> str:
    names = ["bias"] + [f"in{i}" for i in range(pop.n_in - 1)] + \
            [f"out{i}" for i in range(pop.n_out)]
    def nm(i):
        return names[i] if i < len(names) else f"h{i}"
    act, adj = pop.act[p].cpu(), pop.adj[p].cpu()
    nodes = [i for i in range(pop.n_in, pop.max_nodes) if pop.active[p, i]]
    lines = [f"  {nm(j)} [{ACT_NAMES[int(act[j])]}] <- "
             + ", ".join(nm(i) for i in adj[:, j].nonzero().flatten().tolist())
             for j in sorted(nodes, key=lambda j: float(pop.rank[p, j]))]
    return "\n".join(lines)
