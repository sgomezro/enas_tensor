import time
import logging

import torch
from omegaconf import DictConfig

from source.environments import SwingUp, evaluate, rollout_step
from source.helpers.helpers import LOGGER_NAME
from source.wann_engine import (SHARED_WEIGHTS, compile_population, init_population,
                                logits_edges, mutate, reference_forward, wann_forward)

log = logging.getLogger(LOGGER_NAME)


def selftest(device):
    """Check the batched kernel against the scalar reference on random genomes."""
    gen = torch.Generator(device=device).manual_seed(0)
    pop = init_population(64, 6, 1, 32, 0.5, device, gen)
    for _ in range(40):
        mutate(pop, gen)
    weights = torch.tensor(SHARED_WEIGHTS, device=device, dtype=torch.float64)
    comp = compile_population(pop, torch.float64)
    obs = torch.randn(64, len(SHARED_WEIGHTS), 5, device=device,
                      generator=gen, dtype=torch.float64)
    out = wann_forward(obs, comp.adj_f, comp.act_masks, comp.depth_masks,
                       weights[None, :, None], comp.out_idx)
    err = 0.0
    for p in range(64):
        for w_i, w in enumerate(SHARED_WEIGHTS):
            ref = reference_forward(pop, p, obs[p, w_i].tolist(), w)
            err = max(err, abs(ref[0] - float(out[p, w_i, 0])))
    hidden, conns = pop.n_hidden().float().mean(), pop.n_connections().float().mean()
    log.info(f"selftest: max |batched - reference| = {err:.2e} over 64 genomes x 6 weights "
             f"(avg {hidden:.1f} hidden nodes, {conns:.1f} connections)")
    assert err < 1e-9, "batched forward disagrees with scalar reference"
    # DAG check: every edge goes from lower to higher rank
    r = pop.rank
    assert not (pop.adj & (r[:, :, None] >= r[:, None, :])).any(), "cycle risk"

    # Gradient checks: (a) only existing edges get gradient; (b) genomes are
    # independent -- the loss of genome p only produces gradient in theta[p].
    comp32 = compile_population(pop, torch.float64)
    theta = torch.randn(pop.adj.shape, dtype=torch.float64, device=device,
                        generator=gen).requires_grad_(True)
    X = torch.randn(32, 5, dtype=torch.float64, device=device, generator=gen)
    out = logits_edges(comp32, theta, X)
    out[3].pow(2).sum().backward()
    g = theta.grad
    assert bool((g[~pop.adj] == 0).all()), "gradient leaked onto non-edges"
    others = torch.ones(64, dtype=torch.bool, device=device)
    others[3] = False
    assert bool((g[others] == 0).all()), "gradient leaked across genomes"
    # (c) analytic gradient matches finite differences on one edge
    e = pop.adj[3].nonzero()[0]
    eps = 1e-6
    with torch.no_grad():
        th = theta.detach().clone()
        th[3, e[0], e[1]] += eps
        up = logits_edges(comp32, th, X)[3].pow(2).sum()
        th[3, e[0], e[1]] -= 2 * eps
        dn = logits_edges(comp32, th, X)[3].pow(2).sum()
    fd = float((up - dn) / (2 * eps))
    an = float(g[3, e[0], e[1]])
    assert abs(fd - an) <= 1e-5 * max(1.0, abs(fd)), f"grad mismatch {an} vs {fd}"
    log.info(f"gradient checks passed (edge-masked, per-genome independent, "
             f"autograd {an:.6f} vs finite-diff {fd:.6f})")
    # (d) Lamarckian mutations keep weights on edges only
    lam = init_population(32, 6, 1, 32, 0.5, device, gen, with_weights=True)
    for _ in range(40):
        mutate(lam, gen)
    assert bool((lam.weight[~lam.adj] == 0).all()), "Lamarckian weight off-edge"
    log.info("selftest passed")


def benchmark(cfg: DictConfig):
    """Rollout throughput (network + env steps per second) on each available device."""
    env, enas = cfg.environment, cfg.enas
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    step_fn = torch.compile(rollout_step) if cfg.compile else rollout_step
    for dev in devices:
        gen = torch.Generator(device=dev).manual_seed(0)
        pop = init_population(enas.pop_size, 1 + SwingUp.n_obs, SwingUp.n_act,
                              cfg.model.max_nodes or 64, 0.5, dev, gen)
        for _ in range(20):
            mutate(pop, gen)
        weights = torch.tensor(SHARED_WEIGHTS, device=dev)
        evaluate(pop, weights, env.episodes, 20, 0, step_fn=step_fn)  # warm-up (+compile)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        evaluate(pop, weights, env.episodes, env.steps, 0, step_fn=step_fn)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n = enas.pop_size * len(SHARED_WEIGHTS) * env.episodes * env.steps
        log.info(f"[{dev}{' compiled' if cfg.compile else ''}] {n / dt:,.0f} network+env steps/s "
                 f"({dt:.2f}s for pop={enas.pop_size}, W=6, B={env.episodes}, T={env.steps})")
