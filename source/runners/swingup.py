import os
import time
import logging

import torch
from omegaconf import DictConfig

from source.environments import SwingUp, evaluate, rollout_step
from source.helpers.helpers import (LOGGER_NAME, resolve_device, save_champion,
                                    update_config)
from source.wann_engine import (compile_population, describe, init_population,
                                next_generation)

log = logging.getLogger(LOGGER_NAME)


def resolve_defaults(cfg: DictConfig) -> DictConfig:
    """Fill task-dependent values left as null in the YAML defaults."""
    return update_config(cfg, {
        "model.max_nodes": cfg.model.max_nodes or 64,
        "model.p_init_conn": 0.25 if cfg.model.p_init_conn is None else cfg.model.p_init_conn,
        "training.finetune_steps": cfg.training.finetune_steps or 0,
    })


def evolve(cfg: DictConfig):
    """Evolve weight-agnostic topologies on CartPole swing-up."""
    if cfg.training.learning != "none":
        raise SystemExit("training.learning=baldwinian/lamarckian is implemented for supervised "
                         "tasks (environment=spirals|digits). For swing-up, use "
                         "training.finetune_steps to fine-tune the champion by BPTT.")
    resolve_defaults(cfg)
    env, enas, model, training = cfg.environment, cfg.enas, cfg.model, cfg.training
    dev = resolve_device(cfg.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    # mode="reduce-overhead" (CUDA graphs) breaks here: each step's outputs are fed back
    # as the next step's inputs and get overwritten by the graph replay. Default mode
    # (kernel fusion only) is as fast in practice.
    step_fn = torch.compile(rollout_step) if cfg.compile else rollout_step
    pop = init_population(enas.pop_size, 1 + SwingUp.n_obs, SwingUp.n_act,
                          model.max_nodes, model.p_init_conn, dev, gen)
    weights = torch.tensor(list(model.shared_weights), device=dev)
    log.info(f"device={dev}  pop={enas.pop_size}  max_nodes={model.max_nodes}  "
             f"episodes={env.episodes}  steps={env.steps}  weights={weights.tolist()}")
    best = (-float("inf"), None)
    for g in range(enas.max_gen):
        t0 = time.perf_counter()
        fit = evaluate(pop, weights, env.episodes, env.steps, seed=cfg.seed * 100003 + g,
                       step_fn=step_fn)
        mean_f = fit.mean(1)
        i = int(mean_f.argmax())
        if float(mean_f[i]) > best[0]:
            best = (float(mean_f[i]), pop.index(torch.tensor([i], device=dev)))
        if g % cfg.log_every == 0 or g == enas.max_gen - 1:
            log.info(f"gen {g:4d} | best mean {float(mean_f[i]):7.1f}  "
                     f"best max-w {float(fit.max(1).values.max()):7.1f}  "
                     f"pop mean {float(mean_f.mean()):6.1f} | "
                     f"conns {int(pop.n_connections()[i]):3d} hidden {int(pop.n_hidden()[i]):3d} "
                     f"| {time.perf_counter() - t0:5.2f}s")
        pop, _ = next_generation(pop, mean_f, fit.max(1).values, gen, enas.elite,
                                 enas.tournament_size, enas.p_complexity,
                                 enas.prob_add_node, enas.prob_add_conn)

    champ = best[1]
    final = evaluate(champ, weights, 32, env.steps, seed=12345)[0]
    log.info("\nchampion (re-evaluated on 32 fresh episodes per weight):")
    for w, r in zip(weights.tolist(), final.tolist()):
        log.info(f"  w = {w:+.1f}  return {r:7.1f}")
    log.info(f"  mean over weights: {float(final.mean()):.1f}")
    log.info("topology (node [activation] <- inputs):")
    log.info(describe(champ, 0))
    theta = finetune_swingup(champ, cfg, gen) if training.finetune_steps > 0 else None
    save_champion(champ, os.path.join(cfg.out_dir, "champion.pt"), theta)


def finetune_swingup(champ, cfg: DictConfig, gen):
    """Experimental: fine-tune per-edge weights of a swing-up champion by
    backpropagating the return through the torch physics (truncated horizon)."""
    training, steps = cfg.training, cfg.environment.steps
    dev = champ.adj.device
    comp = compile_population(champ)
    weights = torch.tensor(list(cfg.model.shared_weights), device=dev)
    fit = evaluate(champ, weights, 16, steps, seed=777)[0]
    w_best = weights[fit.argmax()]
    theta = (comp.adj_f * w_best).clone().requires_grad_(True)
    one = torch.ones(1, device=dev)
    before = float(evaluate(champ, one, 32, steps, seed=12345, theta=theta.detach())[0, 0])
    opt = torch.optim.Adam([theta], lr=training.finetune_lr)
    B, H = 32, training.finetune_horizon
    for it in range(training.finetune_steps):
        s = SwingUp.reset(B, dev, gen)[None]
        alive = torch.ones(1, B, dtype=torch.bool, device=dev)
        ret = torch.zeros(1, B, device=dev)
        for _ in range(H):
            s, alive, ret = rollout_step(s, alive, ret, comp.adj_f * theta, comp.act_masks,
                                         comp.depth_masks, one.view(1, 1, 1), comp.out_idx)
        loss = -ret.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], 1.0)    # BPTT through chaotic dynamics
        opt.step()
    after = float(evaluate(champ, one, 32, steps, seed=12345, theta=theta.detach())[0, 0])
    log.info(f"\nBPTT fine-tuning ({training.finetune_steps} updates, horizon {H}, start w = "
             f"{float(w_best):+.1f}): return {before:.1f} -> {after:.1f} (32 fresh episodes)")
    return (theta * comp.adj_f).detach()
