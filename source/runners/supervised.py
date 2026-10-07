import os
import time
import logging

import torch
from omegaconf import DictConfig

from source.environments import make_dataset
from source.helpers.helpers import (LOGGER_NAME, resolve_device, save_champion,
                                    update_config)
from source.wann_engine import (ce_acc, compile_population, describe, init_population,
                                logits_edges, logits_shared, next_generation,
                                test_acc_edges, train_edges)

log = logging.getLogger(LOGGER_NAME)


def resolve_defaults(cfg: DictConfig, n_in: int, C: int) -> DictConfig:
    """Fill task-dependent values left as null in the YAML defaults."""
    model, training = cfg.model, cfg.training
    return update_config(cfg, {
        "model.max_nodes": model.max_nodes or (n_in + C + 48),
        "model.p_init_conn": (model.p_init_conn if model.p_init_conn is not None
                              else min(0.25, 20 / (n_in * C))),
        "training.finetune_steps": (500 if training.finetune_steps is None
                                    else training.finetune_steps),
    })


def evolve_supervised(cfg: DictConfig):
    """Evolve topologies on a classification task, with an optional inner loop:
      none        -- pure WANN: fitness = -CE averaged over the shared weights
      baldwinian  -- train per-edge weights for k steps starting from each genome's
                     best shared weight; fitness = post-training -CE; trained weights
                     are discarded (only the topology is inherited)
      lamarckian  -- genomes carry per-edge weights; they are trained for k steps
                     every generation and written back, so offspring inherit them
    Secondary objective (alternating with -connections, as in WANN):
      none -> max over shared weights; baldwinian -> untrained weight-agnostic
      score (keeps the WANN prior); lamarckian -> the primary itself.
    """
    env, enas = cfg.environment, cfg.enas
    dev = resolve_device(cfg.device)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    Xtr, ytr, Xte, yte, C = make_dataset(env.name, cfg.seed, env.n_samples)
    Xtr, ytr, Xte, yte = Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)
    n_in = 1 + Xtr.shape[1]
    resolve_defaults(cfg, n_in, C)
    model, training = cfg.model, cfg.training
    mode = training.learning
    pop = init_population(enas.pop_size, n_in, C, model.max_nodes, model.p_init_conn, dev, gen,
                          with_weights=(mode == "lamarckian"))
    weights = torch.tensor(list(model.shared_weights), device=dev)
    log.info(f"task={env.name} (train {len(ytr)}, test {len(yte)}, {C} classes)  "
             f"learning={mode}  device={dev}  pop={enas.pop_size}  max_nodes={model.max_nodes}"
             + (f"  inner_steps={training.inner_steps}" if mode != "none" else ""))
    best = (-float("inf"), None)
    for g in range(enas.max_gen):
        t0 = time.perf_counter()
        idx = torch.randint(0, len(ytr), (min(enas.eval_batch, len(ytr)),),
                            device=dev, generator=gen)
        Xb, yb = Xtr[idx], ytr[idx]
        comp = compile_population(pop)
        if mode != "lamarckian":
            with torch.no_grad():
                ce_w, acc_w = ce_acc(logits_shared(comp, Xb, weights), yb)   # (P,W)
            agn = -ce_w
        if mode == "none":
            primary, alt, acc = agn.mean(1), agn.max(1).values, acc_w.mean(1)
        else:
            if mode == "baldwinian":
                w_best = weights[agn.argmax(1)]
                theta0 = comp.adj_f * w_best[:, None, None]
            else:
                theta0 = pop.weight
            theta = train_edges(comp, theta0, Xtr, ytr, training.inner_steps,
                                training.inner_lr, training.batch, gen)
            with torch.no_grad():
                ce_t, acc = ce_acc(logits_edges(comp, theta, Xb), yb)
            primary = -ce_t
            if mode == "baldwinian":
                alt = agn.mean(1)
            else:
                alt = primary
                pop.weight = theta               # Lamarckian write-back
        i = int(primary.argmax())
        if float(primary[i]) > best[0]:
            best = (float(primary[i]), pop.index(torch.tensor([i], device=dev)))
        if g % cfg.log_every == 0 or g == enas.max_gen - 1:
            log.info(f"gen {g:4d} | best -CE {float(primary[i]):7.3f}  acc {float(acc[i]):.3f}  "
                     f"pop acc {float(acc.mean()):.3f} | conns {int(pop.n_connections()[i]):3d} "
                     f"hidden {int(pop.n_hidden()[i]):3d} | {time.perf_counter() - t0:5.2f}s")
        pop, _ = next_generation(pop, primary, alt, gen, enas.elite, enas.tournament_size,
                                 enas.p_complexity, enas.prob_add_node, enas.prob_add_conn)

    champ = best[1]
    theta = finetune_supervised(champ, Xtr, ytr, Xte, yte, weights, cfg, gen)
    log.info("topology (node [activation] <- inputs):")
    log.info(describe(champ, 0) if champ.max_nodes <= 96 else
             f"  {int(champ.n_connections()[0])} connections, {int(champ.n_hidden()[0])} hidden")
    save_champion(champ, os.path.join(cfg.out_dir, "champion.pt"), theta)


def finetune_supervised(champ, Xtr, ytr, Xte, yte, weights, cfg: DictConfig, gen):
    """Post-search: report weight-agnostic test accuracy, then fine-tune per-edge weights."""
    training = cfg.training
    comp = compile_population(champ)
    log.info("\nchampion test accuracy")
    if champ.weight is None:
        with torch.no_grad():
            ce_tr, _ = ce_acc(logits_shared(comp, Xtr, weights), ytr)
            _, acc_te = ce_acc(logits_shared(comp, Xte, weights), yte)
        for w, a in zip(weights.tolist(), acc_te[0].tolist()):
            log.info(f"  shared w = {w:+.1f}  acc {a:.3f}")
        w_best = weights[ce_tr[0].argmin()]             # chosen on TRAIN data
        theta0 = comp.adj_f * w_best
        log.info(f"  before fine-tuning (shared w = {float(w_best):+.1f}): "
                 f"{test_acc_edges(comp, theta0, Xte, yte):.3f}")
    else:
        theta0 = champ.weight
        log.info(f"  inherited Lamarckian weights: {test_acc_edges(comp, theta0, Xte, yte):.3f}")
    if training.finetune_steps <= 0:
        return theta0
    theta = train_edges(comp, theta0, Xtr, ytr, training.finetune_steps,
                        training.finetune_lr, training.batch, gen)
    log.info(f"  after {training.finetune_steps} Adam steps on per-edge weights: "
             f"{test_acc_edges(comp, theta, Xte, yte):.3f}")
    return theta
