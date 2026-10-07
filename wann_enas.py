#!/usr/bin/env python3
"""
wann_enas.py -- Tensorized Weight Agnostic Neural Networks (WANN) in PyTorch.

Reference: A. Gaier and D. Ha, "Weight Agnostic Neural Networks", NeurIPS 2019.

Gradient training (topology evolved, per-edge weights learned by backprop):
  * post-search fine-tuning of the champion (supervised; swing-up via BPTT)
  * training.learning=baldwinian : fitness after k Adam steps, weights not inherited
  * training.learning=lamarckian : genomes carry weights, trained weights inherited

Usage (Hydra overrides)
  python wann_enas.py mode=selftest                       # kernel + gradient checks
  python wann_enas.py                                     # evolve swing-up (defaults)
  python wann_enas.py experiments=swingup_quick           # quick CPU demo
  python wann_enas.py compile=true                        # torch.compile the step (GPU)
  python wann_enas.py mode=benchmark                      # rollout throughput per device
  python wann_enas.py environment=spirals                 # pure WANN + fine-tune
  python wann_enas.py environment=spirals training.learning=baldwinian
  python wann_enas.py environment=digits  training.learning=lamarckian
  python wann_enas.py training.finetune_steps=100         # swing-up + BPTT fine-tuning
  python wann_enas.py sweep=sweep_seeds                   # every experiment in a sweep dir
"""
import sys

from omegaconf import DictConfig, OmegaConf

from source.environments import SUPERVISED_TASKS
from source.helpers.helpers import (load_config, load_sweep, resolve_device,
                                    save_run_config, setup_logger)
from source.runners import benchmark, evolve, evolve_supervised, selftest


def run_wann(cfg: DictConfig):
    log = setup_logger(cfg.log_filename)

    if cfg.mode == "selftest":
        selftest(resolve_device(cfg.device, default="cpu"))
        return
    if cfg.mode == "benchmark":
        benchmark(cfg)
        return

    # Print config
    log.info("=" * 60)
    log.info(f"WANN for {cfg.environment.name}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log.info("=" * 60)

    if cfg.environment.name in SUPERVISED_TASKS:
        evolve_supervised(cfg)
    else:
        evolve(cfg)
    # Saved after the run so task-dependent defaults are filled in
    save_run_config(cfg, cfg.out_dir)


if __name__ == "__main__":
    # Detect sweep mode:  python wann_enas.py sweep=sweep_seeds
    sweep_arg = [a for a in sys.argv[1:] if a.startswith("sweep=")]

    if sweep_arg:
        sweep_dir = sweep_arg[0].split("=", 1)[1]
        extra = [a for a in sys.argv[1:] if not a.startswith("sweep=")]
        configs = load_sweep(sweep_dir, extra_overrides=extra if extra else None)
        for i, cfg in enumerate(configs, 1):
            print(f"\n{'#' * 60}")
            print(f"# Sweep experiment {i}/{len(configs)}")
            print(f"{'#' * 60}")
            run_wann(cfg)
    else:
        # Load config from CLI overrides (single-run mode)
        cfg = load_config()
        run_wann(cfg)
