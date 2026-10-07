# enas_tensor
New ENAS engine with tensors and backpropagation treinable weights accelerated by GPU.

Tensorized Weight Agnostic Neural Networks (WANN, Gaier & Ha, NeurIPS 2019) in PyTorch. The whole population lives as padded tensors on one device, so evaluation, mutation and ranking run as batched GPU kernels. Topologies are evolved; per-edge weights can optionally be learned by backprop (Baldwinian / Lamarckian inner loop, or post-search fine-tuning). Configuration uses [Hydra](https://hydra.cc/), with the same layout as DM_ENAS.

## Project Structure

```
enas_tensor/
├── wann_enas.py                        # Main entry point
├── requirements.txt
│
├── config/
│   ├── config.yaml                     # Main Hydra config (defaults + global settings)
│   ├── environment/                    # Task to solve (pick one)
│   │   ├── swingup.yaml                # CartPole swing-up (default)
│   │   ├── spirals.yaml                # Two-spirals classification
│   │   └── digits.yaml                 # sklearn 8x8 digits classification
│   ├── enas/default.yaml               # Population, selection, mutation rates
│   ├── model/default.yaml              # Node slots, initial connectivity, shared weights
│   ├── training/default.yaml           # Inner-loop learning and champion fine-tuning
│   ├── experiments/                    # Experiment override files
│   │   ├── swingup_quick.yaml
│   │   ├── spirals_baldwinian.yaml
│   │   ├── digits_lamarckian.yaml
│   │   └── sweep_seeds/                # Example generated sweep
│   └── generate_experiments/           # Sweep generator tool
│       ├── generator.py
│       ├── sweep_seeds.yaml
│       └── sweep_learning_seeds.yaml
│
├── source/
│   ├── wann_engine/                    # Core tensorized WANN algorithm
│   │   ├── _activations.py             # Activation set, shared-weight values
│   │   ├── _population.py              # Population tensors, depth compilation, describe()
│   │   ├── _forward.py                 # Batched forward pass (+ scalar reference)
│   │   ├── _variation.py               # Vectorized mutation
│   │   ├── _selection.py               # Pareto fronts + tournament selection
│   │   └── _training.py                # Per-edge weight training (Adam, CE loss)
│   ├── environments/
│   │   ├── swingup.py                  # Torch swing-up physics, rollout, evaluate()
│   │   └── datasets.py                 # spirals / digits datasets
│   ├── runners/
│   │   ├── swingup.py                  # Evolution loop + BPTT fine-tuning (RL)
│   │   ├── supervised.py               # Evolution loop + none/baldwinian/lamarckian
│   │   └── diagnostics.py              # selftest, benchmark
│   └── helpers/
│       └── helpers.py                  # Config loading, sweep runner, logging, checkpoints
│
├── logs/                               # Run logs (auto-created, git-ignored)
└── experiments/                        # champion.pt + parameters_used.yaml per run (git-ignored)
```

## Installation

```bash
python -m venv ~/.venv/dm_venv
source ~/.venv/dm_venv/bin/activate
pip install -r requirements.txt
```

## Configuration System

Base parameters are loaded from four config groups, then experiment-specific overrides are applied on top:

```
environment/swingup → enas/default → model/default → training/default → config.yaml globals → experiment overrides
```

Values left as `null` in `model/default.yaml` and `training/default.yaml` are task-dependent and resolved at runtime (e.g. `max_nodes` is 64 for swing-up, `inputs + outputs + 48` for supervised tasks). The resolved config is saved as `parameters_used.yaml` next to the champion.

## Running

| Command | What it does |
|---------|--------------|
| `python wann_enas.py mode=selftest` | Kernel + gradient checks |
| `python wann_enas.py` | Evolve swing-up with defaults |
| `python wann_enas.py experiments=swingup_quick` | Quick CPU demo |
| `python wann_enas.py compile=true` | `torch.compile` the rollout step (GPU) |
| `python wann_enas.py mode=benchmark` | Rollout throughput per device |
| `python wann_enas.py environment=spirals` | Pure WANN + fine-tuning |
| `python wann_enas.py environment=spirals training.learning=baldwinian` | Baldwinian inner loop |
| `python wann_enas.py environment=digits training.learning=lamarckian` | Lamarckian inner loop |
| `python wann_enas.py training.finetune_steps=100` | Swing-up + BPTT fine-tuning |
| `python wann_enas.py sweep=sweep_seeds` | Run every experiment in a sweep directory |
| `python wann_enas.py sweep=sweep_seeds device=cpu` | Sweep with a global override |

### Mapping from the old `wann_torch.py` flags

| Old flag | Hydra override |
|----------|----------------|
| `--task` | `environment=` |
| `--selftest` / `--benchmark` | `mode=selftest` / `mode=benchmark` |
| `--pop`, `--gens`, `--elite`, `--tournament`, `--p-complexity`, `--eval-batch` | `enas.pop_size`, `enas.max_gen`, `enas.elite`, `enas.tournament_size`, `enas.p_complexity`, `enas.eval_batch` |
| `--episodes`, `--steps` | `environment.episodes`, `environment.steps` |
| `--max-nodes`, `--p-init-conn` | `model.max_nodes`, `model.p_init_conn` |
| `--learning`, `--inner-steps`, `--inner-lr`, `--batch` | `training.learning`, `training.inner_steps`, `training.inner_lr`, `training.batch` |
| `--finetune-steps`, `--finetune-lr`, `--finetune-horizon` | `training.finetune_steps`, `training.finetune_lr`, `training.finetune_horizon` |
| `--seed`, `--device`, `--compile`, `--log-every` | `seed`, `device`, `compile`, `log_every` |
| `--out` | `out_dir` (champion saved as `<out_dir>/champion.pt`) |

## Generating Experiment Sweeps

Write a sweep definition in `config/generate_experiments/`:

```yaml
sweep_name: my_sweep

groups:                    # config-group selections for all experiments
  environment: spirals

fixed:                     # copied into every experiment
  enas:
    max_gen: 50

sweep:                     # Cartesian product
  seed: [0, 1, 2]
  enas.pop_size: [128, 512]

zip:                       # linked by index (lists of equal length)
  training.learning: [none, baldwinian]
  training.inner_steps: [0, 10]
```

```bash
python config/generate_experiments/generator.py -e config/generate_experiments/my_sweep --dry-run
python config/generate_experiments/generator.py -e config/generate_experiments/my_sweep
python wann_enas.py sweep=my_sweep
```

Generated files go to `config/experiments/<sweep_name>/`, each with its own `log_filename` and `out_dir`.

## Writing Experiment Configs Manually

Create a YAML file in `config/experiments/` with the `# @package _global_` header, containing only values that differ from the defaults:

```yaml
# @package _global_
defaults:
  - override /environment: spirals

enas:
  pop_size: 256
training:
  learning: baldwinian
seed: 42
log_filename: logs/my_experiment.log
```

```bash
python wann_enas.py experiments=my_experiment
```
