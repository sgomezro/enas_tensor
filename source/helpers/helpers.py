import os
import sys
import logging

import torch
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir

from source.wann_engine import ACT_NAMES

LOGGER_NAME = "wann"


def _get_config_dir():
    """Absolute path to <project_root>/config."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, "config")


#**********************************************************
#* Config functions
#**********************************************************
def load_config(overrides: list = None):
    """
    Load a single config using Hydra compose API.
    CLI args are passed directly as Hydra overrides.

    Returns:
        cfg: Hydra DictConfig (structured config)
    """
    config_dir = _get_config_dir()
    overrides = sys.argv[1:] if overrides is None else overrides

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    return cfg


def load_sweep(sweep_dir: str, extra_overrides: list = None):
    """
    Discover all experiment YAML files in a sweep directory and return
    a list of resolved configs — one per experiment file.

    The sweep directory is relative to config/experiments/.
    For example, sweep_dir="sweep_seeds" scans:
        config/experiments/sweep_seeds/*.yaml

    Args:
        sweep_dir:        Name of the subdirectory under config/experiments/
        extra_overrides:  Optional list of additional CLI-style overrides
                          applied on top of every experiment (e.g. ["seed=42"])

    Returns:
        list[DictConfig]: One resolved config per experiment file, sorted by filename.
    """
    config_dir = _get_config_dir()
    experiment_path = os.path.join(config_dir, "experiments", sweep_dir)

    if not os.path.isdir(experiment_path):
        raise FileNotFoundError(
            f"Sweep directory not found: {experiment_path}\n"
            f"Create it under config/experiments/{sweep_dir}/ with experiment YAML files."
        )

    yaml_files = sorted(
        f for f in os.listdir(experiment_path)
        if f.endswith(".yaml") or f.endswith(".yml")
    )

    if not yaml_files:
        raise FileNotFoundError(f"No YAML files found in {experiment_path}")

    configs = []
    for yaml_file in yaml_files:
        exp_name = f"{sweep_dir}/{os.path.splitext(yaml_file)[0]}"
        overrides = [f"experiments={exp_name}"]
        if extra_overrides:
            overrides.extend(extra_overrides)

        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name="config", overrides=overrides)
        configs.append(cfg)

        print(f"  Loaded experiment: {exp_name}")

    print(f"Sweep '{sweep_dir}': {len(configs)} experiments loaded.")
    return configs


def update_config(cfg: DictConfig, temp_dict: dict) -> DictConfig:
    """
    Update the config with derived fields that are only known at runtime
    (e.g. task-dependent defaults left as null in the YAML files).

    Parameters:
        cfg: Hydra DictConfig (structured config)
        temp_dict: dotted keys -> values, e.g. {"model.max_nodes": 64}
    Returns:
        cfg with updated derived fields
    """
    was_readonly = OmegaConf.is_readonly(cfg)
    OmegaConf.set_readonly(cfg, False)
    for key, value in temp_dict.items():
        OmegaConf.update(cfg, key, value, merge=False)
    OmegaConf.set_readonly(cfg, was_readonly)
    return cfg


def resolve_device(device: str = None, default: str = None) -> torch.device:
    """Explicit device, else `default`, else CUDA when available."""
    return torch.device(device or default or ("cuda" if torch.cuda.is_available() else "cpu"))


#**********************************************************
#* Logging functions
#**********************************************************
def setup_logger(log_file, name=LOGGER_NAME, level=logging.INFO, console=True):
    """
    Function to setup a logger; prints to file and optionally to console.
    Parameters:
        log_file: path to the log file (None = console only)
        name: name of the logger
        level: logging level
        console: whether to print to console
    Returns:
        logger: logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    logger.propagate = False
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handler = logging.FileHandler(log_file, mode='a')
        handler.setFormatter(logging.Formatter('%(asctime)s -- %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
    return logger


#**********************************************************
#* Checkpoint functions
#**********************************************************
def save_champion(champ, path, theta=None):
    d = {k: getattr(champ, k).cpu() for k in ("adj", "act", "rank", "active")}
    d |= {"n_in": champ.n_in, "n_out": champ.n_out, "act_names": ACT_NAMES}
    if theta is not None:
        d["edge_weights"] = theta.cpu()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(d, path)
    logging.getLogger(LOGGER_NAME).info(f"saved champion to {path}")


def save_run_config(cfg: DictConfig, out_dir: str):
    """Write the fully resolved config next to the run outputs."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "parameters_used.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
