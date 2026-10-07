"""
Hydra Experiment Sweep Generator
=================================
Generates Hydra experiment override YAML files from a sweep definition.

Sweep definition YAML format:
    sweep_name: my_sweep              # output folder under config/experiments/
    groups:                           # config-group selections for ALL experiments
      environment: spirals
    fixed:                            # values shared by ALL experiments
      enas:
        max_gen: 50
    sweep:                            # parameters to sweep (Cartesian product)
      seed: [0, 1, 2]
      enas.pop_size: [128, 512]
    zip:                              # parameters linked by index (optional)
      training.learning: [none, baldwinian]
      training.inner_steps: [0, 10]

Usage:
    python config/generate_experiments/generator.py -e config/generate_experiments/my_sweep_def
    python config/generate_experiments/generator.py -e config/generate_experiments/my_sweep_def --dry-run
"""

import argparse
import os
import copy
from itertools import product
from collections import OrderedDict

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_nested(d: dict, dotted_key: str, value):
    """
    Set a value in a nested dict using a dot-separated key.
    e.g. set_nested(d, "enas.pop_size", 50) → d['enas']['pop_size'] = 50
    Top-level keys (no dot) are set directly on d.
    """
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def build_experiment_name(sweep_name: str, combo: dict, index: int) -> str:
    """
    Build a descriptive file-safe experiment name from the swept parameters.
    Falls back to a zero-padded index if the name would be too long.
    """
    parts = [sweep_name]
    for key, val in combo.items():
        short_key = key.split(".")[-1]
        parts.append(f"{short_key}{val}")

    name = "_".join(parts)

    if len(name) > 120:
        name = f"{sweep_name}_{str(index).zfill(4)}"

    return name


def write_experiment_yaml(path: str, params: dict, groups: dict = None):
    """
    Write a Hydra experiment override YAML with the # @package _global_ header.
    Only non-empty groups are written.  Sections are separated by blank lines
    and preceded by a comment for readability.
    """
    # Ordered sections matching the Hydra config groups
    sections = OrderedDict([
        ("environment", "Environment overrides"),
        ("enas",        "ENAS overrides"),
        ("model",       "Model overrides"),
        ("training",    "Training overrides"),
    ])

    with open(path, "w") as f:
        f.write("# @package _global_\n")
        f.write("#\n")
        f.write(f"# Auto-generated experiment configuration\n")
        f.write(f"# Defaults are loaded first, then these values are applied on top.\n")

        # Config-group selections, e.g. environment: spirals
        if groups:
            f.write("\ndefaults:\n")
            for group, option in groups.items():
                f.write(f"  - override /{group}: {option}\n")

        # Write grouped sections (environment, enas, model, training)
        for group, comment in sections.items():
            if group in params and params[group]:
                f.write(f"\n# {comment}\n")
                yaml.dump(
                    {group: params[group]},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )

        # Write top-level (global) overrides: seed, log_filename, etc.
        global_keys = [k for k in params if k not in sections]
        if global_keys:
            f.write("\n# Global overrides\n")
            for k in global_keys:
                yaml.dump(
                    {k: params[k]},
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_sweep(sweep_def_path: str, dry_run: bool = False):
    """Generate experiment YAML files from a sweep definition."""

    with open(sweep_def_path, "r") as f:
        config = yaml.safe_load(f)

    sweep_name = config["sweep_name"]
    groups = config.get("groups", {})
    fixed = config.get("fixed", {})
    sweep_params = config.get("sweep", {})
    zip_params = config.get("zip", {})

    # ---- Cartesian product of sweep parameters ----
    sweep_keys = list(sweep_params.keys())
    sweep_values = [sweep_params[k] for k in sweep_keys]

    if sweep_values:
        cartesian_combos = list(product(*sweep_values))
    else:
        cartesian_combos = [()]

    # ---- Zipped parameters (linked, not crossed) ----
    zip_keys = list(zip_params.keys())
    if zip_keys:
        zip_values = [zip_params[k] for k in zip_keys]
        lengths = [len(v) for v in zip_values]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"All 'zip' parameter lists must have the same length. "
                f"Got lengths: {dict(zip(zip_keys, lengths))}"
            )
        zip_combos = list(zip(*zip_values))
    else:
        zip_combos = [None]

    # ---- Generate all experiments ----
    # Always output to <project_root>/config/experiments/<sweep_name>/
    # Resolve project root from this script's location: config/generate_experiments/
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    output_dir = os.path.join(project_root, "config", "experiments", sweep_name)

    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    experiments = []
    index = 0

    for cart_combo in cartesian_combos:
        for zip_combo in zip_combos:
            index += 1

            # Start from fixed params
            params = copy.deepcopy(fixed)

            # Build a flat combo dict for naming
            combo_dict = {}

            # Apply cartesian sweep values
            for key, val in zip(sweep_keys, cart_combo):
                set_nested(params, key, val)
                combo_dict[key] = val

            # Apply zipped values
            if zip_combo is not None:
                for key, val in zip(zip_keys, zip_combo):
                    set_nested(params, key, val)
                    combo_dict[key] = val

            # Build name and set log path
            exp_name = build_experiment_name(sweep_name, combo_dict, index)
            params.setdefault("log_filename", f"logs/{sweep_name}/{exp_name}.log")
            params.setdefault("out_dir", f"experiments/{sweep_name}/{exp_name}")

            # Write YAML
            out_path = os.path.join(output_dir, f"{exp_name}.yaml")
            experiments.append((exp_name, out_path, params))

            if not dry_run:
                write_experiment_yaml(out_path, params, groups)

    # ---- Summary ----
    print(f"Sweep: {sweep_name}")
    print(f"Output: {output_dir}")
    print(f"Experiments: {len(experiments)}")
    print()
    for exp_name, out_path, params in experiments:
        status = "[DRY RUN]" if dry_run else "[CREATED]"
        swept = {k: params.get(k.split(".")[-1], "?")
                 for k in list(sweep_keys) + list(zip_keys)}
        print(f"  {status} {exp_name}")
    print()
    print(f"Run all:    python wann_enas.py sweep={sweep_name}")
    print(f"Run single: python wann_enas.py experiments={sweep_name}/{experiments[0][0]}")

    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="Generate Hydra experiment sweep YAML files"
    )
    parser.add_argument(
        "-e", "--exp_yaml", type=str, required=True,
        help="Path to sweep definition YAML (without .yaml extension)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview experiments without writing files",
    )
    args = parser.parse_args()

    sweep_def_path = args.exp_yaml
    if not sweep_def_path.endswith(".yaml"):
        sweep_def_path += ".yaml"

    if not os.path.isfile(sweep_def_path):
        raise FileNotFoundError(f"Sweep definition not found: {sweep_def_path}")

    generate_sweep(sweep_def_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
