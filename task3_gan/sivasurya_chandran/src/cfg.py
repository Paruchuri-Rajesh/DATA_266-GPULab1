"""
Config loading with named training variants.

config.yaml holds one base recipe plus a `variants:` section. Each variant overrides a few training or
discriminator settings; `--variant NAME` merges those overrides and moves the run's outputs to
checkpoints/runs/NAME and logs/train_raw_NAME.log so variants never overwrite each other. Variants may not
change the generator architecture, so every variant's G_AB loads with the base config (evaluate/translate/audit).
"""
import os

import yaml

GENERATOR_KEYS = {"ngf", "n_resblocks", "upsample"}


def _merge(base: dict, override: dict):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


def load_config(path: str, variant: str = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    variants = cfg.pop("variants", None) or {}
    if variant:
        if variant not in variants:
            raise SystemExit(f"unknown variant {variant!r}; {path} defines {sorted(variants)}")
        override = variants[variant] or {}
        clash = GENERATOR_KEYS & set(override.get("model", {}))
        if clash:
            raise SystemExit(f"variant {variant} changes the generator architecture ({sorted(clash)}); not allowed")
        _merge(cfg, override)
        t = cfg["train"]
        t["out_dir"] = os.path.join(t["out_dir"], "runs", variant)
        root, ext = os.path.splitext(t["log_file"])
        t["log_file"] = f"{root}_{variant}{ext}"
        cfg["variant"] = variant
    return cfg


def variant_names(path: str) -> list:
    with open(path) as f:
        return list((yaml.safe_load(f).get("variants") or {}).keys())
