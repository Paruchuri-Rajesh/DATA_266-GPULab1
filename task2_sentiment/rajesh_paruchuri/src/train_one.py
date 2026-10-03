"""Train one Task 2 model in its own process: python train_one.py <model_name> <full|smoke>

The notebook calls this once per model. A fresh process per model returns all MPS memory
to the OS between models; in a single long-lived kernel the cache pool kept growing.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sentiment_lib as sl


def main(name, tag):
    P = sl.find_paths(Path(__file__).resolve().parent)
    cfg = sl.load_config(P)
    cfg["smoke"] = tag == "smoke"
    d = np.load(P["data_proc"] / f"train_arrays_{tag}.npz")
    V = int(d["vocab_size"])
    data = (d["tr_ids"], d["tr_lens"], d["tr_y"], d["va_ids"], d["va_lens"], d["va_y"])
    sl.set_seed(cfg["seed"])
    device = sl.pick_device()
    model = sl.build_model(name, cfg["models"][name], V, cfg).to(device)
    print(f"{name}: {sl.count_params(model):,} parameters on {sl.hardware_string(device)}", flush=True)
    hist, summary = sl.train_model(name, model, data, cfg, device,
                                   P["logs"] / f"{name}_{tag}.log", P["ckpt"] / f"{name}_{tag}.pt")
    with open(P["out"] / f"{name}_history_{tag}.json", "w") as f:
        json.dump({"hist": hist, "summary": summary}, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
