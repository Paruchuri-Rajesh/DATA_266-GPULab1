"""
Post-hoc fix for Task 1 runs made before train.py filtered fp16 overflow steps
(2026-09-30). With AMP, the GradScaler occasionally overflows; that step's
gradient norm is inf and the step is skipped. Older runs averaged those inf
values into grad_norm_mean/max. This recomputes the stats over finite norms
from the full per-step list in checkpoints/train_state.pt and updates
metrics_train.json. The original values are kept under *_incl_overflow and the
raw log is not touched.

  python tools/fix_grad_norm_stats.py task1_llm/sivasurya_chandran
"""
import json
import os
import sys

import numpy as np
import torch

member_dir = sys.argv[1]
ckpt = os.path.join(member_dir, "checkpoints")
st = torch.load(os.path.join(ckpt, "train_state.pt"), map_location="cpu", weights_only=False)
g = np.array(st["grad_norms"])
fin = g[np.isfinite(g)]
path = os.path.join(ckpt, "metrics_train.json")
m = json.load(open(path))
if "grad_norm_stats_source" in m:
    sys.exit(f"{path} already fixed")
for k in ("grad_norm_mean", "grad_norm_max", "grad_norm_p95"):
    m[f"{k}_incl_overflow"] = m[k]
m.update(
    grad_norm_mean=float(fin.mean()),
    grad_norm_max=float(fin.max()),
    grad_norm_p95=float(np.percentile(fin, 95)),
    amp_overflow_skipped_steps=int((~np.isfinite(g)).sum()),
    grad_norm_stats_source="recomputed over finite per-step norms from checkpoints/train_state.pt by tools/fix_grad_norm_stats.py",
)
json.dump(m, open(path, "w"), indent=2)
print({k: m[k] for k in ("grad_norm_mean", "grad_norm_p95", "grad_norm_max", "amp_overflow_skipped_steps")})
