# Task 3 run history (sivasurya_chandran)

The graded model is the final run (`config.yaml`, `python src/run_final.py`), which writes `checkpoints/`,
`logs/train_raw.log`, `outputs/` and `submission.csv`. The three earlier runs were moved, unchanged, into
`history/` folders on 2026-10-01 so the final run could use the standard paths. File contents were not edited;
paths recorded inside them (e.g. `checkpoints/G_AB.pt` in v1's `metrics_train.json`) refer to where the files
were when the run wrote them.

| Run | Config as run | Raw log | Weights, metrics, samples | Outputs | Uploaded CSV | Kaggle public (our scorer) |
|---|---|---|---|---|---|---|
| v1 | `configs/history/config_v1.yaml` | `logs/history/train_raw_v1.log` | `checkpoints/history/v1/` (was `checkpoints/`) | `outputs/history/v1/` | `kaggle_submissions/v1_submission.csv` | −49.257 |
| v2 | `configs/history/config_v2.yaml` | `logs/history/train_raw_v2.log` | `checkpoints/history/v2/` (was `checkpoints/v2/`) | `outputs/history/v2/` | `kaggle_submissions/v2_submission.csv` | −46.353 |
| v3 | `configs/history/config_v3.yaml` | `logs/history/train_raw_v3.log` | `checkpoints/history/v3/` (was `checkpoints/v3/`) | `outputs/history/v3/` | `kaggle_submissions/v3_submission.csv` | −45.970 |

The v1–v3 scores and Kaggle entries were computed with our own scorer, `evaluate_competition_stats.py`: photo→Monet only, all 7,038 translations against the competition's `real_stats.npz` (Inception-v3 with `transform_input=True`). The instructor's evaluation script was shared later and is used for the final run below.

- v1: Colab T4, 30K steps, transposed-conv upsampling. Held-out FID 101.9 (1,000 photos); all 7,038 photos FID 98.100, MiFID 0.414.
- v2: Colab T4, 50K steps, resize-conv upsampling, DiffAugment (colour, translation, cutout), all 300 Monet, λ_id 2.5, FID-based checkpoint selection. Best epoch 45: held-out FID 95.8; all photos FID 92.296, MiFID 0.410.
- v3: Apple M5, 12K steps, warm start from v2's best epoch, second LR cycle, translation-only DiffAugment (v3's config says 24 epochs/2 h; it was shortened to 12 epochs before launch when Colab quota ran out). Best epoch 11: held-out FID 95.3; all photos FID 91.537, MiFID 0.402.
- v1's 30 human-audit images in `checkpoints/history/v1/audit/` were generated from v1's G_AB; the final run prepares a new set from the final model.
- `submission/history/` (not committed) keeps the translated-image zips of v1–v3.

## Final run (RTX 4090, 2026-10-01/02)

`V2_CONTINUE`: v2's weights trained further with v3's recipe for 270 epochs × 1,000 steps (6.88 h, bf16), selecting raw/EMA generator pairs on held-out photos. It is promoted to the standard paths (`checkpoints/`, `logs/train_raw.log`, `outputs/`, `submission.csv`).
- Scored with the instructor's `Part3_Evaluation_Script.ipynb` (executed as `src/Part3_Evaluation_Script_run.ipynb`): FID 101.909 (photo→Monet 96.956, Monet→photo 106.862), MiFID 0.4082, i.e. −51.1585. This is `submission.csv`. The same scoring on the RTX 4090 right after training gave FID 101.963 / MiFID 0.4082 (`kaggle_submissions/final_cuda_run_submission*.{csv,json}`); the small difference is GPU vs CPU arithmetic in Inception-v3.
- `MSD_TEXTURE` (fresh two-scale discriminators, cycle weight 10 → 3) ran alongside it and was stopped at step 6,800 (its 7th epoch), when the run was restarted with `V2_CONTINUE` alone (its log is `logs/train_raw_MSD_TEXTURE.log`).
- An earlier attempt with `A_base` and `C_msD_cycdecay` was stopped at step 6,800 (7th epoch); its logs are in `logs/history/run_killed_20261001_1717/`. `logs/history/overnight_partial_copy/` is a mid-run copy of the final run's logs, kept as written. `logs/history/patchnce_continuation_log.txt` is a 24K-step PatchNCE continuation of the final model that was stopped because it scored worse.
- `uvcgan2/` holds a separate UVCGAN2-style experiment from the same session (code, configs, logs; weights not kept in the repo). It was not used for the submission.
