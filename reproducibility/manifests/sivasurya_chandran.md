# Environment manifest — Sivasurya Chandran

The raw logs under `reproducibility/raw_logs/sivasurya_chandran/` are byte-for-byte copies of each task's `logs/train_raw.log`. I haven't edited them; runs that were resumed append to the same log.

The machine-readable manifests, written by the training scripts, are:
- `sivasurya_chandran_task1_manifest.json`
- `sivasurya_chandran_task2_manifest.json`
- `sivasurya_chandran_task3_manifest.json`

Pinned versions are in `sivasurya_chandran_requirements.txt` (the Mac environment, Python 3.9). Task 1 ran on Colab and Task 3's final run on the lab RTX 4090, each with the versions listed in its manifest; `sivasurya_chandran_task3_requirements.txt` is the package list used on the lab machine.

All of my runs use seed 1337.

## Shared code used by my tasks
- `common/utils.py`: device selection, seeding, peak memory, hardware info and the manifest writer.
- `task2_sentiment/eval_metrics.py`: all classification metrics (bootstrap CIs, McNemar, ECE, slices).
- `task3_gan/eval_metrics.py`: FID, KID, precision/recall, density/coverage, LPIPS and cosine metrics for Task 3; `task3_gan/download_data.py` fetches the competition images.
- `tools/`:
  - `make_tables.py`, `export_reports.py`: build `metrics_report.csv` and copy logs and manifests here;
  - `fix_grad_norm_stats.py`: Task 1 gradient-norm statistics over finite steps;
  - `download_raw_data.py`: fetches TinyStories and Yelp polarity into the `data/` folders;
  - `colab.py`, `pack_results.py`: helpers for running on Colab.

## Run — Task 1 LLM (full)
- **Environment:** Google Colab with an NVIDIA Tesla T4 (2 vCPUs, Linux x86_64), Python 3.13, torch 2.11.0+cu128, fp16 autocast.
- **Dates:** 2026-09-30, 02:08 → 06:39 by the log timestamps, which are in Colab's clock (UTC). That includes a pause and resume after epoch 5.
- **Commands** (from `task1_llm/sivasurya_chandran/`):
  1. `python src/preprocess.py --config config.yaml`
  2. `python src/train.py --config config.yaml`, then `python src/train.py --config config.yaml --resume` after the Colab session dropped
  3. `python src/generate.py --config config.yaml --checkpoint checkpoints/gpt_final.pt`
- **Data:**
  - 110K TinyStories stories, seed 1337: 100K train / 10K validation.
  - 89.8M / 9.0M characters, vocabulary of 109 characters.
- **Model:** 6 layers, 8 heads, 256-d, block size 128, 4,827,648 parameters, 10 epochs (109,640 steps).
- **Checkpoint → result:** `checkpoints/gpt_final.pt` gives every number in `task1_llm/sivasurya_chandran/metrics_report.csv` and `results.md`.
- **Headline results:**
  - val CE 0.6385, perplexity 1.894, bits per character 0.921, top-1 accuracy 0.7955
  - 0 NaN steps, 0 loss spikes, 41 fp16-overflow steps skipped by the GradScaler
- **Cost:** 6,735.6 s of training at 133,346 tokens/s, with peak memory of 944 MB.
- **Raw log:** `reproducibility/raw_logs/sivasurya_chandran/task1_llm/train_raw.log`.
- **Notes on the run:**
  - The first resume attempts failed because the saved RNG state was loaded onto the GPU. I fixed it in `src/train.py`, and the successful resume retrained epoch 6 from its start.
  - All attempts are in the raw log.

## Run — Task 2 Sentiment (full)
- **Environment:** Apple M5 MacBook (10-core CPU, MPS), macOS 26.6, Python 3.9.6, torch 2.8.0, fp32.
- **Dates:** 2026-09-29, 18:24 → 23:59 (local time).
- **Commands** (from `task2_sentiment/sivasurya_chandran/`):
  1. `python src/preprocess.py --config config.yaml`
  2. `python src/train.py --config config.yaml`
  3. `python src/error_analysis.py --config config.yaml --model_name experimental_2`
- **Data:** Yelp polarity, 503,979 train / 55,997 validation / 38,000 test.
- **Models:** bag-of-embeddings MLP, multi-width CNN, BiLSTM + attention. All use from-scratch 128-d embeddings over a 50K vocabulary.
- **Checkpoints → results:** `checkpoints/{baseline,experimental_1,experimental_2}.pt` give the numbers in `task2_sentiment/sivasurya_chandran/metrics_report.csv` and `results.md`.
- **Headline test accuracy:** 0.9332 / 0.9502 / 0.9581.
- **Raw log:** `reproducibility/raw_logs/sivasurya_chandran/task2_sentiment/train_raw.log`.
- **Note on timing:** the laptop slept from 20:03 to 22:25 while the BiLSTM was training (seen in `pmset -g log`). Its logged 19,092 s therefore includes about 8,500 s of sleep. The log is unchanged, and `results.md` explains the correction.

## Run — Task 3 CycleGAN (final, `V2_CONTINUE`)
- **Environment:** lab PC with an NVIDIA GeForce RTX 4090 (24 GB), AMD CPU (32 threads), Windows 11, Python 3.9.13, torch 2.6.0+cu124, bf16 autocast.
- **Dates:** 2026-10-01 17:34 → 2026-10-02 02:02 (training), post-processing 2026-10-02 11:38–11:40, local time on the lab PC.
- **Command** (from `task3_gan/sivasurya_chandran/`): `python src/run_final.py` with a 13-hour budget and variants `V2_CONTINUE, MSD_TEXTURE`; restarted at 19:29 with `V2_CONTINUE` only (it resumed from its saved state); re-run at 11:37 with a 2-hour budget, which skipped the finished training and completed post-processing after the Windows encoding fix. Each start is recorded in `logs/final_run_status.log`.
- **Data:** 300 Monet paintings and 6,038 photos for training; 1,000 photos held out for selection and evaluation (seed 1337).
- **Model:** ResNet-9 generators (64 filters, resize-convolution upsampling) and 70×70 PatchGAN discriminators, 28,285,832 parameters; warm-started from v2 (`checkpoints/history/v2/`).
- **Training:** 270 epochs × 1,000 steps, 24,770 s, peak GPU memory 19,863 MB.
- **Checkpoint → result:** `checkpoints/{G_AB,G_BA,D_A,D_B}.pt` (epoch 269, raw weights) → `metrics_report.csv`, `full_metrics_report.csv`, `results.md`.
  - `outputs/pred_A2B` (300 Monet → photo) and `outputs/pred_B2A` (7,038 photos → Monet) → the instructor's evaluation notebook (`src/Part3_Evaluation_Script_run.ipynb`): FID 101.909, MiFID 0.4082 (`kaggle_submissions/final_submission.csv`, Kaggle −51.1585).
- **Stability:** 0 NaN steps, 0 skipped overflow steps.
- **Raw logs:** `reproducibility/raw_logs/sivasurya_chandran/task3_gan/` (a copy of the member folder's `logs/`):
  - `train_raw.log`: the final run (identical to `train_raw_V2_CONTINUE.log`);
  - `train_raw_MSD_TEXTURE.log`: the stopped second variant;
  - `history/`: v1–v3, the killed first attempt, a mid-run copy, and the PatchNCE continuation.
- **Kaggle entry (`submission.csv`):** v3, best epoch 11. Apple M5 MacBook (MPS), Python 3.9.6, torch 2.8.0 (`checkpoints/history/v3/manifest.json`). `checkpoints/history/v3/G_AB.pt` → `outputs/history/v3/pred_B2A` (7,038 images) → `python evaluate_competition_stats.py --gen outputs/history/v3/pred_B2A` → FID 91.537, MiFID 0.402 → public score −45.9695, team rank 11. Raw log: `logs/history/train_raw_v3.log`.
- **Earlier runs (v1–v3):** configs in `task3_gan/sivasurya_chandran/configs/history/`, metrics and samples in `checkpoints/history/`, uploaded CSVs in `kaggle_submissions/`; see `HISTORY.md`. v2's weights (the final run starts from them) and v3's weights (the Kaggle entry) are committed.

## Not in git
- `data_processed/`: the Task 2 arrays alone are about 600 MB. It is regenerated by `src/preprocess.py`.
- `train_state.pt` and the per-epoch GPT checkpoints. Only `gpt_best.pt` and `gpt_final.pt` are committed.
- The raw datasets. They are fetched by `tools/download_raw_data.py` or by `src/preprocess.py` (Tasks 1–2) and `task3_gan/download_data.py` (Task 3).
- Task 3: v1 weights, resume states, the duplicate weights under `checkpoints/runs/`, and the UVCGAN experiment's weights. These are kept in my OneDrive backup of the lab results.
