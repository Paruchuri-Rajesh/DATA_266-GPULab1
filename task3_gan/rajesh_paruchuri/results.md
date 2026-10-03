# Task 3 — CycleGAN photo ↔ Monet (Rajesh Paruchuri)

Code:
- `src/cyclegan.py`: models, DiffAugment, training loop;
- `src/evaluate.py`: metrics;
- `src/human_audit.py`: blinded audit and inter-rater agreement.

All are driven by `src/part3_cyclegan.ipynb` (executed, with outputs). The Kaggle numbers come from the course's `Part3_Evaluation_Script.ipynb`; its executed copy is `src/Part3_Evaluation_Script_run.ipynb`.
Config: `src/config.json`. Seed: 266.

**Integrity:** every image scored or submitted is the direct output of the EMA generators in `checkpoints/cyclegan_full.pt`. The images are not edited, hand-picked or filtered.
No pretrained network is used to train or generate. Inception-v3, VGG16 and LPIPS are used only in `evaluate.py`, to measure finished outputs.

## 3.1 Data
- **Source:** class competition `data-266-fall-2026-gan-image-style-transfer`, downloaded with the Kaggle CLI.
- **Contents:**
  - domain A = 300 Monet paintings;
  - domain B = 7,038 photos;
  - all 256×256 JPEG.
- **Unpaired:** each training step draws an independent random painting and an independent random photo.
- **Augmentation:** random 128×128 crop and horizontal flip.

## 3.1 Architecture and why
| Component | Mine | Standard CycleGAN (Zhu et al., 2017) | Why mine |
|---|---|---|---|
| Generators G_AB (Monet→photo), G_BA (photo→Monet) | ResNet, **ngf 48, 6 residual blocks**, InstanceNorm, reflection padding; decoder = **bilinear ×2 + 3×3 conv** | ngf 64, 9 blocks, transposed-conv upsampling | 4.41M params per G (vs 11.4M), so a laptop GPU can train 24K iterations. Resize-conv avoids checkerboard artifacts |
| Discriminators D_A, D_B | 70×70 PatchGAN C64-128-256-512-1, **spectral norm**, no norm layers | 70×70 PatchGAN with InstanceNorm | spectral norm bounds D's Lipschitz constant and stabilises the min-max game |
| Small-data handling | **DiffAugment** (colour, translation, cutout) on every real and fake D input | none | 300 Monet paintings is tiny; without augmentation D memorises them |
| Inference weights | **EMA** of G (decay 0.999) | raw G | averages out step-to-step GAN noise |
| Losses | LSGAN + λ_cyc 10 · cycle L1 + λ_id 5 · identity L1 | same family | the standard CycleGAN objective |
| Other | 50-image replay buffer | same | reduces oscillation |

Parameters: G 4,412,931 ×2 + D 2,764,737 ×2 = **14,355,336** (the standard 9-block / ngf-64 CycleGAN: about 28.3M).

**Training:**
- Adam lr 2e-4, β = (0.5, 0.999), batch 1;
- 80 epochs × 300 iterations = **24,000 iterations**;
- constant LR for 40 epochs, then linear decay to 0;
- inference on full 256×256 images (the networks are fully convolutional).

## 3.1 Training behaviour (convergence and stability)
`outputs/loss_curves_full.png`, raw log `reproducibility/raw_logs/rajesh_paruchuri/task3_gan/train_full.log`.

| Quantity (mean of final 500 iterations) | Value |
|---|---|
| G adversarial loss (both directions) | 0.645 |
| D_A / D_B loss | 0.214 / 0.216 |
| Cycle L1, unweighted A+B (start → end) | 0.82 → **0.358** |
| Identity L1, unweighted A+B (start → end) | 0.79 → **0.373** |
| Mean D output, real / fake (both domains) | ≈ 0.57 / ≈ 0.43 |
| G gradient norm, mean / max (pre-clip, no clipping) | 44.2 / 99.9 (max at iteration 50) |
| D gradient norm, mean / max | 5.5 / 8.6 |
| NaN/Inf steps | **0** |

- **Stable equilibrium:** from epoch 15 on, neither discriminator wins or collapses. Mean D outputs stay in a 0.4–0.6 band for both domains, and the G adversarial loss drifts slowly down without spikes.
  This is the intended effect of spectral norm plus DiffAugment.
- **Not converged:** cycle and identity losses fall steadily throughout and are still falling at epoch 80 while the LR decays to 0. Longer training is the most obvious improvement.
- **Visual progress** (`outputs/previews/full_epoch005…080.png`):
  - by epoch 10 the photos get the Monet palette;
  - by epoch 40 brush texture appears;
  - the blob and sun-halo artifacts appear by epoch 15 and never fully go away (see `failure_analysis.md`).

## 3.2 Evaluation (both directions)
**Distribution metrics.** Inception-v3 pool-3 features; my extractor reproduces the class `real_stats.npz` features to cosine 0.9999.

| Metric | photo → Monet (B2A, Kaggle direction) | Monet → photo (A2B) |
|---|---|---|
| FID, all images (7,038 vs 300 / 300 vs 7,038) | **105.72** | 148.80 |
| KID ± std (50 subsets of 100) | 0.0308 ± 0.0040 | 0.0776 ± 0.0061 |
| Precision / recall (k = 3) | 0.247 / 0.523 | 0.177 / 0.291 |
| Density / coverage | 0.195 / 0.537 | 0.101 / 0.059 |
| Min cosine distance to a real image (memorisation check) | 0.258 | 0.247 |
| **Official script FID / MiFID (first 300 vs 300)** | **129.21 / 0.3975** | **169.76 / 0.4296** |

Reference: the FID between real photos and real Monet is 110.7.
My photo→Monet outputs (105.7) are slightly closer to Monet than real photos are. That is a modest style shift, consistent with "Monet palette and texture, photographic structure".

**Per-image content and cycle metrics.** 300 random photos and all 300 Monet paintings; L1 in [0, 1] pixel units.

| Metric | photo→Monet→photo | Monet→photo→Monet |
|---|---|---|
| **Cycle-reconstruction L1** | **0.0774** | **0.0775** |
| L1 input vs translation | 0.1064 | 0.0842 |
| L1 input vs an unrelated image (reference) | 0.2810 | 0.2531 |
| LPIPS input vs reconstruction | 0.460 | 0.480 |
| LPIPS input vs translation | 0.523 | 0.468 |
| Content-preservation cosine (VGG16 relu4_3, input vs translation) | 0.373 | 0.424 |

**Cycle-consistency verification:**
- **The constraint is working.** Reconstructions are 3.3–3.6× closer to their inputs than unrelated images of the same domain are, and closer than the translations themselves.
  Visually (`outputs/translation_cycle_grid_full.png`), every reconstruction recovers the layout and colours of its input.
- **But it is partly satisfied by hidden codes.** LPIPS between input and reconstruction is nearly as high as for the translation, and dark blob artifacts reappear in reconstructions.
  So part of the cycle is carried by near-invisible marks (failure 1 in `failure_analysis.md`).

## 3.2 Kaggle submission
The course script averages the two directions on 300-vs-300 subsets:

```
ID,FID,MiFID
1,149.48371141836398,0.41354605555534363
```
(`submission.csv`)

| | |
|---|---|
| Team | **PairProgramming_Team_06** (with Siva Surya Chandran) |
| My submission | ref 56806123, uploaded 2026-10-03 with the Kaggle CLI from this `submission.csv` |
| **My public score** | **−74.9486** (Kaggle's score is −(FID + MiFID)/2, which matches the submitted values) |
| Private score | not shown until the competition closes |
| **Team leaderboard rank** | **11** (public). Kaggle ranks a team by its best submission, which is Siva's −45.9695 |

My entry scores well below Siva's, mainly because my Monet→photo direction is weak (official-script FID 169.8 vs 129.2 for photo→Monet).

The official-script FID (129.2 for photo→Monet) is higher than my all-image FID (105.7), because FID is biased upward with only 300 samples.
The "MiFID" in this script is the mean cosine distance between index-paired real and generated features, not Kaggle's memorisation-informed FID.

## 3.2 Blinded human audit
- **Selection:** 30 photos chosen with seed 266 *before* inspecting outputs.
- **Images:** shown as anonymous side-by-side images (`outputs/human_audit/audit_01…30.png`, shuffled order).
- **Scoring:** style, content and artifacts on a 1–5 scale (`RATER_GUIDE.md`).
- **Raters:** two, scoring independently in `audit_sheet.csv`.
- **Agreement:** % exact, % within-1, Cohen's kappa (unweighted and linear-weighted), computed by `human_audit.score()`.

**Status: pending — the two raters have not filled in the sheet yet.** No ratings were invented.

## Notes on the comparison
- On all 7,038 photos my photo→Monet FID is 105.7, with **half the parameters** of the standard CycleGAN.
  The cycle-reconstruction error is low (0.077) in both directions.
- My **Monet→photo direction is clearly weaker**, which pulls the two-direction submission average up to 149.5.
- The comparison with my Team 6 teammate's model (Siva Surya Chandran, best Kaggle entry −45.9695) belongs in the team report.

## Limitations and next steps
1. **Train longer.** Losses had not plateaued at 24K iterations.
2. **Fine-tune on full 256×256 images for the last epochs.** This targets the sun-halo and border artifacts caused by 128-px crops.
3. **Add a noise perturbation in the cycle path.** This targets hidden-code blobs.
4. **Give G_AB more capacity** (9 blocks / ngf 64) for the weak Monet→photo direction.
5. Every proposed fix has a measurable check in `failure_analysis.md`.

## Hardware (disclosure)
- **Machine:** Apple MacBook Air, **M4** (10-core CPU, 10-core GPU, 16 GB unified memory); PyTorch 2.11.0 MPS, fp32; macOS 26.5.2, Python 3.13.7.
- **Training:** 9,417 s (2.6 h), 5.1 images/s (Monet+photo crops).
- **Peak memory:** 409 MB MPS tensors, 1,009 MB MPS driver pool, 693 MB process RSS.
- **Inference:** photo→Monet at 10.6 images/s at 256×256.
