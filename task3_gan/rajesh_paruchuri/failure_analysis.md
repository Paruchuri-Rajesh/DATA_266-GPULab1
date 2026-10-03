# Task 3 — Failure analysis (Rajesh Paruchuri)

Model: CycleGAN with 6-block / ngf-48 generators, spectral-norm PatchGANs, DiffAugment, and EMA generators (`checkpoints/cyclegan_full.pt`, epoch 80).
Evidence:
- `outputs/translation_cycle_grid_full.png`
- `outputs/previews/full_epoch*.png` (the same 3 Monet and 3 photos every 5 epochs)
- the 30 audit images in `outputs/human_audit/`
- the metrics in `metrics_report.csv`

Every observation below is visible in those files.

---

## 1. Hidden "cycle information" blobs (both directions)
**What it looks like.** Small, high-contrast **black or saturated-green spots** appear on objects:
- the horse and carriage (`audit_09`);
- the hillside of the framed landscape (`audit_14`);
- the water in several Monet→photo outputs (`translation_cycle_grid_full.png`, column "→ photo").

The same spots reappear in the cycle reconstruction.

**Why.** This is the known CycleGAN steganography failure (Chu et al., 2017).
- The cycle loss rewards G_BA(G_AB(x)) ≈ x. When a translation has to discard information (photos have more detail than Monet paintings, and the reverse for colour), the generator can hide it in a low-amplitude or very local signal that the other generator learns to decode.
- My cycle L1 is very low (0.077 in both directions), while the translation itself changes the image much more (L1 0.106 / 0.084).
- LPIPS between input and reconstruction (0.46–0.48) is almost as high as between input and translation (0.47–0.52). The reconstruction matches pixels well but not perceptually, which is consistent with information carried in an encoded form rather than in the visible content.
- The discriminators do not punish the spots enough: DiffAugment's cutout and translation make a 10-pixel blob easy to miss for a 70×70 PatchGAN.

**Testable fix.** Add Gaussian noise (σ ≈ 0.05) to G's output before feeding it to the other generator in the cycle path.
Hidden low-amplitude codes no longer survive, so the generator has to keep the information visibly.
Measure: count of dark-blob pixels (luminance < 0.05 inside a bright neighbourhood) per image, and photo→Monet FID, before and after.

## 2. Bright light sources become white blobs with rainbow halos
**What it looks like.**
- Suns and specular highlights become **flat white blobs** surrounded by **purple, pink or green colour fringes** (`audit_01`, `audit_27`; previews from epoch 10 onward).
- In sunsets the sky gets concentric colour banding.

**Why.**
- Monet's paintings have almost no clipped (pure-white) regions; the paintings render light as coloured strokes. The generator has no learned mapping for saturated input, so tanh saturates and the network fills the area with whatever colour combination keeps D_A unsure.
- Training on 128×128 crops makes this worse: a sun disc often fills most of a crop, so the generator rarely sees a light source together with the sky around it.

**Testable fix.** Fine-tune the last 10 epochs on full 256×256 images (or 192 crops), so light sources are seen in context.
Evaluate on the slice of photos with more than 1% clipped pixels (mean luminance of the brightest 1% above 0.98): compare FID on that slice and the share of output pixels outside the Monet colour gamut, before and after.

## 3. Content outside the Monet domain is copied or garbled, not stylised
**What it looks like.**
- A photo with a **white passe-partout frame** keeps a ghostly frame (`audit_14`).
- A **text watermark** ("Jeffrey Kaphan Photography") becomes scribbled marks (`audit_27`).
- A **bright pink house** and a **metal sculpture** keep photographic edges and colours instead of becoming brushwork (`audit_20`, `audit_01`).
- Mechanical objects (the carriage wheels in `audit_09`) lose structure.

**Why.** The 300 Monet paintings are landscapes, water, gardens and boats. The generator has never seen frames, text, neon colours or close-up machinery in the target domain, so it either passes them through (identity loss pushes towards keeping them) or breaks them into texture.
This also explains the low **precision** of photo→Monet (0.25): a quarter of generated images fall inside the real-Monet manifold, while recall is higher (0.52), meaning the outputs cover much of Monet's variety without each being convincing.

**Testable fix.** Lower the identity-loss weight from 5 to 2.5 after epoch 40 (the original paper's weight is 0.5·λ_cyc, and annealing it is common), so out-of-domain colours are pushed towards the Monet palette.
Measure precision and FID on the 300-image official subset, and re-score `audit_01/09/14/20/27`.

## 4. Monet→photo stays painterly (weak direction)
**What it looks like.** Monet→photo outputs keep brush texture and soft edges; they look like slightly sharpened paintings, not photographs.
Their FID against real photos is **148.8** on all 300 paintings (169.8 on the official 300-vs-300 subset), compared with 105.7 for photo→Monet.

**Why.** It is the harder direction: inventing photographic detail from a painting is under-determined.
With ngf 48 and 6 residual blocks, my generator has about 40% of the parameters of the standard 9-block / ngf-64 one, so it has less capacity to synthesise detail.
The cycle loss also rewards keeping the painting's structure.

**Testable fix.** Raise G_AB capacity only (9 blocks, ngf 64) and keep G_BA small. The Kaggle-relevant direction is not slowed, and the A2B FID change isolates the capacity effect.

## 5. Border streaks
**What it looks like.** Thin **green or magenta streaks along the left or top edge** of several outputs (`translation_cycle_grid_full.png`, rows 1–3; `audit_01`, `audit_27`).

**Why.** Training used random 128×128 crops, and only crops that touch a border see reflection-padded edges. Most crops are interior, so the edge behaviour is under-trained, and at full-image inference every image has four edges.

**Testable fix.** Bias crop sampling so 25% of crops touch an image border, or fine-tune on full images.
Measure the mean colour deviation of the outer 4-pixel band compared with the next band in, across the 300 audit-subset outputs.

---

## Training-stability observations (not failures)
- **0 NaN/Inf steps** in 24,000 iterations. The pre-clip gradient norms stayed bounded: G mean 44, max 100 (only at the start); D mean 5.5, max 8.6. No clipping was needed.
- **No discriminator collapse.** Mean D outputs settled at about 0.57 (real) and 0.43 (fake) for both domains from epoch 15 to the end. Neither D won (outputs near 1/0) nor gave up (both near 0.5).
  With 300 Monet paintings and no augmentation, D_A typically separates them almost perfectly; DiffAugment kept it from doing so.
- **Under-training, not instability, limits quality.** Cycle and identity losses were still falling at epoch 80 (cycle 0.82 → 0.36, unweighted sum over both directions), even as the learning rate decayed to 0.
  More iterations would probably help more than any architecture change.
