# Task 3 failure analysis — Sivasurya Chandran

The visual cases below come from `checkpoints/eval_photo2monet_input_fake_rec.png` and `checkpoints/eval_monet2photo_input_fake_rec.png`. Each grid has eight columns:
- top row: the input;
- middle row: the translation;
- bottom row: the cycle reconstruction.

Numbers come from `checkpoints/metrics_eval.json`, `submission_details.json` and the logs.

## Photo → Monet (the stronger direction, FID 96.96 on the submission set)

Most translations pick up a visible brush texture, broken-up foliage and Monet's softer palette. Four kinds of failure come up repeatedly.

**1. Inventing colour on grey inputs.** A misty, almost colourless field (column 2) comes back tinted purple, with the treeline turned violet. The generator has learned that Monet paintings are colourful, and it adds colour even when the photo gives it none to work with.

**2. Losing contrast in dramatic skies.** The dark storm cloud (column 7) and the heavy sepia sky (column 3) come out pale and washed out. The shapes survive, but the mood of the photo is gone. The result looks like a generic light sky rather than one of Monet's darker ones.

**3. Texture on close-ups.** The macro shot of pebbles (column 5) keeps its shapes, but gets a noisy, speckled colour texture. Brushstrokes sized for a whole landscape don't make sense on small objects, so close-ups look worse than landscapes.

**4. Watermarks pass straight through.** Photographer credits and signatures stay readable in both the translation and the reconstruction: "© 2016 Thomas J. Astle" in column 7, the web address in column 3 and the signature in column 4. Nothing in the loss tells the model to remove text, and the cycle loss actively rewards keeping it.

**Why these failures exist.** The reconstructions (bottom row) are very close to the inputs (cycle L1 0.105), which shows the central trade-off. A strong cycle loss (weight 10) keeps the content safe, but it also pulls the translation back toward the photo and limits how far the style can move.

## Monet → photo (the weaker direction, FID 106.86 on the submission set)

**5. Everything gets darker.** The flower garden (column 1), the tree-lined road (column 3), the cliff with ruins (column 6) and especially Waterloo Bridge (column 8) all come back darker and muddier than the painting. The bridge turns into a near-black night scene under a purple sky.

**6. Subjects with no photo equivalent.** Column 4 is a still life of hanging game birds, painted indoors. The photo set is almost all outdoor landscapes, so the generator has nothing sensible to turn this into, and the output is just a darker, higher-contrast copy of the painting. This failure comes from the data: 300 paintings cover a wider range of subjects than 7,038 photos of scenery.

**7. Streaks in fine foliage.** The poplars reflected in water (column 7) break into white and blue streaks where the model can't turn painted leaves into photographic detail.

There is one success worth noting: the pale, misty poplars in column 5 turn into a convincing blue sky with clouds.

**What the metrics say about this direction.** They agree with what the images show. Monet → photo has good precision (0.70) but low coverage (0.36): individual outputs look like photos, but they all look like the same narrow kind of photo.

## Experiments that didn't work

**UVCGAN-style generators (`uvcgan2/`).** I implemented a UVCGAN-v2-style model:
- a U-Net generator with a small vision-transformer bottleneck;
- spectrally normalised discriminators;
- inpainting pre-training of the generators (about 49K steps).

Both adversarial pilot runs collapsed to the identity function: the "translated" images were almost the same as the inputs. Their selection FIDs (about 124.7–126.2) sat right at the score of untranslated images (126.5).

Two things pushed the model there:
- **The loss weights.** Cycle, identity and low-resolution consistency together outweighed the adversarial term, and returning the input satisfies all three perfectly.
- **The pre-training.** Inpainting pre-training starts both generators off as reconstruction networks.

Lowering those weights 5–10× in the second pilot didn't get it out. The logs are in `uvcgan2/logs/` and `uvcgan2/*_console.txt`.

**PatchNCE continuation** (`logs/history/patchnce_continuation_log.txt`). I added a contrastive PatchNCE loss (from the CUT paper) to the final model and continued training it. I set a rule beforehand: the run had to reach an FID below 100.5 within about 20K steps, or it would be stopped.

The samples looked more painterly, with stronger brushstrokes and softer edges. The score went the wrong way, though: from 101.96 at the start to 104.66 at 24K steps. I stopped it there.

The likely reason is the restart learning rate. The final model had finished at a learning rate of about 1e-6, and the continuation restarted at 5e-5, roughly 40 times higher. That knocked the model out of the spot it had settled into before the new loss could help.

**Second variant (MSD_TEXTURE).** This variant used two-scale discriminators and a cycle weight decaying from 10 to 3. It ran next to the final model at first and was stopped at step 6,800, when the run was restarted with the final model alone. Its log is `logs/train_raw_MSD_TEXTURE.log`.

## A tooling failure

After training finished at 02:02, the post-processing crashed with no visible error, so no `submission.csv` was written. The cause was a `UnicodeEncodeError`. `src/run_final.py` writes a summary table containing "λ" and "→" with plain `open(path, "w")`, and on Windows that defaults to the cp1252 encoding, which can't represent those characters. The code had been written on a Mac, where the default is UTF-8, so the bug never appeared there.

Adding `encoding="utf-8"` fixed it. Re-running `run_final.py` then skipped training, because the run was already finished, and completed promotion, translation and scoring in about two minutes. No training was lost.

## What I would try next

1. **Fix Monet → photo.** It is the weaker direction, about 10 FID behind. One option is to leave paintings with no photographic equivalent (still lifes, interiors) out of `G_BA`'s adversarial target, or to give them less weight.
2. **Continue the finished model gently.** Restart at a small learning rate (around 1e-6 to 5e-6). The PatchNCE run showed that restarting at 5e-5 does damage.
3. **Remove watermarks from the training photos.** Masking or cropping the corner text would stop it surviving into the outputs.
