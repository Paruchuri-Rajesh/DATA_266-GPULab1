# Task 3 final run: variants trained in parallel

Winner (lowest held-out score, instructor's metric): **V2_CONTINUE**

| Variant | D scales | λ_cycle | λ_identity | DiffAugment | Epochs | Steps | s/step (incl. checks) | Best held-out score | FID (photo→Monet / Monet→photo) | Best epoch (weights) |
|---|---|---|---|---|---|---|---|---|---|---|
| **V2_CONTINUE** | 1 | 10.0 | 2.5 | translation | 270 | 270000 | 0.0917 | 49.267 | 98.13 (94.73 / 101.53) | 269 (raw_raw) |
