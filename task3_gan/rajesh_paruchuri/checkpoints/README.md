# Task 3 checkpoint (Rajesh Paruchuri)

`cyclegan_full.pt` (95 MB: G_AB, G_BA, D_A, D_B, their EMA generators, config, epoch 80) is kept out of git by the
rule `task3_gan/**/checkpoints/*.pt` in `.gitignore`.

- Local path after training: `task3_gan/rajesh_paruchuri/checkpoints/cyclegan_full.pt`
- Recreate it: `python task3_gan/rajesh_paruchuri/src/cyclegan.py full` (about 2.6 h on an Apple M4)
- Every metric in `../metrics_report.csv` and `../submission.csv` comes from the EMA generators in this file.
