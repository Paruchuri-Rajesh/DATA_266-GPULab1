import json, os
nb = json.load(open("Part3_Evaluation_Script.ipynb", encoding="utf-8"))
code = []
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if s.strip().startswith("# !pip"):
        continue
    code.append(s)
src = "\n\n".join(code)
# point BASE at the preserved v3 outputs; monet/photo come from ../data
src = src.replace('BASE = "Part 3/Data"', 'BASE = "NBTEST"')
src = src.replace('REAL_MONET = os.path.join(BASE, "monet_jpg")', 'REAL_MONET = "../data/monet_jpg"')
src = src.replace('REAL_PHOTO = os.path.join(BASE, "photo_jpg")', 'REAL_PHOTO = "../data/photo_jpg"')
src = src.replace('GEN_A2B    = os.path.join(BASE, "pred_A2B")', 'GEN_A2B = "outputs_stale_v3_backup_20261001_171326/pred_A2B"')
src = src.replace('GEN_B2A    = os.path.join(BASE, "pred_B2A")', 'GEN_B2A = "outputs_stale_v3_backup_20261001_171326/pred_B2A"')
src = src.replace('submission.to_csv("submission.csv", index=False)', 'submission.to_csv("NB_submission.csv", index=False)')
open("uvcgan2/_notebook_as_script.py", "w", encoding="utf-8").write(src)
print("extracted notebook ->", len(src), "chars")
