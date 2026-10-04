import os, sys, yaml, torch, time
sys.path.insert(0, os.path.abspath("uvcgan2")); sys.path.insert(0, os.path.abspath("src")); sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))
from dataset import list_images, split_files
from dual_selector import DualSelector
from models_uvcgan import build_uvc_generator
cfg = yaml.safe_load(open("uvcgan2/config_uvcgan.yaml"))
d, s = cfg["data"], cfg["select"]
photos, monet = list_images(d["domain_a_dir"]), list_images(d["domain_b_dir"])
_, hold_a = split_files(photos, d["holdout_a"], d["seed"])
t = time.time()
sel = DualSelector(s, hold_a, monet, d["domain_a_dir"], 256, "cuda", "uvcgan2/_seltmp")
print("  built in %.0fs | official=%d heldout=%d monet=%d" % (time.time()-t, sel.n_official, sel.n_heldout, len(sel.monet)))
m = {**cfg["model"], "image_size": 256}
G1, G2 = build_uvc_generator(m).cuda().eval(), build_uvc_generator(m).cuda().eval()
t = time.time(); r = sel(G1, G2)
print("  one candidate scored in %.0fs" % (time.time()-t))
print("  OFFICIAL:", {k: round(v,4) for k,v in r["official"].items()})
print("  HELDOUT :", {k: round(v,4) for k,v in r["heldout"].items()})
