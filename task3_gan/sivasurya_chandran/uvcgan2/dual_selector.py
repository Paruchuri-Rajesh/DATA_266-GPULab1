"""
Two checkpoint-selection protocols, both built on the instructor's exact evaluator functions
(evaluate_local.py == Part3_Evaluation_Script.ipynb, verified to match it to every printed digit).

official  - exactly what the competition scores:
              real photos  = evaluate_local.take_n(list_images(photo_dir), 300)   (first 300 by filename)
              pred_B2A     = G_AB(those same first 300 photos)      -> compared with the 300 real Monet
              pred_A2B     = G_BA(all 300 Monet paintings)          -> compared with those 300 real photos
            Used for checkpoint promotion, because it matches the submission.
            NOTE: those 300 photos are mostly in the training split, so this score is optimistic and is
            not a generalisation estimate. That is what `heldout` is for.

heldout   - the same metric computed on photos the model never trained on (and which the official
            score never looks at). Reported alongside, never used for promotion.

Both return the per-direction breakdown so every number can be reported separately.
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MEMBER, "src"))
sys.path.insert(0, MEMBER)

from dataset import ImageFolderList, save_jpegs  # noqa: E402


class DualSelector:
    """Scores a (G_AB, G_BA) pair under both protocols with the instructor's own functions."""

    def __init__(self, scfg, hold_a, monet_files, photo_dir, image_size, device, tmp_dir):
        import evaluate_local

        self.el, self.device, self.tmp, self.size = evaluate_local, device, tmp_dir, image_size
        self.batch = scfg.get("inception_batch", 32)

        # ---- the official 300: first by filename, exactly as the evaluator picks them
        self.official_photos = evaluate_local.take_n(evaluate_local.list_images(photo_dir), 300)
        official_names = {os.path.basename(p) for p in self.official_photos}

        # ---- held-out photos: never trained on, and disjoint from the official 300
        pool = sorted(p for p in hold_a if os.path.basename(p) not in official_names) or sorted(hold_a)
        n = min(scfg.get("n_images", 300), len(pool))
        n_sets = max(1, min(scfg.get("n_sets_heldout", 1), len(pool) // max(1, n)))
        self.heldout_sets = [pool[i * n:(i + 1) * n] for i in range(n_sets)]

        self.monet = sorted(monet_files)[:300]
        self.n_official = len(self.official_photos)
        self.n_heldout = sum(len(s) for s in self.heldout_sets)

        self.model = evaluate_local.get_inception_model()
        # real-image activations are fixed: compute once
        self.act_real_monet = evaluate_local.get_activations(self.model, self.monet, self.batch)
        self.act_real_photo_official = evaluate_local.get_activations(self.model, self.official_photos, self.batch)
        self.act_real_photo_heldout = [evaluate_local.get_activations(self.model, s, self.batch)
                                       for s in self.heldout_sets]

    @torch.no_grad()
    def _translate_acts(self, G, files, tag):
        paths = []
        for x, names in DataLoader(ImageFolderList(files, self.size), batch_size=16):
            paths += save_jpegs(G(x.to(self.device)), names, os.path.join(self.tmp, tag))
        return self.el.get_activations(self.model, sorted(paths), self.batch)

    def _pair(self, real_act, gen_act):
        m = min(len(real_act), len(gen_act))        # the evaluator truncates both sorted lists equally
        return self.el.fid_mifid_from_acts(real_act[:m], gen_act[:m])

    @torch.no_grad()
    def __call__(self, G_AB, G_BA) -> dict:
        modes = (G_AB.training, G_BA.training)
        G_AB.eval(), G_BA.eval()

        # Monet -> photo is shared by both protocols (same 300 paintings through G_BA)
        act_a2b = self._translate_acts(G_BA, self.monet, "a2b")

        # ---- official protocol
        act_b2a_off = self._translate_acts(G_AB, self.official_photos, "b2a_official")
        fid_b2a, mifid_b2a = self._pair(self.act_real_monet, act_b2a_off)
        fid_a2b, mifid_a2b = self._pair(self.act_real_photo_official, act_a2b)
        official = {"fid_b2a": float(fid_b2a), "mifid_b2a": float(mifid_b2a),
                    "fid_a2b": float(fid_a2b), "mifid_a2b": float(mifid_a2b)}
        official["fid"] = (official["fid_b2a"] + official["fid_a2b"]) / 2
        official["mifid"] = (official["mifid_b2a"] + official["mifid_a2b"]) / 2
        official["score"] = (official["fid"] + official["mifid"]) / 2

        # ---- held-out protocol (generalisation; Monet->photo reuses act_a2b)
        hb = [self._pair(self.act_real_monet, self._translate_acts(G_AB, s, f"b2a_hold{k}"))
              for k, s in enumerate(self.heldout_sets)]
        ha = [self._pair(r, act_a2b) for r in self.act_real_photo_heldout]
        heldout = {"fid_b2a": float(np.mean([f for f, _ in hb])), "mifid_b2a": float(np.mean([m for _, m in hb])),
                   "fid_a2b": float(np.mean([f for f, _ in ha])), "mifid_a2b": float(np.mean([m for _, m in ha]))}
        heldout["fid"] = (heldout["fid_b2a"] + heldout["fid_a2b"]) / 2
        heldout["mifid"] = (heldout["mifid_b2a"] + heldout["mifid_a2b"]) / 2
        heldout["score"] = (heldout["fid"] + heldout["mifid"]) / 2

        G_AB.train(modes[0]), G_BA.train(modes[1])
        # promotion uses the official score; "score" at the top level is therefore the official one
        return {"score": official["score"], "official": official, "heldout": heldout}
