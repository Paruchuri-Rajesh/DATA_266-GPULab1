"""
Creates two small synthetic unpaired domains for the CPU smoke test only
(A: smooth "photo-like" scenes, B: posterised, stroke-textured "paintings").
Real training uses the Kaggle Monet/photo data in ../data/.

Run:
    python src/make_synthetic_data.py --out data_smoketest --n 60 --size 64
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def scene(rng, size):
    top, bottom = rng.integers(0, 255, 3), rng.integers(0, 255, 3)
    t = np.linspace(0, 1, size)[:, None, None]
    img = (top * (1 - t) + bottom * t).repeat(size, axis=1).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for _ in range(rng.integers(2, 5)):
        x0, y0 = rng.integers(0, size, 2)
        r = rng.integers(size // 10, size // 4)
        d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    return im


def paint(im, rng):
    arr = (np.asarray(im.filter(ImageFilter.GaussianBlur(1.5))) // 48 * 48 + 24).astype(np.int16)
    strokes = rng.normal(0, 18, arr.shape[:2])[:, :, None]
    strokes = np.repeat(strokes, 3, axis=2)
    for _ in range(2):
        strokes = (strokes + np.roll(strokes, 1, axis=1) + np.roll(strokes, 2, axis=1)) / 3
    return Image.fromarray(np.clip(arr + strokes + np.array([12, 6, -10]), 0, 255).astype(np.uint8))


def main(out, n, size, seed):
    rng = np.random.default_rng(seed)
    for sub in ("photo_jpg", "monet_jpg"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    for i in range(n):
        scene(rng, size).save(os.path.join(out, "photo_jpg", f"p{i:04d}.jpg"), quality=95)
        paint(scene(rng, size), rng).save(os.path.join(out, "monet_jpg", f"m{i:04d}.jpg"), quality=95)
    print(f"wrote {n} + {n} synthetic images to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_smoketest")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, a.n, a.size, a.seed)
