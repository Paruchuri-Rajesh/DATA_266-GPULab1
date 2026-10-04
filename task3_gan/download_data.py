"""
Downloads the class competition's Monet/photo images into task3_gan/data/{monet_jpg,photo_jpg} and keeps any
other files from the zip in task3_gan/data/competition_files/. Works on Linux, macOS and Windows.

    python download_data.py                        # Kaggle API; token in KAGGLE_API_TOKEN or ~/.kaggle/access_token
    python download_data.py --zip path/to/data-266-fall-2026-gan-image-style-transfer.zip   # a zip downloaded by hand
    python download_data.py --zip path/to/dataset.zip      # the instructor's dataset.zip (identical images, no token needed)

The token (kaggle.com > Settings > API, KGAT_...) needs `pip install "kaggle>=1.8"` (Python >= 3.10).
Never commit it.
"""
import argparse
import os
import shutil
import zipfile

SLUG = "data-266-fall-2026-gan-image-style-transfer"
IMG_EXT = (".jpg", ".jpeg", ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=SLUG)
    ap.add_argument("--zip", default=None, help="use an already-downloaded competition zip instead of the Kaggle API")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)

    zpath = a.zip
    if not zpath:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(a.slug, path=a.dir, quiet=False)
        zpath = os.path.join(a.dir, f"{a.slug}.zip")
    raw = os.path.join(a.dir, "_raw")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(raw)

    # the zip layout is not guaranteed: find the two image folders wherever they are
    for name in ("monet_jpg", "photo_jpg"):
        src = next((os.path.join(dp, d) for dp, dns, _ in os.walk(raw) for d in dns if d == name and "__MACOSX" not in dp), None)
        if not src:
            raise SystemExit(f"no {name}/ folder in {zpath}; see {raw}")
        dst = os.path.join(a.dir, name)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.lower().endswith(IMG_EXT):
                shutil.move(os.path.join(src, f), os.path.join(dst, f))
        shutil.rmtree(src)
    # everything else (real_stats.npz etc.) -> competition_files/
    comp = os.path.join(a.dir, "competition_files")
    for dp, _, files in os.walk(raw):
        for f in files:
            if f == ".DS_Store" or dp.split(os.sep)[-1].startswith("__MACOSX") or "__MACOSX" in dp:
                continue
            os.makedirs(comp, exist_ok=True)
            shutil.copy2(os.path.join(dp, f), os.path.join(comp, f))
    shutil.rmtree(raw)
    if not a.zip:
        os.remove(zpath)
    count = {n: len([f for f in os.listdir(os.path.join(a.dir, n)) if f.lower().endswith(IMG_EXT)]) for n in ("monet_jpg", "photo_jpg")}
    print(f"monet: {count['monet_jpg']} images, photo: {count['photo_jpg']} images")
    print("competition files:", sorted(os.listdir(comp)) if os.path.isdir(comp) else "none")
    if count["monet_jpg"] != 300 or count["photo_jpg"] != 7038:
        print("WARNING: expected 300 Monet and 7,038 photos")


if __name__ == "__main__":
    main()
