"""
Saves the shared raw datasets into the task data/ folders so the team can zip
them to Google Drive (datasets are not pushed to GitHub). Each member's
src/preprocess.py reads these copies when present, else falls back to the HF Hub.

  python tools/download_raw_data.py                 # both
  python tools/download_raw_data.py --only yelp     # or: tinystories

Task 3 images come from Kaggle instead: python task3_gan/download_data.py
"""
import argparse
import os

from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = {
    "tinystories": ("roneneldan/TinyStories", os.path.join(ROOT, "task1_llm", "data", "tinystories")),
    "yelp": ("fancyzhx/yelp_polarity", os.path.join(ROOT, "task2_sentiment", "data", "yelp_polarity")),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(DATASETS))
    a = ap.parse_args()
    for key, (hf_id, out) in DATASETS.items():
        if a.only and key != a.only:
            continue
        ds = load_dataset(hf_id)
        ds.save_to_disk(out)
        print(f"{hf_id} -> {out}  {({k: len(v) for k, v in ds.items()})}")
