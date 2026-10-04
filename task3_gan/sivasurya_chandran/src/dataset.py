import glob
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EXTS = ("*.jpg", "*.jpeg", "*.png")


def list_images(folder: str):
    files = []
    for e in EXTS:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)


def split_files(files, holdout: int, seed: int):
    """Deterministic train/holdout split; holdout images are never trained on and are used for evaluation."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    holdout = min(holdout, len(files) // 5)
    return [files[i] for i in order[holdout:]], [files[i] for i in order[:holdout]]


def eval_transform(size: int):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


class UnpairedImageDataset(Dataset):
    """Two unpaired domains. B is sampled at random each step so A/B pairs never repeat in a fixed order."""

    def __init__(self, files_a, files_b, image_size: int = 256, augment: bool = True):
        self.files_a, self.files_b = files_a, files_b
        if augment:
            load = int(image_size * 1.12)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((load, load), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5] * 3, [0.5] * 3),
                ]
            )
        else:
            self.transform = eval_transform(image_size)

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, idx):
        a = Image.open(self.files_a[idx % len(self.files_a)]).convert("RGB")
        b = Image.open(self.files_b[random.randrange(len(self.files_b))]).convert("RGB")
        return {"A": self.transform(a), "B": self.transform(b)}


def save_jpegs(fake, names, folder, quality: int = 95):
    """Writes generator outputs in [-1, 1] as JPEGs named after their inputs. Every output that gets scored
    (selection, outputs/pred_*, submission) goes through this one function."""
    os.makedirs(folder, exist_ok=True)
    arr = ((fake.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    paths = []
    for a, n in zip(arr, names):
        paths.append(os.path.join(folder, os.path.splitext(n)[0] + ".jpg"))
        Image.fromarray(a).save(paths[-1], quality=quality)
    return paths


class ImageFolderList(Dataset):
    def __init__(self, files, image_size: int = 256):
        self.files = files
        self.transform = eval_transform(image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx]).convert("RGB")), os.path.basename(self.files[idx])
