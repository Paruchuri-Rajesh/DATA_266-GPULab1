import numpy as np
import torch


class CharSequences:
    """Fixed-length input/target pairs: x = ids[s : s+T], y = ids[s+1 : s+T+1],
    with non-overlapping starts s = 0, T, 2T, ... One pass over all starts = one epoch."""

    def __init__(self, ids: np.ndarray, block_size: int):
        self.ids = ids
        self.block_size = block_size
        self.starts = np.arange(0, len(ids) - block_size - 1, block_size)

    def __len__(self):
        return len(self.starts)

    def pair(self, i: int):
        s = self.starts[i]
        x = self.ids[s : s + self.block_size].astype(np.int64)
        y = self.ids[s + 1 : s + 1 + self.block_size].astype(np.int64)
        return x, y

    def epoch_batches(self, batch_size: int, rng: np.random.Generator, device: str):
        order = rng.permutation(len(self.starts))
        for b in range(0, len(order) - batch_size + 1, batch_size):
            starts = self.starts[order[b : b + batch_size]]
            offs = np.arange(self.block_size)
            x = self.ids[starts[:, None] + offs].astype(np.int64)
            y = self.ids[starts[:, None] + offs + 1].astype(np.int64)
            yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

    def steps_per_epoch(self, batch_size: int) -> int:
        return len(self.starts) // batch_size
