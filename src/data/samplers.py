"""Samplers: JointBatchSampler for multi-source training and SqrtInverseClassSampler."""

import numpy as np
from torch.utils.data import Sampler


class JointBatchSampler(Sampler):
    """Batch sampler for ConcatDataset([aptos_train, eyepacs_train]).

    Each batch of size B contains:
      - B/2 APTOS images (shuffled without replacement)
      - B/2 EyePACS images (sampled with replacement with probability ~ 1 / sqrt(class frequency))
    Epoch length is n_aptos // (B // 2) batches so the epoch corresponds to one pass over APTOS.
    """

    def __init__(self, n_a: int, y_e, B: int, seed: int = 0):
        super().__init__()
        self.n_a = int(n_a)
        self.B = int(B)
        self.y_e = np.asarray(y_e, dtype=int)
        f = np.bincount(self.y_e, minlength=5).astype(float)
        # Avoid division by zero if any class has 0 count
        f_safe = np.maximum(f, 1.0)
        inv_sqrt = 1.0 / np.sqrt(f_safe)
        self.pe = inv_sqrt[self.y_e]
        self.pe /= self.pe.sum()
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        h = self.B // 2
        perm = self.rng.permutation(self.n_a)
        for s in range(0, len(perm) - h + 1, h):
            aptos_indices = [int(i) for i in perm[s:s + h]]
            eyepacs_indices = [int(self.n_a + j) for j in self.rng.choice(len(self.y_e), h, p=self.pe)]
            yield aptos_indices + eyepacs_indices

    def __len__(self):
        return self.n_a // (self.B // 2)


class MultiSourceBatchSampler(Sampler):
    """Batch sampler for ConcatDataset([aptos_train, source_1, ..., source_m]).

    Each batch of size B holds B/2 APTOS images (shuffled, without replacement) and B/2 images from
    the extra sources, split as evenly as possible between them (any remainder is assigned to a
    random source per batch). Inside each source images are drawn with replacement with probability
    ~ 1 / sqrt(class frequency), exactly as JointBatchSampler does for EyePACS. One epoch = one pass
    over APTOS. With a single extra source the batch layout equals JointBatchSampler's.
    """

    def __init__(self, n_a: int, source_labels: list, B: int, seed: int = 0):
        super().__init__()
        assert len(source_labels) >= 1, 'MultiSourceBatchSampler needs at least one extra source'
        self.n_a = int(n_a)
        self.B = int(B)
        self.offsets = []
        self.probs = []
        off = self.n_a
        for y in source_labels:
            y = np.asarray(y, dtype=int)
            f = np.maximum(np.bincount(y, minlength=5).astype(float), 1.0)
            p = (1.0 / np.sqrt(f))[y]
            self.probs.append(p / p.sum())
            self.offsets.append(off)
            off += len(y)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        h = self.B // 2
        m = len(self.probs)
        perm = self.rng.permutation(self.n_a)
        for s in range(0, len(perm) - h + 1, h):
            batch = [int(i) for i in perm[s:s + h]]
            counts = [h // m] * m
            for j in self.rng.choice(m, h - sum(counts), replace=False):
                counts[int(j)] += 1
            for src, c in enumerate(counts):
                if c:
                    idx = self.rng.choice(len(self.probs[src]), c, p=self.probs[src])
                    batch.extend(int(self.offsets[src] + j) for j in idx)
            yield batch

    def __len__(self):
        return self.n_a // (self.B // 2)


class SqrtInverseClassSampler(Sampler):
    """Sampler for class-imbalanced datasets with sampling probability ~ 1 / sqrt(class frequency)."""

    def __init__(self, y, num_samples: int = None, seed: int = 0):
        super().__init__()
        self.y = np.asarray(y, dtype=int)
        self.num_samples = len(self.y) if num_samples is None else int(num_samples)
        f = np.bincount(self.y, minlength=5).astype(float)
        inv_sqrt = 1.0 / np.sqrt(np.maximum(f, 1.0))
        p = inv_sqrt[self.y]
        self.p = p / p.sum()
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        indices = self.rng.choice(len(self.y), size=self.num_samples, replace=True, p=self.p)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
