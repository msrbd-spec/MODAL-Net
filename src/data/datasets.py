"""Datasets: DRDataset for APTOS, Messidor, and EyePACS images."""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.augment import build_transform
from src.data.preprocess import preprocess
from src.losses.losses import base_target


class DRDataset(Dataset):
    """Diabetic Retinopathy Dataset.

    Reads fundus images from specified directories, applies fundus preprocessing,
    optional Fourier amplitude mixing with a style bank, augmentations, and prepares
    targets (either hard ordinal-smoothed base_target or precomputed soft labels Q).
    """

    def __init__(
        self,
        df,
        dirs,
        cfg: dict,
        train: bool,
        soft: dict = None,
        domain: int = 0,
        style_bank: list = None,
    ):
        self.df = df.reset_index(drop=True)
        self.dirs = [dirs] if isinstance(dirs, str) else list(dirs)
        self.cfg = cfg
        self.train = train
        self.soft = soft
        self.domain = int(domain)
        self.bank = style_bank
        self.tf = build_transform(cfg, train)

    def _read(self, name: str) -> np.ndarray:
        name_str = str(name).strip()
        has_ext = name_str.lower().endswith(('.png', '.jpg', '.jpeg'))
        for d in self.dirs:
            candidates = (
                [os.path.join(d, name_str)]
                if has_ext
                else [os.path.join(d, name_str + ext) for ext in ('.png', '.jpg', '.jpeg')]
            )
            for p in candidates:
                if os.path.isfile(p):
                    im = cv2.imread(p)
                    if im is not None:
                        return im
        raise FileNotFoundError(f"Image not found in any directory: {name_str} (searched {self.dirs})")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        name = str(r['id_code'])
        y = int(r['diagnosis'])

        amp = None
        sty = None
        if self.train and self.cfg.get('amp_mix') and self.bank and random.random() < float(self.cfg.get('amp_prob', 0.5)):
            amp = {
                'stage': self.cfg.get('amp_stage', 'pre'),
                'beta': random.uniform(
                    float(self.cfg.get('amp_beta_min', 0.01)),
                    float(self.cfg.get('amp_beta_max', 0.10)),
                ),
                'lam': random.random(),
            }
            cand = random.choice(self.bank)
            if isinstance(cand, str):
                im_sty = cv2.imread(cand)
                sty = cv2.cvtColor(im_sty, cv2.COLOR_BGR2RGB) if im_sty is not None else None
            else:
                sty = cand

        raw_bgr = self._read(name)
        pre_bgr = preprocess(
            raw_bgr,
            self.cfg.get('preproc', 'legacy'),
            int(self.cfg.get('img_size', 512)),
            style_rgb=sty,
            amp=amp,
        )
        rgb = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(image=rgb)['image']

        if self.soft is not None and name in self.soft:
            q = self.soft[name]
        else:
            q = base_target(y, self.cfg)

        return {
            'x': x,
            'y': y,
            'q': torch.as_tensor(q, dtype=torch.float32),
            'domain': self.domain,
            'id': name,
        }
