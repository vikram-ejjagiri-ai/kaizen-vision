"""Methods 2-4: deep learning.

All three use the same ImageNet-pretrained convolutional network (Ultralytics YOLO11s classification model,
about 5.5 M parameters), so the comparison is about *how* the network is used, not about different networks:

2. Pretrained features + linear classifier: the network is frozen, only a logistic regression is trained on its
   1,280-number image description. Very little data and seconds of training.
3. Fine-tuned CNN: the whole network is retrained on the task (standard transfer learning).
4. Anomaly detection (PatchCore method, Roth et al. 2022): learns only what *good* parts look like from
   mid-level patch features and flags anything unusual. Needs no images of defects at all.
"""
from __future__ import annotations

import copy
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = Path(__file__).resolve().parents[1] / "models"
WEIGHTS = "yolo11s-cls.pt"
WEIGHTS_URL = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{WEIGHTS}"
IMG = 224
torch.set_num_threads(max(1, torch.get_num_threads()))


def weights_path() -> Path:
    p = MODELS / WEIGHTS
    if not p.exists():
        MODELS.mkdir(exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, p)
    return p


def load_network() -> torch.nn.Module:
    """The ImageNet-pretrained classification network (torch module, eval mode, float32)."""
    from ultralytics import YOLO
    net = YOLO(str(weights_path())).model
    net = net.float().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def to_tensor(images: np.ndarray, size: int = IMG) -> torch.Tensor:
    """Greyscale uint8 (n, h, w) -> float tensor (n, 3, size, size) in [0, 1] (Ultralytics convention)."""
    t = torch.from_numpy(images).float().div_(255.0).unsqueeze(1)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.repeat(1, 3, 1, 1)


def _layers(net: torch.nn.Module) -> torch.nn.ModuleList:
    return net.model  # Sequential: backbone layers 0..n-2, Classify head last


@torch.no_grad()
def embed(net: torch.nn.Module, images: np.ndarray, batch: int = 32) -> np.ndarray:
    """1,280-number global image description from the head's convolution + pooling."""
    layers = _layers(net)
    head = layers[-1]
    out = []
    for i in range(0, len(images), batch):
        x = to_tensor(images[i:i + batch])
        for layer in layers[:-1]:
            x = layer(x)
        x = head.pool(head.conv(x)).flatten(1)
        out.append(x.numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------------- 2. features + classifier
class DeepFeatures:
    name = "Pretrained CNN features + linear classifier"
    short = "features"

    def __init__(self, net: torch.nn.Module, seed: int = 0):
        self.net, self.seed = net, seed

    def fit(self, x, y, emb=None):
        e = embed(self.net, x) if emb is None else emb
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        n_splits = int(min(5, np.bincount(y).min()))
        grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0]},
                            cv=StratifiedKFold(n_splits, shuffle=True, random_state=self.seed))
        grid.fit(e, y)
        self.model = grid.best_estimator_
        return self

    def predict_proba(self, x, emb=None):
        e = embed(self.net, x) if emb is None else emb
        return self.model.predict_proba(e)


# --------------------------------------------------------------------------------- 3. fine-tuned network
class FineTuned:
    name = "Fine-tuned CNN"
    short = "finetuned"

    def __init__(self, n_classes: int, epochs: int = 12, seed: int = 0, lr: float = 1e-3, batch: int = 32,
                 size: int = 192):
        self.n_classes, self.epochs, self.seed, self.lr, self.batch, self.size = n_classes, epochs, seed, lr, batch, size

    def _build(self):
        net = copy.deepcopy(load_network())
        head = _layers(net)[-1]
        head.linear = torch.nn.Linear(head.linear.in_features, self.n_classes)
        for p in net.parameters():
            p.requires_grad_(True)
        return net

    @staticmethod
    def _augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        """Standard geometric augmentation only (flips and 90-degree turns); no brightness or blur changes,
        so the robustness test is not trained in."""
        if torch.rand(1, generator=g) < 0.5:
            x = x.flip(3)
        if torch.rand(1, generator=g) < 0.5:
            x = x.flip(2)
        k = int(torch.randint(0, 4, (1,), generator=g))
        return torch.rot90(x, k, (2, 3))

    def fit(self, x: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        self.net = self._build()
        self.net.train()
        data = to_tensor(x, self.size)
        target = torch.from_numpy(y).long()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * int(np.ceil(len(x) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.15)
        self.history = []
        t0 = time.perf_counter()
        for ep in range(self.epochs):
            perm = torch.randperm(len(x), generator=g)
            total = 0.0
            for i in range(0, len(x), self.batch):
                idx = perm[i:i + self.batch]
                xb = self._augment(data[idx], g)
                logits = self._logits(xb)
                loss = F.cross_entropy(logits, target[idx], label_smoothing=0.05)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                total += float(loss) * len(idx)
            self.history.append(total / len(x))
        self.train_seconds = time.perf_counter() - t0
        self.net.eval()
        return self

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        layers = _layers(self.net)
        for layer in layers[:-1]:
            x = layer(x)
        head = layers[-1]
        return head.linear(head.drop(head.pool(head.conv(x)).flatten(1)))

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(x), batch):
            out.append(torch.softmax(self._logits(to_tensor(x[i:i + batch], self.size)), 1).numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------------- 4. anomaly detection
class PatchCore:
    """Anomaly detection trained on good parts only.

    Patch features from two mid-level layers (stride 8 and 16), locally averaged, form a memory bank of what
    normal surface looks like. A greedy coreset keeps a representative subset. The anomaly score of an image is
    the largest distance of any of its patches to the nearest normal patch.
    """
    name = "Anomaly detection (good parts only)"
    short = "anomaly"

    def __init__(self, net: torch.nn.Module, layers=(4, 6), coreset: float = 0.01, size: int = IMG, seed: int = 0):
        self.net, self.idx, self.coreset, self.size, self.seed = net, layers, coreset, size, seed

    @torch.no_grad()
    def patches(self, images: np.ndarray, batch: int = 16) -> torch.Tensor:
        """(n, p, c) patch features."""
        layers = _layers(self.net)
        out = []
        for i in range(0, len(images), batch):
            x = to_tensor(images[i:i + batch], self.size)
            feats = []
            for k, layer in enumerate(layers[:-1]):
                x = layer(x)
                if k in self.idx:
                    feats.append(x)
                if k >= max(self.idx):
                    break
            h, w = feats[0].shape[-2:]
            f = torch.cat([F.avg_pool2d(F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False),
                                        3, 1, 1) for t in feats], 1)
            out.append(f.flatten(2).transpose(1, 2))
        return torch.cat(out)

    def _greedy_coreset(self, bank: torch.Tensor, n: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed)
        proj = bank @ torch.randn(bank.shape[1], 64, generator=g) / np.sqrt(64)   # faster distances
        sel = [int(torch.randint(len(bank), (1,), generator=g))]
        dmin = torch.cdist(proj, proj[sel]).squeeze(1)
        for _ in range(n - 1):
            j = int(torch.argmax(dmin))
            sel.append(j)
            dmin = torch.minimum(dmin, torch.cdist(proj, proj[j:j + 1]).squeeze(1))
        return bank[sel]

    def fit(self, x_good: np.ndarray):
        p = self.patches(x_good)
        bank = p.reshape(-1, p.shape[-1])
        n = max(1000, int(len(bank) * self.coreset))
        self.bank = self._greedy_coreset(bank, min(n, len(bank)))
        self.grid = int(np.sqrt(p.shape[1]))
        return self

    @torch.no_grad()
    def score_maps(self, images: np.ndarray, batch: int = 16) -> np.ndarray:
        maps = []
        for i in range(0, len(images), batch):
            p = self.patches(images[i:i + batch])
            d = torch.cdist(p.reshape(-1, p.shape[-1]), self.bank).min(1).values
            maps.append(d.reshape(len(p), self.grid, self.grid).numpy())
        return np.concatenate(maps)

    def score(self, images: np.ndarray) -> np.ndarray:
        return self.score_maps(images).reshape(len(images), -1).max(1)

    def calibrate(self, x_good_holdout: np.ndarray, quantile: float = 0.97) -> float:
        """Decision threshold from good parts that were not in the memory bank."""
        self.threshold = float(np.quantile(self.score(x_good_holdout), quantile))
        return self.threshold
