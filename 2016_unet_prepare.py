#首先是路径的检查
#然后是找到313 个颜色 bin 文件 pts_in_hull.npy，并且 PATHS["ab_bins_npy"] 也还是占位符 /path/to/pts_in_hull.npy
#最后是环境检查
import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

import torch
from torch.utils.data import Dataset, DataLoader


PATHS = {
    "imagenet100_root": "/autodl-tmp/ImageNet100",
    "ab_bins_npy": "/autodl-tmp/pts_in_hull.npy",
    "split_json": "/autodl-tmp/op_2/cache/imagenet100_split_80_10_10.json",
    "ab_prior_npy": "/autodl-tmp/op_2/cache/ab_prior_313.npy",
    "ab_weights_npy": "/autodl-tmp/op_2/cache/ab_class_weights_313.npy",
}

CFG = {
    "seed": 42,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "image_size": 224,
    "batch_size": 16,
    "num_workers": 8,
    "pin_memory": True,
    "num_color_bins": 313,
    "soft_k": 5,
    "soft_sigma": 5.0,
    "rebalance_lambda": 0.5,
    "prior_resize_for_stats": 224,
    "normalize_L": True,
    "L_mean": 50.0,
    "L_std": 50.0,
    "allowed_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"],
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path, allowed_exts):
    return path.suffix in set(allowed_exts)


def pil_load_rgb(path: str):
    return Image.open(path).convert("RGB")


def resize_rgb_image(img: Image.Image, image_size: int):
    return img.resize((image_size, image_size), resample=Image.BICUBIC)


def pil_to_lab(img: Image.Image):
    rgb = np.asarray(img).astype(np.float32) / 255.0
    lab = rgb2lab(rgb).astype(np.float32)
    return lab


def normalize_L_channel(L: np.ndarray, cfg: dict):
    if not cfg["normalize_L"]:
        return L
    return (L - cfg["L_mean"]) / cfg["L_std"]


def scan_imagenet_style_dataset(root_dir: str, allowed_exts):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    grouped = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p, allowed_exts):
            class_name = p.parent.name
            grouped[class_name].append(str(p))

    grouped = {k: sorted(v) for k, v in grouped.items() if len(v) > 0}
    if len(grouped) == 0:
        raise RuntimeError(f"No images found under: {root_dir}")
    return grouped


def stratified_split(grouped_paths: dict, cfg: dict):
    train_ratio = cfg["train_ratio"]
    val_ratio = cfg["val_ratio"]
    test_ratio = cfg["test_ratio"]

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1")

    rng = random.Random(cfg["seed"])
    class_names = sorted(grouped_paths.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    splits = {"train": [], "val": [], "test": []}

    for class_name in class_names:
        paths = grouped_paths[class_name][:]
        rng.shuffle(paths)

        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        if n >= 3:
            if n_val == 0:
                n_val = 1
                n_train = max(n_train - 1, 1)
            if n_test == 0:
                n_test = 1
                n_train = max(n_train - 1, 1)

        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        for split_name, split_paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            for p in split_paths:
                splits[split_name].append({
                    "path": p,
                    "class_name": class_name,
                    "class_idx": class_to_idx[class_name],
                })

    rng.shuffle(splits["train"])
    rng.shuffle(splits["val"])
    rng.shuffle(splits["test"])
    return splits, class_to_idx


def save_splits(splits: dict, class_to_idx: dict, save_path: str):
    ensure_parent(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"splits": splits, "class_to_idx": class_to_idx}, f, ensure_ascii=False, indent=2)


def load_splits(load_path: str):
    with open(load_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["splits"], obj["class_to_idx"]


def load_or_create_splits(paths: dict, cfg: dict):
    split_json = paths["split_json"]
    if os.path.exists(split_json):
        return load_splits(split_json)

    grouped = scan_imagenet_style_dataset(paths["imagenet100_root"], cfg["allowed_exts"])
    splits, class_to_idx = stratified_split(grouped, cfg)
    save_splits(splits, class_to_idx, split_json)
    return splits, class_to_idx


def load_ab_bins(ab_bins_path: str, expected_bins: int = 313):
    ab_bins = np.load(ab_bins_path).astype(np.float32)
    if ab_bins.shape != (expected_bins, 2):
        raise ValueError(f"ab_bins shape should be ({expected_bins}, 2), got {ab_bins.shape}")
    return ab_bins


def soft_encode_ab_sparse(ab_hw2: np.ndarray, ab_bins_q2: np.ndarray, k: int = 5, sigma: float = 5.0):
    H, W, _ = ab_hw2.shape
    flat_ab = ab_hw2.reshape(-1, 2)
    d2 = np.sum((flat_ab[:, None, :] - ab_bins_q2[None, :, :]) ** 2, axis=2)

    knn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    knn_d2 = np.take_along_axis(d2, knn_idx, axis=1)

    order = np.argsort(knn_d2, axis=1)
    knn_idx = np.take_along_axis(knn_idx, order, axis=1)
    knn_d2 = np.take_along_axis(knn_d2, order, axis=1)

    soft_w = np.exp(-knn_d2 / (2.0 * sigma * sigma)).astype(np.float32)
    soft_w = soft_w / np.maximum(soft_w.sum(axis=1, keepdims=True), 1e-12)

    q_gt = knn_idx[:, 0].astype(np.int64)

    soft_idx = knn_idx.T.reshape(k, H, W).astype(np.int64)
    soft_w = soft_w.T.reshape(k, H, W).astype(np.float32)
    q_gt = q_gt.reshape(H, W)
    return soft_idx, soft_w, q_gt


def compute_ab_prior_from_samples(samples, ab_bins_q2, cfg):
    Q = ab_bins_q2.shape[0]
    counts = np.zeros(Q, dtype=np.float64)

    for item in samples:
        img = pil_load_rgb(item["path"])
        img = resize_rgb_image(img, cfg["prior_resize_for_stats"])
        lab = pil_to_lab(img)
        ab = lab[:, :, 1:3]

        flat_ab = ab.reshape(-1, 2)
        d2 = np.sum((flat_ab[:, None, :] - ab_bins_q2[None, :, :]) ** 2, axis=2)
        q = np.argmin(d2, axis=1)
        counts += np.bincount(q, minlength=Q).astype(np.float64)

    prior = counts / np.maximum(counts.sum(), 1e-12)
    return prior.astype(np.float32)


def smooth_prior_with_gaussian_kernel(prior_q, ab_bins_q2, sigma=5.0):
    diff = ab_bins_q2[:, None, :] - ab_bins_q2[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)

    K = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float64)
    K = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-12)

    prior_smooth = K @ prior_q.astype(np.float64)
    prior_smooth = prior_smooth / np.maximum(prior_smooth.sum(), 1e-12)
    return prior_smooth.astype(np.float32)


def compute_class_rebalancing_weights(prior_q, ab_bins_q2, cfg):
    lam = cfg["rebalance_lambda"]
    Q = len(prior_q)

    prior_smooth = smooth_prior_with_gaussian_kernel(
        prior_q, ab_bins_q2, sigma=cfg["soft_sigma"]
    )
    mixed = (1.0 - lam) * prior_smooth + lam / Q
    weights = 1.0 / np.maximum(mixed, 1e-12)
    weights = weights / np.maximum(np.sum(prior_smooth * weights), 1e-12)
    return prior_smooth.astype(np.float32), weights.astype(np.float32)


def load_or_compute_prior_and_weights(paths: dict, cfg: dict, train_samples, ab_bins_q2):
    prior_path = paths["ab_prior_npy"]
    weights_path = paths["ab_weights_npy"]

    if os.path.exists(prior_path) and os.path.exists(weights_path):
        prior = np.load(prior_path).astype(np.float32)
        weights = np.load(weights_path).astype(np.float32)
        return prior, weights

    ensure_parent(prior_path)
    ensure_parent(weights_path)

    prior = compute_ab_prior_from_samples(train_samples, ab_bins_q2, cfg)
    _, weights = compute_class_rebalancing_weights(prior, ab_bins_q2, cfg)

    np.save(prior_path, prior)
    np.save(weights_path, weights)
    return prior, weights


class ColorizationDataset(Dataset):
    def __init__(self, samples, ab_bins_q2, class_weights_q, cfg):
        self.samples = samples
        self.ab_bins = ab_bins_q2.astype(np.float32)
        self.class_weights_q = class_weights_q.astype(np.float32)
        self.cfg = cfg
        self.image_size = cfg["image_size"]
        self.k = cfg["soft_k"]
        self.sigma = cfg["soft_sigma"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = pil_load_rgb(item["path"])
        img = resize_rgb_image(img, self.image_size)
        lab = pil_to_lab(img)

        L = lab[:, :, 0]
        ab = lab[:, :, 1:3]

        soft_idx, soft_w, q_gt = soft_encode_ab_sparse(
            ab_hw2=ab, ab_bins_q2=self.ab_bins, k=self.k, sigma=self.sigma
        )
        class_weight = self.class_weights_q[q_gt]
        L = normalize_L_channel(L, self.cfg)

        return {
            "L": torch.from_numpy(L[None, :, :]).float(),
            "ab": torch.from_numpy(ab.transpose(2, 0, 1)).float(),
            "soft_idx": torch.from_numpy(soft_idx).long(),
            "soft_w": torch.from_numpy(soft_w).float(),
            "q_gt": torch.from_numpy(q_gt).long(),
            "class_weight": torch.from_numpy(class_weight).float(),
            "class_idx": torch.tensor(item["class_idx"]).long(),
            "path": item["path"],
        }


def build_dataloaders(paths: dict, cfg: dict):
    set_seed(cfg["seed"])

    splits, class_to_idx = load_or_create_splits(paths, cfg)
    ab_bins = load_ab_bins(paths["ab_bins_npy"], expected_bins=cfg["num_color_bins"])
    prior, class_weights = load_or_compute_prior_and_weights(
        paths=paths,
        cfg=cfg,
        train_samples=splits["train"],
        ab_bins_q2=ab_bins,
    )

    train_dataset = ColorizationDataset(splits["train"], ab_bins, class_weights, cfg)
    val_dataset = ColorizationDataset(splits["val"], ab_bins, class_weights, cfg)
    test_dataset = ColorizationDataset(splits["test"], ab_bins, class_weights, cfg)

    loader_kwargs = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        persistent_workers=cfg["num_workers"] > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    meta = {
        "class_to_idx": class_to_idx,
        "ab_bins": torch.from_numpy(ab_bins).float(),
        "ab_prior": torch.from_numpy(prior).float(),
        "ab_class_weights": torch.from_numpy(class_weights).float(),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
    }
    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    train_loader, val_loader, test_loader, meta = build_dataloaders(PATHS, CFG)

    print(f"train: {meta['num_train']}")
    print(f"val  : {meta['num_val']}")
    print(f"test : {meta['num_test']}")
    print(f"ab_bins shape: {tuple(meta['ab_bins'].shape)}")
    print(f"class_weights shape: {tuple(meta['ab_class_weights'].shape)}")

    batch = next(iter(train_loader))
    print("L            :", tuple(batch["L"].shape))
    print("ab           :", tuple(batch["ab"].shape))
    print("soft_idx     :", tuple(batch["soft_idx"].shape))
    print("soft_w       :", tuple(batch["soft_w"].shape))
    print("q_gt         :", tuple(batch["q_gt"].shape))
    print("class_weight :", tuple(batch["class_weight"].shape))
