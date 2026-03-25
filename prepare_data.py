# prepare_data.py
import os
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from tqdm import tqdm

# =========================================================
# 1) 路径与核心参数配置 (与你的主程序严格对齐)
# =========================================================
PATHS = {
    "imagenet100_root": "/root/autodl-tmp/ImageNet100",
    "ab_bins_npy": "/root/autodl-tmp/pts_in_hull.npy",
    "split_json": "/root/autodl-tmp/op_2/imagenet100_split_80_10_10.json",
    "ab_prior_npy": "/root/autodl-tmp/op_2/ab_prior_313.npy",
    "ab_weights_npy": "/root/autodl-tmp/op_2/ab_class_weights_313.npy",
}

CFG = {
    "seed": 42,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "prior_resize_for_stats": 224,
    "soft_sigma": 5.0,
    "rebalance_lambda": 0.5,
    "allowed_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
}


# =========================================================
# 2) 基础工具函数
# =========================================================
def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path, allowed_exts):
    return path.suffix.lower() in set(allowed_exts)


def pil_load_rgb(path: str):
    return Image.open(path).convert("RGB")


def resize_rgb_image(img: Image.Image, image_size: int):
    return img.resize((image_size, image_size), resample=Image.BICUBIC)


def pil_to_lab(img: Image.Image):
    rgb = np.asarray(img).astype(np.float32) / 255.0
    return rgb2lab(rgb).astype(np.float32)


def save_json(obj, path):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_ab_bins(path: str):
    return np.load(path).astype(np.float32)


# =========================================================
# 3) 数据集切分逻辑 (保证与训练时绝对一致)
# =========================================================
def scan_imagenet_style_dataset(root_dir: str, allowed_exts):
    root = Path(root_dir)
    grouped = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p, allowed_exts):
            grouped[p.parent.name].append(str(p))
    return {k: sorted(v) for k, v in grouped.items() if len(v) > 0}


def stratified_split(grouped_paths: dict, cfg: dict):
    train_ratio, val_ratio, test_ratio = cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"]
    rng = random.Random(cfg["seed"])
    class_names = sorted(grouped_paths.keys())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    splits = {"train": [], "val": [], "test": []}

    for class_name in class_names:
        paths = grouped_paths[class_name][:]
        rng.shuffle(paths)
        n = len(paths)
        n_train, n_val = int(n * train_ratio), int(n * val_ratio)
        n_test = n - n_train - n_val

        for split_name, split_paths in [("train", paths[:n_train]), ("val", paths[n_train:n_train + n_val]),
                                        ("test", paths[n_train + n_val:])]:
            for p in split_paths:
                splits[split_name].append({"path": p, "class_name": class_name, "class_idx": class_to_idx[class_name]})
    return splits, class_to_idx


def load_or_create_splits(paths: dict, cfg: dict):
    if os.path.exists(paths["split_json"]):
        print(f"[Info] 发现已有数据集切分文件: {paths['split_json']}")
        with open(paths["split_json"], "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj["splits"], obj["class_to_idx"]
    print("[Info] 扫描数据集并生成切分文件...")
    grouped = scan_imagenet_style_dataset(paths["imagenet100_root"], cfg["allowed_exts"])
    splits, class_to_idx = stratified_split(grouped, cfg)
    save_json({"splits": splits, "class_to_idx": class_to_idx}, paths["split_json"])
    return splits, class_to_idx


# =========================================================
# 4) 核心：全局色彩先验计算
# =========================================================
def compute_and_save_priors(paths, cfg, train_samples, ab_bins_q2):
    if os.path.exists(paths["ab_prior_npy"]) and os.path.exists(paths["ab_weights_npy"]):
        print(f"\n[Success] 目标文件已存在！无需重复计算。\n -> {paths['ab_prior_npy']}\n -> {paths['ab_weights_npy']}")
        return

    print(f"\n[Task] 开始计算 {len(train_samples)} 张训练图片的全局色彩先验...")
    Q = ab_bins_q2.shape[0]
    counts = np.zeros(Q, dtype=np.float64)

    # 核心进度条循环
    for item in tqdm(train_samples, desc="统计色彩分布 (CPU 计算中)"):
        img = resize_rgb_image(pil_load_rgb(item["path"]), cfg["prior_resize_for_stats"])
        ab = pil_to_lab(img)[:, :, 1:3]
        d2 = np.sum((ab.reshape(-1, 2)[:, None, :] - ab_bins_q2[None, :, :]) ** 2, axis=2)
        counts += np.bincount(np.argmin(d2, axis=1), minlength=Q).astype(np.float64)

    print("\n[Task] 正在生成平滑先验与权重矩阵...")
    prior = counts / np.maximum(counts.sum(), 1e-12)

    d2_smooth = np.sum((ab_bins_q2[:, None, :] - ab_bins_q2[None, :, :]) ** 2, axis=2)
    K = np.exp(-d2_smooth / (2.0 * cfg["soft_sigma"] ** 2))
    K = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-12)
    prior_smooth = K @ prior
    prior_smooth = prior_smooth / np.maximum(prior_smooth.sum(), 1e-12)

    mixed = (1.0 - cfg["rebalance_lambda"]) * prior_smooth + cfg["rebalance_lambda"] / Q
    weights = 1.0 / np.maximum(mixed, 1e-12)
    weights = weights / np.maximum(np.sum(prior_smooth * weights), 1e-12)

    ensure_parent(paths["ab_prior_npy"])
    np.save(paths["ab_prior_npy"], prior_smooth.astype(np.float32))
    np.save(paths["ab_weights_npy"], weights.astype(np.float32))
    print(f"[Success] 计算完成！文件已永久保存至：\n -> {paths['ab_prior_npy']}\n -> {paths['ab_weights_npy']}")


if __name__ == "__main__":
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    print("=== 离线数据预处理脚本启动 ===")
    ab_bins = load_ab_bins(PATHS["ab_bins_npy"])
    splits, _ = load_or_create_splits(PATHS, CFG)

    compute_and_save_priors(PATHS, CFG, splits["train"], ab_bins)
    print("=== 全部预处理流程圆满结束，你可以关机并切换至 GPU 模式了！ ===")