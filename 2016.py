#完全的完整版本
import os
import math
import time
import copy
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from tqdm import tqdm  # 【新增】引入进度条

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

import matplotlib

matplotlib.use('Agg')  # 无头模式，防止服务器端绘图报错
import matplotlib.pyplot as plt

# =========================================================
# 1) 路径与参数配置 (严格对齐 AutoDL 路径)
# =========================================================

PATHS = {
    "imagenet100_root": "/root/autodl-tmp/ImageNet100",
    "ab_bins_npy": "/root/autodl-tmp/pts_in_hull.npy",

    # 所有的中间产物和输出结果都保存在 op_2 文件夹下
    "split_json": "/root/autodl-tmp/op_2/imagenet100_split_80_10_10.json",
    "ab_prior_npy": "/root/autodl-tmp/op_2/ab_prior_313.npy",
    "ab_weights_npy": "/root/autodl-tmp/op_2/ab_class_weights_313.npy",
}

CFG = {
    "seed": 42,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,

    "image_size": 224,
    "batch_size": 128,  # 双卡 5090 显存充足，建议 64 或 128
    "num_workers": 24,
    "pin_memory": True,

    "num_color_bins": 313,
    "soft_k": 5,
    "soft_sigma": 5.0,
    "rebalance_lambda": 0.5,
    "prior_resize_for_stats": 224,

    "normalize_L": True,
    "L_mean": 50.0,
    "L_std": 50.0,
    "allowed_exts": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 50,
    "lr": 1e-4,
    "betas": (0.9, 0.99),
    "weight_decay": 1e-3,

    "use_amp": True,
    "grad_clip_norm": 1.0,
    "log_interval": 50,
    "save_every": 1,

    "annealed_T": 0.38,
    "eps": 1e-8,

    "num_classes": 313,
    "in_channels": 1,
    "base_channels": 64,

    # 统一输出路径
    "checkpoint_dir": "/root/autodl-tmp/op_2/checkpoints",
    "sample_dir": "/root/autodl-tmp/op_2/samples",
    "test_output_dir": "/root/autodl-tmp/op_2/test_results",
    "loss_curve_path": "/root/autodl-tmp/op_2/loss_curve.png",

    "resume_path": "",
    "save_best_only": True,

    "scheduler_type": "multistep",
    "lr_milestones_epoch": [25, 40],
    "lr_gamma": 0.316227766,
}


# =========================================================
# 2) 基础工具函数
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_image_file(path: Path, allowed_exts):
    return path.suffix.lower() in set(allowed_exts)


def pil_load_rgb(path: str):
    return Image.open(path).convert("RGB")


def resize_rgb_image(img: Image.Image, image_size: int):
    return img.resize((image_size, image_size), resample=Image.BICUBIC)


def pil_to_lab(img: Image.Image):
    rgb = np.asarray(img).astype(np.float32) / 255.0
    return rgb2lab(rgb).astype(np.float32)


def normalize_L_channel(L: np.ndarray, cfg: dict):
    if not cfg["normalize_L"]: return L
    return (L - cfg["L_mean"]) / cfg["L_std"]


def denorm_L_tensor(L, cfg):
    if cfg.get("normalize_L", False):
        return L * cfg["L_std"] + cfg["L_mean"]
    return L


def tensor_lab_to_rgb_image(L_1hw, ab_2hw):
    L = L_1hw.detach().cpu().numpy()[0]
    ab = ab_2hw.detach().cpu().numpy().transpose(1, 2, 0)
    lab = np.concatenate([L[..., None], ab], axis=-1).astype(np.float32)
    rgb = lab2rgb(lab)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


# =========================================================
# 3) 数据集解析与切分
# =========================================================

def scan_imagenet_style_dataset(root_dir: str, allowed_exts):
    root = Path(root_dir)
    assert root.exists(), f"数据集路径不存在: {root_dir}"
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

        if n >= 3:
            if n_val == 0: n_val, n_train = 1, max(n_train - 1, 1)
            if n_test == 0: n_test, n_train = 1, max(n_train - 1, 1)

        for split_name, split_paths in [("train", paths[:n_train]), ("val", paths[n_train:n_train + n_val]),
                                        ("test", paths[n_train + n_val:])]:
            for p in split_paths:
                splits[split_name].append({"path": p, "class_name": class_name, "class_idx": class_to_idx[class_name]})

    for k in splits: rng.shuffle(splits[k])
    return splits, class_to_idx


def load_or_create_splits(paths: dict, cfg: dict):
    if os.path.exists(paths["split_json"]):
        with open(paths["split_json"], "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj["splits"], obj["class_to_idx"]
    grouped = scan_imagenet_style_dataset(paths["imagenet100_root"], cfg["allowed_exts"])
    splits, class_to_idx = stratified_split(grouped, cfg)
    save_json({"splits": splits, "class_to_idx": class_to_idx}, paths["split_json"])
    return splits, class_to_idx


def load_ab_bins(path: str, expected_bins: int = 313):
    ab_bins = np.load(path).astype(np.float32)
    assert ab_bins.shape == (expected_bins, 2), "ab_bins shape mismatch"
    return ab_bins


# =========================================================
# 4) 量化与统计先验
# =========================================================

def soft_encode_ab_sparse(ab_hw2: np.ndarray, ab_bins_q2: np.ndarray, k: int = 5, sigma: float = 5.0):
    H, W, _ = ab_hw2.shape
    flat_ab = ab_hw2.reshape(-1, 2)
    d2 = np.sum((flat_ab[:, None, :] - ab_bins_q2[None, :, :]) ** 2, axis=2)
    knn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    knn_d2 = np.take_along_axis(d2, knn_idx, axis=1)

    order = np.argsort(knn_d2, axis=1)
    knn_idx, knn_d2 = np.take_along_axis(knn_idx, order, axis=1), np.take_along_axis(knn_d2, order, axis=1)

    soft_w = np.exp(-knn_d2 / (2.0 * sigma * sigma)).astype(np.float32)
    soft_w = soft_w / np.maximum(soft_w.sum(axis=1, keepdims=True), 1e-12)

    return knn_idx.T.reshape(k, H, W).astype(np.int64), soft_w.T.reshape(k, H, W).astype(np.float32), knn_idx[
        :, 0].reshape(H, W).astype(np.int64)


def load_or_compute_prior_and_weights(paths, cfg, train_samples, ab_bins_q2):
    # 1. 检查缓存机制：只要存在 .npy 文件，就会瞬间读取，跳过计算
    if os.path.exists(paths["ab_prior_npy"]) and os.path.exists(paths["ab_weights_npy"]):
        print("\n[Data] 找到已缓存的色彩先验文件，直接跳过统计并加载！")
        return np.load(paths["ab_prior_npy"]).astype(np.float32), np.load(paths["ab_weights_npy"]).astype(np.float32)

    print(f"\n[Data] 未找到先验缓存，正在全局统计 {len(train_samples)} 张图片的色彩分布 (仅需执行一次)...")
    Q = ab_bins_q2.shape[0]
    counts = np.zeros(Q, dtype=np.float64)

    # 2. 加入 tqdm 进度条，遍历全量 train_samples
    for item in tqdm(train_samples, desc="计算全量色彩先验"):
        img = resize_rgb_image(pil_load_rgb(item["path"]), cfg["prior_resize_for_stats"])
        ab = pil_to_lab(img)[:, :, 1:3]
        d2 = np.sum((ab.reshape(-1, 2)[:, None, :] - ab_bins_q2[None, :, :]) ** 2, axis=2)
        counts += np.bincount(np.argmin(d2, axis=1), minlength=Q).astype(np.float64)

    prior = counts / np.maximum(counts.sum(), 1e-12)

    d2_smooth = np.sum((ab_bins_q2[:, None, :] - ab_bins_q2[None, :, :]) ** 2, axis=2)
    K = np.exp(-d2_smooth / (2.0 * cfg["soft_sigma"] ** 2))
    K = K / np.maximum(K.sum(axis=1, keepdims=True), 1e-12)
    prior_smooth = K @ prior
    prior_smooth = prior_smooth / np.maximum(prior_smooth.sum(), 1e-12)

    mixed = (1.0 - cfg["rebalance_lambda"]) * prior_smooth + cfg["rebalance_lambda"] / Q
    weights = 1.0 / np.maximum(mixed, 1e-12)
    weights = weights / np.maximum(np.sum(prior_smooth * weights), 1e-12)

    # 3. 第一次计算完成后，保存为 .npy 供以后永久使用
    ensure_parent(paths["ab_prior_npy"])
    np.save(paths["ab_prior_npy"], prior_smooth.astype(np.float32))
    np.save(paths["ab_weights_npy"], weights.astype(np.float32))
    print("[Data] 全局色彩先验统计完成，并已永久保存至缓存！\n")

    return prior_smooth.astype(np.float32), weights.astype(np.float32)


# =========================================================
# 5) Dataset & DataLoader
# =========================================================

class ColorizationDataset(Dataset):
    def __init__(self, samples, ab_bins, class_weights, cfg):
        self.samples, self.ab_bins, self.class_weights, self.cfg = samples, ab_bins.astype(
            np.float32), class_weights.astype(np.float32), cfg

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        lab = pil_to_lab(resize_rgb_image(pil_load_rgb(item["path"]), self.cfg["image_size"]))
        L, ab = lab[:, :, 0], lab[:, :, 1:3]
        return {
            "L": torch.from_numpy(normalize_L_channel(L, self.cfg)[None, :, :]).float(),
            "ab": torch.from_numpy(ab.transpose(2, 0, 1)).float(),
            "path": item["path"]
        }


def build_dataloaders(paths, cfg):
    splits, class_to_idx = load_or_create_splits(paths, cfg)
    ab_bins = load_ab_bins(paths["ab_bins_npy"], cfg["num_color_bins"])
    prior, class_weights = load_or_compute_prior_and_weights(paths, cfg, splits["train"], ab_bins)

    loaders = []
    for split_key, is_train in [("train", True), ("val", False), ("test", False)]:
        ds = ColorizationDataset(splits[split_key], ab_bins, class_weights, cfg)
        loaders.append(DataLoader(ds, batch_size=cfg["batch_size"], shuffle=is_train, num_workers=cfg["num_workers"],
                                  pin_memory=cfg["pin_memory"], drop_last=is_train,
                                  persistent_workers=cfg["num_workers"] > 0))

    return loaders[0], loaders[1], loaders[2], {
        "ab_bins": torch.from_numpy(ab_bins).float(),
        "class_weights": torch.from_numpy(class_weights).float()  # 新增：将权重传出
    }


# =========================================================
# 6) 模型架构与损失函数
# =========================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x): return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetColorization(nn.Module):
    def __init__(self, in_channels=1, num_classes=313, base_ch=64):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = Down(base_ch, base_ch * 2)
        self.enc3 = Down(base_ch * 2, base_ch * 4)
        self.enc4 = Down(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)
        self.up4 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1, e2, e3, e4 = self.enc1(x), self.enc2(self.enc1(x)), self.enc3(self.enc2(self.enc1(x))), self.enc4(
            self.enc3(self.enc2(self.enc1(x))))
        return self.head(self.up1(self.up2(self.up3(self.up4(self.bottleneck(self.pool(e4)), e4), e3), e2), e1))


class RebalancedSoftCrossEntropyLoss(nn.Module):
    def forward(self, logits, soft_idx, soft_w, class_weight):
        log_probs = F.log_softmax(logits, dim=1)
        gathered = torch.gather(log_probs, dim=1, index=soft_idx)
        return (-(soft_w * gathered).sum(dim=1) * class_weight).mean()


# =========================================================
# 7) 核心计算与评估逻辑
# =========================================================

@torch.no_grad()
def annealed_mean_from_logits(logits, ab_bins, T=0.38):
    # 直接在 Softmax 内部处理温度 T，原生的算子高度优化且绝对稳定
    probs = F.softmax(logits / T, dim=1)
    return torch.einsum("bqhw,qc->bchw", probs, ab_bins.to(logits.device).float())


# =========================================================
# 8) 训练/验证循环 (修复版)
# =========================================================
def soft_encode_ab_gpu(ab_b2hw, ab_bins_q2, k=5, sigma=5.0):
    B, _, H, W = ab_b2hw.shape
    ab_flat = ab_b2hw.view(B, 2, -1).permute(0, 2, 1)
    d2 = torch.cdist(ab_flat, ab_bins_q2) ** 2
    knn_d2, knn_idx = torch.topk(d2, k, dim=-1, largest=False)

    soft_w = torch.exp(-knn_d2 / (2.0 * sigma ** 2))
    soft_w = soft_w / (soft_w.sum(dim=-1, keepdim=True) + 1e-12)

    return knn_idx.permute(0, 2, 1).view(B, k, H, W), soft_w.permute(0, 2, 1).view(B, k, H, W)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, ab_bins, cfg, class_weights):
    model.train()
    running_loss = 0.0
    class_weights = class_weights.to(device)

    for batch in loader:
        L = batch["L"].to(device)
        ab = batch["ab"].to(device)

        with torch.no_grad():
            soft_idx, soft_w = soft_encode_ab_gpu(ab, ab_bins, cfg["soft_k"], cfg["soft_sigma"])
            q_gt = soft_idx[:, 0, :, :]
            class_weight = class_weights[q_gt]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg["use_amp"]):
            loss = criterion(model(L), soft_idx, soft_w, class_weight)

        # 【修复】：反向传播与优化器更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    return running_loss / len(loader)  # 【修复】：返回 loss


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, ab_bins, cfg, class_weights):
    model.eval()
    running_loss = 0.0
    class_weights = class_weights.to(device)  # 【修复】：传入类别权重

    for batch in loader:
        L = batch["L"].to(device)
        ab = batch["ab"].to(device)

        # 【修复】：验证集同样需要在 GPU 上实时计算 Target
        soft_idx, soft_w = soft_encode_ab_gpu(ab, ab_bins, cfg["soft_k"], cfg["soft_sigma"])
        q_gt = soft_idx[:, 0, :, :]
        class_weight = class_weights[q_gt]

        with autocast(enabled=cfg["use_amp"]):
            loss = criterion(model(L), soft_idx, soft_w, class_weight)
            running_loss += loss.item()

    return running_loss / len(loader)


# =========================================================
# 9) 测试集绘图 (严格 10 张三图拼接)
# =========================================================
@torch.no_grad()
def test_and_visualize(model, loader, device, ab_bins, cfg):
    model.eval()
    ensure_dir(cfg["test_output_dir"])
    count = 0

    print("\n[Test] 开始生成 10 张对比图并保存至: ", cfg["test_output_dir"])
    for batch in loader:
        L = batch["L"].to(device)
        gt_ab = batch["ab"]
        logits = model(L)
        pred_ab = annealed_mean_from_logits(logits, ab_bins, T=cfg["annealed_T"]).cpu()
        L_cpu = denorm_L_tensor(batch["L"].cpu(), cfg)

        for i in range(L.shape[0]):
            if count >= 10: return  # 只抽 10 张图

            # 还原三张图
            gt_rgb = tensor_lab_to_rgb_image(L_cpu[i], gt_ab[i])
            pred_rgb = tensor_lab_to_rgb_image(L_cpu[i], pred_ab[i])
            gray_rgb = tensor_lab_to_rgb_image(L_cpu[i], torch.zeros_like(gt_ab[i]))

            # 绘制 1x3 画布
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(gt_rgb)
            axes[0].set_title("Original RGB")
            axes[0].axis("off")

            axes[1].imshow(gray_rgb)
            axes[1].set_title("Grayscale Input")
            axes[1].axis("off")

            axes[2].imshow(pred_rgb)
            axes[2].set_title("Predicted RGB")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = os.path.join(cfg["test_output_dir"], f"test_result_{count:02d}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            count += 1


# =========================================================
# 10) 主控制流 (修正版)
# =========================================================

def main():
    try:
        set_seed(CFG["seed"])
        ensure_dir(CFG["checkpoint_dir"])

        print("构建数据流...")
        train_loader, val_loader, test_loader, meta = build_dataloaders(PATHS, CFG)
        ab_bins = meta["ab_bins"].to(CFG["device"])
        # 【修改 1】：提取类别权重
        class_weights = meta["class_weights"]

        print("初始化模型 (多卡并行检测)...")
        model = UNetColorization(in_channels=CFG["in_channels"], num_classes=CFG["num_classes"],
                                 base_ch=CFG["base_channels"])

        # 【多卡配置核心】利用 nn.DataParallel 调用多卡
        if torch.cuda.device_count() > 1:
            print(f"检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel 模式。")
            model = nn.DataParallel(model)
        model = model.to(CFG["device"])

        criterion = RebalancedSoftCrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"], betas=CFG["betas"],
                                     weight_decay=CFG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=CFG["lr_milestones_epoch"],
                                                         gamma=CFG["lr_gamma"])
        scaler = GradScaler(enabled=CFG["use_amp"])

        best_val_loss = float("inf")
        train_losses_history, val_losses_history = [], []

        print("开始训练...")
        for epoch in range(1, CFG["epochs"] + 1):
            # 【修改 2】：将 class_weights 传入训练和验证函数
            t_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, CFG["device"], ab_bins, CFG,
                                     class_weights)
            v_loss = validate_one_epoch(model, val_loader, criterion, CFG["device"], ab_bins, CFG, class_weights)
            scheduler.step()

            train_losses_history.append(t_loss)
            val_losses_history.append(v_loss)

            print(
                f"Epoch [{epoch:03d}/{CFG['epochs']}] | lr: {optimizer.param_groups[0]['lr']:.6f} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

            # 保存权重
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                # 剥离 DataParallel 外壳，防止键值污染
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({"model": model_to_save.state_dict()}, os.path.join(CFG["checkpoint_dir"], "best.pt"))

        # 【生成并保存 Loss 可视化图】
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, CFG["epochs"] + 1), train_losses_history, label='Train Loss', marker='o')
        plt.plot(range(1, CFG["epochs"] + 1), val_losses_history, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(CFG["loss_curve_path"])
        plt.close()
        print(f"训练完成！Loss 曲线已保存至: {CFG['loss_curve_path']}")

        # 【测试并提取 10 张结果图】
        print("加载最优权重进入测试阶段...")
        model.load_state_dict(torch.load(os.path.join(CFG["checkpoint_dir"], "best.pt"))["model"])
        test_and_visualize(model, test_loader, CFG["device"], ab_bins, CFG)
        print("全部流程执行完毕！")

    finally:
        # 无论正常结束还是中途报错，都会执行此处的关机命令
        print("\n[System] 正在执行自动关机指令...")
        os.system("shutdown -h now")


if __name__ == "__main__":
    main()