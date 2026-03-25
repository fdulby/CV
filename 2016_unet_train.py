import os
import math
import time
import copy
import json
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import lab2rgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler



#  动态加载 2016_unet_prepare.py

PREPARE_FILE = "2016_unet_prepare.py"

spec = importlib.util.spec_from_file_location("prepare_mod", PREPARE_FILE)
prepare_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_mod)

PATHS = copy.deepcopy(prepare_mod.PATHS)
CFG = copy.deepcopy(prepare_mod.CFG)

build_dataloaders = prepare_mod.build_dataloaders
set_seed = prepare_mod.set_seed
pil_load_rgb = prepare_mod.pil_load_rgb
resize_rgb_image = prepare_mod.resize_rgb_image
pil_to_lab = prepare_mod.pil_to_lab


#  在原 CFG 基础上补充训练/推理配置

CFG.update({
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "epochs": 50,
    "lr": 3e-5,
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

    "checkpoint_dir": "./artifacts/checkpoints",
    "sample_dir": "./artifacts/samples",
    "test_output_dir": "./artifacts/test_colorized",

    "resume_path": "",
    "save_best_only": True,

    "scheduler_type": "multistep",
    "lr_milestones_epoch": [25, 40],
    "lr_gamma": 0.316227766,  # 约等于从 3e-5 到 1e-5 的倍率
})


#  工具函数

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def denorm_L_tensor(L, cfg):
    if cfg.get("normalize_L", False):
        return L * cfg["L_std"] + cfg["L_mean"]
    return L


def tensor_lab_to_rgb_image(L_1hw, ab_2hw):
    """
    输入:
      L_1hw: torch.Tensor [1, H, W]
      ab_2hw: torch.Tensor [2, H, W]
    输出:
      rgb_uint8: np.ndarray [H, W, 3], uint8
    """
    L = L_1hw.detach().cpu().numpy()[0]
    ab = ab_2hw.detach().cpu().numpy().transpose(1, 2, 0)

    lab = np.concatenate([L[..., None], ab], axis=-1).astype(np.float32)
    rgb = lab2rgb(lab)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).round().astype(np.uint8)
    return rgb


def save_rgb_image(rgb_uint8, save_path):
    ensure_dir(Path(save_path).parent)
    Image.fromarray(rgb_uint8).save(save_path)


# U-Net

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNetColorization(nn.Module):
    def __init__(self, in_channels=1, num_classes=313, base_ch=64):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = Down(base_ch, base_ch * 2)
        self.enc3 = Down(base_ch * 2, base_ch * 4)
        self.enc4 = Down(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        self.up4 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch, base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)          # [B, 64, 224, 224]
        e2 = self.enc2(e1)         # [B, 128, 112, 112]
        e3 = self.enc3(e2)         # [B, 256, 56, 56]
        e4 = self.enc4(e3)         # [B, 512, 28, 28]

        b = self.bottleneck(self.pool(e4))  # [B, 1024, 14, 14]

        d4 = self.up4(b, e4)       # [B, 512, 28, 28]
        d3 = self.up3(d4, e3)      # [B, 256, 56, 56]
        d2 = self.up2(d3, e2)      # [B, 128, 112, 112]
        d1 = self.up1(d2, e1)      # [B, 64, 224, 224]

        logits = self.head(d1)     # [B, 313, 224, 224]
        return logits


# 稀疏 soft target + 重加权交叉熵

class RebalancedSoftCrossEntropyLoss(nn.Module):
    """
    输入:
      logits       : [B, Q, H, W]
      soft_idx     : [B, K, H, W]
      soft_w       : [B, K, H, W]
      class_weight : [B, H, W]
    逻辑:
      - 先算 log_softmax(logits)
      - 只在 K 个近邻 bin 上 gather
      - 做 soft target 交叉熵
      - 再乘每像素类别权重
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, soft_idx, soft_w, class_weight):
        log_probs = F.log_softmax(logits, dim=1)                      # [B, Q, H, W]
        gathered = torch.gather(log_probs, dim=1, index=soft_idx)    # [B, K, H, W]
        per_pixel_ce = -(soft_w * gathered).sum(dim=1)               # [B, H, W]
        per_pixel_ce = per_pixel_ce * class_weight                   # [B, H, W]
        return per_pixel_ce.mean()


#  annealed-mean 推理

@torch.no_grad()
def annealed_mean_from_logits(logits, ab_bins, T=0.38, eps=1e-8):
    """
    logits:  [B, Q, H, W]
    ab_bins: [Q, 2]
    return:  [B, 2, H, W]
    """
    probs = F.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=eps)

    annealed = torch.exp(torch.log(probs) / T)
    annealed = annealed / torch.clamp(annealed.sum(dim=1, keepdim=True), min=eps)

    ab_bins = ab_bins.to(logits.device).float()                      # [Q, 2]
    ab = torch.einsum("bqhw,qc->bchw", annealed, ab_bins)            # [B, 2, H, W]
    return ab


@torch.no_grad()
def top1_ab_from_logits(logits, ab_bins):
    """
    调试用: 直接取 mode 对应的颜色中心
    """
    idx = torch.argmax(logits, dim=1)                                # [B, H, W]
    ab_bins = ab_bins.to(logits.device).float()
    ab = ab_bins[idx]                                                # [B, H, W, 2]
    ab = ab.permute(0, 3, 1, 2).contiguous()                         # [B, 2, H, W]
    return ab


# 评估指标

@torch.no_grad()
def compute_ab_mae(pred_ab, gt_ab):
    return (pred_ab - gt_ab).abs().mean().item()


@torch.no_grad()
def compute_ab_rmse(pred_ab, gt_ab):
    return torch.sqrt(((pred_ab - gt_ab) ** 2).mean()).item()


#  可视化样本保存

@torch.no_grad()
def save_sample_grid(batch, pred_ab, save_dir, prefix, cfg, max_items=4):
    ensure_dir(save_dir)

    L = denorm_L_tensor(batch["L"], cfg)
    gt_ab = batch["ab"]

    B = min(L.shape[0], max_items)
    for i in range(B):
        l_i = L[i]
        gt_ab_i = gt_ab[i]
        pred_ab_i = pred_ab[i]

        gt_rgb = tensor_lab_to_rgb_image(l_i, gt_ab_i)
        pred_rgb = tensor_lab_to_rgb_image(l_i, pred_ab_i)

        gray_rgb = tensor_lab_to_rgb_image(
            l_i,
            torch.zeros_like(gt_ab_i)
        )

        save_rgb_image(gray_rgb, os.path.join(save_dir, f"{prefix}_{i:02d}_gray.png"))
        save_rgb_image(gt_rgb, os.path.join(save_dir, f"{prefix}_{i:02d}_gt.png"))
        save_rgb_image(pred_rgb, os.path.join(save_dir, f"{prefix}_{i:02d}_pred.png"))


# 单轮训练 / 验证

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, ab_bins, cfg, epoch):
    model.train()

    running_loss = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    num_batches = 0

    start_time = time.time()

    for step, batch in enumerate(loader, start=1):
        L = batch["L"].to(device, non_blocking=True)
        gt_ab = batch["ab"].to(device, non_blocking=True)
        soft_idx = batch["soft_idx"].to(device, non_blocking=True)
        soft_w = batch["soft_w"].to(device, non_blocking=True)
        class_weight = batch["class_weight"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg["use_amp"] and device.startswith("cuda")):
            logits = model(L)
            loss = criterion(logits, soft_idx, soft_w, class_weight)

        scaler.scale(loss).backward()

        if cfg["grad_clip_norm"] is not None and cfg["grad_clip_norm"] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred_ab = annealed_mean_from_logits(
                logits=logits,
                ab_bins=ab_bins,
                T=cfg["annealed_T"],
                eps=cfg["eps"]
            )
            mae = compute_ab_mae(pred_ab, gt_ab)
            rmse = compute_ab_rmse(pred_ab, gt_ab)

        running_loss += loss.item()
        running_mae += mae
        running_rmse += rmse
        num_batches += 1

        if step % cfg["log_interval"] == 0:
            print(
                f"[Train] Epoch {epoch:03d} | Step {step:04d}/{len(loader):04d} "
                f"| loss={running_loss/num_batches:.6f} "
                f"| mae={running_mae/num_batches:.4f} "
                f"| rmse={running_rmse/num_batches:.4f}"
            )

    elapsed = time.time() - start_time
    return {
        "loss": running_loss / max(num_batches, 1),
        "mae": running_mae / max(num_batches, 1),
        "rmse": running_rmse / max(num_batches, 1),
        "time_sec": elapsed,
    }


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, ab_bins, cfg, epoch, save_samples=False):
    model.eval()

    running_loss = 0.0
    running_mae = 0.0
    running_rmse = 0.0
    num_batches = 0

    first_batch = None
    first_pred_ab = None

    start_time = time.time()

    for batch in loader:
        L = batch["L"].to(device, non_blocking=True)
        gt_ab = batch["ab"].to(device, non_blocking=True)
        soft_idx = batch["soft_idx"].to(device, non_blocking=True)
        soft_w = batch["soft_w"].to(device, non_blocking=True)
        class_weight = batch["class_weight"].to(device, non_blocking=True)

        with autocast(enabled=cfg["use_amp"] and device.startswith("cuda")):
            logits = model(L)
            loss = criterion(logits, soft_idx, soft_w, class_weight)

        pred_ab = annealed_mean_from_logits(
            logits=logits,
            ab_bins=ab_bins,
            T=cfg["annealed_T"],
            eps=cfg["eps"]
        )

        mae = compute_ab_mae(pred_ab, gt_ab)
        rmse = compute_ab_rmse(pred_ab, gt_ab)

        running_loss += loss.item()
        running_mae += mae
        running_rmse += rmse
        num_batches += 1

        if first_batch is None:
            first_batch = {
                "L": batch["L"].cpu(),
                "ab": batch["ab"].cpu(),
            }
            first_pred_ab = pred_ab.cpu()

    elapsed = time.time() - start_time

    if save_samples and first_batch is not None:
        save_sample_grid(
            batch=first_batch,
            pred_ab=first_pred_ab,
            save_dir=CFG["sample_dir"],
            prefix=f"epoch_{epoch:03d}",
            cfg=cfg,
            max_items=4
        )

    return {
        "loss": running_loss / max(num_batches, 1),
        "mae": running_mae / max(num_batches, 1),
        "rmse": running_rmse / max(num_batches, 1),
        "time_sec": elapsed,
    }


# checkpoint

def save_checkpoint(state, path):
    ensure_dir(Path(path).parent)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return ckpt


# =========================================================
# 10) 训练主流程
# =========================================================
def build_optimizer(model, cfg):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )
    return optimizer


def build_scheduler(optimizer, cfg):
    if cfg["scheduler_type"] == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["lr_milestones_epoch"],
            gamma=cfg["lr_gamma"],
        )
        return scheduler
    return None


def train_main():
    set_seed(CFG["seed"])

    ensure_dir(CFG["checkpoint_dir"])
    ensure_dir(CFG["sample_dir"])
    ensure_dir(CFG["test_output_dir"])

    train_loader, val_loader, test_loader, meta = build_dataloaders(PATHS, CFG)

    device = CFG["device"]
    ab_bins = meta["ab_bins"].to(device)

    model = UNetColorization(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"],
        base_ch=CFG["base_channels"],
    ).to(device)

    criterion = RebalancedSoftCrossEntropyLoss()
    optimizer = build_optimizer(model, CFG)
    scheduler = build_scheduler(optimizer, CFG)
    scaler = GradScaler(enabled=CFG["use_amp"] and device.startswith("cuda"))

    start_epoch = 1
    best_val_loss = float("inf")

    if CFG["resume_path"]:
        ckpt = load_checkpoint(
            CFG["resume_path"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from: {CFG['resume_path']} | start_epoch={start_epoch}")

    history = []

    for epoch in range(start_epoch, CFG["epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            ab_bins=ab_bins,
            cfg=CFG,
            epoch=epoch,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            ab_bins=ab_bins,
            cfg=CFG,
            epoch=epoch,
            save_samples=True,
        )

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        log_item = {
            "epoch": epoch,
            "lr": lr_now,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(log_item)

        print(
            f"\n[Epoch {epoch:03d}] "
            f"lr={lr_now:.8f} | "
            f"train_loss={train_metrics['loss']:.6f} | train_mae={train_metrics['mae']:.4f} | "
            f"val_loss={val_metrics['loss']:.6f} | val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f}\n"
        )

        latest_ckpt_path = os.path.join(CFG["checkpoint_dir"], "latest.pt")
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "cfg": CFG,
        }
        save_checkpoint(state, latest_ckpt_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_ckpt_path = os.path.join(CFG["checkpoint_dir"], "best.pt")
            state["best_val_loss"] = best_val_loss
            save_checkpoint(state, best_ckpt_path)
            print(f"Saved best checkpoint to: {best_ckpt_path}")

        elif not CFG["save_best_only"] and (epoch % CFG["save_every"] == 0):
            epoch_ckpt_path = os.path.join(CFG["checkpoint_dir"], f"epoch_{epoch:03d}.pt")
            save_checkpoint(state, epoch_ckpt_path)

    history_path = os.path.join(CFG["checkpoint_dir"], "train_history.json")
    save_json(history, history_path)
    print(f"Training history saved to: {history_path}")

    test_main(ckpt_path=os.path.join(CFG["checkpoint_dir"], "best.pt"))


#  测试集推理

@torch.no_grad()
def test_main(ckpt_path=None):
    _, _, test_loader, meta = build_dataloaders(PATHS, CFG)

    device = CFG["device"]
    ab_bins = meta["ab_bins"].to(device)

    model = UNetColorization(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"],
        base_ch=CFG["base_channels"],
    ).to(device)

    if ckpt_path is None:
        ckpt_path = os.path.join(CFG["checkpoint_dir"], "best.pt")

    ckpt = load_checkpoint(
        ckpt_path,
        model=model,
        map_location=device
    )
    print(f"Loaded checkpoint for test: {ckpt_path}")

    model.eval()

    running_mae = 0.0
    running_rmse = 0.0
    num_batches = 0

    ensure_dir(CFG["test_output_dir"])

    for batch_idx, batch in enumerate(test_loader):
        L = batch["L"].to(device, non_blocking=True)
        gt_ab = batch["ab"].to(device, non_blocking=True)

        logits = model(L)
        pred_ab = annealed_mean_from_logits(
            logits=logits,
            ab_bins=ab_bins,
            T=CFG["annealed_T"],
            eps=CFG["eps"]
        )

        mae = compute_ab_mae(pred_ab, gt_ab)
        rmse = compute_ab_rmse(pred_ab, gt_ab)

        running_mae += mae
        running_rmse += rmse
        num_batches += 1

        L_cpu = denorm_L_tensor(batch["L"].cpu(), CFG)
        pred_ab_cpu = pred_ab.cpu()

        for i in range(L_cpu.shape[0]):
            rgb_pred = tensor_lab_to_rgb_image(L_cpu[i], pred_ab_cpu[i])

            src_path = batch["path"][i]
            stem = Path(src_path).stem
            save_path = os.path.join(CFG["test_output_dir"], f"{batch_idx:04d}_{i:02d}_{stem}_pred.png")
            save_rgb_image(rgb_pred, save_path)

    print(
        f"[Test] mae={running_mae / max(num_batches, 1):.4f} | "
        f"rmse={running_rmse / max(num_batches, 1):.4f}"
    )


# 单张图推理

@torch.no_grad()
def colorize_single_image(image_path, ckpt_path=None, save_path=None):
    device = CFG["device"]

    _, _, _, meta = build_dataloaders(PATHS, CFG)
    ab_bins = meta["ab_bins"].to(device)

    model = UNetColorization(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"],
        base_ch=CFG["base_channels"],
    ).to(device)

    if ckpt_path is None:
        ckpt_path = os.path.join(CFG["checkpoint_dir"], "best.pt")

    load_checkpoint(ckpt_path, model=model, map_location=device)
    model.eval()

    img = pil_load_rgb(image_path)
    img = resize_rgb_image(img, CFG["image_size"])
    lab = pil_to_lab(img)

    L = lab[:, :, 0].astype(np.float32)
    if CFG.get("normalize_L", False):
        L_in = (L - CFG["L_mean"]) / CFG["L_std"]
    else:
        L_in = L

    x = torch.from_numpy(L_in[None, None, :, :]).float().to(device)

    logits = model(x)
    pred_ab = annealed_mean_from_logits(
        logits=logits,
        ab_bins=ab_bins,
        T=CFG["annealed_T"],
        eps=CFG["eps"]
    )[0].cpu()

    L_tensor = torch.from_numpy(L[None, :, :]).float()
    rgb_pred = tensor_lab_to_rgb_image(L_tensor, pred_ab)

    if save_path is None:
        stem = Path(image_path).stem
        save_path = os.path.join(CFG["test_output_dir"], f"{stem}_single_pred.png")

    save_rgb_image(rgb_pred, save_path)
    print(f"Saved colorized image to: {save_path}")


# =========================================================
if __name__ == "__main__":
    train_main()
