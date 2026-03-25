import os
import json
import copy
import time
import random
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import lab2rgb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


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
    "checkpoint_dir": "/autodl-tmp/op_2/checkpoints",
    "sample_dir": "/autodl-tmp/op_2/val_samples",
    "test_output_dir": "/autodl-tmp/op_2/test_preds",
    "test_triplet_dir": "/autodl-tmp/op_2/test_triplets",
    "plot_dir": "/autodl-tmp/op_2/plots",
    "history_path": "/autodl-tmp/op_2/train_history.json",
    "resume_path": "",
    "save_best_only": True,
    "scheduler_type": "multistep",
    "lr_milestones_epoch": [25, 40],
    "lr_gamma": 0.316227766,
    "use_data_parallel": True,
    "test_visual_num": 10,
})


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
        self.pool = nn.MaxPool2d(2, 2)
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
        return self.conv(x)


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
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


class RebalancedSoftCrossEntropyLoss(nn.Module):
    def forward(self, logits, soft_idx, soft_w, class_weight):
        log_probs = F.log_softmax(logits, dim=1)
        gathered = torch.gather(log_probs, dim=1, index=soft_idx)
        per_pixel_ce = -(soft_w * gathered).sum(dim=1)
        per_pixel_ce = per_pixel_ce * class_weight
        return per_pixel_ce.mean()


@torch.no_grad()
def annealed_mean_from_logits(logits, ab_bins, T=0.38, eps=1e-8):
    probs = F.softmax(logits, dim=1)
    probs = torch.clamp(probs, min=eps)
    annealed = torch.exp(torch.log(probs) / T)
    annealed = annealed / torch.clamp(annealed.sum(dim=1, keepdim=True), min=eps)
    ab_bins = ab_bins.to(logits.device).float()
    return torch.einsum("bqhw,qc->bchw", annealed, ab_bins)


@torch.no_grad()
def compute_ab_mae(pred_ab, gt_ab):
    return (pred_ab - gt_ab).abs().mean().item()


@torch.no_grad()
def compute_ab_rmse(pred_ab, gt_ab):
    return torch.sqrt(((pred_ab - gt_ab) ** 2).mean()).item()


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
        gray_rgb = tensor_lab_to_rgb_image(l_i, torch.zeros_like(gt_ab_i))

        save_rgb_image(gray_rgb, os.path.join(save_dir, f"{prefix}_{i:02d}_gray.png"))
        save_rgb_image(gt_rgb, os.path.join(save_dir, f"{prefix}_{i:02d}_gt.png"))
        save_rgb_image(pred_rgb, os.path.join(save_dir, f"{prefix}_{i:02d}_pred.png"))


def plot_loss_curves(history, save_path):
    ensure_dir(Path(save_path).parent)
    epochs = [x["epoch"] for x in history]
    train_loss = [x["train"]["loss"] for x in history]
    val_loss = [x["val"]["loss"] for x in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="train_loss")
    plt.plot(epochs, val_loss, marker="s", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Val Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_triplet_figure(original_rgb, gray_rgb, pred_rgb, save_path, title_text):
    ensure_dir(Path(save_path).parent)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(gray_rgb)
    axes[1].set_title("Gray")
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Pred")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title_text)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(state, path):
    ensure_dir(Path(path).parent)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def build_optimizer(model, cfg):
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )


def build_scheduler(optimizer, cfg):
    if cfg["scheduler_type"] == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg["lr_milestones_epoch"],
            gamma=cfg["lr_gamma"],
        )
    return None


def maybe_wrap_dataparallel(model, cfg):
    if cfg["device"].startswith("cuda") and torch.cuda.device_count() >= 2 and cfg.get("use_data_parallel", True):
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model, device_ids=[0, 1])
    else:
        print(f"Using single device: {cfg['device']}")
    return model


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
                eps=cfg["eps"],
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

    return {
        "loss": running_loss / max(num_batches, 1),
        "mae": running_mae / max(num_batches, 1),
        "rmse": running_rmse / max(num_batches, 1),
        "time_sec": time.time() - start_time,
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
            eps=cfg["eps"],
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

    if save_samples and first_batch is not None:
        save_sample_grid(
            batch=first_batch,
            pred_ab=first_pred_ab,
            save_dir=cfg["sample_dir"],
            prefix=f"epoch_{epoch:03d}",
            cfg=cfg,
            max_items=4,
        )

    return {
        "loss": running_loss / max(num_batches, 1),
        "mae": running_mae / max(num_batches, 1),
        "rmse": running_rmse / max(num_batches, 1),
        "time_sec": time.time() - start_time,
    }


def train_main():
    set_seed(CFG["seed"])

    for d in [
        CFG["checkpoint_dir"],
        CFG["sample_dir"],
        CFG["test_output_dir"],
        CFG["plot_dir"],
        CFG["test_triplet_dir"],
    ]:
        ensure_dir(d)

    train_loader, val_loader, _, meta = build_dataloaders(PATHS, CFG)

    device = CFG["device"]
    ab_bins = meta["ab_bins"].to(device)

    model = UNetColorization(
        in_channels=CFG["in_channels"],
        num_classes=CFG["num_classes"],
        base_ch=CFG["base_channels"],
    ).to(device)

    model = maybe_wrap_dataparallel(model, CFG)

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
            map_location=device,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from: {CFG['resume_path']} | start_epoch={start_epoch}")

    history = []

    for epoch in range(start_epoch, CFG["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, ab_bins, CFG, epoch
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, ab_bins, CFG, epoch, save_samples=True
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
            f"[Epoch {epoch:03d}] lr={lr_now:.8f} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"train_mae={train_metrics['mae']:.4f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_val_loss": best_val_loss,
            "cfg": CFG,
        }

        save_checkpoint(state, os.path.join(CFG["checkpoint_dir"], "latest.pt"))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            state["best_val_loss"] = best_val_loss
            save_checkpoint(state, os.path.join(CFG["checkpoint_dir"], "best.pt"))
            print("Saved new best checkpoint")
        elif (not CFG["save_best_only"]) and (epoch % CFG["save_every"] == 0):
            save_checkpoint(state, os.path.join(CFG["checkpoint_dir"], f"epoch_{epoch:03d}.pt"))

        save_json(history, CFG["history_path"])
        plot_loss_curves(history, os.path.join(CFG["plot_dir"], "loss_curve.png"))

    test_main(ckpt_path=os.path.join(CFG["checkpoint_dir"], "best.pt"))


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

    model = maybe_wrap_dataparallel(model, CFG)

    if ckpt_path is None:
        ckpt_path = os.path.join(CFG["checkpoint_dir"], "best.pt")

    load_checkpoint(ckpt_path, model=model, map_location=device)
    model.eval()

    ensure_dir(CFG["test_output_dir"])
    ensure_dir(CFG["test_triplet_dir"])

    running_mae = 0.0
    running_rmse = 0.0
    num_batches = 0
    saved_triplets = 0

    rng = random.Random(CFG["seed"])
    sample_targets = set(
        rng.sample(range(meta["num_test"]), min(CFG["test_visual_num"], meta["num_test"]))
    )

    global_idx = 0

    for batch_idx, batch in enumerate(test_loader):
        L = batch["L"].to(device, non_blocking=True)
        gt_ab = batch["ab"].to(device, non_blocking=True)

        logits = model(L)
        pred_ab = annealed_mean_from_logits(
            logits=logits,
            ab_bins=ab_bins,
            T=CFG["annealed_T"],
            eps=CFG["eps"],
        )

        running_mae += compute_ab_mae(pred_ab, gt_ab)
        running_rmse += compute_ab_rmse(pred_ab, gt_ab)
        num_batches += 1

        L_cpu = denorm_L_tensor(batch["L"].cpu(), CFG)
        gt_ab_cpu = batch["ab"].cpu()
        pred_ab_cpu = pred_ab.cpu()

        for i in range(L_cpu.shape[0]):
            src_path = batch["path"][i]
            stem = Path(src_path).stem

            pred_rgb = tensor_lab_to_rgb_image(L_cpu[i], pred_ab_cpu[i])
            gt_rgb = tensor_lab_to_rgb_image(L_cpu[i], gt_ab_cpu[i])
            gray_rgb = tensor_lab_to_rgb_image(L_cpu[i], torch.zeros_like(gt_ab_cpu[i]))

            save_rgb_image(
                pred_rgb,
                os.path.join(CFG["test_output_dir"], f"{batch_idx:04d}_{i:02d}_{stem}_pred.png")
            )

            if global_idx in sample_targets:
                save_triplet_figure(
                    gt_rgb,
                    gray_rgb,
                    pred_rgb,
                    os.path.join(CFG["test_triplet_dir"], f"{saved_triplets:02d}_{stem}_triplet.png"),
                    title_text=stem,
                )
                saved_triplets += 1

            global_idx += 1

    print(
        f"[Test] mae={running_mae / max(num_batches, 1):.4f} | "
        f"rmse={running_rmse / max(num_batches, 1):.4f} | "
        f"saved_triplets={saved_triplets}"
    )


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

    model = maybe_wrap_dataparallel(model, CFG)

    if ckpt_path is None:
        ckpt_path = os.path.join(CFG["checkpoint_dir"], "best.pt")

    load_checkpoint(ckpt_path, model=model, map_location=device)
    model.eval()

    img = pil_load_rgb(image_path)
    img = resize_rgb_image(img, CFG["image_size"])
    lab = pil_to_lab(img)

    L = lab[:, :, 0].astype(np.float32)
    L_in = (L - CFG["L_mean"]) / CFG["L_std"] if CFG.get("normalize_L", False) else L

    x = torch.from_numpy(L_in[None, None, :, :]).float().to(device)

    logits = model(x)
    pred_ab = annealed_mean_from_logits(
        logits=logits,
        ab_bins=ab_bins,
        T=CFG["annealed_T"],
        eps=CFG["eps"],
    )[0].cpu()

    L_tensor = torch.from_numpy(L[None, :, :]).float()
    pred_rgb = tensor_lab_to_rgb_image(L_tensor, pred_ab)

    if save_path is None:
        stem = Path(image_path).stem
        save_path = os.path.join(CFG["test_output_dir"], f"{stem}_single_pred.png")

    save_rgb_image(pred_rgb, save_path)
    print(f"Saved colorized image to: {save_path}")


if __name__ == "__main__":
    train_main()

