# ============================================================
# Attention U-Net 图像着色任务 — 完整训练 / 验证 / 测试脚本
# ============================================================
# 运行环境：AutoDL  |  GPU 推荐：单卡或多卡均可自动适配
#
# 依赖安装（首次运行前执行一次）：
#   pip install lpips
#
# 输出目录：/root/autodl-tmp/op_1
#   ├── unet_best.pth          — 验证集上表现最优的模型权重
#   ├── loss_curve.png         — 训练 & 验证 Loss 曲线
#   ├── test_metrics.txt       — PSNR / SSIM / LPIPS 数值结果
#   └── test_visualization.png — 10 张"原图-灰度图-预测图"对比
# ============================================================

import os
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # 无 GUI 服务器环境必须使用非交互后端
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import lpips


# ============================================================
# ★★★  超参数集中配置区  ★★★
# ★  所有关键参数均在此处定义，修改此处即可，无需深入代码  ★
# ============================================================
CFG = {
    # ---------- 路径配置 ----------
    "train_dir"   : "/root/autodl-tmp/ImageNet100/train",   # 训练集目录（ImageFolder 格式）
    "val_dir"     : "/root/autodl-tmp/ImageNet100/val",     # 验证集目录
    "test_dir"    : "/root/autodl-tmp/ImageNet100/test",    # 测试集目录
    "output_dir"  : "/root/autodl-tmp/op_1",                # 权重 & 可视化输出目录

    # ---------- 图像尺寸 ----------
    "img_size"    : 256,     # 所有图像统一缩放至此尺寸（像素），建议 128/256/512

    # ---------- 训练超参数 ----------
    "batch_size"  : 64,      # 每批次样本数量；显存不足时可调小至 32
    "num_epochs"  : 50,      # 总训练轮数
    "lr"          : 0.0002,  # Adam 优化器学习率
    "perc_weight" : 0.5,     # 感知损失（Perceptual Loss）的权重系数；越大色彩越鲜艳
    "num_workers" : 16,      # DataLoader 并行加载线程数；CPU 核数较少时可减小

    # ---------- 可视化 ----------
    "vis_test_num": 10,      # 测试集随机可视化图片数量
}
# ============================================================


# ============================================================
# 1. 数据集定义
# ============================================================
class ImageNetColorizationDataset(Dataset):
    """
    自定义数据集：将彩色 RGB 图像转换至 CIE Lab 颜色空间，
    以 L 通道（灰度/亮度）作为模型输入，ab 通道（色度）作为预测目标。

    参数
    ----
    root_dir : str
        符合 ImageFolder 目录结构的图像根目录
        （子目录为类别名，内含图像文件）
    img_size : int
        统一缩放后的图像边长（像素），默认 256
    is_train : bool
        True  → 训练集模式：加入随机水平翻转做数据增强
        False → 验证/测试集模式：仅做 Resize + CenterCrop
    """
    def __init__(self, root_dir, img_size=256, is_train=False):
        self.is_train = is_train
        self.img_size = img_size

        # 基础变换：缩放 + 中心裁剪 + ToTensor（输出 [0,1] 的 float tensor）
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]

        # 仅训练集启用随机水平翻转（插入在 Resize 之后）
        if is_train:
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=transforms.Compose(transform_list)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        返回
        ----
        L_tensor  : Tensor (1, H, W)  — 归一化 L 通道，范围 [-1, 1]
        ab_tensor : Tensor (2, H, W)  — 归一化 ab 通道，范围 [-1, 1]
        """
        # ImageFolder 返回 (tensor[C,H,W], class_label)，着色任务不需要标签
        img_tensor, _ = self.dataset[idx]

        # tensor (C,H,W) [0,1] → numpy (H,W,C) uint8，送入 OpenCV
        img_rgb = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # RGB → BGR（OpenCV 默认 BGR 通道顺序）
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # BGR [0,255] → Lab（OpenCV 要求输入为 float32 且在 [0,1]）
        img_lab = cv2.cvtColor(
            img_bgr.astype(np.float32) / 255.0,
            cv2.COLOR_BGR2Lab
        )

        # 分离亮度 L 与色度 ab
        L  = img_lab[:, :, 0]    # L 通道，原始范围 [0, 100]
        ab = img_lab[:, :, 1:]   # ab 通道，原始范围约 [-127, 127]

        # 归一化至 [-1, 1]，便于网络训练
        L_norm  = (L / 50.0) - 1.0   # [0,100]    → [-1, 1]
        ab_norm = ab / 127.0          # [-127,127] → [-1, 1]

        # 转换为 Tensor
        L_tensor  = torch.from_numpy(L_norm).unsqueeze(0).float()       # (1, H, W)
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()  # (2, H, W)

        return L_tensor, ab_tensor


# ============================================================
# 2. 模型定义：Attention U-Net（结构与原始代码完全一致）
# ============================================================

class AttentionGate(nn.Module):
    """
    注意力门控模块（Attention Gate, AG）。

    在 U-Net 跳跃连接（skip connection）上加入注意力机制：
    利用解码器 gating signal (g) 对编码器特征 (x) 进行空间权重调制，
    抑制无关背景区域的响应，让解码器聚焦于显著目标区域。

    参数
    ----
    F_g   : int — gating signal 通道数（来自解码器上采样输出）
    F_l   : int — 跳跃连接特征通道数（来自编码器对应层）
    F_int : int — 中间层通道数（通常取 min(F_g, F_l) 的一半）
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # 对 gating signal 做 1×1 卷积降维
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 对跳跃连接特征做 1×1 卷积降维
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 将两路特征融合后生成 [0,1] 空间权重图（注意力系数）
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g : Tensor — gating signal（解码器上采样结果）
        x : Tensor — 跳跃连接特征（编码器对应层输出）
        返回：经注意力加权后的跳跃连接特征
        """
        g1    = self.W_g(g)
        x1    = self.W_l(x)
        psi   = self.relu(g1 + x1)   # 两路特征相加后 ReLU 激活
        alpha = self.psi(psi)         # 生成空间注意力权重图
        return x * alpha              # 对跳跃连接特征进行逐像素加权


class AttentionUNet(nn.Module):
    """
    Attention U-Net：在标准 U-Net 基础上，于每个 skip connection
    处加入 AttentionGate，用于图像着色任务。

    输入  : (B, 1, H, W)  — 单通道 L 图（灰度）
    输出  : (B, 2, H, W)  — 双通道 ab 预测（tanh 激活，范围 [-1,1]）

    网络结构
    --------
    编码器（3 层下采样 + MaxPool）：
        enc1: 1   → 64ch
        enc2: 64  → 128ch
        enc3: 128 → 256ch
    瓶颈层：
        bottleneck: 256 → 512ch
    解码器（3 层上采样 + AttentionGate + skip concat）：
        up3 + att3 + dec3: 512 → 256ch
        up2 + att2 + dec2: 256 → 128ch
        up1 + att1 + dec1: 128 → 64ch
    输出层：
        final: 64 → 2ch (1×1 conv + tanh)
    """
    def __init__(self):
        super(AttentionUNet, self).__init__()

        # ---- 辅助函数：双层卷积块（Conv2d-BN-ReLU × 2）----
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # ---- 编码器 ----
        self.enc1 = conv_block(1,   64)
        self.enc2 = conv_block(64,  128)
        self.enc3 = conv_block(128, 256)

        # ---- 瓶颈层 ----
        self.bottleneck = conv_block(256, 512)

        # ---- 解码器 ----
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = conv_block(512, 256)   # 上采样 256 + skip 256 = 512 输入

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = conv_block(256, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = conv_block(128, 64)

        # ---- 输出层：1×1 卷积 + tanh ----
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # === 编码路径 ===
        e1 = self.enc1(x)                      # (B, 64,  H,   W)
        e2 = self.enc2(F.max_pool2d(e1, 2))    # (B, 128, H/2, W/2)
        e3 = self.enc3(F.max_pool2d(e2, 2))    # (B, 256, H/4, W/4)
        b  = self.bottleneck(F.max_pool2d(e3, 2))  # (B, 512, H/8, W/8)

        # === 解码路径（含注意力门控 + 跳跃连接拼接）===
        d3 = self.up3(b)                        # (B, 256, H/4, W/4)
        x3 = self.att3(g=d3, x=e3)             # 注意力加权的 e3
        d3 = self.dec3(torch.cat([d3, x3], dim=1))  # (B, 256, H/4, W/4)

        d2 = self.up2(d3)                       # (B, 128, H/2, W/2)
        x2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))  # (B, 128, H/2, W/2)

        d1 = self.up1(d2)                       # (B, 64, H, W)
        x1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))  # (B, 64, H, W)

        return torch.tanh(self.final(d1))       # (B, 2, H, W)，输出 ab 预测


# ============================================================
# 3. 感知损失（Perceptual Loss，基于 VGG16 深层特征）
# ============================================================
class PerceptualLoss(nn.Module):
    """
    利用预训练 VGG16 的中间层特征计算感知损失。

    原理：直接 MSE 约束像素值容易产生平均色（灰色/棕色）；
    而感知损失约束高层语义特征，能鼓励模型预测出更鲜明的颜色。

    使用 VGG16 前 16 层（到 relu3_3），权重固定不参与训练。
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        # 截取 VGG16 前 16 层，设置为评估模式并冻结权重
        self.loss_network = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

    def forward(self, pred_rgb, gt_rgb):
        """
        pred_rgb : Tensor (B, 3, H, W) — 预测图像
        gt_rgb   : Tensor (B, 3, H, W) — 真实图像
        返回：两者 VGG 特征之间的 MSE 损失值
        """
        pred_feat = self.loss_network(pred_rgb)
        with torch.no_grad():
            gt_feat = self.loss_network(gt_rgb)
        return F.mse_loss(pred_feat, gt_feat)


# ============================================================
# 4. 工具函数
# ============================================================
def tensor2rgb(L_tensor, ab_tensor):
    """
    将归一化后的 L 通道 tensor 与 ab 通道 tensor 拼合，
    转换回 RGB numpy 图像。

    参数
    ----
    L_tensor  : Tensor (1, H, W)，归一化 L，范围 [-1, 1]
    ab_tensor : Tensor (2, H, W)，归一化 ab，范围 [-1, 1]

    返回
    ----
    rgb : numpy array (H, W, 3)，float32，范围 [0, 1]
    """
    img_size = L_tensor.shape[-1]

    # 反归一化：还原到 OpenCV Lab 的原始数值范围
    L  = (L_tensor.cpu().squeeze().numpy() + 1.0) * 50.0        # [-1,1] → [0,100]
    ab = ab_tensor.cpu().permute(1, 2, 0).numpy() * 127.0        # [-1,1] → [-127,127]

    # 合并为三通道 Lab 图像
    lab = np.zeros((img_size, img_size, 3), dtype=np.float32)
    lab[:, :, 0]  = L
    lab[:, :, 1:] = ab

    # Lab → BGR → RGB
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return np.clip(rgb, 0, 1)   # 防止数值越界


# ============================================================
# 5. 独立验证函数
# ============================================================
def validate(model, val_loader, criterion_mse, criterion_perceptual,
             device, perc_weight):
    """
    在验证集上计算平均总损失（MSE + 感知损失），不更新参数。

    参数
    ----
    model                : AttentionUNet 模型
    val_loader           : 验证集 DataLoader
    criterion_mse        : nn.MSELoss()
    criterion_perceptual : PerceptualLoss()
    device               : torch.device
    perc_weight          : float — 感知损失权重系数

    返回
    ----
    avg_val_loss : float — 验证集平均总损失
    """
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for L, ab in val_loader:
            L, ab = L.to(device), ab.to(device)

            ab_pred = model(L)

            # MSE 损失
            loss_mse = criterion_mse(ab_pred, ab)

            # 感知损失（构造 3 通道输入，扩展 Lab → 伪 RGB）
            input_vgg_pred = torch.cat([L, ab_pred], dim=1).expand(-1, 3, -1, -1)
            input_vgg_gt   = torch.cat([L, ab],      dim=1).expand(-1, 3, -1, -1)
            loss_p         = criterion_perceptual(input_vgg_pred, input_vgg_gt)

            total_loss = loss_mse + perc_weight * loss_p
            running_val_loss += total_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    return avg_val_loss


# ============================================================
# 6. 独立测试函数（PSNR / SSIM / LPIPS + 可视化）
# ============================================================
def test(model, test_loader, device, output_dir, vis_num=10):
    """
    在测试集上执行完整评估，输出三项图像质量指标并保存对比可视化图。

    指标说明
    --------
    PSNR  (Peak Signal-to-Noise Ratio) ↑ 越高越好，单位 dB
          反映像素级重建精度，一般 >30dB 为较好。
    SSIM  (Structural Similarity Index) ↑ 越高越好，范围 [0,1]
          综合亮度、对比度、结构三维度的感知相似度。
    LPIPS (Learned Perceptual Image Patch Similarity) ↓ 越低越好
          基于深度网络特征的感知距离，与人类视觉判断高度相关。

    参数
    ----
    model       : 加载了最优权重的 AttentionUNet 模型
    test_loader : 测试集 DataLoader（shuffle=True 保证可视化随机性）
    device      : torch.device
    output_dir  : str — 结果保存路径
    vis_num     : int — 可视化对比图的图像数量（默认 10）

    返回
    ----
    (mean_psnr, mean_ssim, mean_lpips) : tuple of float
    """
    print("\\n" + "=" * 60)
    print("★  开始测试（PSNR / SSIM / LPIPS）")
    print("=" * 60)

    # 初始化 LPIPS 评估器（AlexNet 特征，速度快且效果好）
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    model.eval()

    psnr_list  = []
    ssim_list  = []
    lpips_list = []
    vis_samples = []   # 收集用于可视化的样本

    with torch.no_grad():
        for L, ab in test_loader:
            L, ab = L.to(device), ab.to(device)
            ab_pred = model(L)   # 前向推理

            # 逐样本（逐 batch 内每张图）计算指标
            for b in range(L.size(0)):
                # 转换为 RGB numpy 图像，范围 [0, 1]
                img_gt   = tensor2rgb(L[b], ab[b])       # 真实彩色图（Ground Truth）
                img_pred = tensor2rgb(L[b], ab_pred[b])  # 模型预测着色图

                # ---- PSNR ----
                psnr_val = psnr_fn(img_gt, img_pred, data_range=1.0)
                psnr_list.append(psnr_val)

                # ---- SSIM（channel_axis=2 表示 HWC 的 RGB 三通道）----
                ssim_val = ssim_fn(
                    img_gt, img_pred,
                    data_range=1.0,
                    channel_axis=2
                )
                ssim_list.append(ssim_val)

                # ---- LPIPS（输入需要 [-1,1] 范围的 tensor）----
                gt_t   = (torch.from_numpy(img_gt).permute(2, 0, 1)
                          .unsqueeze(0).float().to(device)) * 2 - 1
                pred_t = (torch.from_numpy(img_pred).permute(2, 0, 1)
                          .unsqueeze(0).float().to(device)) * 2 - 1
                lpips_val = lpips_fn(pred_t, gt_t).item()
                lpips_list.append(lpips_val)

                # 收集可视化样本（仅收集前 vis_num 张）
                if len(vis_samples) < vis_num:
                    # 灰度图：L 通道反归一化至 [0,1]
                    gray_img = (L[b].cpu().squeeze().numpy() + 1.0) / 2.0
                    vis_samples.append({
                        "original": img_gt,
                        "gray"    : gray_img,
                        "pred"    : img_pred,
                        "psnr"    : psnr_val,
                        "ssim"    : ssim_val,
                        "lpips"   : lpips_val,
                    })

    # ---- 汇总并打印平均指标 ----
    mean_psnr  = float(np.mean(psnr_list))
    mean_ssim  = float(np.mean(ssim_list))
    mean_lpips = float(np.mean(lpips_list))

    print(f"\\n  测试集共 {len(psnr_list)} 张图像：")
    print(f"  PSNR  (↑) : {mean_psnr:.4f} dB")
    print(f"  SSIM  (↑) : {mean_ssim:.4f}")
    print(f"  LPIPS (↓) : {mean_lpips:.4f}")

    # 将指标写入文本文件
    metrics_path = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"测试集评估结果（共 {len(psnr_list)} 张图像）\\n")
        f.write(f"PSNR  (↑): {mean_psnr:.4f} dB\\n")
        f.write(f"SSIM  (↑): {mean_ssim:.4f}\\n")
        f.write(f"LPIPS (↓): {mean_lpips:.4f}\\n")
    print(f"  [*] 指标已保存至：{metrics_path}")

    # ---- 可视化：原图 - 灰度图 - 预测图 对比 ----
    if len(vis_samples) > 0:
        actual_num = len(vis_samples)
        fig, axes = plt.subplots(actual_num, 3, figsize=(12, 4 * actual_num))

        # actual_num == 1 时 axes 不是二维数组，统一处理为列表
        if actual_num == 1:
            axes = [axes]

        for i, sample in enumerate(vis_samples):
            # 第 1 列：原图（Ground Truth 彩色图）
            axes[i][0].imshow(sample["original"])
            axes[i][0].set_title("Original (GT)", fontsize=10)
            axes[i][0].axis("off")

            # 第 2 列：灰度图（L 通道）
            axes[i][1].imshow(sample["gray"], cmap="gray")
            axes[i][1].set_title("Grayscale (L)", fontsize=10)
            axes[i][1].axis("off")

            # 第 3 列：预测图（模型上色结果），标题附带指标数值
            axes[i][2].imshow(sample["pred"])
            axes[i][2].set_title(
                f"Pred | PSNR:{sample[\'psnr\']:.2f}dB "
                f"SSIM:{sample[\'ssim\']:.3f} LPIPS:{sample[\'lpips\']:.3f}",
                fontsize=8
            )
            axes[i][2].axis("off")

        plt.suptitle(
            f"Test Visualization — {actual_num} Random Samples",
            fontsize=14, y=1.005
        )
        plt.tight_layout()
        vis_path = os.path.join(output_dir, "test_visualization.png")
        plt.savefig(vis_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  [*] 可视化对比图已保存至：{vis_path}")

    return mean_psnr, mean_ssim, mean_lpips


# ============================================================
# 7. 主程序
# ============================================================
if __name__ == "__main__":

    # ---- 从超参数配置区读取所有参数 ----
    TRAIN_DIR   = CFG["train_dir"]
    VAL_DIR     = CFG["val_dir"]
    TEST_DIR    = CFG["test_dir"]
    OUTPUT_DIR  = CFG["output_dir"]
    IMG_SIZE    = CFG["img_size"]
    BATCH_SIZE  = CFG["batch_size"]
    NUM_EPOCHS  = CFG["num_epochs"]
    LR          = CFG["lr"]
    PERC_WEIGHT = CFG["perc_weight"]
    NUM_WORKERS = CFG["num_workers"]
    VIS_NUM     = CFG["vis_test_num"]

    # 创建输出目录（不存在时自动创建）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 自动选择计算设备（优先 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用：{device}")
    print(f"[超参数] img_size={IMG_SIZE} | batch_size={BATCH_SIZE} | "
          f"num_epochs={NUM_EPOCHS} | lr={LR} | perc_weight={PERC_WEIGHT} | "
          f"num_workers={NUM_WORKERS}")

    # ----------------------------------------------------------
    # 7.1 构建三个独立数据集 & DataLoader
    #     训练 / 验证 / 测试 分别使用各自目录，互不混用
    # ----------------------------------------------------------
    print("\\n[数据] 正在加载三个数据集...")

    # 训练集：开启随机水平翻转做数据增强，shuffle=True
    train_dataset = ImageNetColorizationDataset(
        root_dir=TRAIN_DIR, img_size=IMG_SIZE, is_train=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    # 验证集：不做增强，shuffle=False（保证评估结果可复现）
    val_dataset = ImageNetColorizationDataset(
        root_dir=VAL_DIR, img_size=IMG_SIZE, is_train=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    # 测试集：不做增强，shuffle=True（保证可视化抽样的随机性）
    test_dataset = ImageNetColorizationDataset(
        root_dir=TEST_DIR, img_size=IMG_SIZE, is_train=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
    )

    print(f"  训练集：{len(train_dataset)} 张图像")
    print(f"  验证集：{len(val_dataset)} 张图像")
    print(f"  测试集：{len(test_dataset)} 张图像")

    # ----------------------------------------------------------
    # 7.2 初始化模型、损失函数、优化器
    # ----------------------------------------------------------
    model = AttentionUNet().to(device)

    # 多 GPU 自动并行（若检测到多张 GPU）
    if torch.cuda.device_count() > 1:
        print(f"[GPU] 检测到 {torch.cuda.device_count()} 张 GPU，启用 DataParallel")
        model = nn.DataParallel(model)

    criterion_mse        = nn.MSELoss()
    criterion_perceptual = PerceptualLoss().to(device)
    optimizer            = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")   # 用于判断是否保存最优权重
    train_losses  = []             # 记录每 epoch 训练平均 Loss
    val_losses    = []             # 记录每 epoch 验证平均 Loss

    # ----------------------------------------------------------
    # 7.3 训练 & 验证循环
    # ----------------------------------------------------------
    print("\\n" + "=" * 60)
    print("★  训练启动")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):

        # ===== 训练阶段 =====
        model.train()
        running_loss = 0.0

        for i, (L, ab) in enumerate(train_loader):
            L, ab = L.to(device), ab.to(device)

            optimizer.zero_grad()
            ab_pred = model(L)

            # MSE 损失：逐像素颜色误差
            loss_mse = criterion_mse(ab_pred, ab)

            # 感知损失：将预测/真实 Lab 拼合后 expand 为 3ch 送入 VGG
            input_vgg_pred = torch.cat([L, ab_pred], dim=1).expand(-1, 3, -1, -1)
            input_vgg_gt   = torch.cat([L, ab],      dim=1).expand(-1, 3, -1, -1)
            loss_p         = criterion_perceptual(input_vgg_pred, input_vgg_gt)

            # 总损失 = MSE + 感知权重 × 感知损失
            total_loss = loss_mse + PERC_WEIGHT * loss_p
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            # 每 100 步打印一次批次损失（方便观察训练动态）
            if (i + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] "
                      f"Step [{i+1}/{len(train_loader)}]  "
                      f"Batch Loss: {total_loss.item():.4f}")

        # 记录本 epoch 训练平均 Loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ===== 验证阶段（独立函数，不影响训练逻辑）=====
        avg_val_loss = validate(
            model, val_loader,
            criterion_mse, criterion_perceptual,
            device, PERC_WEIGHT
        )
        val_losses.append(avg_val_loss)

        print(f"=== Epoch [{epoch+1}/{NUM_EPOCHS}]  "
              f"Train Loss: {avg_train_loss:.4f}  |  "
              f"Val Loss: {avg_val_loss:.4f} ===")

        # ===== 保存最优权重（以验证集 Loss 为判断依据）=====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch"               : epoch + 1,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss"          : avg_train_loss,
                "val_loss"            : avg_val_loss,
                "train_loss_history"  : train_losses,
                "val_loss_history"    : val_losses,
            }
            save_path = os.path.join(OUTPUT_DIR, "unet_best.pth")
            torch.save(checkpoint, save_path)
            print(f"  [*] 更新最优权重 "
                  f"(val_loss={best_val_loss:.6f}) → {save_path}")

    # ----------------------------------------------------------
    # 7.4 绘制训练 & 验证 Loss 曲线并保存
    # ----------------------------------------------------------
    print("\\n[可视化] 正在绘制 Loss 曲线...")
    epoch_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_range, train_losses, "b-o", markersize=4, label="Train Loss")
    plt.plot(epoch_range, val_losses,   "r-s", markersize=4, label="Val Loss")
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()
    print(f"[*] Loss 曲线已保存至：{loss_curve_path}")

    # ----------------------------------------------------------
    # 7.5 加载最优权重，在测试集上完成最终评估
    # ----------------------------------------------------------
    print("\\n[测试] 加载最优权重进行测试...")
    best_ckpt_path = os.path.join(OUTPUT_DIR, "unet_best.pth")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[*] 已加载 Epoch {checkpoint[\'epoch\']} 的最优权重")

    # 执行完整测试评估（PSNR / SSIM / LPIPS + 可视化对比图）
    mean_psnr, mean_ssim, mean_lpips = test(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=OUTPUT_DIR,
        vis_num=VIS_NUM
    )

    print("\\n" + "=" * 60)
    print("★  全部流程完成！")
    print(f"   PSNR  : {mean_psnr:.4f} dB")
    print(f"   SSIM  : {mean_ssim:.4f}")
    print(f"   LPIPS : {mean_lpips:.4f}")
    print(f"   所有输出文件已保存至：{OUTPUT_DIR}")
    print("=" * 60)

    # 训练结束后自动关机（AutoDL 环境）
    os.system("shutdown")

os.makedirs("output", exist_ok=True)
with open("output/cv_1_final.py", "w", encoding="utf-8") as f:
    f.write(code)

print(f"代码已生成，字符数：{len(code)}")