import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips  # 用于计算 LPIPS 评估指标

# ==========================================
# 💡 超参数与配置中心 (方便统一查看与调节)
# ==========================================
CONFIG = {
    'train_dir': '/root/autodl-tmp/ImageNet100/train',
    'val_dir': '/root/autodl-tmp/ImageNet100/val',
    'test_dir': '/root/autodl-tmp/ImageNet100/test',
    'save_dir': '/root/autodl-tmp/op_1',
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 0.0002,
    'img_size': 256,
    'num_workers': 16,
    'perceptual_weight': 0.5,  # 感知损失的权重
    'seed': 42  # 随机种子，保证测试集抽图可复现
}

# 设置随机种子
torch.manual_seed(CONFIG['seed'])
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])


# ==========================================
# 1. 数据集定义
# ==========================================
class ImageNetColorizationDataset(Dataset):
    def __init__(self, root_dir, img_size=256, is_train=True):
        self.is_train = is_train
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size)
        ]
        # 仅在训练集应用数据增强
        if is_train:
            transform_list.insert(1, transforms.RandomHorizontalFlip())

        self.dataset = datasets.ImageFolder(root=root_dir, transform=transforms.Compose(transform_list))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_pil, _ = self.dataset[idx]
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

        L = img_lab[:, :, 0]
        ab = img_lab[:, :, 1:]

        L_norm = (L / 50.0) - 1.0
        ab_norm = ab / 127.0

        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()

        return L_tensor, ab_tensor


# ==========================================
# 2. 模型定义 (保持原有的 Attention U-Net)
# ==========================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        psi = self.relu(g1 + x1)
        return x * self.psi(psi)


class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.bottleneck = conv_block(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.up3(b)
        x3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))
        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))
        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))
        return torch.tanh(self.final(d1))


# ==========================================
# 3. 损失函数与工具函数
# ==========================================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.loss_network = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

    def forward(self, pred_rgb, gt_rgb):
        pred_feat = self.loss_network(pred_rgb)
        with torch.no_grad():
            gt_feat = self.loss_network(gt_rgb)
        return F.mse_loss(pred_feat, gt_feat)


def tensor2rgb(L_tensor, ab_tensor):
    """将单张图片的 L 和 ab 张量转换为 RGB 的 numpy 数组 (范围 [0, 1])"""
    L = ((L_tensor.cpu().squeeze().numpy() + 1.0) * 50.0)
    ab = (ab_tensor.cpu().permute(1, 2, 0).numpy() * 127.0)
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.clip(rgb, 0, 1)


# ==========================================
# 4. 主程序 (Train, Val, Test)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # --- 数据集加载 ---
    train_dataset = ImageNetColorizationDataset(root_dir=CONFIG['train_dir'], img_size=CONFIG['img_size'],
                                                is_train=True)
    val_dataset = ImageNetColorizationDataset(root_dir=CONFIG['val_dir'], img_size=CONFIG['img_size'], is_train=False)
    test_dataset = ImageNetColorizationDataset(root_dir=CONFIG['test_dir'], img_size=CONFIG['img_size'], is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    # 测试集设置 batch_size 为 1 方便逐张计算指标和提图
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    # --- 模型与优化器 ---
    model = AttentionUNet().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion_mse = nn.MSELoss()
    criterion_perceptual = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    # 💡 新增：初始化 AMP 的梯度缩放器
    scaler = torch.cuda.amp.GradScaler()
    # 记录列表
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"========== 开始训练 (共 {CONFIG['num_epochs']} Epochs) ==========")
    for epoch in range(CONFIG['num_epochs']):

        # ---------------- 训练阶段 ----------------
        model.train()
        running_train_loss = 0.0
        for i, (L, ab) in enumerate(train_loader):
            L, ab = L.to(device), ab.to(device)

            optimizer.zero_grad()

            # 💡 新增：开启自动混合精度上下文
            with torch.cuda.amp.autocast():
                # --- 前向传播与计算 Loss (这些会在半精度下超快运行) ---
                ab_pred = model(L)

                loss_mse = criterion_mse(ab_pred, ab)
                input_vgg_pred = torch.cat([L, ab_pred], dim=1)
                input_vgg_gt = torch.cat([L, ab], dim=1)
                loss_p = criterion_perceptual(input_vgg_pred, input_vgg_gt)

                total_loss = loss_mse + CONFIG['perceptual_weight'] * loss_p

            # 💡 修改：使用 scaler 进行反向传播和参数更新
            scaler.scale(total_loss).backward()  # 缩放 Loss 并反向传播
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器的比例因子

            running_train_loss += total_loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f"Train - Epoch [{epoch + 1}/{CONFIG['num_epochs']}], Step [{i + 1}], Loss: {total_loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------------- 验证阶段 ----------------
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for L_val, ab_val in val_loader:
                L_val, ab_val = L_val.to(device), ab_val.to(device)
                ab_pred_val = model(L_val)

                loss_mse = criterion_mse(ab_pred_val, ab_val)
                input_vgg_pred = torch.cat([L_val, ab_pred_val], dim=1).expand(-1, 3, -1, -1)
                input_vgg_gt = torch.cat([L_val, ab_val], dim=1).expand(-1, 3, -1, -1)
                loss_p = criterion_perceptual(input_vgg_pred, input_vgg_gt)

                val_loss = loss_mse + CONFIG['perceptual_weight'] * loss_p
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"=== Epoch [{epoch + 1}/{CONFIG['num_epochs']}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ===")

        # ---------------- 保存最优权重 ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, os.path.join(CONFIG['save_dir'], 'unet_best.pth'))
            print(f"[*] 发现更优模型！已将最佳权重保存至 {CONFIG['save_dir']} (Val Loss={best_val_loss:.6f})")

    # ---------------- 训练结束，绘制 Loss 曲线 ----------------
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, CONFIG['num_epochs'] + 1), train_losses, 'b-o', label='Train Loss')
    plt.plot(range(1, CONFIG['num_epochs'] + 1), val_losses, 'r-o', label='Val Loss')
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(CONFIG['save_dir'], 'loss_curve.png'))
    plt.close()
    print("已保存 Loss 曲线。准备进入测试阶段...")

    # ==========================================
    # 5. 全新测试阶段 (PSNR, SSIM, LPIPS 及 可视化)
    # ==========================================
    print(f"\n========== 开始测试 ==========")
    # 加载最佳模型
    best_checkpoint = torch.load(os.path.join(CONFIG['save_dir'], 'unet_best.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()

    # 初始化评估指标和 LPIPS 模型
    lpips_fn = lpips.LPIPS(net='alex').to(device)  # 测试标准通常使用 alex net
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_test_samples = len(test_loader)

    # 用于存储需要可视化的 10 张图
    visualize_images = []

    with torch.no_grad():
        for i, (L_test, ab_test) in enumerate(test_loader):
            L_test, ab_test = L_test.to(device), ab_test.to(device)
            ab_pred_test = model(L_test)

            # 转为 numpy 以计算 PSNR 和 SSIM
            pred_rgb_np = tensor2rgb(L_test[0], ab_pred_test[0])  # 范围 [0, 1]
            gt_rgb_np = tensor2rgb(L_test[0], ab_test[0])  # 范围 [0, 1]
            gray_np = (L_test[0].cpu().squeeze().numpy() + 1.0) / 2.0  # 灰度图范围 [0, 1]

            # 1. 计算 PSNR / SSIM (使用 skimage, 数据范围设定为 1.0)
            batch_psnr = psnr(gt_rgb_np, pred_rgb_np, data_range=1.0)
            batch_ssim = ssim(gt_rgb_np, pred_rgb_np, data_range=1.0, channel_axis=-1)

            # 2. 计算 LPIPS (LPIPS 需要 [-1, 1] 的 Tensor)
            pred_tensor_lpips = torch.from_numpy(pred_rgb_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            gt_tensor_lpips = torch.from_numpy(gt_rgb_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
            batch_lpips = lpips_fn(pred_tensor_lpips, gt_tensor_lpips).item()

            total_psnr += batch_psnr
            total_ssim += batch_ssim
            total_lpips += batch_lpips

            # 收集前 10 张图用于后续可视化作对比
            if i < 10:
                visualize_images.append({
                    'gray': gray_np,
                    'gt': gt_rgb_np,
                    'pred': pred_rgb_np,
                    'psnr': batch_psnr,
                    'ssim': batch_ssim
                })

    # 输出平均指标
    avg_psnr = total_psnr / num_test_samples
    avg_ssim = total_ssim / num_test_samples
    avg_lpips = total_lpips / num_test_samples

    print(f"=== 测试集评估结果 ===")
    print(f"Avg PSNR:  {avg_psnr:.4f}")
    print(f"Avg SSIM:  {avg_ssim:.4f}")
    print(f"Avg LPIPS: {avg_lpips:.4f}")

    # ---------------- 6. 测试集抽样 10 张可视化对比 ----------------
    print(f"正在生成 10 张测试集的原图-灰度-预测对比图...")
    fig, axes = plt.subplots(10, 3, figsize=(15, 40))  # 10行3列
    plt.subplots_adjust(hspace=0.3)

    for idx, img_data in enumerate(visualize_images):
        # 原图 (GT)
        axes[idx, 0].imshow(img_data['gt'])
        axes[idx, 0].set_title(f"Original (GT)")
        axes[idx, 0].axis('off')

        # 灰度图 (Input)
        axes[idx, 1].imshow(img_data['gray'], cmap='gray')
        axes[idx, 1].set_title(f"Grayscale Input")
        axes[idx, 1].axis('off')

        # 预测图 (Pred)
        axes[idx, 2].imshow(img_data['pred'])
        axes[idx, 2].set_title(f"Prediction\nPSNR: {img_data['psnr']:.2f} | SSIM: {img_data['ssim']:.3f}")
        axes[idx, 2].axis('off')

    plt.savefig(os.path.join(CONFIG['save_dir'], 'test_10_samples_comparison.png'), bbox_inches='tight')
    plt.close()

    print(f"所有训练、测试及可视化结果均已保存在目录: {CONFIG['save_dir']}")
    print("运行完毕，正在关机...")
    os.system("shutdown")