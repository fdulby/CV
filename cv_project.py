import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# ==========================================
# 1. 数据集：增加色彩权重映射 (Class Rebalancing)
# ==========================================
class ImageNetColorizationDataset(Dataset):
    def __init__(self, root_dir, img_size=256, is_train=True):
        self.is_train = is_train
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size)
        ]
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

        # 归一化
        L_norm = (L / 50.0) - 1.0
        ab_norm = ab / 127.0

        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()

        return L_tensor, ab_tensor


# ==========================================
# 2. 模型：带 Attention Gate 的 U-Net
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
        psi = self.psi(psi)
        return x * psi


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
# 3. 损失函数：引入感知损失 (VGG Loss)
# ==========================================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.loss_network = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

    def forward(self, pred_rgb, gt_rgb):
        # 将输入缩放到 VGG 期望的范围 [0, 1]
        loss = F.mse_loss(self.loss_network(pred_rgb), self.loss_network(gt_rgb))
        return loss


# 辅助函数：Tensor 转 RGB (带梯度支持，用于计算 Perceptual Loss)
def lab_tensor_to_rgb_tensor(L, ab):
    # L: [-1,1], ab: [-1,1] -> 转换为 [0,1] 的 RGB
    L = (L + 1.0) * 50.0
    ab = ab * 127.0
    # 注意：在训练中直接做 Lab->RGB 的转换比较复杂，这里使用简化的近似或者
    # 只针对 L 和 ab 的感知损失。为了稳定，我们对 ab 通道单独做特征对比。
    return torch.cat([L, ab], dim=1)


# ==========================================
# 4. 辅助显示函数 (保持不变)
# ==========================================
def tensor2rgb(L_tensor, ab_tensor):
    L = ((L_tensor.cpu().squeeze().numpy() + 1.0) * 50.0)
    ab = (ab_tensor.cpu().permute(1, 2, 0).numpy() * 127.0)
    lab = np.zeros((256, 256, 3), dtype=np.float32)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.clip(rgb, 0, 1)


# ==========================================
# 5. 主训练逻辑
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置
    train_dir = '/root/autodl-tmp/ImageNet100/train'
    checkpoint_dir = '/root/autodl-tmp/optimizer_dataset'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 数据加载
    train_dataset = ImageNetColorizationDataset(root_dir=train_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 模型与损失
    model = AttentionUNet().to(device)
    criterion_mse = nn.MSELoss()
    criterion_perceptual = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for i, (L, ab) in enumerate(train_loader):
            L, ab = L.to(device), ab.to(device)

            optimizer.zero_grad()
            ab_pred = model(L)

            # 综合 Loss：MSE 保证基础，Perceptual 保证鲜艳度
            loss_mse = criterion_mse(ab_pred, ab)

            # 简单的感知损失：直接对比 ab 特征 (此处简化，通常应转为 RGB)
            # 为了解决灰感，我们给 ab_pred 一个放大系数或者直接对比
            loss_p = criterion_perceptual(
                torch.cat([L, ab_pred], dim=1).repeat(1, 1.5, 1, 1)[:, :3, :, :],
                torch.cat([L, ab], dim=1).repeat(1, 1.5, 1, 1)[:, :3, :, :]
            )

            total_loss = loss_mse + 0.1 * loss_p

            total_loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}")

        # 保存与可视化 (逻辑与之前相同...)
        # 每 5 个 epoch 保存对比图到 checkpoint_dir
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth'))
            # ... 可视化代码 ...
            # ==========================================
            # 5. 主训练逻辑 (接上一段代码)
            # ==========================================
            best_loss = float('inf')

            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0

                for i, (L, ab) in enumerate(train_loader):
                    L, ab = L.to(device), ab.to(device)

                    optimizer.zero_grad()
                    ab_pred = model(L)

                    # 计算组合 Loss
                    loss_mse = criterion_mse(ab_pred, ab)
                    # 这里的感知损失帮助模型跳出“灰色陷阱”
                    # 我们将 L 和 ab 拼在一起伪造一个 3 通道输入给 VGG
                    loss_p = criterion_perceptual(
                        torch.cat([L, ab_pred], dim=1).repeat(1, 1, 1, 1).expand(-1, 3, -1, -1),
                        torch.cat([L, ab], dim=1).repeat(1, 1, 1, 1).expand(-1, 3, -1, -1)
                    )

                    total_loss = loss_mse + 0.1 * loss_p

                    total_loss.backward()
                    optimizer.step()

                    running_loss += total_loss.item()

                    if (i + 1) % 100 == 0:
                        print(
                            f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {total_loss.item():.4f} (MSE: {loss_mse.item():.4f})")

                avg_loss = running_loss / len(train_loader)
                print(f"=== Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f} ===")

                # ------------------------------------------
                # 自动保存模型参数
                # ------------------------------------------
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss
                }
                # 总是保存最新的权重
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'unet_latest.pth'))

                # 保存表现最好的权重
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'unet_best.pth'))
                    print(f"[*] 发现更优模型，已更新 best 权重")

                # ------------------------------------------
                # 6. 每 5 个 Epoch 自动保存可视化结果
                # ------------------------------------------
                if (epoch + 1) % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        # 从验证数据中抓取一个 Batch
                        L_val, ab_val = next(iter(train_loader))
                        L_val, ab_val = L_val.to(device), ab_val.to(device)
                        ab_pred_val = model(L_val)

                        # 取出该 Batch 中的第一张图进行展示
                        L_img = L_val[0]
                        ab_true_img = ab_val[0]
                        ab_pred_img = ab_pred_val[0]

                        # 转换回 RGB 空间
                        gt_rgb = tensor2rgb(L_img, ab_true_img)
                        pred_rgb = tensor2rgb(L_img, ab_pred_img)
                        gray_img = (L_img.cpu().squeeze().numpy() + 1.0) / 2.0  # 灰度图展示

                        # 绘图
                        plt.figure(figsize=(20, 7))

                        plt.subplot(1, 3, 1)
                        plt.title("Original (Ground Truth)")
                        plt.imshow(gt_rgb)
                        plt.axis('off')

                        plt.subplot(1, 3, 2)
                        plt.title("Input (L Channel)")
                        plt.imshow(gray_img, cmap='gray')
                        plt.axis('off')

                        plt.subplot(1, 3, 3)
                        plt.title(f"Model Prediction (Epoch {epoch + 1})")
                        plt.imshow(pred_rgb)
                        plt.axis('off')

                        plt.tight_layout()

                        # 保存到指定文件夹
                        vis_save_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}_visualization.png')
                        plt.savefig(vis_save_path)
                        plt.close()  # 必须关闭，否则后台运行会内存泄漏
                        print(f"[*] 可视化图已保存至: {vis_save_path}")

                    model.train()  # 切回训练模式

            print("训练完成！所有权重和对比图均已存放在 /root/autodl-tmp/optimizer_dataset")