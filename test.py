import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random


# 1. 数据集定义 (保持不变)
class CIFAR10ColorizationDataset(Dataset):
    def __init__(self, root='./data', train=True):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_pil, _ = self.dataset[idx]
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr_f32 = img_bgr.astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_bgr_f32, cv2.COLOR_BGR2Lab)

        L = img_lab[:, :, 0]
        ab = img_lab[:, :, 1:]

        L_norm = (L / 50.0) - 1.0
        ab_norm = ab / 127.0

        # 转换为 Tensor: (C, H, W)
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_norm.transpose(2, 0, 1)).float()

        return L_tensor, ab_tensor, img_rgb  # 额外返回原图用于对比


# 2. 增强版 U-Net 模型 (增加通道数)
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        # 基础通道数从 64 增加到 128
        base = 128
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.final = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool(s1)
        s2 = self.enc2(p1)
        p2 = self.pool(s2)
        s3 = self.enc3(p2)

        u2 = self.up2(s3)
        # CIFAR-10 图片小，通常不需要 padding 调整，直接拼接
        m2 = torch.cat([u2, s2], dim=1)
        d2 = self.dec2(m2)

        u1 = self.up1(d2)
        m1 = torch.cat([u1, s1], dim=1)
        d1 = self.dec1(m1)

        return torch.tanh(self.final(d1))


# 3. 可视化函数 (抽取 5 张图)
def visualize_results(model, dataset, device, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.4)

    indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(indices):
        L_tensor, ab_true, img_rgb = dataset[idx]

        # 预测
        input_L = L_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output_ab = model(input_L).cpu().squeeze(0).numpy()

        # 后处理预测图
        L_norm = L_tensor.squeeze().numpy()
        L_rescaled = (L_norm + 1.0) * 50.0
        ab_rescaled = output_ab.transpose(1, 2, 0) * 127.0

        lab_pred = np.concatenate([L_rescaled[:, :, np.newaxis], ab_rescaled], axis=2).astype(np.float32)
        # Lab 转 RGB
        bgr_pred = cv2.cvtColor(lab_pred, cv2.COLOR_Lab2BGR)
        rgb_pred = cv2.cvtColor(bgr_pred, cv2.COLOR_BGR2RGB)
        rgb_pred = np.clip(rgb_pred * 255, 0, 255).astype(np.uint8)

        # 展示原图
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title("Original RGB")
        axes[i, 0].axis('off')

        # 展示灰度图 (L 通道)
        axes[i, 1].imshow(L_rescaled, cmap='gray')
        axes[i, 1].set_title("Input L (Gray)")
        axes[i, 1].axis('off')

        # 展示预测图
        axes[i, 2].imshow(rgb_pred)
        axes[i, 2].set_title("Predicted RGB")
        axes[i, 2].axis('off')

    plt.show()


# 4. 主训练流程
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 针对 RTX 5060 的优化参数
    BATCH_SIZE = 128
    EPOCHS = 30

    train_dataset = CIFAR10ColorizationDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = CIFAR10ColorizationDataset(train=False)

    model = UNet().to(device)

    # 修改 1: 损失函数换成 L1Loss
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        for L_batch, ab_batch, _ in train_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_ab = model(L_batch)
                loss = criterion(pred_ab, ab_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {loss.item():.4f}")

    # 训练结束后展示结果
    print("生成对比图...")
    visualize_results(model, test_dataset, device, num_samples=5)