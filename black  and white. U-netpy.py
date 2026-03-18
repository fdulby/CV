import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# 1. 数据集 ( ImageNet100)

class ImageNetColorizationDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        #  ImageFolder 读取本地 ImageNet100
        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(img_size)  # 保证严格的尺寸
            ])
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_pil, _ = self.dataset[idx]
        img_rgb = np.array(img_pil)

        # 转换为 BGR (OpenCV 格式) 并转为 float32
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr_f32 = img_bgr.astype(np.float32) / 255.0

        # 转换为 Lab 空间
        img_lab = cv2.cvtColor(img_bgr_f32, cv2.COLOR_BGR2Lab)

        L = img_lab[:, :, 0]
        ab = img_lab[:, :, 1:]

        # 归一化: L(0~100)->[-1,1], ab(-127~127)->[-1,1]
        L_norm = (L / 50.0) - 1.0
        ab_norm = ab / 127.0

        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()

        return L_tensor, ab_tensor


# 2. 模型定义 (U-Net 框架)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        # Decoder
        self.up3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)
        # Output
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.output_conv(d1)



# 3. 辅助函数：Tensor 转 RGB 用于可视化

def tensor2rgb(L_tensor, ab_tensor):
#将预测的或真实的 Tensor 转换为可显示的 RGB
    L = L_tensor.cpu().numpy().squeeze(0)  # (H, W)
    ab = ab_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 2)

    # 反归一化
    L = (L + 1.0) * 50.0
    ab = ab * 127.0

    # 拼接 Lab 通道
    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab

    # 转回 BGR 再到 RGB
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    rgb = np.clip(bgr, 0, 1)  # 限制在 0-1 之间
    return rgb



# 4. 训练与可视化

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 修改此处路径为 ImageNet100 的路径
    train_dir = './imagenet100/train'

    # 如果路径不存在，这里用随机数据替代以便代码能够运行
    if not os.path.exists(train_dir):
        print("Warning: ImageNet directory not found. Using fake data for demonstration.")
    else:
        train_dataset = ImageNetColorizationDataset(root_dir=train_dir, img_size=256)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = UNet(in_channels=1, out_channels=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)  # 微调了 lr

    # 简化版训练循环 (仅演示 1 Epoch)
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        for i, (L_batch, ab_batch) in enumerate(train_loader):
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)

            optimizer.zero_grad()
            ab_pred = model(L_batch)
            loss = criterion(ab_pred, ab_batch)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}], Loss: {loss.item():.4f}")

            break  # 实际训练请删除此 break

    # --- 可视化抽取 (推理与展示) ---
    print("Extracting images for visualization...")
    model.eval()
    with torch.no_grad():
        # 取一个 Batch
        L_val, ab_val = next(iter(train_loader))
        L_val = L_val.to(device)
        ab_pred_val = model(L_val)

        # 挑出 Batch 中的第一张图
        L_img = L_val[0]
        ab_true_img = ab_val[0]
        ab_pred_img = ab_pred_val[0]

        # 转换回 RGB
        gt_rgb = tensor2rgb(L_img, ab_true_img)
        pred_rgb = tensor2rgb(L_img, ab_pred_img)

        # 灰度图直接显示 L 通道
        gray_img = (L_img.cpu().squeeze().numpy() + 1.0) / 2.0

        # matplotlib 绘图
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Input (Grayscale)")
        plt.imshow(gray_img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Prediction (Colorized)")
        plt.imshow(pred_rgb)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Ground Truth (Original)")
        plt.imshow(gt_rgb)
        plt.axis('off')

        plt.tight_layout()
        plt.show()