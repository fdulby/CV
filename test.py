import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


# 1. 数据集定义：修复了原图返回格式，确保 DataLoader 兼容
class CIFAR10ColorizationDataset(Dataset):
    def __init__(self, root='./data', train=True):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_pil, _ = self.dataset[idx]
        img_rgb = np.array(img_pil)

        # 转换到 Lab 空间
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr_f32 = img_bgr.astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_bgr_f32, cv2.COLOR_BGR2Lab)

        L = img_lab[:, :, 0]
        ab = img_lab[:, :, 1:]

        # 归一化到 [-1, 1]
        L_norm = (L / 50.0) - 1.0
        ab_norm = ab / 127.0

        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_norm.transpose(2, 0, 1)).float()

        # 修正：将原图转为 Tensor 避免 DataLoader 在 batch 拼接时报错
        img_rgb_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)

        return L_tensor, ab_tensor, img_rgb_tensor


# 2. 增强版 U-Net：基础通道数提升至 128
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
        base = 128  # 增加通道数以利用 5060 性能
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
        m2 = torch.cat([u2, s2], dim=1)
        d2 = self.dec2(m2)

        u1 = self.up1(d2)
        m1 = torch.cat([u1, s1], dim=1)
        d1 = self.dec1(m1)

        return torch.tanh(self.final(d1))


# 3. 可视化函数：修复了色彩转换逻辑和维度拼接
def visualize_results(model, dataset, device, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 15))

    indices = random.sample(range(len(dataset)), num_samples)

    for i, idx in enumerate(indices):
        L_tensor, _, img_rgb_tensor = dataset[idx]
        original_rgb = img_rgb_tensor.permute(1, 2, 0).numpy()

        input_L = L_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output_ab = model(input_L).cpu().squeeze(0).numpy()

        # 反归一化
        L_norm = L_tensor.squeeze().numpy()
        L_rescaled = (L_norm + 1.0) * 50.0
        ab_rescaled = output_ab.transpose(1, 2, 0) * 127.0

        # 构造 Lab 图像
        lab_pred = np.zeros((32, 32, 3), dtype=np.float32)
        lab_pred[:, :, 0] = L_rescaled
        lab_pred[:, :, 1:] = ab_rescaled

        # 修正：Lab -> BGR -> RGB 的转换逻辑
        bgr_pred = cv2.cvtColor(lab_pred, cv2.COLOR_Lab2BGR)
        rgb_pred = cv2.cvtColor(np.clip(bgr_pred, 0, 1), cv2.COLOR_BGR2RGB)
        rgb_pred = (rgb_pred * 255).astype(np.uint8)

        axes[i, 0].imshow(original_rgb)
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(L_rescaled, cmap='gray')
        axes[i, 1].set_title("Input (Gray)")
        axes[i, 2].imshow(rgb_pred)
        axes[i, 2].set_title("Predicted")
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.show()


# 4. 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")

    # 超参数优化
    BATCH_SIZE = 128
    EPOCHS = 30

    train_dataset = CIFAR10ColorizationDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = CIFAR10ColorizationDataset(train=False)

    model = UNet().to(device)

    # 修改：使用 L1Loss 获得更清晰的边缘
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 针对 5060 开启自动混合精度加速
    scaler = torch.amp.GradScaler('cuda')

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, leave=False)
        for L_batch, ab_batch, _ in loop:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_ab = model(L_batch)
                loss = criterion(pred_ab, ab_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch + 1}/{EPOCHS}] 完成，平均 Loss: {total_loss / len(train_loader):.4f}")

    # 结果展示
    print("正在生成可视化对比图...")
    visualize_results(model, test_dataset, device, num_samples=5)