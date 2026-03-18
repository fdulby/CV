import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import cv2

#颜色转换
class CIFAR10ColorizationDataset(Dataset):
    def __init__(self, root='./data', train=True):
        #初始 CIFAR-10 数据集
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # RGB 图像格式转 np
        img_pil, _ = self.dataset[idx]
        # shape: (32, 32, 3)
        img_rgb = np.array(img_pil)

        # OpenCV 色彩转换
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_bgr_f32 = img_bgr.astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_bgr_f32, cv2.COLOR_BGR2Lab)

        # 分离 L , ab
        L = img_lab[:, :, 0]
        ab = img_lab[:, :, 1:]

        # 归一化到：[-1, 1]，原来的L为0-100，ab为+-127
        L_norm = (L / 50.0) - 1.0
        ab_norm = ab / 127.0

        #注意，opencv中的通道与Tensor不同，后者维度为 (C, H, W)
        # shape: (1, 32, 32)
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
        # shape: (2, 32, 32)
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()

        return L_tensor, ab_tensor


# U-Net
class UNet(nn.Module):
    #in为L，out为预测的ab
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()

        # --- 编码器 (Encoder) ---
        #信息不丢失的情况为通道数增加，池化持续减少
        #所以需要padding=1
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        # (Bottleneck)
        #对于U-net，一般在forward中是maxpool，意味着CIFAR-10 (32x32) 下采样3次，成为4*4
        self.bottleneck = self.conv_block(256, 512)

        # --- 解码器 (Decoder) ---
        # 1. 上采样UpConv
        self.up3 = self.upconv_block(512, 256)
        # 2. 特征融合块 (ConvBlock): 输入是拼接后的通道数512，输出压缩回 256
        self.dec3 = self.conv_block(512, 256)

        self.up2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)  # 128+128=256 -> 128

        self.up1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)  # 64+64=128 -> 64

        # 输出层，用Tanh 将输出限制在 [-1, 1]
        self.output_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels):
        #卷积，采用padding=1
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #BN加速收敛
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        #反卷积
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        # 瓶颈层
        b = self.bottleneck(F.max_pool2d(e3, 2))

        # 解码
        d3 = self.up3(b)
        #跳跃连接
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # 输出
        out = self.output_conv(d1)
        return out


# 训练 U-Net 模型
if __name__ == "__main__":
    # 检查设备，确保使用 CPU
    device = torch.device("cpu")  # 强制使用 CPU

    # 实例化数据集和数据加载器
    train_dataset = CIFAR10ColorizationDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 实例化 U-Net 模型
    model = UNet(in_channels=1, out_channels=2)
    model = model.to(device)  # 将模型移动到 CPU

    # 损失函数和优化器
    criterion = nn.MSELoss()  # 你可以选择 Smooth L1 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 训练过程
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for L_batch, ab_batch in train_loader:
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            ab_pred = model(L_batch)

            # 计算损失
            loss = criterion(ab_pred, ab_batch)

            # 反向传播
            loss.backward()

            # 优化
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
