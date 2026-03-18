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
    def __init__(self, root_dir, img_size=256, is_train=True):
        # 训练集增加随机水平翻转，验证/测试集仅做中心裁剪
        if is_train:
            transform_list = [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(img_size)
            ]
        else:
            transform_list = [
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(img_size)
            ]

        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=transforms.Compose(transform_list)
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


    rgb_img = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    rgb_img = np.clip(rgb_img, 0, 1) # 限制在 0-1 之间供 plt 显示
    return rgb_img


# 4. 训练与可视化

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 更新为真实的 ImageNet100 路径
    train_dir = '/root/autodl-tmp/ImageNet100'

    if not os.path.exists(train_dir):
        print(f"Error: 找不到路径 {train_dir}，请检查路径是否正确！")
        exit()

    train_dataset = ImageNetColorizationDataset(root_dir=train_dir, img_size=256, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # 2. 模型、损失函数与优化器定义
    model = UNet(in_channels=1, out_channels=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # 3. 指定参数和图片保存路径 (自动创建文件夹)
    checkpoint_dir = '/root/autodl-tmp/optimizer_dataset'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')

    # 4. 正式训练循环
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (L_batch, ab_batch) in enumerate(train_loader):
            L_batch, ab_batch = L_batch.to(device), ab_batch.to(device)

            optimizer.zero_grad()
            ab_pred = model(L_batch)
            loss = criterion(ab_pred, ab_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {loss.item():.4f}")

        # 计算本轮平均 Loss
        avg_loss = running_loss / len(train_loader)
        print(f"=== Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f} ===")

        # 5. 自动保存模型参数
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, 'unet_latest.pth'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'unet_best.pth'))
            print(f"[*] 发现更优模型！已将参数保存至 {checkpoint_dir}/unet_best.pth")

        # ==========================================
        # 6. 核心修改：每 5 个 Epoch 自动保存一次可视化结果，不阻塞进程
        # ==========================================
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                L_val, ab_val = next(iter(train_loader))
                L_val = L_val.to(device)
                ab_pred_val = model(L_val)

                L_img = L_val[0]
                ab_true_img = ab_val[0]
                ab_pred_img = ab_pred_val[0]

                gt_rgb = tensor2rgb(L_img, ab_true_img)
                pred_rgb = tensor2rgb(L_img, ab_pred_img)
                gray_img = (L_img.cpu().squeeze().numpy() + 1.0) / 2.0

                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.title("Original Image")
                plt.imshow(gt_rgb)
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title("L Channel (Grayscale)")
                plt.imshow(gray_img, cmap='gray')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title(f"Synthesized Image (Epoch {epoch + 1})")
                plt.imshow(pred_rgb)
                plt.axis('off')

                plt.tight_layout()

                # 直接保存为图片到权重同一个文件夹下，并关闭画板
                save_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}_vis.png')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()  # 释放内存，防止 OOM
                print(f"[*] 已生成对比图并自动保存至: {save_path}")