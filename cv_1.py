import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# 1. 数据集定义
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
        L_norm = (L / 50.0) - 1.0
        ab_norm = ab / 127.0
        L_tensor = torch.from_numpy(L_norm).unsqueeze(0).float()
        ab_tensor = torch.from_numpy(ab_norm).permute(2, 0, 1).float()
        return L_tensor, ab_tensor


# 2. 模型定义 (Attention U-Net)
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


# 3. 优化后的感知损失
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
    L = ((L_tensor.cpu().squeeze().numpy() + 1.0) * 50.0)
    ab = (ab_tensor.cpu().permute(1, 2, 0).numpy() * 127.0)
    lab = np.zeros((256, 256, 3), dtype=np.float32)
    lab[:, :, 0] = L
    lab[:, :, 1:] = ab
    bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.clip(rgb, 0, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = '/root/autodl-tmp/ImageNet100'
    checkpoint_dir = '/root/autodl-tmp/optimizer_dataset'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 参数设置
    batch_size = 96
    num_epochs = 50
    train_losses = []  # 用于记录每轮 Loss

    train_dataset = ImageNetColorizationDataset(root_dir=train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=16, pin_memory=True, persistent_workers=True)

    model = AttentionUNet().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion_mse = nn.MSELoss()
    criterion_perceptual = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    best_loss = float('inf')

    print("训练启动")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (L, ab) in enumerate(train_loader):
            L, ab = L.to(device), ab.to(device)
            optimizer.zero_grad()
            ab_pred = model(L)

            loss_mse = criterion_mse(ab_pred, ab)
            input_vgg_pred = torch.cat([L, ab_pred], dim=1).expand(-1, 3, -1, -1)
            input_vgg_gt = torch.cat([L, ab], dim=1).expand(-1, 3, -1, -1)
            loss_p = criterion_perceptual(input_vgg_pred, input_vgg_gt)

            # 💡 核心修改：将感知损失权重从 0.2 提升至 0.5，对抗灰色
            total_loss = loss_mse + 0.5 * loss_p
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {total_loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"=== Epoch [{epoch + 1}/{num_epochs}] Avg Loss: {avg_loss:.4f} ===")

        # 保存当前轮次的独立权重
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'loss_history': train_losses
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pth'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'unet_best.pth'))

        # 可视化
        model.eval()
        with torch.no_grad():
            L_val, ab_val = next(iter(train_loader))
            L_val, ab_val = L_val.to(device), ab_val.to(device)
            ab_pred_val = model(L_val)
            vis_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}_vis.png')
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1);
            plt.title("Original");
            plt.imshow(tensor2rgb(L_val[0], ab_val[0]))
            plt.subplot(1, 3, 2);
            plt.title("Grayscale");
            plt.imshow((L_val[0].cpu().squeeze().numpy() + 1) / 2, cmap='gray')
            plt.subplot(1, 3, 3);
            plt.title(f"Pred E{epoch + 1}");
            plt.imshow(tensor2rgb(L_val[0], ab_pred_val[0]))
            plt.savefig(vis_path);
            plt.close()

    # 训练结束自动画 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.grid(True);
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_curve.png'))
    plt.close()

    print("训练结束，正在关机...")
    os.system("shutdown")