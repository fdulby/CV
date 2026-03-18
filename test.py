import torch
print(torch.__version__)  # 打印 PyTorch 版本
print(torch.cuda.is_available())  # 检查是否有可用的 GPU

import torch

path = '/root/autodl-tmp/optimizer_dataset/unet_best.pth'
checkpoint = torch.load(path, map_location='cpu')

print("文件加载成功！")
print("包含的键值有:", checkpoint.keys())
print("最后保存时的 Epoch:", checkpoint['epoch'])
print("当时达到的最低 Loss:", checkpoint['loss'])