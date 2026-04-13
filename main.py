# main.py
# 条件扩散模型训练与采样主程序
# 运行前请确保已安装所需依赖：torch, torchvision, einops, matplotlib, PIL

import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
import os

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import save_image, make_grid

# 用户自定义库
import other_utils
from configs import IMG_CH, IMG_SIZE, BATCH_SIZE, T, nrows, ncols, N_CLASSES

# ------------------------- 设备配置 -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------- 数据加载 -------------------------
def load_MNIST(data_transform, train=True):
    """加载 MNIST 数据集"""
    return torchvision.datasets.MNIST(
        "../data",
        download=True,
        train=train,
        transform=data_transform,
    )

def load_transformed_MNIST(img_size, batch_size):
    """返回经过缩放和归一化的 MNIST 数据集及 DataLoader"""
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # 自动归一化到 [0,1]
    ]
    data_transform = transforms.Compose(data_transforms)
    train_set = load_MNIST(data_transform, train=True)
    test_set = load_MNIST(data_transform, train=False)
    data = torch.utils.data.ConcatDataset([train_set, test_set])
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    return data, dataloader

data, dataloader = load_transformed_MNIST(IMG_SIZE, BATCH_SIZE)

# ------------------------- 扩散参数 -------------------------
B_start = 0.0001
B_end = 0.02
B = torch.linspace(B_start, B_end, T).to(device)          # β_t 噪声调度

a = 1.0 - B                                               # α_t = 1 - β_t
a_bar = torch.cumprod(a, dim=0)                           # ᾱ_t 累积乘积
sqrt_a_bar = torch.sqrt(a_bar)                            # 前向过程均值系数
sqrt_one_minus_a_bar = torch.sqrt(1 - a_bar)              # 前向过程标准差系数

# 逆向过程用到的系数
sqrt_a_inv = torch.sqrt(1 / a)                            # 1/√α_t
pred_noise_coeff = (1 - a) / torch.sqrt(1 - a_bar)        # 预测噪声系数

# ------------------------- 前向加噪函数 -------------------------
def q(x_0, t):
    """
    前向扩散过程：从真实图像 x_0 加噪得到 x_t，并返回添加的噪声
    x_0: (B, C, H, W) 原始图像
    t: (B,) 时间步索引
    """
    t = t.int()
    noise = torch.randn_like(x_0)
    sqrt_a_bar_t = sqrt_a_bar[t, None, None, None]
    sqrt_one_minus_a_bar_t = sqrt_one_minus_a_bar[t, None, None, None]
    x_t = sqrt_a_bar_t * x_0 + sqrt_one_minus_a_bar_t * noise
    return x_t, noise

# ------------------------- 可视化前向过程 -------------------------
plt.figure(figsize=(8, 8))
x_0 = data[0][0].to(device)          # 取一张图像展示
for t in range(T):
    t_tensor = torch.Tensor([t]).type(torch.int64)
    x_t, _ = q(x_0, t_tensor)
    ax = plt.subplot(nrows, ncols, t + 1)
    ax.axis('off')
    other_utils.show_tensor_image(x_t)
plt.savefig("forward_process.png", bbox_inches='tight')
plt.show()

# ------------------------- 逆向去噪函数 -------------------------
@torch.no_grad()
def reverse_q(x_t, t, e_t):
    """
    单步逆向采样：根据预测噪声 e_t 从 x_t 计算 x_{t-1}
    x_t: (B, C, H, W)
    t: (B,) 时间步索引
    e_t: (B, C, H, W) 预测噪声
    """
    t = t.int()
    # 获取对应时间步的系数（形状为 (B,)）
    pred_noise_coeff_t = pred_noise_coeff[t]
    sqrt_a_inv_t = sqrt_a_inv[t]

    # 扩展为 (B, 1, 1, 1) 以匹配图像张量形状
    pred_noise_coeff_t = pred_noise_coeff_t[:, None, None, None]
    sqrt_a_inv_t = sqrt_a_inv_t[:, None, None, None]

    # 估计的 x_0 均值
    u_t = sqrt_a_inv_t * (x_t - pred_noise_coeff_t * e_t)

    # 若所有样本的时间步均为 0，则直接返回 u_t（最后一步不加噪声）
    if (t == 0).all():
        return u_t
    else:
        # 加上上一时间步的噪声
        B_t = B[t - 1]                      # shape: (B,)
        B_t = B_t[:, None, None, None]      # 扩展为 (B, 1, 1, 1)
        new_noise = torch.randn_like(x_t)
        return u_t + torch.sqrt(B_t) * new_noise

# ------------------------- 上下文掩码与独热编码 -------------------------
def get_context_mask(c, drop_prob):
    """
    将类别标签转为独热向量，并按概率 drop_prob 随机丢弃（置零）
    c: (B,) 类别索引
    返回:
        c_hot: (B, N_CLASSES) 独热向量（部分被置零）
        c_mask: (B, N_CLASSES) 掩码指示器（1 表示保留，0 表示丢弃）
    """
    c_hot = F.one_hot(c.to(torch.int64), num_classes=N_CLASSES)
    c_mask = torch.bernoulli(torch.ones_like(c_hot).float() - drop_prob)
    return c_hot, c_mask

# ------------------------- U-Net 模型定义（直接复用 Functions.py 中的定义） -------------------------
class GELUConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, group_size):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.GroupNorm(group_size, out_ch),
            nn.GELU(),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_chs, out_chs, group_size):
        super().__init__()
        self.conv1 = GELUConvBlock(in_chs, out_chs, group_size)
        self.conv2 = GELUConvBlock(out_chs, out_chs, group_size)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        return x1 + x2

class DownBlock(nn.Module):
    def __init__(self, in_chs, out_chs, group_size):
        super(DownBlock, self).__init__()
        layers = [
            GELUConvBlock(in_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            RearrangePoolBlock(out_chs, group_size),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class RearrangePoolBlock(nn.Module):
    def __init__(self, in_chs, group_size):
        super().__init__()
        self.rearrange = Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        self.conv = GELUConvBlock(4 * in_chs, in_chs, group_size)
    def forward(self, x):
        x = self.rearrange(x)
        return self.conv(x)

class SinusoidalPositionEmbedBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class EmbedBlock(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedBlock, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Unflatten(1, (emb_dim, 1, 1)),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_chs, out_chs, group_size):
        super(UpBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(2 * in_chs, out_chs, 2, 2),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
            GELUConvBlock(out_chs, out_chs, group_size),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x

class UNet(nn.Module):
    def __init__(
        self, T, img_ch, img_size, down_chs=(64, 64, 128), t_embed_dim=8, c_embed_dim=10
    ):
        super().__init__()
        self.T = T
        up_chs = down_chs[::-1]
        latent_image_size = img_size // 4
        small_group_size = 8
        big_group_size = 32

        self.down0 = ResidualConvBlock(img_ch, down_chs[0], small_group_size)
        self.down1 = DownBlock(down_chs[0], down_chs[1], big_group_size)
        self.down2 = DownBlock(down_chs[1], down_chs[2], big_group_size)
        self.to_vec = nn.Sequential(nn.Flatten(), nn.GELU())

        self.dense_emb = nn.Sequential(
            nn.Linear(down_chs[2] * latent_image_size**2, down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[1]),
            nn.ReLU(),
            nn.Linear(down_chs[1], down_chs[2] * latent_image_size**2),
            nn.ReLU(),
        )

        self.sinusoidaltime = SinusoidalPositionEmbedBlock(t_embed_dim)
        self.t_emb1 = EmbedBlock(t_embed_dim, up_chs[0])
        self.t_emb2 = EmbedBlock(t_embed_dim, up_chs[1])
        self.c_embed1 = EmbedBlock(c_embed_dim, up_chs[0])
        self.c_embed2 = EmbedBlock(c_embed_dim, up_chs[1])

        self.up0 = nn.Sequential(
            nn.Unflatten(1, (up_chs[0], latent_image_size, latent_image_size)),
            GELUConvBlock(up_chs[0], up_chs[0], big_group_size),
        )
        self.up1 = UpBlock(up_chs[0], up_chs[1], big_group_size)
        self.up2 = UpBlock(up_chs[1], up_chs[2], big_group_size)

        self.out = nn.Sequential(
            nn.Conv2d(2 * up_chs[-1], up_chs[-1], 3, 1, 1),
            nn.GroupNorm(small_group_size, up_chs[-1]),
            nn.ReLU(),
            nn.Conv2d(up_chs[-1], img_ch, 3, 1, 1),
        )

    def forward(self, x, t, c, c_mask):
        down0 = self.down0(x)
        down1 = self.down1(down0)
        down2 = self.down2(down1)
        latent_vec = self.to_vec(down2)

        latent_vec = self.dense_emb(latent_vec)
        t = t.float() / self.T
        t = self.sinusoidaltime(t)
        t_emb1 = self.t_emb1(t)
        t_emb2 = self.t_emb2(t)

        c = c * c_mask
        c = c.float()
        c_emb1 = self.c_embed1(c)
        c_emb2 = self.c_embed2(c)

        up0 = self.up0(latent_vec)
        up1 = self.up1(c_emb1 * up0 + t_emb1, down2)
        up2 = self.up2(c_emb2 * up1 + t_emb2, down1)
        return self.out(torch.cat((up2, down0), 1))

# ------------------------- 初始化模型 -------------------------
model = UNet(
    T, IMG_CH, IMG_SIZE, down_chs=(64, 64, 128), t_embed_dim=8, c_embed_dim=N_CLASSES
)
print("Num params: ", sum(p.numel() for p in model.parameters()))
model = model.to(device)

# ------------------------- 损失函数 -------------------------
def get_loss(model, x_0, t, *model_args):
    x_noisy, noise = q(x_0, t)
    noise_pred = model(x_noisy, t, *model_args)
    return F.mse_loss(noise, noise_pred)

# ------------------------- 采样并可视化 -------------------------
def sample_images(model, img_ch, img_size, ncols, *model_args, axis_on=False, save_path=None):
    x_t = torch.randn((1, img_ch, img_size, img_size), device=device)
    plt.figure(figsize=(8, 8))
    hidden_rows = T / ncols
    plot_number = 1
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device).float()
        e_t = model(x_t, t, *model_args)
        x_t = reverse_q(x_t, t, e_t)
        if i % hidden_rows == 0:
            ax = plt.subplot(1, ncols + 1, plot_number)
            if not axis_on:
                ax.axis('off')
            other_utils.show_tensor_image(x_t.detach().cpu())
            plot_number += 1
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# ------------------------- 训练循环 -------------------------
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 5
preview_c = 0

model.train()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        c_drop_prob = 0.1
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).float()
        x = batch[0].to(device)
        c_hot, c_mask = get_context_mask(batch[1].to(device), c_drop_prob)
        c_hot = c_hot.float().to(device)
        c_mask = c_mask.to(device)

        loss = get_loss(model, x, t, c_hot, c_mask)
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0 and step % 100 == 0:
            print(f"Epoch {epoch} | Step {step:03d} | Loss: {loss.item():.4f} | Preview class: {preview_c}")
            # 预览生成时不丢弃条件
            c_drop_prob = 0
            c_hot_preview, c_mask_preview = get_context_mask(torch.Tensor([preview_c]), c_drop_prob)
            c_hot_preview = c_hot_preview.float().to(device)
            c_mask_preview = c_mask_preview.to(device)
            sample_images(model, IMG_CH, IMG_SIZE, ncols, c_hot_preview, c_mask_preview,
                          save_path=f"preview_epoch{epoch}_step{step}.png")
            preview_c = (preview_c + 1) % N_CLASSES

# ------------------------- 无分类器引导采样函数 -------------------------
@torch.no_grad()
def sample_w(model, c, w):
    """
    使用无分类器引导生成指定类别图像
    c: (n_samples, N_CLASSES) 独热条件向量
    w: 引导强度标量
    """
    input_size = (IMG_CH, IMG_SIZE, IMG_SIZE)
    n_samples = len(c)
    w_tensor = torch.tensor([w]).float().to(device)
    w_tensor = w_tensor[:, None, None, None]   # 便于广播

    x_t = torch.randn(n_samples, *input_size).to(device)
    c = c.repeat(len(w_tensor), 1)             # 为每个 w 复制
    c = c.repeat(2, 1)                         # 双倍 batch（有条件和无条件）
    c_mask = torch.ones_like(c).to(device)
    c_mask[n_samples:] = 0.0                   # 后半部分为无条件

    for i in range(0, T)[::-1]:
        t = torch.full((n_samples,), i, device=device).float()
        t = t.repeat(2)                        # 同样双倍
        x_t_in = x_t.repeat(2, 1, 1, 1)
        e_t = model(x_t_in, t, c, c_mask)
        e_t_keep_c = e_t[:n_samples]
        e_t_drop_c = e_t[n_samples:]
        e_t = (1 + w_tensor) * e_t_keep_c - w_tensor * e_t_drop_c

        x_t = reverse_q(x_t, t[:n_samples], e_t)
    return x_t

# ------------------------- 测试指定数字生成 -------------------------
model.eval()
w = 2.0
c = torch.arange(N_CLASSES).to(device)        # 0~9 所有类别
c_hot, c_mask = get_context_mask(c, 0.0)
x_gen = sample_w(model, c_hot.float(), w)

# 诊断输出值域
print(f"x_gen range: [{x_gen.min().item():.4f}, {x_gen.max().item():.4f}]")

# 方法一：使用 torchvision 直接保存（自动处理值域缩放）
# 若 x_gen 值域为 [-1, 1]，normalize=True 会将其映射到 [0, 1]
save_image(x_gen.cpu(), 'generated_digits_w2.png', nrow=N_CLASSES, normalize=True)

# 方法二（备选）：手动映射后保存
# x_gen_vis = (x_gen + 1) / 2  # 将 [-1, 1] 映射到 [0, 1]
# save_image(x_gen_vis.cpu(), 'generated_digits_w2.png', nrow=N_CLASSES)

print(f"Generated images shape: {x_gen.shape}")
print("图片已保存为 generated_digits_w2.png")