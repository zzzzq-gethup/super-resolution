
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from mmcv.ops import CARAFE

from einops import rearrange
from timm.models.layers import trunc_normal_

import torch.nn.functional as F


class ALC(nn.Module):
    def __init__(self, in_channels):
        """
        高效拉普拉斯增强模块
        结合了拉普拉斯先验和自适应学习能力，同时保持计算效率

        参数:
            in_channels: 输入通道数
        """
        super(ALC, self).__init__()

        # 可学习的拉普拉斯核作为先验知识
        self.laplacian_kernel = nn.Parameter(torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            dtype=torch.float32, requires_grad=True).view(1, 1, 3, 3))

        # 轻量级高频提取网络 (与HighFrequencyEnhancer类似的计算量)
        self.hf_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )

        # 初始化最后层权重为零，确保训练稳定
        self.hf_extractor[-1].weight.data.zero_()
        self.hf_extractor[-1].bias.data.zero_()

        # 自适应融合参数
        self.blpha = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # 特征细化层 (可选)
        self.refine = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        """
        前向传播:
        1. 提取拉普拉斯高频分量
        2. 学习自适应高频分量
        3. 融合两种高频信息
        4. 残差连接增强
        """
        # 1. 拉普拉斯高频分量提取
        laplacian_kernels = self.laplacian_kernel.repeat(x.size(1), 1, 1, 1)
        laplacian_hf = F.conv2d(x, laplacian_kernels, padding=1, groups=x.size(1))

        # 2. 自适应高频分量学习
        learned_hf = self.hf_extractor(x)

        # 3. 融合两种高频信息
        combined_hf = laplacian_hf * self.blpha + learned_hf

        # 4. 残差连接增强
        enhanced = x + self.alpha * combined_hf

        # 特征细化
        return self.refine(enhanced)

class DDG(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale,
                 drop, attn_drop, drop_path, norm_layer, gc, patch_size, img_size, ape, patch_norm, fusex = "fscb"):
        super(DDG, self).__init__()
        self.ape = ape
        self.patch_norm = patch_norm
        self.SDRCAB = SDRCAB(dim=dim, input_resolution=input_resolution,
                        num_heads=num_heads, window_size=window_size, depth=depth,
                        shift_size=shift_size, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path,
                        norm_layer=norm_layer, gc=gc, img_size=img_size, patch_size=patch_size)

        self.pre_hf_enhancer = ALC(dim)
        self.post_hf_enhancer = ALC(dim)


        self.norm = norm_layer(dim)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dim,
            embed_dim=dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop)

        self.freq_processor = FDB(in_channels=dim)

        self.s_ffusion = FSILB(dim, dim, fusex)



    def forward(self, x):

        x_trans = self.pre_hf_enhancer(x)

        x_size = (x_trans.shape[2], x_trans.shape[3])
        x_trans = self.patch_embed(x_trans)
        if self.ape:
            x_trans = x_trans + self.absolute_pos_embed
        x_trans = self.pos_drop(x_trans)
        x_trans = self.SDRCAB(x_trans, x_size)
        x_trans = self.norm(x_trans)
        x_trans = self.patch_unembed(x_trans, x_size)

        x_trans = self.post_hf_enhancer(x_trans)

        with torch.no_grad():  # 限制梯度反传
            x_fdb = self.freq_processor(x)

        gated_fused = self.s_ffusion(x_fdb, x_trans)



        return gated_fused


class FusionCF(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 特征融合层（带残差连接）
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels * 2, 2, 3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, feat_trans, feat_cnn):
        """
        输入:
            feat_trans: [B,C,H,W] Transformer特征（全局信息）
            feat_cnn: [B,C,H,W] CNN特征（局部细节）
        输出: [B,C,H,W] 融合特征
        """
        # 通道级融合
        combined = torch.cat([feat_trans, feat_cnn], dim=1)  # [B,2C,H,W]
        x = self.conv1(combined)
        w_tf, w_cnn = x.chunk(2, dim=1)
        x1 = feat_trans * w_tf + feat_cnn * w_cnn

        return x1

class FSILB(nn.Module):
    def __init__(self, in_channels, out_channels, fusex = "fsca"):
        super().__init__()

        self.fusex = fusex
        # 两个 Conv 模块
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # 两个 FSCA 模块
        self.FSCA1 = FSCA(out_channels)
        self.FSCA2 = FSCA(out_channels)

        self.fuse1 = FusionCF(out_channels)
        self.fuse2 = FusionCF(out_channels)
        # Concatenate 后的处理模块
        self.concat_conv = nn.Conv2d(out_channels * 2, out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.split = nn.Conv2d(out_channels, 2, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))


    def forward(self, x1, x2):
        xx = x2
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        if self.fusex == "fsca":
            print("zzzzzzzzzzzqeqe")
            out1 = self.FSCA1(x1, x2)
            out2 = self.FSCA2(x2, x1)
            print("zzzzzzzzzzzqe313131")
        else:
            out1 = self.fuse1(x1, x2)
            out2 = self.fuse2(x2, x1)

        concat_out = torch.cat([out1, out2], dim=1)
        conv_out = self.concat_conv(concat_out)
        sigmoid_out = self.sigmoid(conv_out)
        mask = self.split(sigmoid_out)
        mask1, mask2 = mask.chunk(2, dim=1)
        out1_masked = out1 * mask1
        out2_masked = out2 * mask2
        final_out = out1_masked + out2_masked

        return  final_out*self.alpha +xx

class FSCA(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Q生成路径（XH分支）
        self.q_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # K生成路径（XH分支）
        self.k_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # V生成路径（XS分支）
        self.v_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # 输出卷积
        self.out_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, xh, xs):
        # 生成Q, K, V
        Q = self.q_conv(xh)
        K = self.k_conv(xh)
        V = self.v_conv(xs)

        # 调整维度用于注意力计算
        B, C, H, W = Q.shape
        Q = Q.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        K = K.view(B, C, -1)  # [B, C, HW]
        V = V.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        # 计算注意力权重
        attn_weights = torch.matmul(Q, K) / (C ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 应用注意力到V
        attn_output = torch.matmul(attn_weights, V).permute(0, 2, 1)
        attn_output = attn_output.view(B, C, H, W)

        # 最终输出
        output = self.out_conv(attn_output) + xh
        return output


class FDB(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # P分支 (相位分支) - 添加批量归一化
        self.p_branch = nn.Sequential(
            nn.Conv2d(2 * in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

        # A分支 (振幅分支) - 使用更高效的分组卷积
        self.a_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),  # 深度卷积
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, 1),  # 点卷积
            nn.BatchNorm2d(in_channels)
        )

        # 多尺度复数卷积
        self.multi_scale = nn.ModuleList([
            ComplexConv2d(in_channels, in_channels // 2, 3),
            ComplexConv2d(in_channels, in_channels // 2, 5)
        ])

        # 后处理模块 - 更高效的深度可分离卷积
        self.post_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # 深度可分离卷积分解
            nn.Conv2d(128, 128, 3, padding=1, groups=128),  # 深度卷积
            nn.Conv2d(128, 256, 1),  # 点卷积
            nn.BatchNorm2d(256),
            nn.PReLU()
        )

        # 最终调制层 - 添加残差连接
        self.final = nn.Sequential(
            nn.Conv2d(256, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # FFT变换
        with torch.cuda.amp.autocast(enabled=False):  # 防止半精度问题
            fft = torch.fft.rfft2(x, norm='ortho')
            real, imag = fft.real, fft.imag

        # 相位分支处理
        p_out = self.p_branch(torch.cat([real, imag], dim=1))

        # 振幅分支处理
        a_input = torch.sqrt(real.square() + imag.square() + 1e-6)
        a_out = self.a_branch(a_input)

        # 多尺度复数卷积
        real1, imag1 = self.multi_scale[0](p_out, a_out)
        real2, imag2 = self.multi_scale[1](p_out, a_out)
        real = torch.cat([real1, real2], dim=1)
        imag = torch.cat([imag1, imag2], dim=1)

        # 特征融合
        mag = torch.sqrt(real.square() + imag.square() + 1e-6)
        x = self.post_conv(mag)

        # 生成调制掩码并应用
        mask = self.final(x)

        # 应用调制并逆变换
        with torch.cuda.amp.autocast(enabled=False):
            # 掩码应用
            modulated_real = real * mask
            modulated_imag = imag * mask

            # 逆FFT
            complex_tensor = torch.complex(modulated_real, modulated_imag)
            restored = torch.fft.irfft2(complex_tensor, s=(H, W), norm='ortho')

        # 添加残差连接
        return restored + identity



class ComplexConv2d(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        if k==3:
            p = 1
        else:
            p = 2
        self.real_conv = nn.Conv2d(in_c, out_c, k, padding=p)
        self.imag_conv = nn.Conv2d(in_c, out_c, k, padding=p)

    def forward(self, x_real, x_imag):
        real_part = self.real_conv(x_real) - self.imag_conv(x_imag)
        imag_part = self.real_conv(x_imag) + self.imag_conv(x_real)
        return real_part, imag_part


class WeightedFusion(nn.Module):
    def __init__(self, num_bands):
        super().__init__()
        # 初始化权重参数（默认等权重）
        self.weights = nn.Parameter(torch.ones(num_bands) / num_bands)

    def forward(self, processed_bands):
        # 确保权重和为1（可选）
        normalized_weights = torch.softmax(self.weights, dim=0)
        # 加权融合 [N, C, H, W] * [N] -> [C, H, W]
        combined = sum(w * band for w, band in zip(normalized_weights, processed_bands))
        return combined

class LightCrossAttention(nn.Module):
    def __init__(self, dim, reduction_ratio=9):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.GELU()
        )

        self.spatial_proj = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.GELU(),
            nn.Linear(dim // reduction_ratio, dim)
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.GELU(),
            nn.Linear(dim // reduction_ratio, dim)
        )

    def forward(self, spatial, freq):
        spatial_red = self.reduction(spatial)
        freq_red = self.reduction(freq)
        attn = torch.einsum('bic,bjc->bij', spatial_red, freq_red)
        attn = F.softmax(attn, dim=-1)
        fused_spatial = torch.einsum('bij,bjc->bic', attn, freq)
        fused_freq = torch.einsum('bij,bjc->bic', attn.transpose(1, 2), spatial)
        a = self.spatial_proj(fused_spatial)
        b = self.freq_proj(fused_freq)
        return a + b








def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class ChannelAdjust(nn.Module):
    """SE Block动态调整通道权重"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局压缩
            nn.Conv2d(channels, channels // reduction, 1),  # 降维
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),  # 升维
            nn.Sigmoid()  # 权重归一化
        )

    def forward(self, x):
        return x * self.se(x)  # 通道重标定


class SDRCAB(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, shift_size, mlp_ratio, qkv_bias, qk_scale,
                 drop, attn_drop, drop_path, norm_layer, gc, patch_size, img_size):
        super(SDRCAB, self).__init__()

        self.swin1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                          num_heads=num_heads, window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.se1 = ChannelAdjust(dim)
        self.adjust1 = nn.Conv2d(dim, gc, 1)

        self.swin2 = SwinTransformerBlock(dim + gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + gc) % num_heads), window_size=window_size,
                                          shift_size=window_size // 2,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.se2 = ChannelAdjust(dim + gc)
        self.adjust2 = nn.Conv2d(dim + gc, gc, 1)

        self.swin3 = SwinTransformerBlock(dim + 2 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 2 * gc) % num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.se3 = ChannelAdjust(dim + gc * 2)
        self.adjust3 = nn.Conv2d(dim + gc * 2, gc, 1)

        self.swin4 = SwinTransformerBlock(dim + 3 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 3 * gc) % num_heads), window_size=window_size,
                                          shift_size=8,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.se4 = ChannelAdjust(dim + gc * 3)
        self.adjust4 = nn.Conv2d(dim + gc * 3, gc, 1)

        self.swin5 = SwinTransformerBlock(dim + 4 * gc, input_resolution=input_resolution,
                                          num_heads=num_heads - ((dim + 4 * gc) % num_heads), window_size=window_size,
                                          shift_size=0,  # For first block
                                          mlp_ratio=1,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                          norm_layer=norm_layer)
        self.se5 = ChannelAdjust(dim + gc * 4)
        self.adjust5 = nn.Conv2d(dim + gc * 4, dim, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.pe = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.pue = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, xsize):
        x1 = self.pe(self.lrelu(self.adjust1(self.se1(self.pue(self.swin1(x, xsize), xsize)))))
        x2 = self.pe(self.lrelu(self.adjust2(self.se2(self.pue(self.swin2(torch.cat((x, x1), -1), xsize), xsize)))))
        x3 = self.pe(self.lrelu(self.adjust3(self.se3(self.pue(self.swin3(torch.cat((x, x1, x2), -1), xsize), xsize)))))
        x4 = self.pe(
            self.lrelu(self.adjust4(self.se4(self.pue(self.swin4(torch.cat((x, x1, x2, x3), -1), xsize), xsize)))))
        x5 = self.pe(self.adjust5(self.se5(self.pue(self.swin5(torch.cat((x, x1, x2, x3, x4), -1), xsize), xsize))))

        return x5 * 0.2 + x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        self.cab = CAB(dim)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        cabx = x.transpose(1, 3)
        cabx = self.cab(cabx)
        cabx = cabx.transpose(1, 3)
        cabx = cabx.reshape(B, H * W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = x + cabx
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 结构为 [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def forward(self, x, x_size):
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])  # 输出结构为 [B, Ph*Pw, C]
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class DualGateFusion(nn.Module):
    """双重门控融合模块，包含空间门控和通道门控"""

    def __init__(self, channels, reduction_ratio=8):
        super().__init__()

        # 空间门控（关注特征空间重要性）
        self.spatial_gate = nn.Sequential(
            # 联合使用平均池和最大池
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # 通道门控（动态调整通道权重）
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels * 2, 1),
            nn.Sigmoid()
        )

        # 特征融合层（带残差连接）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 1)
        )

    def forward(self, feat_trans, feat_cnn):
        """
        输入:
            feat_trans: [B,C,H,W] Transformer特征（全局信息）
            feat_cnn: [B,C,H,W] CNN特征（局部细节）
        输出: [B,C,H,W] 融合特征
        """
        # 通道级融合
        combined = torch.cat([feat_trans, feat_cnn], dim=1)  # [B,2C,H,W]

        # 通道门控
        channel_attn = self.channel_gate(combined)  # [B,2C,1,1]
        c_attn_trans, c_attn_cnn = channel_attn.chunk(2, dim=1)
        weighted_trans = feat_trans * c_attn_trans
        weighted_cnn = feat_cnn * c_attn_cnn

        # 空间门控（基于CNN特征生成空间权重）
        spatial_avg = torch.mean(weighted_cnn, dim=1, keepdim=True)  # [B,1,H,W]
        spatial_max, _ = torch.max(weighted_cnn, dim=1, keepdim=True)  # [B,1,H,W]
        spatial_attn = self.spatial_gate(torch.cat([spatial_avg, spatial_max], dim=1))  # [B,1,H,W]

        # 门控融合
        gated_trans = weighted_trans * spatial_attn  # 空间权重调整
        fused = self.fusion_conv(torch.cat([gated_trans, weighted_cnn], dim=1))

        # 残差连接
        return fused + feat_cnn  # 保留原始CNN特征

@ARCH_REGISTRY.register()
class FSCSR(nn.Module):

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 gc=32,
                 num_bands = 3,
                 **kwargs):
        super(FSCSR, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.freq_processor = FDB(in_channels=embed_dim)
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # self.fusion = DualGateFusion(embed_dim)

        # self.freq_fusion = nn.Sequential(
        #     nn.Conv2d(2 * embed_dim, embed_dim // 8, 1),
        #     nn.LeakyReLU(0.2),
        #     nn.Conv2d(embed_dim // 8, 2, 1),
        #     nn.Softmax(dim=1)
        # )

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution


        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = DDG(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                        num_heads=num_heads[i_layer], window_size=window_size, depth=0,
                        shift_size=window_size // 2, mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        norm_layer=norm_layer, gc=gc, img_size=img_size, patch_size=patch_size,
                         ape=self.ape, patch_norm=self.patch_norm)

            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)

            fused_with_freq = self.forward_features(x)

            x = self.conv_after_body(fused_with_freq) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x
