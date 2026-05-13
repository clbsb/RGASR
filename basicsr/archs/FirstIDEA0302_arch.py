import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


# ================ 核心辅助函数 ================

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
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


# ================================ 创新模块 (RMSCA & GACA) ================================

class RMSCA(nn.Module):
    """
    优化后的 RMSCA (Regional Multi-Scale Cross Attention)
    逻辑：
    1. Pad 图片以适配 region_size。
    2. 切分窗口 (Window Partition)。
    3. 在窗口内部进行多尺度 AvgPool，生成多尺度 Key/Value。
    4. Query 为窗口内原始像素，进行 Cross Attention。
    """

    def __init__(self, dim, num_heads, region_size=30, scales=[2, 3, 5]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size
        self.scales = scales
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

        # 归一化，有助于收敛
        self.norm_input = nn.LayerNorm(dim)

        # 【新增】关键修改：对最后一层进行零初始化
        # 这样初始输出为0，不会干扰 x_local，等效于 gamma=0 的效果
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, x_size):
        # x: (B, N, C), N=H*W
        B, N, C = x.shape
        H, W = x_size

        # 恢复空间结构用于处理
        x = x.view(B, H, W, C)

        # 1. Padding 逻辑 (使其能被 region_size 整除)
        mod_h = (self.region_size - H % self.region_size) % self.region_size
        mod_w = (self.region_size - W % self.region_size) % self.region_size

        # pad 输入: (left, right, top, bottom) for last 2 dims if NCHW, but here we have BHWC?
        # F.pad 对 BHWC 操作比较麻烦，通常转为 BCHW 处理
        x_perm = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x_pad = F.pad(x_perm, (0, mod_w, 0, mod_h), 'reflect')
        _, _, Hp, Wp = x_pad.shape
        x_pad = x_pad.permute(0, 2, 3, 1)  # (B, Hp, Wp, C)

        # 2. Window Partition -> (B*n_wins, R, R, C)
        x_wins = window_partition(x_pad, self.region_size)  # (BW, 30, 30, C)

        # 归一化输入特征
        x_wins = self.norm_input(x_wins)

        # 3. 构建 Query (原始尺度)
        # (BW, 900, C)
        q = self.q(x_wins).view(-1, self.region_size * self.region_size, self.dim)

        # 4. 构建 Multi-scale Key/Value
        # 在每个窗口内部进行下采样
        ms_tokens_list = []

        # 为了高效，我们将 x_wins 转为 NCHW 进行池化
        x_wins_perm = x_wins.permute(0, 3, 1, 2)  # (BW, C, R, R)

        for s in self.scales:
            # 确保 region_size 能被 scale 处理，即使不能整除 avg_pool 也能工作
            # (BW, C, R/s, R/s)
            pooled = F.avg_pool2d(x_wins_perm, kernel_size=s, stride=s)
            # Flatten -> (BW, C, N_s) -> (BW, N_s, C)
            ms_tokens_list.append(pooled.flatten(2).transpose(1, 2))

        # 拼接所有尺度的 Token
        k_v_input = torch.cat(ms_tokens_list, dim=1)  # (BW, N_total_ms, C)

        kv = self.kv(k_v_input)  # (BW, N_total_ms, 2*C)
        k, v = torch.chunk(kv, 2, dim=-1)

        # 5. Multi-head Cross Attention
        # Reshape for multi-head: (BW, N, nH, d) -> (BW, nH, N, d)
        q = q.view(-1, q.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(-1, k.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(-1, v.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_out = (attn @ v).transpose(1, 2).reshape(-1, self.region_size * self.region_size, C)
        x_out = self.proj(x_out)  # (BW, 900, C)

        # 6. Reverse Windows & Unpad
        # reshape back to (BW, R, R, C)
        x_out = x_out.view(-1, self.region_size, self.region_size, C)
        x_out = window_reverse(x_out, self.region_size, Hp, Wp)  # (B, Hp, Wp, C)

        # Crop padding
        if mod_h > 0 or mod_w > 0:
            x_out = x_out[:, :H, :W, :]

        return x_out.reshape(B, N, C)


class GACA(nn.Module):
    """
    优化后的 GACA (Global Aggregated Cross Attention)
    逻辑：
    1. 在全图生成多尺度 Tokens (Original + Downsampled)。
    2. 聚合 Tokens -> 软聚类 (Soft Clustering) -> 生成 K 个全局 Tokens。
    3. Query 为 Original, Key/Value 为全局 Tokens 进行 Cross Attention。
    """

    def __init__(self, dim, num_heads, num_clusters=64, scales=[2, 3, 5]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.scales = scales
        self.scale = (dim // num_heads) ** -0.5

        # 聚类生成器
        self.norm_cluster = nn.LayerNorm(dim)
        self.cluster_proj = nn.Linear(dim, num_clusters)  # Project to K clusters

        # Attention Projections
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out_proj = nn.Linear(dim, dim)

        # 【新增】关键修改：对最后一层进行零初始化
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, x, x_size):
        # x: (B, N, C)
        B, N, C = x.shape
        H, W = x_size

        x_spatial = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # 1. 生成多尺度 Tokens
        # 包含原始 Token
        all_tokens_list = [x]

        for s in self.scales:
            # 使用 adaptive_avg_pool2d 确保在不同分辨率下都能得到合理的特征
            # 目标尺寸 H//s, W//s
            pooled = F.adaptive_avg_pool2d(x_spatial, (H // s, W // s))
            all_tokens_list.append(pooled.flatten(2).transpose(1, 2))  # (B, N_s, C)

        # 拼接所有来源的 tokens: (B, N_total, C)
        input_tokens = torch.cat(all_tokens_list, dim=1)

        # 2. 软聚类生成全局 Global Tokens
        # (B, N_total, C)
        norm_tokens = self.norm_cluster(input_tokens)

        # 计算聚类权重: (B, N_total, K)
        # 加上温度系数 sqrt(dim) 防止 softmax 过于尖锐
        attn_logits = self.cluster_proj(norm_tokens)
        attn_weights = (attn_logits / math.sqrt(self.dim)).softmax(dim=1)

        # 聚合生成 Global Tokens: (B, K, C) = (B, K, N_total) @ (B, N_total, C)
        # 注意转置 weights
        global_tokens = attn_weights.transpose(1, 2) @ input_tokens

        # 3. Cross Attention
        # Q: Original Image (B, N, C)
        # K, V: Global Tokens (B, K, C)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(global_tokens).reshape(B, self.num_clusters, 2, self.num_heads, C // self.num_heads).permute(2, 0,
                                                                                                                  3, 1,
                                                                                                                  4)
        k, v = kv[0], kv[1]  # (B, nH, K, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class AdaptiveFusion(nn.Module):
    """
    修改为单纯的残差连接，移除可学习系数 gamma。
    """
    def __init__(self, dim):
        super().__init__()
        # 不再需要定义参数

    def forward(self, x_local, x_global):
        # 直接相加，梯度更直接
        return x_local + x_global


# ================================ 基础模块 (Standard SwinIR Components) ================================

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)


class CA(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CA, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        return self.cab(x) + x


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        B, N, C = x.shape
        H, W = x_size
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B_ // nw, nw, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        x = self.proj(x)
        return x


# ================================ 构建层级结构 ================================

class BasicLayer(nn.Module):
    """
    基础层：支持 SW-MSA (Local) 以及并行 RMSCA/GACA (Global)
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 mlp_ratio,
                 norm_layer,
                 shift_size=0,
                 convffn_kernel_size=5,
                 act_layer=nn.GELU,
                 attn_mode='SW-MSA'  # 'SW-MSA' | 'RMSCA' | 'GACA'
                 ):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = window_size
        self.attn_mode = attn_mode
        self.dim = dim

        self.norm1 = norm_layer(dim)

        # 1. 创新模块 (Parallel Branch)
        if attn_mode == 'RMSCA':
            self.attn_innov = RMSCA(dim=dim, num_heads=num_heads, region_size=30, scales=[2, 3, 5])
        elif attn_mode == 'GACA':
            self.attn_innov = GACA(dim=dim, num_heads=num_heads, num_clusters=64, scales=[2, 3, 5])
        else:
            self.attn_innov = None

        # 2. 基础窗口注意力 (Local Branch)
        self.attn_local = WindowAttention(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
        )

        # 3. 融合模块 (仅当有创新模块时使用)
        if self.attn_innov is not None:
            self.fusion = AdaptiveFusion(dim=dim)

        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,
                               act_layer=act_layer)
        self.norm2 = norm_layer(dim)

    def _run_window_attention(self, wa_module, x, h, w, params):
        B, N, C = x.shape
        # Cyclic Shift
        if self.shift_size > 0:
            x_reshaped = x.view(B, h, w, C)
            shifted_x = torch.roll(x_reshaped, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_x = x.view(B, h, w, C)
            attn_mask = None

        # Partition
        x_windows = window_partition(shifted_x, self.window_size)  # (B*nW, Wh, Ww, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Attention
        attn_windows = wa_module(x_windows, rpi=params['rpi_sa'], mask=attn_mask)

        # Merge
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        # Reverse Shift
        if self.shift_size > 0:
            x_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_out = shifted_x

        return x_out.view(B, h * w, C)

    def forward(self, x, x_size, params):
        h, w = x_size
        shortcut = x
        x_norm = self.norm1(x)

        # === Path 1: Local Window Attention (Always Run) ===
        out_local = self._run_window_attention(self.attn_local, x_norm, h, w, params)

        # === Path 2: Global/Regional Innovation (Conditional) ===
        if self.attn_innov is not None:
            # RMSCA 和 GACA 内部自己处理了 norm 或 多尺度生成
            out_global = self.attn_innov(x_norm, x_size)
            # 自适应融合：Local + gamma * Global
            attn_out = self.fusion(out_local, out_global)
        else:
            attn_out = out_local

        # Residual
        x = shortcut + attn_out

        # FFN
        x = x + self.convffn(self.norm2(x), x_size)

        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_layer,
                 num_heads,
                 window_size,
                 mlp_ratio,
                 norm_layer,
                 convffn_kernel_size):
        super().__init__()
        self.dim = dim
        self.conv = CA(num_feat=dim, compress_ratio=3, squeeze_factor=10)
        self.layers = nn.ModuleList()

        for i in range(num_layer):
            # 策略: [WA, WA, RMSCA, WA, WA, GACA]
            attn_mode = 'SW-MSA'
            if i == 2:
                attn_mode = 'RMSCA'
            elif i == 5:
                attn_mode = 'GACA'

            # 严格交替 Shift 策略
            if i % 2 != 0:
                shift_size = window_size // 2
            else:
                shift_size = 0

            layer = BasicLayer(dim=dim,
                               num_heads=num_heads,
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               norm_layer=norm_layer,
                               shift_size=shift_size,
                               convffn_kernel_size=convffn_kernel_size,
                               attn_mode=attn_mode)
            self.layers.append(layer)

    def forward(self, x, x_size, params):
        # x: (B, C, H, W) 输入
        ori_x = x
        B, C, H, W = x.shape

        # Flatten for Transformer: (B, C, H, W) -> (B, N, C)
        x = x.flatten(2).transpose(1, 2).contiguous()

        # 串行经过 Layers
        for layer in self.layers:
            x = layer(x, x_size, params)

        # Unembed: (B, N, C) -> (B, C, H, W)
        x = x.transpose(1, 2).view(B, self.dim, H, W).contiguous()

        # 卷积与残差
        x = self.conv(x)
        x = x + ori_x

        return x


# ================================ Main Network ================================

@ARCH_REGISTRY.register()
class FirstIDEA0302_CA(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=60,
                 num_basicblock=4,
                 num_layer=6,
                 num_heads=6,
                 window_size=8,
                 mlp_ratio=1.,
                 norm_layer=nn.LayerNorm,
                 upscale=2,
                 img_range=1.,
                 resi_connection='1conv',
                 convffn_kernel_size=7):
        super().__init__()
        self.img_range = img_range
        self.window_size = window_size
        self.dim = embed_dim
        self.upscale = upscale

        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep Feature Extraction
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.blocks = nn.ModuleList()
        for i in range(num_basicblock):
            block = BasicBlock(dim=embed_dim,
                               num_layer=num_layer,
                               num_heads=num_heads,
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               norm_layer=norm_layer,
                               convffn_kernel_size=convffn_kernel_size)
            self.blocks.append(block)

        self.norm = norm_layer(embed_dim)
        self.after_blocks = nn.Sequential(nn.Conv2d(embed_dim * num_basicblock, embed_dim, 1),
                                          nn.LeakyReLU(inplace=True))

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # 3. High Quality Reconstruction
        self.upsample = UpsampleOneStep(upscale, embed_dim, in_chans)
        self.apply(self._init_weights)

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])

        dense = []
        for block in self.blocks:
            x = block(x, x_size, params)
            dense.append(x)

        x = self.after_blocks(torch.cat(dense, dim=1))

        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1, 2).view(x.shape[0], self.dim, x_size[0], x_size[1]).contiguous()
        return x

    def forward(self, x):
        # Check Image Size & Padding
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        _, _, h, w = x.size()

        # MeanShift
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # Pre-calc Res
        res = F.interpolate(x, scale_factor=self.upscale, mode="bicubic", align_corners=False)

        # Mask & Params
        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        # Main Body
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x, params)) + x

        # Upsample
        x = self.upsample(x)

        # Residual add & Unpad
        x = x + res
        x = x / self.img_range + self.mean
        x = x[:, :, :H * self.upscale, :W * self.upscale]
        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask