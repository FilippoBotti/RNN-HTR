# --------------------------------------------------------
# Swin Transformer 1D
# Adapted from 2D Swin Transformer for sequence features [B, L, C]
# --------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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


def window_partition_1d(x, window_size):
    """
    Args:
        x: [B, L, C]
        window_size: int

    Returns:
        windows: [B * num_windows, window_size, C]
    """
    B, L, C = x.shape
    x = x.view(B, L // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows


def window_reverse_1d(windows, window_size, L):
    """
    Args:
        windows: [B * num_windows, window_size, C]
        window_size: int
        L: int

    Returns:
        x: [B, L, C]
    """
    B = int(windows.shape[0] / (L / window_size))
    x = windows.view(B, L // window_size, window_size, -1)
    x = x.contiguous().view(B, L, -1)
    return x


class WindowAttention1D(nn.Module):
    r"""
    Window based multi-head self attention for 1D sequences.

    Args:
        dim: number of input channels
        window_size: local window size along sequence dimension
        num_heads: number of attention heads
    """

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias 1D:
        # possible relative positions: -(M-1), ..., 0, ..., +(M-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads)
        )

        coords = torch.arange(self.window_size)
        relative_coords = coords[:, None] - coords[None, :]  # [M, M]
        relative_coords += self.window_size - 1              # shift to 0 ... 2M-2
        self.register_buffer("relative_position_index", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: [num_windows * B, M, C]
            mask: [num_windows, M, M] or None

        Returns:
            x: [num_windows * B, M, C]
        """
        B_, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.reshape(
            self.window_size, self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW,
                nW,
                self.num_heads,
                N,
                N
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"window_size={self.window_size}, "
            f"num_heads={self.num_heads}"
        )

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""
    1D Swin Transformer Block.

    Input:
        x: [B, L, C]

    Output:
        x: [B, L, C]

    This is intended to replace the 2D SwinTransformerBlock in your HTR pipeline
    after the CNN has produced a 1D sequence.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()

        self.dim = dim

        # Accept both input_resolution=128 and input_resolution=[128, 1]
        if isinstance(input_resolution, (tuple, list)):
            self.input_resolution = input_resolution[0]
        else:
            self.input_resolution = input_resolution

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if self.input_resolution <= self.window_size:
            self.shift_size = 0
            self.window_size = self.input_resolution

        assert 0 <= self.shift_size < self.window_size, \
            "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention1D(
            dim=dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, L):
        """
        Attention mask for shifted windows.

        This prevents tokens from attending across artificial cyclic boundaries.
        """
        M = self.window_size
        S = self.shift_size

        pad_len = (M - L % M) % M
        Lp = L + pad_len

        img_mask = torch.zeros((1, Lp, 1))

        slices = (
            slice(0, -M),
            slice(-M, -S),
            slice(-S, None),
        )

        cnt = 0
        for s in slices:
            img_mask[:, s, :] = cnt
            cnt += 1

        mask_windows = window_partition_1d(img_mask, M)
        mask_windows = mask_windows.view(-1, M)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        """
        Args:
            x: [B, L, C]

        Returns:
            x: [B, L, C]
        """
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)

        M = self.window_size

        pad_len = (M - L % M) % M
        if pad_len > 0:
            x = torch.cat([x, x.new_zeros(B, pad_len, C)], dim=1)

        Lp = x.shape[1]

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            attn_mask = self.attn_mask
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition_1d(shifted_x, M)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        shifted_x = window_reverse_1d(attn_windows, M, Lp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        if pad_len > 0:
            x = x[:, :L, :]

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, "
            f"window_size={self.window_size}, "
            f"shift_size={self.shift_size}, "
            f"mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        L = self.input_resolution

        flops += self.dim * L

        nW = L / self.window_size
        flops += nW * self.attn.flops(self.window_size)

        flops += 2 * L * self.dim * self.dim * self.mlp_ratio

        flops += self.dim * L

        return flops