import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, DropPath

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LongformerLocalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=17,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert window_size % 2 == 1, "Use an odd window_size, e.g. 17 or 33"

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.radius = window_size // 2
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        r = self.radius

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        outputs = []

        for i in range(L):
            start = max(0, i - r)
            end = min(L, i + r + 1)

            qi = q[:, :, i:i + 1, :]
            ki = k[:, :, start:end, :]
            vi = v[:, :, start:end, :]

            attn = (qi @ ki.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = attn @ vi
            outputs.append(out)

        x = torch.cat(outputs, dim=2)
        x = x.transpose(1, 2).reshape(B, L, D)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class LongformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=16,
        mlp_ratio=4.,
        qkv_bias=True,
        drop=0.,
        attn_drop=0.,
        init_values=None,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        args=None,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim, elementwise_affine=True)
        self.attn = LongformerLocalAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    

if __name__ == "__main__":
    B, L, D = 2, 64, 128
    x = torch.randn(B, L, D)
    block = LongformerBlock(dim=D, num_heads=8, window_size=17)
    out = block(x)
    print(out.shape)  # Should be [B, L, D]