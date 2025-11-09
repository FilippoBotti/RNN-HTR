import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from timm.models.vision_transformer import Mlp, DropPath

class BiMamba(nn.Module):
    """BiMamba head for sequence modeling"""
    def __init__(self, input_dim, d_state=16,):
        super().__init__()
        # BiMamba layer
        self.fwd_mamba = Mamba2(
            input_dim,
            d_state=d_state,
        )
        self.bwd_mamba = Mamba2(
            input_dim,
            d_state=d_state,
        )
        
        self.proj = nn.Linear(input_dim * 2, input_dim)
        
    def forward(self, x):
        fwd = self.fwd_mamba(x)
        x_flip = x.flip(dims=[1])
        bimamba_out_flip = self.bwd_mamba(x_flip)
        bwd = bimamba_out_flip.flip(dims=[1])
        output = self.proj(torch.cat([fwd, bwd], dim=-1))
        return output
    
class BiMambaBlock(nn.Module):
    """BiMamba block for sequence modeling"""

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.0,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            args=None
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)

        self.attn = BiMamba(dim, d_state=16)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=True)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

    