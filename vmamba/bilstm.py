import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, DropPath

class BiLSTM(nn.Module):
    """BiMamba head for sequence modeling"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        # BiMamba layer
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        lstm_out = self.bilstm(x)[0]  # Get the output from LSTM
        # output shape: [batch_size, sequence_length, nb_cls]
        
        return lstm_out
    
class BiLSTMBlock(nn.Module):
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
            use_bimamba_arch_proj=False,
            args=None
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)

        self.attn = BiLSTM(dim, dim//2, num_layers=1, dropout=drop)
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

    