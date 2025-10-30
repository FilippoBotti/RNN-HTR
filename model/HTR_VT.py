import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath

import numpy as np
from model import resnet18
from functools import partial
import math

from vmamba import VSSBlock
from mamba_ssm import Mamba2
from vmamba.single_direction_vssm import VSSBlockSingle
from vmamba.double_direction_vssm import VSSBlockDouble, SS2D as SS2D_Double
from rwkv.rwkv_model import RWKV_Block


class BiMambaHead(nn.Module):
    """BiMamba head for sequence modeling"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, nb_cls, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiMamba layer
        self.bimamba = Mamba2(
            input_dim,
            d_state=16,
        )
        
        # Final projection layer
        self.fc = nn.Linear(input_dim, nb_cls)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        bimamba_out = self.bimamba(x)
        # bimamba_out shape: [batch_size, sequence_length, input_dim]
        x_flip = x.flip(dims=[1])
        bimamba_out_flip = self.bimamba(x_flip)
        bimamba_out_flip = bimamba_out_flip.flip(dims=[1])
        # Apply final linear layer
        output = self.fc(bimamba_out + bimamba_out_flip)
        # output shape: [batch_size, sequence_length, nb_cls]
        
        return output
    
class BiLSTMHead(nn.Module):
    """BiLSTM head for sequence modeling"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, nb_cls, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Final projection layer
        self.fc = nn.Linear(hidden_dim * 2, nb_cls)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        lstm_out, _ = self.bilstm(x)
        # lstm_out shape: [batch_size, sequence_length, hidden_dim * 2]
        
        # Apply final linear layer
        output = self.fc(lstm_out)
        # output shape: [batch_size, sequence_length, nb_cls]
        
        return output


class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.num_patches = num_patches
        self.bias = torch.ones(1, 1, self.num_patches, self.num_patches)
        self.back_bias = torch.triu(self.bias)
        self.forward_bias = torch.tril(self.bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            num_patches,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.0,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            args=None
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=True)

        self.attn = Attention(dim, num_patches, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
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


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LayerNorm(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 nb_cls=80,
                 img_size=[512, 32] ,
                 patch_size=[8, 32],
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 args=None,
                 **kwargs):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.layer_norm = LayerNorm()
        self.patch_embed = resnet18.ResNet18(embed_dim)
        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        # --------------------------------------------------------------------------

        if args.architecture == 'mamba':
            if args.mamba_scan_type == 'single':
                self.blocks = nn.ModuleList([
                    VSSBlockSingle(
                        hidden_dim=embed_dim, 
                        norm_layer=norm_layer,
                        mlp_ratio=mlp_ratio,
                        use_checkpoint=True,
                        ssm_init="v2",
                        forward_type="v3",
                        num_patches=self.num_patches,
                    )
                for _ in range(depth)])
            elif args.mamba_scan_type == 'double':
                self.blocks = nn.ModuleList([
                    VSSBlockDouble(
                        hidden_dim=embed_dim, 
                        norm_layer=norm_layer,
                        mlp_ratio=mlp_ratio,
                        use_checkpoint=True,
                        ssm_init="v2",
                        forward_type="v3",
                        num_patches=self.num_patches,
                    )
                for _ in range(depth)])
            elif args.mamba_scan_type == 'quad':
                self.blocks = nn.ModuleList([
                    VSSBlock(
                        hidden_dim=embed_dim, 
                        norm_layer=norm_layer,
                        mlp_ratio=mlp_ratio,
                        use_checkpoint=True,
                        ssm_init="v2",
                        forward_type="v3",
                        num_patches=self.num_patches,
                    )
                for _ in range(depth)])
        elif args.architecture == 'rwkv':
            self.blocks = nn.ModuleList([
                RWKV_Block(n_embd=768, n_head=8, n_layer=12, layer_id=i, patch_resolution=self.grid_size, shift_mode='q_shift_multihead', shift_pixel=1,
                          drop_path=0.1, hidden_rate=4, init_mode='fancy', init_values=0.1,
                          post_norm=False, key_norm=False, with_cls_token=False, with_cp=False)
            for i in range(depth)])
        elif args.architecture == 'transformer':
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, self.num_patches,
                      mlp_ratio, qkv_bias=True, norm_layer=norm_layer, args=args)
                for i in range(depth)])
        elif args.architecture == 'hybrid':
            layers = []
            for i in range(depth):
                if i % 2 == 0:
                    if args.mamba_scan_type == 'single':        
                        layers.append(VSSBlockSingle(
                            hidden_dim=embed_dim, 
                            norm_layer=norm_layer,
                            mlp_ratio=mlp_ratio,
                            use_checkpoint=True,
                            ssm_init="v2",
                            forward_type="v3",
                            scan_mode="uni",
                            num_patches=self.num_patches,
                        ))
                    elif args.mamba_scan_type == 'double':
                        layers.append(VSSBlockDouble(
                            hidden_dim=embed_dim, 
                            norm_layer=norm_layer,
                            mlp_ratio=mlp_ratio,
                            use_checkpoint=True,
                            ssm_init="v2",
                            forward_type="v3",
                            scan_mode="bi",
                            num_patches=self.num_patches,
                        ))
                    elif args.mamba_scan_type == 'quad':
                        layers.append(VSSBlock(
                            hidden_dim=embed_dim, 
                            norm_layer=norm_layer,
                            mlp_ratio=mlp_ratio,
                            use_checkpoint=True,
                            ssm_init="v2",
                            forward_type="v3",
                            scan_mode="quad",
                            num_patches=self.num_patches,
                        ))
                else:
                    layers.append(Block(
                        embed_dim, num_heads, self.num_patches,
                        mlp_ratio, qkv_bias=True, norm_layer=norm_layer, args=args
                    ))
            self.blocks = nn.ModuleList(layers)
        
        self.norm = norm_layer(embed_dim, elementwise_affine=True)
        
        self.head_type = getattr(args, 'head_type', 'linear') if args is not None else 'linear'
        # Head configuration: BiLSTM or Linear
        if self.head_type == 'bilstm':
            bilstm_hidden_dim = getattr(args, 'bilstm_hidden_dim', 256)
            bilstm_num_layers = getattr(args, 'bilstm_num_layers', 2)
            bilstm_dropout = getattr(args, 'bilstm_dropout', 0.1)
            
            self.head = BiLSTMHead(
                input_dim=embed_dim,
                hidden_dim=bilstm_hidden_dim,
                num_layers=bilstm_num_layers,
                nb_cls=nb_cls,
                dropout=bilstm_dropout
            )
        elif self.head_type == 'bimamba':
            self.head = BiMambaHead(
                input_dim=embed_dim,
                hidden_dim=embed_dim,
                num_layers=2,
                nb_cls=nb_cls,
                dropout=0.1
            )
        elif self.head_type == 'linear':
            self.head = torch.nn.Linear(embed_dim, nb_cls)
        else:   
            raise ValueError(f"Unsupported head type: {self.head_type}")


        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # pos_embed = get_2d_sincos_pos_embed(self.embed_dim, [1, self.nb_query])
        # self.qry_tokens.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def generate_span_mask(self, x, mask_ratio, max_span_length):
        N, L, D = x.shape  # batch, length, dim
        mask = torch.ones(N, L, 1).to(x.device)
        span_length = int(L * mask_ratio)
        num_spans = span_length // max_span_length
        for i in range(num_spans):
            idx = torch.randint(L - max_span_length, (1,))
            mask[:,idx:idx + max_span_length,:] = 0
        return mask

    def random_masking(self, x, mask_ratio, max_span_length):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        mask = self.generate_span_mask(x, mask_ratio, max_span_length)
        x_masked = x * mask + (1 - mask) * self.mask_token
        return x_masked

    def forward(self, x, mask_ratio=0.0, max_span_length=1, use_masking=False):
        # embed patches
        x = self.layer_norm(x)
        x = self.patch_embed(x)
        b, c, w, h = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        # masking: length -> length * mask_ratio
        if use_masking:
            x = self.random_masking(x, mask_ratio, max_span_length)
        x = x + self.pos_embed
        # apply Transformer blocks

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # To CTC Loss     
        # Apply head (BiLSTM or Linear)
        x = self.head(x)
        # Only apply layer norm if not using BiLSTM (BiLSTM already has proper output)
        if  self.head_type == 'linear':
            x = self.layer_norm(x)


        return x


def create_model(nb_cls, img_size, args,**kwargs):
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=img_size,
                                 patch_size=(4, 64),
                                 embed_dim=768,
                                 depth=4,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 args=args,
                                 **kwargs)
    return model

