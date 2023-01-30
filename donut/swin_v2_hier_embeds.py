import torch.nn as nn

from timm.models.swin_transformer_v2 import SwinTransformerV2
from timm.models.vision_transformer import Block
import torch


class SwinV2WithVit(SwinTransformerV2):
    def __init__(
            self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, global_pool='avg',
            embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
            window_size=7, mlp_ratio=4., qkv_bias=True,  # or False
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            pretrained_window_sizes=(0, 0, 0, 0),
            depth_last_block=2,
            num_heads_last_block=8,  # must be "d_model / num_heads_last_block = 0"
            drop_path_rate_last_block=0.,# or 0.1
            init_values_last_block=None,  # or 1e-5
            ape_last_block=False,
            **kwargs):
        super().__init__(img_size, patch_size, in_chans, num_classes, global_pool,
                         embed_dim, depths, num_heads,
                         window_size, mlp_ratio, qkv_bias,
                         drop_rate, attn_drop_rate, drop_path_rate,
                         norm_layer, ape, patch_norm,
                         pretrained_window_sizes, **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate_last_block, depth_last_block)]  # stochastic depth decay rule
        self.ape_last_block = ape_last_block
        if ape_last_block:
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_features, self.num_features) * .02)
            self.pos_drop = nn.Dropout(p=drop_rate)

        self.last_blocks = nn.Sequential(*[
            Block(
                dim=self.num_features, num_heads=num_heads_last_block, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values_last_block,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=nn.GELU)
            for i in range(depth_last_block)])

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
