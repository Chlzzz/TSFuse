import torch
import torch.nn as nn

from model.Spatial_Transformer import encoder
from model.patch_embed import PatchEmbed, DePatch

class FuseLayer(nn.Module):
    def __init__(self, size=128, embed_dim=128, depth=2, channel=32,
                 num_heads=8, mlp_ratio=2., patch_size=16, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,drop_path_rate=0., norm_layer=nn.LayerNorm, embed_layer=PatchEmbed, recover_layer=DePatch,):
        super().__init__()

        self.patch_embed = embed_layer(
            img_size=size, patch_size=patch_size, channel=channel, embed_dim=embed_dim)
        self.patch_recover = recover_layer()

        self.shape2vi = encoder(
            embed_dim=embed_dim, num_heads=num_heads, depth=2,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer
        )
        self.ir2vi = encoder(
            embed_dim=embed_dim, num_heads=num_heads, depth=2,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer
        )
        self.shape2ir = encoder(
            embed_dim=embed_dim, num_heads=num_heads, depth=2,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer
        )
        self.vi2ir = encoder(
            embed_dim=embed_dim, num_heads=num_heads, depth=2,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer
        )

    def forward(self, x_vi, x_ir, sfuse):
        # x_vi,x_ir [B, C, H, W]
        ori = x_vi.shape
        x_vi = self.patch_embed(x_vi)
        x_ir = self.patch_embed(x_ir)
        sfuse = self.patch_embed(sfuse)
        
        # sfuse = self.shape2vi(sfuse, x_vi)
        # vi_fused_sir = self.ir2vi(x_ir, sfuse)
        # sfuse = self.shape2ir(sfuse, x_ir)
        # ir_fused_svi = self.vi2ir(x_vi, sfuse)
        vi_fused_ir = self.ir2vi(x_vi, x_ir)
        ir_fused_vi = self.vi2ir(x_ir, x_vi)
        sfuse = self.shape2vi(sfuse, vi_fused_ir)
        sfuse = self.shape2ir(sfuse, ir_fused_vi)


        vi_fused_ir = self.patch_recover(vi_fused_ir, ori)
        ir_fused_vi = self.patch_recover(ir_fused_vi, ori)
        sfuse = self.patch_recover(sfuse, ori)

        return vi_fused_ir, ir_fused_vi, sfuse