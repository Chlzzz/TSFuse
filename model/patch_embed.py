import torch.nn as nn

from timm.models.layers import to_2tuple
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, channel=32, embed_dim=128, norm_layer=None):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size ** 2 * channel, embed_dim),
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.embedding(x)  # BCHW -> BNC
        x = self.norm(x)
        return x
    


class DePatch(nn.Module):
    def __init__(self, channel=32, embed_dim=128, patch_size=16):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, patch_size**2*channel),
        )

    def forward(self, x, ori):
        b, c, h, w = ori
        h_ = h // self.patch_size
        w_ = w // self.patch_size
        x = self.projection(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h_, w=w_, p1=self.patch_size, p2=self.patch_size)
        return x
