import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange 
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np


from model.t2t_vit import Spatial

class ResidualBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, 1)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out

class DenseBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch*2, out_ch, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv3 =nn.Conv2d(out_ch*3, out_ch, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(inplace=True)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out1 = self.conv2(torch.cat([x, out], 1))
        out1 = self.leaky_relu(out1)
        out2 = self.conv3(torch.cat([x, out, out1], 1))
        out2 = self.leaky_relu(out2)

        out = out + out1 + out2 + identity
        return out
    
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU())

class MulRecep(nn.Module):
    def __init__(self, in_channels=128, stride=1, kernel_size=3, padding=1):
        super(MulRecep, self).__init__()
        self.layer1 = conv_batch(in_channels, in_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(in_channels, in_channels, kernel_size=3, padding=1)
        self.layer3 = conv_batch(in_channels, in_channels, kernel_size=5, padding=2)
        self.convout = conv_batch(3*in_channels, in_channels)

    def forward(self, x):
        residual = x
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)
        out = self.convout(torch.cat([out1, out2, out3], 1))
        out += residual
        return out   
    
class TinyUNet(nn.Module):
    def __init__(self, in_channels, stride=1, kernel_size=3, padding=1):
        super(TinyUNet, self).__init__()
        self.downlayer = nn.Sequential(
                nn.MaxPool2d(2),
                conv_batch(in_channels, in_channels, kernel_size=3, padding=1, stride=1))
        self.paraconv = conv_batch(in_channels, in_channels, kernel_size, stride, padding)
        self.uplayer = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU())
        self.convout = conv_batch(2*in_channels, in_channels, kernel_size, stride, padding)

    def forward(self, x):
        residual = x
        x_down = self.downlayer(x)
        x_para = self.paraconv(x_down)
        x_up = self.uplayer(x_para)
        out = self.convout(torch.cat([x, x_up], 1))
        out += residual
        return out
    
class ConvTransBlock(nn.Module):
    def __init__(self, size=128, embed_dim=128, depth=1, channel=32, patch_size=16,
                 num_heads=4, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, modal='VI'):
        super(ConvTransBlock, self).__init__()

        self.modal = modal
        if self.modal == 'VI':
            self.conv_x = MulRecep(in_channels=channel, stride=1, kernel_size=3, padding=1)
        elif self.modal == 'IR':
            self.conv_x = TinyUNet(in_channels=channel, stride=1, kernel_size=3, padding=1)
        else:
            self.conv_x = DenseBlock(in_ch=channel, out_ch=channel)

        self.strans1 = Spatial(size=size, embed_dim=embed_dim, patch_size=patch_size, channel=channel, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer)

        self.resn = ResidualBlock(2*channel, channel, stride=1)

    
    def forward(self, x):
        conv = self.conv_x(x)
        trans = self.strans1(x)
        result = self.resn(torch.cat([conv, trans], 1))
        x = x + result
        return x
    