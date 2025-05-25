import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from model.SAFM import SAFM
from model.convtrans import ConvTransBlock, ResidualBlock
from model.fuse_layer import FuseLayer


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out
    
class decode_out(nn.Module):
    def __init__(self, inchannel=32, kernel=3, stride=1, padding=1):
        super(decode_out, self).__init__()
        # self.conv_in = ConvLayer(2*inchannel, inchannel, kernel_size = 1, stride = 1)
        # self.conv_in0 = ConvLayer(2*inchannel, inchannel, kernel_size = 1, stride = 1)
        self.conv_in1 = ConvLayer(2*inchannel, 2*inchannel, kernel_size = 1, stride = 1)
        self.conv_in2 = ConvLayer(2*inchannel, 2*inchannel, kernel_size = 1, stride = 1)
        self.conv_t1 = ConvLayer(2*inchannel, inchannel, kernel_size=3, stride=1)
        self.conv_t2 = ConvLayer(2*inchannel, 1, kernel_size=3, stride=1, is_last=True)
        
    def forward(self, f1,f2,f3):
        r1 = self.conv_t1(self.conv_in1(torch.cat([f3, f2], 1)))
        result = self.conv_t2(self.conv_in2(torch.cat([f1, r1], 1)))
        return result
    

class TSFuse(nn.Module):
    def __init__(self, config=[2,2,2], img_size=128, patch_size=16, in_chans=32, embed_dim=128,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., norm_layer=None, weight_init='',
                 tbsi_loc=None, tbsi_drop_path=0.):
        
        super(TSFuse, self).__init__()

        self.config = config
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.res_vi = ResidualBlock(1, in_chans, stride=1)
        self.res_ir = ResidualBlock(1, in_chans, stride=1)
        self.res_vr = ResidualBlock(2, in_chans, stride=1)

        self.down1_vi = nn.Sequential(*[ConvTransBlock(embed_dim=64, patch_size=8, channel=32, depth=2, modal='VI') for i in range(config[0])])
        self.down2_vi = nn.Sequential(*[ConvTransBlock(embed_dim=64, patch_size=8, channel=32, depth=2, modal='VI') for i in range(config[0])])
        
        self.down1_ir = nn.Sequential(*[ConvTransBlock(embed_dim=64, patch_size=8, channel=32, depth=2, modal='VR') for i in range(config[0])])
        self.down2_ir = nn.Sequential(*[ConvTransBlock(embed_dim=64, patch_size=8, channel=32, depth=2, modal='VR') for i in range(config[0])])

        self.down1_vr = nn.Sequential(*[ConvTransBlock(embed_dim=64, patch_size=8, channel=32, depth=2, modal='VR') for i in range(config[0])])
        self.down2_vr = nn.Sequential(*[ConvTransBlock(embed_dim=64, patch_size=8, channel=32, depth=2, modal='VR') for i in range(config[0])])
    
        self.vrfuse11 = SAFM(in_channels=32,out_channels=32)
        self.vrfuse21 = SAFM(in_channels=32,out_channels=32)
        # self.vrfuse11 = ResidualBlock(in_ch=64,out_ch=32)
        # self.vrfuse21 = ResidualBlock(in_ch=64,out_ch=32)

        
        self.tbsi_loc = tbsi_loc
        self.tbsi_drop_path = tbsi_drop_path
        self.tbsi_loc1 = FuseLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate, drop_path_rate=self.tbsi_drop_path, norm_layer=norm_layer)
        self.tbsi_loc2 = FuseLayer(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate, drop_path_rate=self.tbsi_drop_path, norm_layer=norm_layer)
       
        self.final_fuse = decode_out(32, 3, 1, 1)

        

    def forward(self, x_vi, x_ir):
        
        x_vr = self.res_vr(torch.cat([x_ir, x_vi], 1))
        x_vi = self.res_vi(x_vi)
        x_ir = self.res_ir(x_ir)
        
        x_vid = self.down1_vi(x_vi) 
        x_ird = self.down1_ir(x_ir)
        x_vrd = self.down1_vr(x_vr)
        vi_feature, ir_feature, x_vrd= self.tbsi_loc1(x_vid, x_ird, x_vrd)
        fuse1 = self.vrfuse11(vi_feature, ir_feature)
        #fuse1 = self.vrfuse11(torch.cat([vi_feature, ir_feature], 1))


        x_vid = self.down2_vi(vi_feature) 
        x_ird = self.down2_ir(ir_feature)
        x_vrd = self.down2_vr(x_vrd)
        vi_feature2, ir_feature2, x_vrd = self.tbsi_loc2(x_vid, x_ird, x_vrd)
        fuse2 = self.vrfuse21(vi_feature2, ir_feature2)
        #fuse2 = self.vrfuse21(torch.cat([vi_feature2, ir_feature2], 1))

        result = self.final_fuse(x_vrd, fuse1, fuse2)

        return result

if __name__ == '__main__':
    net = TSFuse()
    print(net)