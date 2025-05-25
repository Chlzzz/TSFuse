import torch
import torch.nn as nn


class SAFM(nn.Module):
    def __init__(self,in_channels,out_channels,c_kernel=3,r_kernel=3,use_process=True):
        '''
                  :param in_channels:
                  :param out_channels: 
                  :param c_kernel: colum dcn kernels kx1 just use k
                  :param r_kernel: row dcn kernels 1xk just use k
                  :param use_att: bools
                  :param use_process: bools
                  '''
        super(SAFM, self).__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.c_k = c_kernel
        self.r_k = r_kernel
        self.process = use_process

        if use_process == True:
            self.preprocess_v = nn.Sequential(nn.Conv2d(self.in_ch,self.in_ch//2,1,1,0),nn.Conv2d(self.in_ch//2,self.in_ch,1,1,0))
            self.preprocess_i = nn.Sequential(nn.Conv2d(self.in_ch,self.in_ch//2,1,1,0),nn.Conv2d(self.in_ch//2,self.in_ch,1,1,0))
    

        self.conv11 = nn.Conv2d(self.in_ch, self.in_ch, 1, 1, 0)
        self.conv12 = nn.Conv2d(self.in_ch, self.in_ch, 1, 1, 0)

        self.conv21 = nn.Sequential(nn.Conv2d(self.in_ch, self.in_ch, 3, 1, 1), nn.Sigmoid())
        self.conv22 = nn.Sequential(nn.Conv2d(self.in_ch, self.in_ch, 3, 1, 1), nn.Sigmoid())

        self.conv31 = nn.Conv2d(self.in_ch, self.in_ch, 3, 1, 1)
        self.conv32 = nn.Conv2d(self.in_ch, self.in_ch, 3, 1, 1)

        self.bottleneck1 = nn.Sequential(
                            nn.Conv2d(self.in_ch,self.in_ch//2,1,1,0),
                            nn.Conv2d(self.in_ch//2,self.in_ch,1,1,0),
                            nn.Sigmoid()
                        )
        self.bottleneck2 = nn.Sequential(
                            nn.Conv2d(self.in_ch,self.in_ch//2,1,1,0),
                            nn.Conv2d(self.in_ch//2,self.in_ch,1,1,0),
                            nn.Sigmoid()
                        )
        
        # self.dcn_row = DCN(2*self.in_ch,2*self.in_ch,kernel_size=(1,self.r_k),stride=1,padding=(0,self.r_k//2))
        # self.dcn_colum = DCN(2*self.in_ch,2*self.in_ch,kernel_size=(self.c_k,1),stride=1,padding=(self.c_k//2,0))
        
        self.average_pool = nn.AvgPool2d(3, 1, 1)
        self.conv_row = nn.Conv2d(2*self.in_ch, 2*self.in_ch, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv_col = nn.Conv2d(2*self.in_ch, 2*self.in_ch, kernel_size=(3, 1), stride=1, padding=(1, 0))
        
        self.conv41 = nn.Conv2d(2 * self.in_ch, self.in_ch, 3, 1, 1)
        self.conv42 = nn.Sequential(
                nn.Conv2d(6 * self.in_ch, self.in_ch, 7, 1, 3),
                nn.Sigmoid()
                )

        self.convf = nn.Conv2d(self.in_ch, self.out_ch, 3, 1, 1)
        

    def forward(self,x_v,x_i):
        if self.process == True:
            x_v = self.preprocess_v(x_v)
            x_i = self.preprocess_i(x_i)
        else:
            x_v = x_v
            x_i = x_i

        x_v1 = self.conv11(x_v)
        x_i1 = self.conv12(x_i)

        x_v2 = self.conv21(x_v1)
        x_i2 = self.conv22(x_i1)

        x_v2 = x_v2 * x_i1 
        x_i2 = x_i2 * x_v1

        x_v2 = x_v1 + x_i2
        x_i2 = x_i1 + x_v2

        x_v3 = self.conv31(x_v2)
        x_i3 = self.conv32(x_i2)

        x_v3 = x_v3 * self.bottleneck1(x_v3)
        x_i3 = x_i3 * self.bottleneck2(x_i3)

        x_vi = torch.cat([x_v3, x_i3], 1)

        x_vi2 = self.conv41(x_vi)
        x_vimask = self.conv42(torch.cat([self.average_pool(x_vi),  
                                   self.conv_row(x_vi),
                                   self.conv_col(x_vi)], 1))
        
        result = self.convf(x_vi2 * x_vimask)

        return result


