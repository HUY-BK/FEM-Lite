import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.MASF import MASF_Encoder, CCMA_Encoder
from Model.CCMA import MASF_Decoder, CCMA_Decoder
from Bottleneck import Bottleneck


class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class ConvBlock(nn.Module):
  def __init__(self, in_c,mixer_kernel = (7,7)):
    super().__init__()
    self.adw = AxialDW(in_c, mixer_kernel = mixer_kernel, dilation = 1)
    self.dw = nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 1, groups = in_c )
  def forward(self, x):
    residual = x
    x = self.adw(x)
    x = self.dw(x)
    return x

class AC_Encoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.block = ConvBlock(in_c)
    self.bn = nn.BatchNorm2d(in_c)
    self.act = nn.ReLU()
    self.pw  = nn.Conv2d(in_c, out_c, kernel_size = 1)
    self.down = nn.MaxPool2d((2,2))
  def forward(self, x):
    x = self.block(x)

    x = self.act(self.bn(x))
    skip = x
    x = self.pw(x)
    x = self.down(x)
    return x, skip
  
class AC_Decoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.block = ConvBlock(in_c)
    self.pw = nn.Conv2d(in_c, out_c, kernel_size = 1)
  def forward(self, x):
    x = self.block(x)
    x = self.pw(x)

    return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=int((kernel_size-1)/2))

    def forward(self, x):
        max_pool = x.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        avg_pool = x.mean(dim=1, keepdim=True)    # [B, 1, H, W]
        pool = torch.cat([max_pool, avg_pool], dim=1)  # [B, 2, H, W]
        att = torch.sigmoid(self.conv(pool))  # [B, 1, H, W]
        return att
class ChannelAttention(nn.Module):
    def __init__(self, in_c, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Linear(in_c, in_c // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // reduction, in_c)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.avg_pool(x)  # [B, C, 1, 1]
        max = self.max_pool(x)  # [B, C, 1, 1]

        avg = avg.view(x.size(0), -1)  # [B, C]
        max = max.view(x.size(0), -1)  # [B, C]

        avg_out = self.shared_mlp(avg)  # [B, C]
        max_out = self.shared_mlp(max)  # [B, C]

        pool_sum = avg_out + max_out  # [B, C]
        sig_pool = self.sigmoid(pool_sum)  # [B, C]
        sig_pool = sig_pool.view(x.size(0), x.size(1), 1, 1)  # [B, C, 1, 1]

        return sig_pool
class CBAM(nn.Module):
    def __init__(self, in_c, reduction=16, kernel_size=3):
        super().__init__()
        self.in_c = in_c
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.channel_attn = ChannelAttention(in_c, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        chan_att = self.channel_attn(x)  # [B, C, 1, 1]
        fp = chan_att * x  # [B, C, H, W]
        spat_att = self.spatial_attn(fp)  # [B, 1, H, W]
        fpp = spat_att * fp  # [B, C, H, W]
        return fpp



class ConCat(nn.Module):
  def __init__(self, in_c):
    super().__init__()
    self.in_channels1 = in_c
    self.pw = nn.Conv2d(in_c *3 , in_c, kernel_size = 1)
    self.cbam = CBAM(in_c)
    self.bn = nn.BatchNorm2d(in_c)
  def forward(self, skip, x):

    skip = self.cbam(skip)
    x = torch.cat([skip,x], dim = 1)
    x = self.bn(self.pw(x))
    return x


class FEM_Lite(nn.Module):
  def __init__(self, num_class,  c_list=[16, 32, 64, 128, 256, 512]):
    super().__init__()
    self.conv1 = nn.Conv2d(3, c_list[0], kernel_size = 3, padding = 1)
    self.up = nn.Upsample(scale_factor=2)

    self.e1= AC_Encoder(c_list[0], c_list[1])
    self.e2= MASF_Encoder(c_list[1], c_list[2])

    self.e3= MASF_Encoder(c_list[2], c_list[3])
    self.e4= CCMA_Encoder(c_list[3], c_list[4])

    self.e5 = CCMA_Encoder(c_list[4], c_list[5])

    self.bt = Bottleneck(c_list[5],8,8)

    self.d5 = CCMA_Decoder(c_list[4], c_list[4])
    self.d4 = CCMA_Decoder(c_list[3], c_list[3])
    self.d3 = MASF_Decoder(c_list[2], c_list[2])
    self.d2=  MASF_Decoder(c_list[1], c_list[1])
    self.d1 = AC_Decoder(c_list[0], c_list[0])

    self.ra1 = ConCat(c_list[0])
    self.ra2 = ConCat(c_list[1])
    self.ra3 = ConCat(c_list[2])
    self.ra4 = ConCat(c_list[3])
    self.ra5 = ConCat(c_list[4])

    self.out = nn.Conv2d(c_list[0], num_class, kernel_size=1)


  def forward(self, x):
    H, W = x.shape[2:]
    x= self.conv1(x)

    x , skip1 = self.e1(x)

    x , skip2 = self.e2(x)

    x, skip3 = self.e3(x)

    x , skip4 = self.e4(x)

    x , skip5 = self.e5(x)

    x = self.bt(x)
    x = self.up(x)
    x = self.ra5(skip5, x)
    x = self.d5(x)
    x = self.up(x)
    x = self.ra4(skip4, x)
    x = self.d4(x)
    x = self.up(x)
    x = self.ra3(skip3, x)
    x = self.d3(x)
    x = self.up(x)
    x = self.ra2(skip2, x)
    x = self.d2(x)
    x = self.up(x)
    x = self.ra1(skip1, x)
    x = self.d1(x)

    out = self.out(x)
    #out = x
    return out

