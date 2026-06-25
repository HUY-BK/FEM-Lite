import torch
import torch.nn as nn
from Model.FEM_Lite import AxialDW
from Model.FMVSS import FMVSSBlock

class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class ChannelFusion(nn.Module):
  def __init__(self, in_c, reduction = 8):
    super().__init__()
    self.shared_mlp = nn.Sequential(
      nn.Linear(2 * in_c, in_c// reduction),
      nn.LayerNorm(in_c//reduction),
      nn.ReLU(inplace=True),
    )
    self.linear1 = nn.Linear(in_c//reduction , in_c)
    self.linear2 = nn.Linear(in_c // reduction, in_c)
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)

  def forward(self, x1, x2):
   #x = self.adw1(x1) + self.adw2(x2)
    x = x1 + x2
    x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim = 1)
    x = x.view(x.size(0), -1)
    x = self.shared_mlp(x)
    s1 = torch.sigmoid(self.linear1(x))
    s2 = torch.sigmoid(self.linear2(x))
    s1 = s1.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
    s2 = s2.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
    return s1, s2

class PSF(nn.Module):
  def __init__(self, in_c):
    super().__init__()
    self.conv = nn.Conv2d(in_c*2, in_c*2, kernel_size = 5, padding = 2, groups = in_c*2)
    self.simAM = simam_module(channels = in_c)
    self.ca = ChannelFusion(in_c)

  def forward(self, x1,x2):
    c1,c2 = self.ca(x1,x2)
    s1 = x1 * c1
    s2 = x2 * c2

    x1 = x1 + s1
    x2 = x2 + s2

    x = torch.cat([s1,s2], dim = 1)
    x = self.conv(x)
    x = self.simAM(x)
    x = x + torch.cat([x1, x2], dim = 1)
    return x
  

class MASF_Block(nn.Module): 
    def __init__(self, in_c):
      super().__init__()
      self.dw = nn.Conv2d(in_c//2, in_c//2, kernel_size=3, padding=1, groups=in_c//2)
      self.block = FMVSSBlock(hidden_dim=in_c//2)
      self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
      self.act = nn.LeakyReLU(negative_slope=0.01)
      self.scale = nn.Parameter(torch.ones(1))
      self.dsf = PSF(in_c//2)
      self.adw = AxialDW(in_c, mixer_kernel = (3,3))
      self.adw7_1 = nn.Conv2d(in_c//2 , in_c//2 , kernel_size = (7,1), padding = 'same')
      self.adw1_7 = nn.Conv2d(in_c//2 , in_c//2 , kernel_size = (1,7), padding = 'same')

    def forward(self, x):
      residual = x

      x_1, x_2 = torch.chunk(x, 2, dim=1)
      x_1 = self.dw(x_1)
      x1 = x_1.permute(0, 2, 3, 1)
      x1 = self.block(x1)
      x1 = x1.permute(0, 3, 1, 2)

      x2 = self.adw7_1(self.adw1_7((x_2)))

      x = self.dsf(x1, x2)
      x = self.act(self.ins_norm(x)) + self.scale * residual
      out = self.adw(x)
      return out

class MASF_Encoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.block = MASF_Block(in_c)
    self.pw = nn.Conv2d(in_c, out_c, kernel_size = 1)
    self.bn = nn.BatchNorm2d(in_c)
    self.act = nn.ReLU()
    self.down = nn.MaxPool2d((2,2))
  def forward(self, x):
    x = self.block(x)

    x = self.act(self.bn(x))
    skip = x
    x = self.pw(x)
    out = self.down(x)
    return out, skip
  
class MASF_Decoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.block = MASF_Block(in_c)
  def forward(self, x):
    x = self.block(x)
    out = x
    return out
