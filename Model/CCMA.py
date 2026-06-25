import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.FMVSS import FMVSSBlock
from Model.FEM_Lite import AxialDW
from Model.MASF import PSF


class CFA(nn.Module):
    def __init__(self, dim):
        super(CFA, self).__init__()
        self.dim = dim
    def forward(self, q, k, v):
        B, C, H, W = q.shape
        query = q

        q_weight = q.view(B, C, -1)
        q_weight = F.softmax(q_weight, dim=-1)
        q_weight = q_weight.view(B, C, H, W)
        global_query = (q_weight * query).sum(dim=[2, 3])

        global_query = global_query.unsqueeze(-1).unsqueeze(-1)

        key_mod = k * global_query

        k_weight = key_mod.view(B, C, -1)
        k_weight = F.softmax(k_weight, dim=-1)
        k_weight = k_weight.view(B, C, H, W)
        global_key = (k_weight * key_mod).sum(dim=[2, 3])
        global_key = global_key.unsqueeze(-1).unsqueeze(-1)

        # Step 4: apply global_key to value
        out = v * global_key  # shape: [B, C, H, W]

        xc  = q - global_query.view(B,C,1,1)
        var = (q_weight * xc*xc).sum(dim=[2,3]) + 1e-6            # (B,C)
        conf_q = (1.0/var.sqrt()).clamp(max=10.0).view(B,C,1,1)
        # Step 5: add residual connection
        return out + query * conf_q  # output shape: [B, C, H, W]

class CCMA_Block(nn.Module): # Mamba Axial Selective Fusion
    def __init__(self, in_c):
      super().__init__()
      self.dw1 = nn.Conv2d(in_c//4, in_c//4, kernel_size=3, padding='same',dilation = 1, groups=in_c//4)
      self.dw2= nn.Conv2d(in_c//4, in_c//4, kernel_size=3, padding='same', dilation = 2, groups=in_c//4)
      self.dw3 = nn.Conv2d(in_c//4, in_c//4, kernel_size=3, padding='same', dilation = 3, groups=in_c//4)
      self.dw4 = nn.Conv2d(in_c//4, in_c//4, kernel_size=3, padding='same',dilation = 4,  groups=in_c//4)
      self.block = FMVSSBlock(hidden_dim=in_c // 4)
      self.ins_norm = nn.InstanceNorm2d(in_c, affine=True)
      self.act = nn.LeakyReLU(negative_slope=0.01)
      self.scale = nn.Parameter(torch.ones(1))
      self.dsf = PSF(in_c//2)
      self.attn = CFA(in_c//4)
      self.attn2 = CFA(in_c//4)
      self.adw = AxialDW(in_c, mixer_kernel = (7,7))
      self.norm = nn.LayerNorm(in_c)

    def forward(self, x):
      residual = x
      x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()

      x_1, x_2, x_3 ,x_4= torch.chunk(x, 4, dim=1)

      x1 = self.dw1(x_1)
      x1 = x1.permute(0, 2, 3, 1)
      x1 = self.block(x1)
      x1 = x1.permute(0, 3, 1, 2)
      x1 = self.scale * x_1 + x1

      x2 = self.dw2(x_2)
      x2 = x2.permute(0, 2, 3, 1)
      x2 = self.block(x2)
      x2 = x2.permute(0, 3, 1, 2)
      x2 = self.scale * x_2 + x2

      x3 = self.dw3(x_3)
      x3 = x3.permute(0, 2, 3, 1)
      x3 = self.block(x3)
      x3 = x3.permute(0, 3, 1, 2)
      x3 = self.scale * x_3 + x3

      x4 = self.dw4(x_4)
      x4 = x4.permute(0, 2, 3, 1)
      x4 = self.block(x4)
      x4 = x4.permute(0, 3, 1, 2)
      x4 = self.scale * x_4 + x4

      out1 = self.attn(x1,x2,x3) +x3
      out2 = self.attn2(x4,x3,x2) +x2

      out1 = torch.cat([x1,out1], dim = 1)
      out2 = torch.cat([x4,out2], dim = 1)

      out = self.dsf(out1, out2)
      out = self.act(self.ins_norm(out))
      out = self.adw(out)

      return out


class CCMA_Encoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.block = CCMA_Block(in_c)
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
  
class CCMA_Decoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.block = CCMA_Block(in_c)
  def forward(self, x):
    x = self.block(x)
    out = x
    return out