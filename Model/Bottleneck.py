
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']

    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method) # Lấy tọa độ tần số (u,v) sẽ dùng, theo freq_sel_method.
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return y.expand_as(x) * x


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) # Tính cosin basis
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):#Xếp filter cho từng kênh
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)
        return dct_filter


class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16):
        super(SpatialGate, self).__init__()
        inter_channel = gate_channel // reduction_ratio

        self.reduce = nn.Sequential(
            nn.Conv2d(gate_channel, inter_channel, kernel_size=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )

        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inter_channel, inter_channel, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm2d(inter_channel),
                nn.ReLU()
            ) for d in [1, 3, 5]
        ])

        self.fuse = nn.Conv2d(inter_channel * 3, 1, kernel_size=1)

    def forward(self, x):
        x_reduced = self.reduce(x)
        features = [conv(x_reduced) for conv in self.dilated_convs]
        out = torch.cat(features, dim=1)
        out = self.fuse(out)
        return out.expand_as(x)


class MSA(nn.Module):
  def __init__(self, in_c, H, W, kernel_size = 3, reduction = 16, method='top16'):
    super().__init__()
    self.ca = MultiSpectralAttentionLayer(in_c,H,W, reduction = reduction , freq_sel_method= method)
    self.sa = SpatialGate(in_c, reduction_ratio = reduction)
  def forward(self, x):
    y = self.ca(x)
    att = y + x
    x = y
    att = self.sa(att)
    att = torch.sigmoid(att)
    x = att * x
    return x
  
  
class CDGA(nn.Module):
  def __init__(self, in_c):
    super().__init__()

    self.in_c = in_c
    self.maxP = nn.AdaptiveMaxPool2d(1)
    self.avgP = nn.AdaptiveAvgPool2d(1)
    self.conv2d = nn.Conv2d(2,1, kernel_size = 7, padding = 'same')
    self.conv1d = nn.Conv1d(2,1, kernel_size = 7, padding = 'same')


  def forward(self, q,k,v):
    B, C, H, W = v.shape

    k_flat = k.view(B, C, -1)  # [B, C, H*W]

    # tính theo spatial
    q1_s = self.maxP(q).squeeze(-1)
    q2_s = self.avgP(q).squeeze(-1)
    q_sa = torch.cat([q1_s,q2_s], dim = 2).permute(0,2,1) # B,2,C
    q_sa = F.softmax(q_sa, dim = -1)
    attn = q_sa @ k_flat # B, 2,  H* W
    attn = self.conv2d(attn.view(B, 2, H, W))
    attn = attn.expand_as(v)


    # tinh theo channel
    q1_c = torch.max(q, dim = 1, keepdim = True)[0]
    q2_c = torch.mean(q, dim = 1, keepdim = True)
    q_ca = torch.cat([q1_c,q2_c], dim = 1).view(B, -1 , H*W) # B, 2, H, W
    q_ca = F.softmax(q_ca, dim = -1)
    attn2 = q_ca @ k_flat.permute(0,2,1).contiguous() # B,2 ,C
    attn2 = self.conv1d(attn2).view(B,C,1,1)
    attn2 = attn2.expand_as(v)

    att = torch.sigmoid(attn+ attn2)
    out = att * v
    return out


class Bottleneck(nn.Module):
  def __init__(self, in_c,H =8, W =8 , reduction = 16):
    super().__init__()
    self.atHW = MSA(in_c,H,W,reduction = reduction, method='top16' )
    self.atCH = MSA(W,in_c,H,reduction = 2, method='top4' )
    self.atCW = MSA(H,in_c,W,reduction = 2, method='top4' )
    self.cdga = CDGA(in_c)

    self.dw1 = nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 1 , groups = in_c)
    self.dw2 = nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 1 , groups = in_c)
    self.dw3 = nn.Conv2d(in_c, in_c, kernel_size = 3, padding = 1 , groups = in_c)

    self.pw1 = nn.Conv2d(in_c, in_c, kernel_size = 1, groups = 16)
    self.pw2 = nn.Conv2d(in_c, in_c, kernel_size = 1, groups = 16)
    self.pw3 = nn.Conv2d(in_c, in_c, kernel_size = 1, groups = 16)

    self.norm = nn.LayerNorm(in_c)

  def forward(self, x):
    B, C, H, W = x.shape
    residual = x
    x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
    q = self.pw1(self.dw1(x))
    v = self.pw2(self.dw2(x))
    k = self.pw3(self.dw3(x))

    x = self.cdga(q,k,v) + residual
    residual = x
    out1 = self.atHW(x)
    out2 = self.atCH(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    out3 = self.atCW(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    out = out1+ out2 + out3
    att = torch.sigmoid(out)
    out = att * residual
    return  out
