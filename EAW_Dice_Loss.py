def dice_score(y_pred, y_true, smooth=1e-6):

    y_pred = torch.sigmoid(y_pred)
    B = y_true.size(0)

    y_pred = y_pred.view(B, -1)
    y_true = y_true.view(B, -1)

    intersection = (y_pred * y_true).sum(dim=1)
    dice = (2 * intersection + smooth) / (y_pred.sum(dim=1) + y_true.sum(dim=1) + smooth)

    return dice.mean() 

def dice_loss(y_pred, y_true, smooth=1e-6):
    return 1 - dice_score(y_pred, y_true, smooth)

class EAW_DiceLoss(nn.Module):
    def __init__(self, alpha = 0.6 , beta = 0.4, delta = 1 , gamma = 1.5, smooth = 1e-6, lamda = 0.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.lamda = lamda
        self.edge_kernel_size = 5
    def get_soft_edge_mask(self, y_true):
        y_true = y_true.float()
        lap_kernel = torch.tensor([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]], dtype=torch.float32, device=y_true.device).unsqueeze(0).unsqueeze(0)

        edge = F.conv2d(y_true, lap_kernel, padding=1).abs()
        edge = (edge > 0.01).float() 

        edge = F.avg_pool2d(edge, kernel_size=self.edge_kernel_size, stride=1, padding=self.edge_kernel_size // 2)
        edge = (edge > 0).float()

        return edge 

    def forward(self, y_pred, y_true):

        y_pred = torch.sigmoid(y_pred).float()
        edge_mask = self.get_soft_edge_mask(y_true)

        term1 = 1
        term2 = self.alpha * y_true * (1 - y_pred)
        term3 = self.beta * (1 - y_true) * y_pred

        uncertainty = y_pred * (1 - y_pred)
        term4 = self.delta * uncertainty

        weight = (term1 + (term2 + term3 ) * edge_mask + term4) ** self.gamma

        intersection = (weight * y_pred * y_true).sum(dim = (1,2,3))
        union = (weight * y_pred).sum(dim = (1,2,3)) + (weight * y_true).sum(dim = (1,2,3))
        dice = 1 - (2. * intersection +  self.smooth)/ (union + self.smooth)
        dice = dice.mean()
        return dice
