import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import torch.nn.functional as F
import numpy as np
!pip install medpy
from medpy.metric.binary import hd95 as hd95_metric
from Model.FEM_Lite import *
from metrics import *
from Dataloader import *
from EAW_Dice_Loss import EAW_DiceLoss

x_test = np.load(args.path_image_test)
y_test = np.load(args.path_label_test)
@torch.no_grad()
def compute_hd95(y_pred, y_true):
    """Compute HD95 for binary masks."""
    y_pred = (y_pred > 0.5).detach().cpu().numpy().astype(bool)
    y_true = (y_true > 0.5).detach().cpu().numpy().astype(bool)
    if y_pred.sum() == 0 and y_true.sum() == 0:
        return 0.0
    try:
        return hd95_metric(y_pred, y_true)
    except:
        return 100.0  
@torch.no_grad()
def compute_hd95_batch(y_pred_bin, y_true_bin):

    yb = y_pred_bin.detach().cpu().numpy().astype(bool)
    yt = y_true_bin.detach().cpu().numpy().astype(bool)
    B = yb.shape[0]
    vals = []
    for b in range(B):
        yp = np.squeeze(yb[b])  # [H,W]
        yt_ = np.squeeze(yt[b]) # [H,W]
        if yp.sum() == 0 and yt_.sum() == 0:
            vals.append(0.0)
        else:
            try:
                vals.append(float(hd95_metric(yp, yt_)))
            except Exception:
                vals.append(100.0)
    return float(np.mean(vals)) if len(vals) else 100.0


model = FEM_Lite(1)
model.eval()
CHECKPOINT_PATH = ""
test_dataset = DataLoader(PH2Loader(x_test, y_test, typeData="test"), batch_size=1, num_workers=2, prefetch_factor=16)

# Lightning module
class Segmentor(pl.LightningModule):
    def __init__(self, model=model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        loss = EAW_DiceLoss()(y_pred, y_true)
        #print(loss.cpu().numpy(), end = ' ')
        dice = dice_score(y_pred, y_true)
        iou = iou_score(y_pred, y_true)
        hd95 = compute_hd95(y_pred, y_true)
        # Compute TP, FP, FN, TN for Precision, Recall, and F-Score
        y_pred_bin = (y_pred > 0.5).float()  # Threshold predictions to binary
        hd95_batch = compute_hd95_batch(y_pred_bin, y_true)
        TP = (y_pred_bin * y_true).sum(dim=(1, 2, 3))
        FP = ((y_pred_bin == 1) & (y_true == 0)).sum(dim=(1, 2, 3))
        FN = ((y_pred_bin == 0) & (y_true == 1)).sum(dim=(1, 2, 3))

        # Precision, Recall, and F-Score
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Average metrics over the batch
        precision_mean = precision.mean().item()
        recall_mean = recall.mean().item()
        f_score_mean = f_score.mean().item()
        metrics = {"loss": loss,
                   "test_dice": dice,
                   "test_iou": iou,
                   "test_hd95": hd95,
                   "test_hd95_batch":hd95_batch,
                   "test_precision" : precision_mean,
                   "test_Recall": recall_mean,
                   "test_F1-Score": f_score_mean}
        self.log_dict(metrics, prog_bar=True)
        return metrics

@torch.no_grad()
def dice_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
@torch.no_grad()
def iou_score(y_pred, y_true):
  smooth = 1e-5
  y_pred = torch.sigmoid(y_pred)
  y_pred = y_pred.data.cpu().numpy()
  y_true = y_true.data.cpu().numpy()

  y_pred = y_pred > 0.5
  y_true = y_true > 0.5

  intersection = (y_pred & y_true).sum()
  union = (y_pred | y_true).sum()

  return (intersection + smooth) / (union + smooth)
trainer = pl.Trainer()
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model = model)
trainer.test(segmentor, test_dataset)


