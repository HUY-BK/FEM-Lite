
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
from Model.FEM_Lite import *
from Metrics import *
from Dataloader import *
from EAW_Dice_Loss import EAW_DiceLoss

import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = FEM_Lite(1).to(device)
model.train()


class Segmentor(pl.LightningModule):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x):
    return self.model(x)

  def _step(self, batch):
    image, y_true = batch
    y_pred = self.model(image)
    loss = EAW_DiceLoss()(y_pred, y_true)
    dice = dice_score(y_pred, y_true)
    iou = iou_score(y_pred, y_true)

    return loss , dice, iou

  def training_step(self, batch, batch_idx):
    loss, dice, iou = self._step(batch)
    metrics = {"loss": loss, "train_dice": dice, "train_iou":iou}
    self.log_dict(metrics,on_step = False, on_epoch = True, prog_bar = True)
    return loss


  def validation_step(self, batch, batch_idx):
    loss, dice, iou = self._step(batch)
    metrics = {"val_loss":loss, "val_dice": dice, "val_iou": iou}
    self.log_dict(metrics, prog_bar=True)
    return metrics


  def test_step(self, batch, batch_idx):
    loss, dice, iou = self._step(batch)
    metrics = {"loss":loss, "test_dice": dice, "test_iou": iou}
    self.log_dict(metrics, prog_bar=True)
    return metrics

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr = (1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max", factor = 0.5, patience = 8)
    lr_schedulers = {"scheduler":scheduler, "monitor": "val_dice"}
    return [optimizer], lr_schedulers
# Training config


import argparse

parser = argparse.ArgumentParser(description="Train a segmentation model.")
parser.add_argument("--path_checkpoint", type=str, default="./checkpoints", help="Path to save checkpoints")
parser.add_argument("--path_image_train", type=str, required=True, help="Path to training images .npy file")
parser.add_argument("--path_label_train", type=str, required=True, help="Path to training labels .npy file")
parser.add_argument("--path_image_test", type=str, required=True, help="Path to testing images .npy file")
parser.add_argument("--path_label_test", type=str, required=True, help="Path to testing labels .npy file")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for DataLoader")
parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs for training")
args = parser.parse_args()

# Create checkpoint directory
os.makedirs(args.path_checkpoint, exist_ok=True)

# Define checkpoint callback
check_point = pl.callbacks.ModelCheckpoint(
    dirpath=args.path_checkpoint,
    filename="ckpt{val_dice:0.4f}_wo_all",
    monitor="val_dice",
    mode="max",
    save_top_k=1,
    verbose=True,
    save_weights_only=True,
    auto_insert_metric_name=False
)
progress_bar = pl.callbacks.TQDMProgressBar()

# Trainer parameters
PARAMS = {
    "benchmark": True,
    "enable_progress_bar": True,
    "logger": True,
    "callbacks": [check_point, progress_bar],
    "log_every_n_steps": 1,
    "num_sanity_val_steps": 0,
    "max_epochs": args.max_epochs,
    "precision": 16,
}
trainer = pl.Trainer(**PARAMS)

# Model and datasets
segmentor = Segmentor(model=model)

# Load ISIC data
x_train = np.load(args.path_image_train)
y_train = np.load(args.path_label_train)
x_test = np.load(args.path_image_test)
y_test = np.load(args.path_label_test)

print(f'Number of samples for training: {x_train.shape[0]}')
print(f'Number of samples for evaluation: {x_test.shape[0]}')

# Dataloaders
train_dataset = DataLoader(
    PH2Loader(x_train, y_train),
    batch_size=args.batch_size,
    pin_memory=True,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
    prefetch_factor=8,
)
test_dataset = DataLoader(
    PH2Loader(x_test, y_test, typeData="test"),
    batch_size=1,
    num_workers=args.num_workers,
    prefetch_factor=16,
)

# Train
trainer.fit(segmentor, train_dataset, test_dataset)