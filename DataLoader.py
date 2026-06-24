

import numpy as np
import torch
from PIL import Image
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms


class RandomCrop(transforms.RandomResizedCrop):
  def __call__(self, imgs):
    i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio) # lay tham so cat ngau nhien
    for imgCount in range(len(imgs)):
        imgs[imgCount] = transforms.functional.resized_crop(imgs[imgCount], i, j, h, w, self.size, self.interpolation)
    return imgs

class PH2Loader(Dataset):
  def __init__(self, images, masks,
              transform=True, typeData = "train"):
    self.transform = transform if typeData == "train" else False  # augment data bool
    self.typeData = typeData
    self.images = images
    self.masks = masks
  def __len__(self):
    return len(self.images)

  def rotate(self, image, mask, degrees=(-15,15), p=0.5):
    if torch.rand(1) < p:
        degree = np.random.uniform(*degrees)
        image = image.rotate(degree, Image.NEAREST)
        mask = mask.rotate(degree, Image.NEAREST)
    return image, mask
  def horizontal_flip(self, image, mask, p=0.5):
    if torch.rand(1) < p:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask
  def vertical_flip(self, image, mask, p=0.5):
    if torch.rand(1) < p:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return image, mask
  def random_resized_crop(self, image, mask, p=0.1):
    if torch.rand(1) < p:
        image, mask = RandomCrop((256, 256), scale=(0.8, 0.95))([image, mask])
    return image, mask

  def augment(self, image, mask):
    image, mask = self.random_resized_crop(image, mask)
    image, mask = self.rotate(image, mask)
    image, mask = self.horizontal_flip(image, mask)
    image, mask = self.vertical_flip(image, mask)
    return image, mask

  def augment2(self, image, mask):

    image, mask = self.rotate(image, mask)
    image, mask = self.horizontal_flip(image, mask)
    image, mask = self.vertical_flip(image, mask)
    return image, mask

  def __getitem__(self, idx):
    image = self.images[idx]
    mask = self.masks[idx]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
        mask = mask.astype(np.uint8)
    if len(mask.shape) == 3:
        mask = mask.squeeze()

    image = Image.fromarray(image)
    mask = Image.fromarray(mask)
    if self.transform :
        image, mask = self.augment(image, mask)

    image = transforms.ToTensor()(image)
    mask = np.asarray(mask, np.int64)
    mask = torch.from_numpy(mask[np.newaxis])
    return image, mask

# dataloader for ultrasound dataset
############ dataset busi
import numpy as np
import torch
from PIL import Image
import random
import cv2

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF   # 👈 functional từ torchvision
                  # 👈 nếu sau này cần dùng hàm loss, conv... của torch

from torch.utils.data import Dataset, DataLoader

# --------- Elastic deformation (dùng OpenCV) ----------
def elastic_deformation(image, mask, alpha=50, sigma=6, p=0.3):
    if torch.rand(1) > p:
        return image, mask
    random_state = np.random.RandomState(None)
    shape = image.size[::-1]  # PIL: (W,H), numpy: (H,W)
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    img_np = np.array(image)
    mask_np = np.array(mask)

    distorted_img = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    distorted_mask = cv2.remap(mask_np, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    return Image.fromarray(distorted_img), Image.fromarray(distorted_mask)


# --------- Speckle noise ----------
def add_speckle_noise(image, p=0.3, sigma=0.1):
    if torch.rand(1) > p:
        return image
    img_np = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img_np.shape)
    noisy_img = img_np + img_np * noise
    noisy_img = np.clip(noisy_img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


# --------- Dataset loader ----------
class BUSILoader(Dataset):
    def __init__(self, images, masks, transform=True, typeData="train"):
        self.transform = transform if typeData == "train" else False
        self.typeData = typeData
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def rotate(self, image, mask, degrees=(-15, 15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask

    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def vertical_flip(self, image, mask, p=0.2):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def random_resized_crop(self, image, mask, p=0.3):
        if torch.rand(1) < p:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, i, j, h, w, (256, 256))
            mask = TF.resized_crop(mask, i, j, h, w, (256, 256), interpolation=Image.NEAREST)
        return image, mask

    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        image, mask = elastic_deformation(image, mask)

        # brightness/contrast jitter
        color_aug = transforms.ColorJitter(brightness=0.1, contrast=0.1)
        image = color_aug(image)
        image = add_speckle_noise(image, p=0.3)
        return image, mask

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            mask = mask.astype(np.uint8)

        if len(mask.shape) == 3:
            mask = mask.squeeze()

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if self.transform:
            image, mask = self.augment(image, mask)

        image = transforms.ToTensor()(image)
        mask = np.asarray(mask, np.uint8)
        mask = (mask > 0).astype(np.uint8)  # binarize
        mask = torch.from_numpy(mask[np.newaxis])

        return image, mask


# dataloader for polyp dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import Dataset, DataLoader
class RandomCropPair:
    def __init__(self, size=(256, 256), scale=(0.8, 0.95), ratio=(3/4, 4/3)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image: Image.Image, mask: Image.Image):
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = TF.resized_crop(image, i, j, h, w, self.size, InterpolationMode.BILINEAR)
        mask  = TF.resized_crop(mask,  i, j, h, w, self.size, InterpolationMode.NEAREST)
        return image, mask

class PolypLoader(Dataset):
    def __init__(self, images, masks, transform=True, typeData="train",
                 out_size=(256, 256), binarize_mask=True, mask_for_bce=True):
        self.transform = (transform and typeData == "train")
        self.images = images
        self.masks = masks
        self.out_size = out_size
        self.binarize_mask = binarize_mask
        self.mask_for_bce = mask_for_bce

        self.rcrop = RandomCropPair(size=out_size, scale=(0.8, 0.95))
        self.hflip_p = 0.5
        self.vflip_p = 0.5
        self.rotate_p = 0.5
        self.deg_range = (-15, 15)

        # 🎨 Thêm ColorJitter cho image
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )

    def __len__(self):
        return len(self.images)

    def rotate(self, image, mask):
        if torch.rand(1) < self.rotate_p:
            deg = float(np.random.uniform(*self.deg_range))
            image = TF.rotate(image, deg, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask  = TF.rotate(mask,  deg, interpolation=InterpolationMode.NEAREST,  fill=0)
        return image, mask

    def hflip(self, image, mask):
        if torch.rand(1) < self.hflip_p:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        return image, mask

    def vflip(self, image, mask):
        if torch.rand(1) < self.vflip_p:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        return image, mask

    def random_resized_crop(self, image, mask):
        return self.rcrop(image, mask)

    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.hflip(image, mask)
        image, mask = self.vflip(image, mask)

        # 🎨 Áp dụng ColorJitter CHỈ CHO IMAGE
        if torch.rand(1) < 0.8:   # tỉ lệ áp dụng 80%
            image = self.color_jitter(image)

        return image, mask

    def __getitem__(self, idx):
        img = self.images[idx]
        msk = self.masks[idx]

        if img.dtype != np.uint8: img = img.astype(np.uint8)
        if msk.dtype != np.uint8: msk = msk.astype(np.uint8)

        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        image = Image.fromarray(img, mode="RGB")
        mask  = Image.fromarray(msk)

        if not self.transform:
            image = image.resize(self.out_size, Image.BILINEAR)
            mask  = mask.resize(self.out_size, Image.NEAREST)

        if self.transform:
            image, mask = self.augment(image, mask)

        image = transforms.ToTensor()(image)

        mask_np = np.array(mask, dtype=np.uint8)
        if self.binarize_mask:
            mask_np = (mask_np > 0).astype(np.uint8)

        if self.mask_for_bce:
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        else:
            mask_t = torch.from_numpy(mask_np).long()

        return image, mask_t
