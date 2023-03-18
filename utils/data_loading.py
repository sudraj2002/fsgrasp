import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import glob

import numpy as np
import torch
from PIL import Image
from utils.utils import *
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import natsort
import imageio
import numpy


class BasicDataset(Dataset):
    def __init__(self, task_names, data_dir: str, image_size: int = 182, scale: float = 1.0, use_aug=False,
                 use_depth=False, mode='Dice_BCE'):
        self.data_dir = data_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.tasks = []
        self.image_size = image_size
        self.task_names = task_names
        self.aug = use_aug
        self.depth = use_depth
        self.mode = mode

    def zoom(self, image, mask):
        # Performs zooming augmentation on image and mask
        # image: PIL image, mask: PIL image; h x w x c
        # Returns h x w x c PIL images
        
        image = np.array(image)
        mask = np.array(mask)
        x, y = mask.shape
        locs = np.where(mask == 255)
        i = int(np.mean(locs[0]))
        j = int(np.mean(locs[1]))
        
        sizes = [64, 92, 128, 182, 256, 512]
        size = random.choice(sizes)
        up_i = i + size
        down_i = i - size
        up_j = j + size
        down_j = j - size
        
        if up_i > x - 1:
            up_i = x - 1
        if up_j > y - 1:
            up_j = y - 1
        if down_i < 0:
            down_i = 0
        if down_j < 0:
            down_j = 0
        
        new_img = image[down_i:up_i, down_j:up_j, :]
        new_mask = mask[down_i:up_i, down_j:up_j]
        
        new_img, new_mask = Image.fromarray(new_img), Image.fromarray(new_mask)
        new_img = new_img.resize((self.image_size, self.image_size), resample=Image.Resampling.BICUBIC)
        new_mask = new_mask.resize((self.image_size, self.image_size), resample=Image.Resampling.NEAREST)
        # new_img = new_img.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
        # new_mask = new_mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        return new_img, new_mask
        
        
    def transform(self, image, mask):
        angles = [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]
        trs = [transforms.GaussianBlur(kernel_size=3),
                                  transforms.ColorJitter(brightness=0.45, saturation=0.35, contrast=0.35, hue=0.3)]
        img = torch.tensor(image)
        mask = torch.tensor(mask)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        angle = random.choice(angles)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)
        for trans in trs:
            img = trans(img)
        return np.asarray(img), np.asarray(mask)


    def taskload(self, idx):
        task_images = []
        task_labels = []
        task = self.task_names[idx]
        image_names = []
        mask_names = []
        taskpath = os.path.join(self.data_dir, task)
        for name in natsort.natsorted(os.listdir(taskpath)):
            if "_" in name.split(".")[0]:
                mask_names.append(name)
            else:
                image_names.append(name)
        assert len(image_names) == len(mask_names)
        
        if random.random() > 0.5:
            zoom_aug = True
        else:
            zoom_aug = False
        
        if not self.aug:
            zoom_aug = False
            
        for i in range(len(image_names)):
            iname = image_names[i]
            mname = mask_names[i]
            img, mask = self.load(os.path.join(taskpath, iname)), self.load(os.path.join(taskpath, mname))
            img, mask = self.preprocess(img, mask, zoom_aug=zoom_aug)
            task_images.append(img)
            task_labels.append(mask)
        task_labels = np.stack(task_labels)
        task_images = np.stack(task_images)
        return task_images, task_labels

    def __len__(self):
        return len(self.task_names)

    def preprocess(self, img, mask, zoom_aug):
        w, h = self.image_size, self.image_size
        assert w > 0 and h > 0, 'Scale is too small, resized images would have no pixel'
        if zoom_aug is False:
            if self.depth:
                img = img.resize((w, h),
                                 resample=Image.Resampling.NEAREST)
                mask = mask.resize((w, h),
                                 resample=Image.Resampling.NEAREST)
            else:
                img = img.resize((w, h),
                                 resample=Image.Resampling.BICUBIC)
                mask = mask.resize((w, h),
                                   resample=Image.Resampling.NEAREST)
        else:
            img, mask = self.zoom(img, mask)
        img_ndarray = np.asarray(img)
        mask_ndarray = np.asarray(mask)

        if img_ndarray.ndim == 2:
            if self.depth:
                img_ndarray = img_ndarray[np.newaxis, ...]
                mask_ndarray = np.expand_dims(mask_ndarray, axis=0)
                if self.aug:
                    img_ndarray, mask_ndarray = self.transform(img_ndarray, mask_ndarray)
                mask_ndarray = mask_ndarray / 255
                if self.mode == 'Dice_BCE' or self.mode == 'BCE':
                    mask = np.ones(mask_ndarray[0].shape)
                    mask = np.multiply(mask, mask_ndarray[0] == 1)
                    
                if self.mode == 'Dice_CE' or self.mode == 'CE':
                    mask = np.ones(mask_ndarray[0].shape)
                    mask = np.multiply(mask, mask_ndarray[0] == 1)
                img = img_ndarray / 255
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
            mask_ndarray = np.expand_dims(mask_ndarray, axis=0)
            if self.aug:
                img_ndarray, mask_ndarray = self.transform(img_ndarray, mask_ndarray)
            mask_ndarray = mask_ndarray / 255
            mask_ndarray[mask_ndarray < 0.5] = 0
            mask_ndarray[mask_ndarray >= 0.5] = 1
            if self.mode == 'Dice_BCE' or self.mode == 'BCE' or self.mode == 'Dice':
                # mask = np.ones((1,) + mask_ndarray[0].shape)
                # mask[0] = np.multiply(mask[0], mask_ndarray[0] == 1)
                mask = np.ones(mask_ndarray[0].shape)
                mask = np.multiply(mask, mask_ndarray[0] == 1)
                    
            if self.mode == 'Dice_CE' or self.mode == 'CE':
                mask = np.ones(mask_ndarray[0].shape)
                mask = np.multiply(mask, mask_ndarray[0] == 1)
            img = img_ndarray / 255
        return img, mask

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        img, mask = self.taskload(idx)
        return {
            'taskimage': torch.as_tensor(img.copy()).float().contiguous(),
            'taskmask': torch.as_tensor(mask.copy()).long().contiguous()
        }
