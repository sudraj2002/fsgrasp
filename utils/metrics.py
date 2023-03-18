import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.init as initer

import imageio
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_iou(outputs: torch.Tensor, labels: torch.Tensor, mode = 'Dice_BCE'):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    SMOOTH = 1e-6
    if mode == 'Dice_CE' or mode == 'CE':
        outputs = outputs.argmax(dim=1)
    elif mode == 'Dice_BCE' or mode == 'BCE' or mode == 'Dice':
        outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
        outputs[outputs > 0.5] = 1
        outputs[outputs <= 0.5] = 0
        outputs = outputs.byte().detach()
    
    # imageio.imsave('res.png', (outputs[0] * 255).cpu().numpy().astype(np.uint8))
    labels = labels.byte().detach()

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch
