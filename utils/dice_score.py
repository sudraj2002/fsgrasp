import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MainLoss(nn.Module):
    def __init__(self, loss_type='Dice_BCE', loss_weight=1.0):
        super(MainLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight

    def forward(self, inputs, target, smooth=1):
        if self.loss_type == 'Dice_BCE':
            input_nsig = inputs.clone()
            inputs = F.sigmoid(inputs)
            iflat = inputs.view(-1)
            tflat = target.view(-1)
            input_nsig = input_nsig.view(-1)
            intersection = (iflat * tflat).sum()                          
            dice = (2.*intersection + smooth)/(iflat.sum() + tflat.sum() + smooth)  
            loss = 1 - dice
            loss += F.binary_cross_entropy_with_logits(input_nsig, tflat, reduction='mean',
                                                       pos_weight=self.loss_weight)

        elif self.loss_type == 'Dice_CE':
            targets = F.one_hot(target, 2).permute(0, 3, 1, 2)
            loss = 0.0
            for channel in range(inputs.shape(1)):
                insig = F.sigmoid(inputs[:, channel, ...])
                iflat = insig.contiguous().view(-1)
                tflat = targets[:, channel, ...].contiguous().view(-1)
                intersection = (iflat * tflat).sum()                            
                dice = (2.*intersection + smooth)/(iflat.sum() + tflat.sum() + smooth)  
                loss += 1 - dice
            loss += F.cross_entropy(inputs, target, reduction='mean', weight=self.loss_weight)

        elif self.loss_type == 'BCE':
            inputs = inputs.view(-1)
            targets = target.view(-1)
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean',
                                                           pos_weight=self.loss_weight)
        elif self.loss_type == 'CE':
            loss = F.cross_entropy(inputs, target, reduction='mean', weight=self.loss_weight)
        
        elif self.loss_type == 'Dice':
            inputs = F.sigmoid(inputs)
            iflat = inputs.view(-1)
            tflat = target.view(-1)
            intersection = (iflat * tflat).sum()                            
            dice = (2.*intersection + smooth)/(iflat.sum() + tflat.sum() + smooth)  
            loss = 1 - dice

        else:
            raise ValueError('Enter a valid loss function')
        return loss
