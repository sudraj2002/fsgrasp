#!/usr/bin/env python3

import random
import copy
import numpy as np
import torch

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from utils import metrics
from utils.utils import split_train_test_tasks, merge, overlay
from unet import UNet
import imageio
import os
import warnings
from utils.dice_score import MainLoss
from utils.utils import save_images, postprocess

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--dir_checkpoint', '-d', type=str, default=None, help='Checkpoint saving directory')
    parser.add_argument('--depth', '-z', type=bool, default=False, help='Whether to train with depth maps')
    parser.add_argument('--aug', '-a', type=bool, default=False, help='Whether to use augmentation')
    parser.add_argument('--image_size', '-i', type=int, default=360, help='Image size')
    parser.add_argument('--shots', '-s', type=int, default=10, help='Number of test shots')
    parser.add_argument('--bsz', '-b', type=int, default=4, help='Inner batch size')
    parser.add_argument('--steps', '-p', type=int, default=30, help='Adaptation steps')
    parser.add_argument('--data_dir', '-t', type=str, default='test_tasks', help='Test directory')
    parser.add_argument('--classes', '-l', type=int, default=2, help='classes')
    parser.add_argument('--save_tr', type=bool, default=False, help='Whether to save train inputs')
    parser.add_argument('--loss', type=str, default='Dice_BCE', help='Loss function')
    parser.add_argument('--loss_weight', type=float, default=None, help='Loss weight')
    parser.add_argument('--over', type=bool, default=False, help='Overlay')
    parser.add_argument('--no_seed', type=bool, default=False, help='Use seed or not')
    parser.add_argument('--postprocess', type=bool, default=False, help='Do postprocessing')
    return parser.parse_args()


def fast_adapt(batch, learner, adapt_opt, loss, adaptation_steps, shots, ways, batch_size, device, return_preds=False, post=False):
    data, labels, name = batch["taskimage"], batch["taskmask"], batch["name"]
    data, labels = torch.squeeze(data, axis=0), torch.squeeze(labels, axis=0)
    data, labels = data.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    print("Name: ", name[0])

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    shots = torch.randperm(data.size(0))[:shots]
    # shots = np.array([1, 5, 2, 9, 4, 6, 3, 7, 0, 8])[:shots] # Fix shots to avoid randomness
    adaptation_indices[shots] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    if adaptation_data.size(0) < batch_size:
        warnings.warn('Batch size higher than available tasks {}'.format(adaptation_data.size(0)))

    # Adapt the model
    for step in range(adaptation_steps):
        idx = torch.randperm(adaptation_data.size(0))[:batch_size]
        adapt_X = adaptation_data[idx]
        adapt_y = adaptation_labels[idx]
        adapt_opt.zero_grad()
        prediction_train = learner(adapt_X)
        error = loss(prediction_train, adapt_y)
        error.backward()
        adapt_opt.step()
    # Evaluate the adapted model
    predictions = []
    valid_error = 0
    mIoU = 0
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    for i in range(len(evaluation_data)):
        eval_data = evaluation_data[i]
        eval_label = evaluation_labels[i]
        eval_data, eval_label = eval_data.unsqueeze(0), eval_label.unsqueeze(0)
        prediction = learner(eval_data)
        val_error = loss(prediction, eval_label)
        s_mask = prediction[0].detach().cpu()
        if post:
            s_mask[s_mask < 0.5] = 0
            s_mask[s_mask >= 0.5] = 1
            s_mask = postprocess(np.array(s_mask).astype(np.uint8))
            predictions.append(s_mask)
            s_mask = np.expand_dims(s_mask, axis=0)
            s_mask = torch.tensor(s_mask).to(device)
            mi = metrics.compute_iou(s_mask, eval_label)
        else:
            predictions.append(s_mask)
            mi = metrics.compute_iou(prediction, eval_label)
        valid_error += val_error
        mIoU += mi
    predictions = np.stack(predictions, axis=0)
    valid_error /= len(evaluation_data)
    mIoU /= len(evaluation_data)
    valid_accuracy = mIoU
    if return_preds:
        return valid_error, valid_accuracy, [evaluation_data, evaluation_labels, predictions]
    return valid_error, valid_accuracy


def main(
        fast_lr=3e-4,
        meta_test_bsz=1,
        ways=1,
        seed=42,
        cuda=1,
):
    test = True
    depth = args.depth
    im_size = args.image_size
    data_dir = args.data_dir
    eval_steps = args.steps
    test_shots = args.shots
    test_inner_bsz = args.bsz
    loss_type = args.loss
    loss_weight = args.loss_weight
    st = args.save_tr
    post = args.postprocess
    if loss_weight is not None:
        if loss_type == 'Dice_BCE' or loss_type == 'BCE':
            loss_weight = torch.tensor([loss_weight], dtype=torch.float32).cuda()
        elif loss_type == 'Dice_CE' or loss_type == 'CE':
            loss_weight = torch.tensor([loss_weight, 1], dtype=torch.float32).cuda()
    
    no_seed = args.no_seed
    cuda = bool(cuda)
    if not no_seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        if not no_seed:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.use_deterministic_algorithms(True)
        device = torch.device('cuda')
    if test:
        assert args.load is not None

    # Create model
    if depth:
        num_channels = 1
    else:
        num_channels = 3
    model = UNet(n_channels=num_channels, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device)

    adapt_opt = torch.optim.Adam(model.parameters(), lr=fast_lr, betas=(0, 0.999), weight_decay=1e-7)

    if args.load:
        info = torch.load(args.load, map_location=device)
        m_dict = info["model_state_dict"]
        adapt_opt_dict = info["adapt_opt_state_dict"]
        contd = info["iteration"]
        best_acc = info["acc"]
        model.load_state_dict(m_dict)
        adapt_opt.load_state_dict(adapt_opt_dict)
        logging.info(f'Model loaded from {args.load}')
        print(f"Model loaded from {args.load}")
        print("Iteration: {}".format(contd))
        print("Model accuracy: {}".format(best_acc))
        del info
        del m_dict
        del adapt_opt_dict


    adapt_opt_state = adapt_opt.state_dict()
    loss = MainLoss(loss_type=loss_type, loss_weight=loss_weight)
    # Split the dataset into train, val and test tasks
    tasks = []
    for task in os.listdir(data_dir):
        tasks.append(task)

    # train_names, test_names = split_train_test_tasks(tasks, num_test_tasks)
    test_names = tasks
    test_dataset = BasicDataset(test_names, data_dir=data_dir, image_size=im_size, scale=1.0, use_depth=depth,
                                use_aug=False)

    # 3. Create data loaders
    test_loader_args = dict(batch_size=1, num_workers=4, pin_memory=True)
    g = torch.Generator()
    g.manual_seed(seed)
    if not no_seed:
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, generator=g, **test_loader_args)
    else:
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **test_loader_args)
    # (Initialize logging)

    if test:
        loader = iter(test_loader)
        meta_test_error = 0
        meta_test_accuracy = 0
        for iterate in range(len(test_loader)):
            learner = copy.deepcopy(model)
            adapt_opt = torch.optim.Adam(
                learner.parameters(),
                lr=fast_lr,
                betas=(0, 0.999)
            )
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(test_loader)
                batch = next(loader)
            adapt_opt.load_state_dict(adapt_opt_state)
            evaluation_error, evaluation_accuracy, prediction_op = fast_adapt(batch,
                                                                              learner,
                                                                              adapt_opt,
                                                                              loss,
                                                                              eval_steps,
                                                                              test_shots,
                                                                              ways,
                                                                              test_inner_bsz,
                                                                              device, return_preds=True, post=post)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
            print("Task IoU: ", evaluation_accuracy.item())
            ims = prediction_op[0]
            labels = prediction_op[1]
            predicts = prediction_op[2]
            save_images(ims, labels, torch.tensor(predicts), mode=loss_type, extra=iterate, over=args.over)
        print('Meta Test Accuracy', meta_test_accuracy / len(test_loader))
        return meta_test_accuracy/len(test_loader)

if __name__ == '__main__':
    args = get_args()
    if args.no_seed is True:
        tot = 0
        for i in range(10):
            miou = main()
            tot += miou
        print("Average miou: ", tot / 10)
    else:
        miou = main()