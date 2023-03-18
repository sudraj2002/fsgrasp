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
from utils.utils import save_images
from utils.utils import postprocess



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
    parser.add_argument('--adapt_dir', '-c', type=str, default='fssd_adapt', help='Adaptation directory')
    parser.add_argument('--test_dir', '-t', type=str, default='fssd_test', help='Test directory')
    parser.add_argument('--classes', '-l', type=int, default=2, help='classes')
    parser.add_argument('--save_tr', type=bool, default=False, help='Whether to save train inputs')
    parser.add_argument('--loss', type=str, default='Dice_BCE', help='Loss function')
    parser.add_argument('--loss_weight', type=float, default=None, help='Loss weight')
    parser.add_argument('--over', type=bool, default=False, help='Overlay')
    parser.add_argument('--no_seed', type=bool, default=False, help='Use seed or not')
    parser.add_argument('--postprocess', type=bool, default=False, help='Do postprocessing')
    return parser.parse_args()


def fast_adapt(batch_adapt, learner, adapt_opt, loss, adaptation_steps, shots, batch_size, device, return_preds=False):
    data, labels, name = batch_adapt["taskimage"], batch_adapt["taskmask"], batch_adapt["images"]
    print(name)
    data, labels = torch.squeeze(data, axis=0), torch.squeeze(labels, axis=0)
    data, labels = data.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    shots = torch.randperm(data.size(0))[:shots]
    print(shots)
    adaptation_data, adaptation_labels = data[shots], labels[shots]
    if adaptation_data.size(0) < batch_size:
        warnings.warn('Batch size higher than available tasks {}'.format(adaptation_data.size(0)))
    adapt_data = []
    adapt_label = []
    pred = []
    # Adapt the model
    for step in range(adaptation_steps):
        idx = torch.randperm(adaptation_data.size(0))[:batch_size]
        adapt_X = adaptation_data[idx]
        adapt_y = adaptation_labels[idx]
        adapt_opt.zero_grad()
        prediction_train = learner(adapt_X)
        error = loss(prediction_train, adapt_y)
        adapt_data.append(adapt_X[0])
        adapt_label.append(adapt_y[0])
        pred.append(prediction_train[0])
        error.backward()
        adapt_opt.step()


def main(
        fast_lr=3e-4,
        meta_test_bsz=1,
        seed=42,
        cuda=1,
):
    test = True
    depth = args.depth
    im_size = args.image_size
    data_dir_adapt = args.adapt_dir
    data_dir_test = args.test_dir
    test_steps = args.steps
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

    loss = MainLoss(loss_type=loss_type, loss_weight=loss_weight)

    # Split the dataset into train, val and test tasks
    tasks_adapt = []
    tasks_test = []
    for task in os.listdir(data_dir_adapt):
        tasks_adapt.append(task)
    num_test_tasks_adapt = len(tasks_adapt)
    _, test_names_adapt = split_train_test_tasks(tasks_adapt, num_test_tasks_adapt)

    for task in os.listdir(data_dir_test):
        tasks_test.append(task)
    num_test_tasks_test = len(tasks_test)
    _, test_names = split_train_test_tasks(tasks_test, num_test_tasks_test)

    adapt_dataset = BasicDataset(test_names_adapt, data_dir=data_dir_adapt, image_size=im_size, scale=1.0, use_depth=depth, use_aug=False)
    test_dataset = BasicDataset(test_names, data_dir=data_dir_test, image_size=im_size, scale=1.0, use_depth=depth, use_aug=False)
    # 3. Create data loaders
    test_loader_args = dict(batch_size=meta_test_bsz, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **test_loader_args)
    adapt_loader = DataLoader(adapt_dataset, shuffle=False, drop_last=False, **test_loader_args)

    adapt_opt_state = adapt_opt.state_dict()

    assert args.load is not None
    loader_test = iter(test_loader)
    loader_adapt = iter(adapt_loader)
    for iterate in range(len(adapt_loader)):
        learner = copy.deepcopy(model)
        adapt_opt = torch.optim.Adam(
            learner.parameters(),
            lr=fast_lr,
            betas=(0, 0.999)
        )
        try:
            batch_adapt = next(loader_adapt)
        except StopIteration:
            loader_adapt = iter(adapt_loader)
            batch_adapt = next(loader_adapt)
        adapt_opt.load_state_dict(adapt_opt_state)
        fast_adapt(batch_adapt, learner, adapt_opt, loss, test_steps, test_shots, test_inner_bsz, device,
                   return_preds=True)
        for i in range(len(test_loader)):
            batch = next(loader_test)
            image, label, image_names = batch["taskimage"], batch["taskmask"], batch["images"]
            data, labels = torch.squeeze(image, axis=0), torch.squeeze(label, axis=0)
            data, labels = data.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            tot_miou = 0
            predicts = []
            for (d, l, n) in zip(data, labels, image_names):
                d, l = torch.unsqueeze(d, axis=0), torch.unsqueeze(l, axis=0)
                pred = learner(d)
                s_mask = torch.squeeze(pred, axis=0).detach().cpu()
                if post:
                    s_mask[s_mask < 0.5] = 0
                    s_mask[s_mask >= 0.5] = 1
                    s_mask = postprocess(np.array(s_mask).astype(np.uint8))
                    predicts.append(s_mask)
                    s_mask = np.expand_dims(s_mask, axis=0)
                    s_mask = torch.tensor(s_mask).to(device)
                    miou = metrics.compute_iou(s_mask, l)
                    tot_miou += miou
                else:
                    predicts.append(s_mask)
                    miou = metrics.compute_iou(pred, l)
                    tot_miou += miou
                print(n[0], ':', miou)
            miou = tot_miou / int(data.size(0))
            ims = data
            predicts = np.stack(predicts, axis=0)
            save_images(ims, labels, torch.tensor(predicts), mode=loss_type, extra=iterate, sep=True, over=args.over)
        print('Meta Test Accuracy', miou.item() / len(test_loader))
    return miou.item()/len(test_loader)


if __name__ == '__main__':
    args = get_args()
    miou = main()
    print('Meta Test Accuracy', miou)
