import matplotlib.pyplot as plt
import glob
import hashlib
import os
import re
import time
import warnings
from typing import List, Optional, Tuple, Dict, Union
import numpy as np
import random
import imageio
from skimage.measure import label, regionprops, regionprops_table
from skimage.io import imread, imshow
import cv2
from PIL import Image


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def count_examples_in_tfrecords(data_dir, path):
    c = 0
    for i in os.listdir(os.path.join(data_dir, path)):
        c += 1
    return c


def split_train_test_tasks(all_tasks: List[str], n_test, reproducbile_splits: bool = False):
    if not isinstance(all_tasks, list):
        all_tasks = list(all_tasks)
    if reproducbile_splits:
        all_tasks = sorted(all_tasks)
    else:
        random.shuffle(all_tasks)
    test_set = []
    for i in range(n_test):
        test_set.append(all_tasks.pop())
    # assert_train_test_split(all_tasks, test_set)
    return all_tasks, test_set


def get_fss_tasks(data_dir):
    return glob.glob(os.path.join(data_dir, "*.png*"))


def merge(im1, im2, data_dir):
    # Merges 2 images of the same size into a grid for comparison
    h, w = im1.shape
    grid = np.zeros((h * 2, w))
    grid[0:h, 0:w] = im1
    grid[h:, 0:w] = im2
    imageio.imsave(data_dir, grid.astype(np.uint8))


def save_images(ims, labels, predicts, mode, extra=0, st=False):
    if not st:
        if mode == 'Dice_CE' or mode == 'CE':
            for i in range(ims.size(0)):
                im = (ims[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                lab = (labels[i].detach().cpu().numpy() * 255).astype(np.uint8)
                predict = (predicts[i].argmax(dim=0).cpu().numpy() * 255).astype(np.uint8)
                dire_images = "outputs_images/" + str(extra) + "image" + str(i) + ".png"
                dire_masks = "outputs_masks/" + str(extra) + "preds" + str(i) + ".png"
                imageio.imsave(dire_images, im)
                merge(lab, predict, dire_masks)

        elif mode == 'Dice_BCE' or mode == 'BCE' or mode == 'Dice':
            for i in range(ims.size(0)):
                im = (ims[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                lab = (labels[i].detach().cpu().numpy() * 255).astype(np.uint8)
                predicts[predicts > 0.5] = 1
                predicts[predicts <= 0.5] = 0
                predict = (predicts[i][0].detach().cpu().numpy() * 255).astype(np.uint8)
                dire_images = "outputs_images/" + str(extra) + "image" + str(i) + ".png"
                dire_masks = "outputs_masks/" + str(extra) + "preds" + str(i) + ".png"
                imageio.imsave(dire_images, im)
                imageio.imsave(dire_masks, lab)
                # merge(lab, predict, dire_masks)
    else:
        if mode == 'Dice_CE' or mode == 'CE':
            for i in range(ims.size(0)):
                im = (ims[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                lab = (labels[i].detach().cpu().numpy() * 255).astype(np.uint8)
                predict = (predicts[i].argmax(dim=0).cpu().numpy() * 255).astype(np.uint8)
                dire_images = "train_images/" + str(extra) + "image" + str(i) + ".png"
                dire_masks = "train_masks/" + str(extra) + "preds" + str(i) + ".png"
                imageio.imsave(dire_images, im)
                merge(lab, predict, dire_masks)

        elif mode == 'Dice_BCE' or mode == 'BCE' or mode == 'Dice':
            for i in range(ims.size(0)):
                im = (ims[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                lab = (labels[i].detach().cpu().numpy() * 255).astype(np.uint8)
                predicts[predicts > 0.5] = 1
                predicts[predicts <= 0.5] = 0
                predict = (predicts[i][0].detach().cpu().numpy() * 255).astype(np.uint8)
                dire_images = "train_images/" + str(extra) + "image" + str(i) + ".png"
                dire_masks = "train_masks/" + str(extra) + "preds" + str(i) + ".png"
                imageio.imsave(dire_images, im)
                imageio.imsave(dire_masks, lab)
                # merge(lab, predict, dire_masks)


def postprocess(mask):
    mask = mask[0] * 255
    label_im = label(mask)
    regions = regionprops(label_im)
    masks = []
    bbox = []
    list_of_index = []
    max_area = 0
    for num, x in enumerate(regions):
        area = x.area
        if area > max_area:
            max_area = area
    for num, x in enumerate(regions):
        area = x.area
        if area > 0.5 * max_area:
            masks.append(regions[num].image)
            bbox.append(regions[num].bbox)
            list_of_index.append(num)
    new_mask = np.zeros_like(label_im)
    for x in list_of_index:
        new_mask += (label_im == x + 1).astype(np.uint8)
    result = mask * new_mask
    result[result > 100] = 255
    result[result <= 100] = 0

    return np.expand_dims(result, axis=0)


def overlay(img, mask, data_dir):
    # Import and convert the mask from binary to RGB
    mask = Image.fromarray(mask).convert('RGB')
    width, height = mask.size

    # Convert the white color (for blobs) to magenta
    mask_colored = change_color(mask, width, height, (255, 255, 255), (15, 255, 80))
    # Convert the black (for background) to white --> important to make a good overlapping
    mask_colored = change_color(mask_colored, width, height, (0, 0, 0), (255, 255, 255))
    res = cv2.addWeighted(np.array(img), 0.6, np.array(mask_colored), 0.4, 0)
    imageio.imsave(data_dir, res.astype(np.uint8))


def change_color(picture, width, height, ex_color, new_color):
    # Process every pixel
    for x in range(width):
        for y in range(height):
            current_color = picture.getpixel((x, y))
            if current_color == ex_color:
                picture.putpixel((x, y), new_color)
    return picture
