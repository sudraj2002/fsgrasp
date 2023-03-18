import os
import imageio
import numpy as np

main_dir = r'/home/sudarshan/Downloads/masks/'
"""for mask in os.listdir(main_dir):
    img = imageio.imread(main_dir + mask).transpose(2, 0, 1)
    img = img[0]
    img[img != 255] = 0
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j] == 255:
                img[i][j] = 0
            else:
                img[i][j] = 255
    imageio.imsave('/home/sudarshan/Downloads/masks/new' + mask, img)"""
for mask in os.listdir(main_dir):
    img = imageio.imread(main_dir + mask).transpose(2, 0, 1)
    img = img[0]
    imageio.imsave('/home/sudarshan/Downloads/masks/new' + mask, img)