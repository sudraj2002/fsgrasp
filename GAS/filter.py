import os
import shutil
import imageio
import numpy as np
import natsort
from contact_points import *

mask_dir = r"/home/sudarshan/Downloads/grasp_masks_new2/"
grasp_dir = r"/home/sudarshan/Downloads/graspnet"
camera_type = 'kinect'
g = loadData(camera='kinect', grasp_dir=grasp_dir)
scene_ids = g.sceneIds
for scene_id in scene_ids:
    if scene_id < 100:
        add1 = r"scene_00%02d/" % (scene_id)
    else:
        add1 = r"scene_0%02d/" % (scene_id)
    add2 = r"%s/" % (camera_type)
    task_dir = mask_dir + add1 + add2
    for object in natsort.natsorted(os.listdir(task_dir)):
        object_dir = task_dir + '/' + object
        fr = 0
        for img in natsort.natsorted(os.listdir(object_dir)):
            image = imageio.imread(object_dir + '/' + img)
            high = (image == 255).sum()
            low = (image == 0).sum()
            frac = high / (low + high)
            fr = frac
            if frac < 0.002:
                shutil.rmtree(object_dir)
                break
