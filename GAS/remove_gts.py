import shutil
import os
import natsort
import numpy as np

src_dir = r'/home/sudarshan/Downloads/grasp_rgbs/'
scene_ids = np.arange(190)
for scene_id in scene_ids:
    if scene_id < 100:
        add1 = r"scene_00%02d/" % (scene_id)
    else:
        add1 = r"scene_0%02d/" % (scene_id)
    scene_dir = src_dir + '/' + add1
    for name in os.listdir(scene_dir):
        if int(name.split('.')[0]) % 10 != 0:
            os.remove(os.path.join(scene_dir, name))