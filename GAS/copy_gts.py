import shutil
import numpy as np

src_dir = r'/home/sudarshan/Downloads/graspnet/scenes/'
scene_ids = np.arange(190)
add2 = r'kinect/depth'

dest_dir = r'/home/sudarshan/Downloads/grasp_depths/'

for scene_id in scene_ids:
    if scene_id < 100:
        add1 = r"scene_00%02d/" % (scene_id)
    else:
        add1 = r"scene_0%02d/" % (scene_id)
    shutil.copytree(src_dir + add1 + add2, dest_dir + add1)