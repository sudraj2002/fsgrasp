from contact_points import *

grasp_dir = r"/home/sudarshan/Downloads/graspnet"

g = loadData(camera='kinect', grasp_dir=grasp_dir)
scene_ids = g.sceneIds
ann_ids = g.annId[:256]
camera_type = "kinect"
im = g.loadRGB(0, camera_type, 0)
size = im.shape
start = 0

for scene_id in scene_ids:
    if scene_id < start:
        continue
    if scene_id < 100:
        add1 = r"scene_00%02d/" % (scene_id)
    else:
        add1 = r"scene_0%02d/" % (scene_id)
    add2 = r"%s/" % (camera_type)
    create_class(g, scene_id, add1, add2)
    i = len(loadGraspGroup(g, scene_id, 0, camera_type))
    print(i)
    for ann_id in ann_ids[::20]:
        contact_points = plot_gripper_mine(g, i, ann_id, scene_id)
        plot_points(g, contact_points, ann_id, scene_id, size, add1, add2)
