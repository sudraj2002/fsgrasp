from contact_points import *
import imageio
import os
import natsort
import shutil
import random

grasp_dir = r"/home/sudarshan/Downloads/graspnet/"
g = loadData(camera='kinect', grasp_dir=grasp_dir)
scene_ids = g.sceneIds
grasp_dir = r"/home/sudarshan/Downloads/graspnet/scenes/"
grasp2_dir = r"/home/sudarshan/Downloads/grasp_masks_new2/"
new_dir = r"/home/sudarshan/Downloads/objs/"
merge_dir = r"/home/sudarshan/Downloads/objs_merged/"
task_dir = r"/home/sudarshan/Downloads/fssd/"
rename_dir = r"/home/sudarshan/Downloads/objs_renamed/"


def form_dict(sceneids):
    objs = {}
    for scene in sceneids:
        if scene < 100:
            add1 = r"scene_00%02d/" % (scene)
        else:
            add1 = r"scene_0%02d/" % (scene)
        main_dir = grasp_dir + add1 + 'object_id_list.txt'
        f = open(main_dir, "r")
        x = f.readlines()
        for obj_id in x:
            obj_id = obj_id.rstrip()
            if obj_id in objs:
                objs[obj_id].append(scene)
            else:
                objs[obj_id] = [scene]
        f.close()
    return sorted(objs.items())


def count_unique(sceneids):
    obj = []
    for scene in sceneids:
        if scene < 100:
            add1 = r"scene_00%02d/" % (scene)
        else:
            add1 = r"scene_0%02d/" % (scene)
        main_dir = grasp2_dir + add1
        for x in os.listdir(main_dir):
            for file in os.listdir(main_dir + '/' + x):
                obj.append(file)
    print(np.unique(obj))


def copy_folders(sceneids):
    objs = np.arange(100)
    for obj in objs:
        count = 0
        if not os.path.exists(new_dir + str(obj)):
            os.mkdir(new_dir + str(obj))
        for scene in sceneids:
            if scene < 100:
                add1 = r"scene_00%02d/" % (scene)
            else:
                add1 = r"scene_0%02d/" % (scene)
            main_dir = grasp2_dir + add1
            for x in os.listdir(main_dir):
                for obj_file in os.listdir(main_dir + x):
                    if int(obj) == int(obj_file):
                        shutil.move(main_dir + x + '/' + obj_file, new_dir + str(obj) + '/' + str(count))
                        count += 1


def rename():
    objs = np.arange(89)
    for obj in objs:
        count_mask = 0
        count_img = 0
        if not os.path.exists(rename_dir + str(obj)):
            os.mkdir(rename_dir + str(obj))
        for folder in natsort.natsorted(os.listdir(new_dir + str(obj))):
            folder_dir = new_dir + str(obj) + '/' + folder
            rename_folder = rename_dir + str(obj) + '/' + folder
            if not os.path.exists(rename_folder):
                os.mkdir(rename_folder)
            for name in natsort.natsorted(os.listdir(folder_dir)):
                if "_" in name.split(".")[0]:
                    # x = name.split('.')[0]
                    # mask_val = int(x.split('_')[0])
                    shutil.copy(folder_dir + '/' + name, rename_folder + '/' + str(count_mask) + '_.png')
                    count_mask += 1
                else:
                    # img_val = int(name.split('.')[0])
                    shutil.copy(folder_dir + '/' + name, rename_folder + '/' + str(count_img) + '.png')
                    count_img += 1


def merge_folders():
    objs = np.arange(89)
    for obj in objs:
        if not os.path.exists(merge_dir + str(obj)):
            os.mkdir(merge_dir + str(obj))
        for folder in natsort.natsorted(os.listdir(rename_dir + str(obj))):
            folder_dir = rename_dir + str(obj) + '/' + folder
            for name in natsort.natsorted(os.listdir(folder_dir)):
                shutil.move(folder_dir + '/' + name, merge_dir + str(obj) + '/' + name)

def make_tasks():
    task_num = 0
    shots = 10
    objs = np.arange(89)
    for obj in objs:
        print(obj)
        image_list = []
        mask_list = []
        for name in natsort.natsorted(os.listdir(merge_dir + str(obj))):
            if '_' in name:
                mask_list.append(name)
            else:
                image_list.append(name)
        num_images = len(image_list)
        while num_images > 0:
            if num_images < 10:
                break
            if not os.path.exists(task_dir + str(task_num)):
                os.mkdir(task_dir + str(task_num))
                task_num += 1
            idxs = np.random.choice(num_images, shots, replace = False)
            image_list = np.array(image_list)
            mask_list = np.array(mask_list)
            selected_imgs = image_list[idxs]
            selected_masks = mask_list[idxs]
            for (image, mask) in zip(selected_imgs, selected_masks):
                shutil.move(merge_dir + str(obj) + '/' + image, task_dir + str(task_num - 1) + '/' + image)
                shutil.move(merge_dir + str(obj) + '/' + mask, task_dir + str(task_num - 1) + '/' + mask)
            image_list = []
            mask_list = []
            for name in natsort.natsorted(os.listdir(merge_dir + str(obj))):
                if '_' in name:
                    mask_list.append(name)
                else:
                    image_list.append(name)
            num_images = len(image_list)
            num_masks = len(mask_list)


def full_match(n_dir, obj):
    for task in os.listdir(n_dir):
        task_dir = n_dir + '/' + task
        img_list = []
        mask_list = []
        for name in natsort.natsorted(os.listdir(task_dir)):
            if "_" in name.split(".")[0]:
                x = name.split('.')[0]
                mask_val = int(x.split('_')[0])
                mask_list.append(mask_val)
            else:
                img_val = int(name.split('.')[0])
                img_list.append(img_val)

        check = mask_list == img_list
        if len(mask_list) != len(img_list):
            print(task, obj)
        elif not check:
            print(task, obj)
for obj in os.listdir(new_dir):
    d = new_dir + obj
    full_match(d, obj)




