from graspnetAPI import GraspNet, Grasp, GraspGroup
import open3d as o3d
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os
import glob
from scipy.spatial.transform import Rotation as Rot
from zipfile import ZipFile
import torch
from torchvision import datasets, transforms
from PIL import Image
import random
import pickle
import imageio
from scipy import ndimage

matplotlib.use("TkAgg")

grasp_dir = r"/home/sudarshan/Downloads/graspnet"
save_dir = r"/home/sudarshan/Downloads/grasp_cp/"


def loadData(grasp_dir, camera='kinect'):
    ### root path for graspnet
    graspnet_root = grasp_dir
    ### initialize a GraspNet instance
    g = GraspNet(graspnet_root, camera='kinect', split='all')
    return g


def loadGraspGroup(g, scene_id, ann_id, camera_type='kinect'):
    ### load a grasp group from Graspnet
    grasp_group = g.loadGrasp(sceneId=scene_id, annId=ann_id, format='6d', camera=camera_type, fric_coef_thresh=0.1)
    ### pick fric_coef_thresh as 0.1 to only select grasps with score 1
    return grasp_group


def transform_to_camera_frame(grasps_mat, points):
    points = np.append(points, np.ones([1]), axis=0)
    points = grasps_mat @ points
    return points[:3]


def transform_to_2d(cam_intr, point):
    point = cam_intr @ point
    point = point / point[2]
    return point


def transform_points(grasps_rt, cam_intr, height, width, depth, finger_width=0.004, tail_length=0.04, depth_base=0.02):
    ### coordinate of the 5 points under gripper frame
    a = np.zeros((5, 3))
    # top left
    a[0] = np.array((depth, -(0.5 * width + 0.5 * finger_width), 0))
    # top right
    a[1] = np.array((depth, (0.5 * width + 0.5 * finger_width), 0))
    # bottom left
    a[2] = np.array((-(depth_base + 0.5 * finger_width), -(0.5 * width + 0.5 * finger_width), 0))
    # bottom right
    a[3] = np.array((-(depth_base + 0.5 * finger_width), (0.5 * width + 0.5 * finger_width), 0))
    # tail
    a[4] = np.array((-(depth_base + finger_width + tail_length), 0, 0))

    ### add noise(independent noise for each axis-rotation)
    ### set rotation around x-/y- axis to zero
    ###transform to camera frame
    for i in range(a.shape[0]):
        a[i] = transform_to_camera_frame(grasps_rt, a[i])

    ###transform to pixel
    for i in range(a.shape[0]):
        a[i] = transform_to_2d(cam_intr, a[i])
    a = np.delete(a, 2, axis=1)
    return a


def draw_2d_grasps(points, img, add1, add2, ann_id, i):
    plt.imshow(img)
    plt.plot([round(points[0, 0]), round(points[2, 0])], [round(points[0, 1]), round(points[2, 1])], color="white",
             linewidth=1)
    plt.plot([round(points[1, 0]), round(points[3, 0])], [round(points[1, 1]), round(points[3, 1])], color="white",
             linewidth=1)
    plt.plot([round(points[2, 0]), round(points[3, 0])], [round(points[2, 1]), round(points[3, 1])], color="white",
             linewidth=1)
    plt.plot([round((points[2, 0] + points[3, 0]) / 2), round(points[4, 0])],
             [round((points[2, 1] + points[3, 1]) / 2), round(points[4, 1])], color="white", linewidth=1)
    if not os.path.exists(save_dir + add1 + add2):
        os.makedirs(save_dir + add1 + add2)
    plt.savefig(save_dir + add1 + add2 + str(ann_id) + "grasp_id_" + str(i) + ".png")
    plt.close()


def plot_gripper_mine(g, i, ann_id, scene_id, camera_type="kinect"):
    gg = loadGraspGroup(g, scene_id, ann_id, camera_type)
    cps = []
    for j in range(i):
        gr = gg[j]
        ### grasp matrix
        grasps_rt = np.append(gr.rotation_matrix, np.zeros([1, 3]), axis=0)
        translation = np.append(gr.translation, np.ones([1]), axis=0)
        grasps_rt = np.append(grasps_rt, np.reshape(translation, (4, 1)), axis=1)
        ###load camera intrinsic
        add1 = r"/home/sudarshan/Downloads/graspnet/scenes/"
        if scene_id < 100:
            add2 = r"scene_00%02d/" % (scene_id)
        else:
            add2 = r"scene_0%02d/" % (scene_id)
        add3 = r"%s/" % (camera_type)
        add4 = r"camK.npy"
        address = add1 + add2 + add3 + add4
        cam_intr = np.load(address)
        ###transform points to 2d
        a = transform_points(grasps_rt, cam_intr, gr.height, gr.width, gr.depth, finger_width=0.004, tail_length=0.04,
                             depth_base=0.02)
        cps.append((a[0], a[1]))
    del gg
    return cps


def plot_points(g, contact_points, ann_id, scene_id, size, add1, add2, camera_type="kinect"):
    ###draw on corresponding image
    img = np.zeros(size)
    t = 1
    for cp in contact_points:
        p1 = cp[0]
        p2 = cp[1]
        p = (p1 + p2) / 2
        p = np.round(p)
        if p[0] < size[1] and p[1] < size[0]:
            img[round(p[1])][round(p[0])][:] = 255
    cv2.imwrite(save_dir + add1 + add2 + str(ann_id) + "pt.png", img.astype(np.uint8))
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    kernel_erode = np.ones((7, 7), np.uint8)
    kernel_dilate = np.ones((15, 15), np.uint8)
    img = cv2.erode(img, kernel_erode, iterations=1)
    img = cv2.dilate(img, kernel_dilate, iterations=2)
    img[img > t] = 255
    img[img <= t] = 0
    img = img[:, :, 0]

    if not os.path.exists(save_dir + add1 + add2):
        os.makedirs(save_dir + add1 + add2)
    # cv2.imwrite(save_dir + add1 + add2 + str(ann_id) + ".png", img.astype(np.uint8))
    smoothen(g, scene_id, ann_id, img, add1, add2)


def draw_2d_points(contact_points, img, add1, add2, ann_id):
    plt.imshow(img, aspect='auto')
    plt.axis('off')
    for cp in contact_points:
        p1 = cp[0]
        p2 = cp[1]
        p = (p1 + p2) / 2
        plt.plot(p[0], p[1], marker='.', color='white')
    if not os.path.exists(save_dir + add1 + add2):
        os.makedirs(save_dir + add1 + add2)
    plt.savefig(save_dir + add1 + add2 + str(ann_id) + "grasp_id_" + ".png", bbox_inches='tight', pad_inches=0)
    plt.savefig(save_dir + add1 + add2 + str(ann_id) + "image" + ".png", bbox_inches='tight', pad_inches=0)
    plt.close()


def smoothen(g, sceneid, ann_id, img, add1, add2, camera='kinect'):
    mask = g.loadMask(sceneid, camera, ann_id)
    mask_ids = np.unique(mask)
    for id in mask_ids:
        if id == 0:
            continue
        masked_img = save_dir + add1 + add2 + str(id) + '/' + str(ann_id) + '_' + str(id) + '.png'
        new_mask = (mask == id).astype(int)
        new_img = np.multiply(img, new_mask).astype(np.uint8)
        t = 100
        new_img[new_img > t] = 255
        new_img[new_img <= t] = 0
        cv2.imwrite(masked_img, new_img)


def create_class(g, sceneid, add1, add2, camera='kinect'):
    mask = g.loadMask(sceneid, camera, 0)
    mask_ids = np.unique(mask)
    for id in mask_ids:
        if not os.path.exists(save_dir + add1 + add2 + str(id)):
            os.makedirs(save_dir + add1 + add2 + str(id))
