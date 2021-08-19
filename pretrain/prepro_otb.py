import os
import numpy as np
import pickle
from collections import OrderedDict

seq_home = "/sshd/mamrez/Projects/Dataset"
seqlist_path = "datasets/list/otb-vot15.txt"
output_path = "pretrain/data/otb-vot15.pkl"

with open(seqlist_path, "r") as fp:
    seq_list = fp.read().splitlines()

# Construct db
data = OrderedDict()
for i, seq in enumerate(seq_list):
    # img_list = sorted([p for p in os.listdir(os.path.join(seq_home, seq, 'img')) if os.path.splitext(p)[1] == '.jpg'])
    # gt_path= os.path.join(seq_home, seq, 'groundtruth_rect.txt')

    img_dir = os.path.join(seq_home, seq, "img")
    gt_path = os.path.join(seq_home, seq, "groundtruth_rect.txt")

    if seq == "David":
        Initial_frame = 300
    elif seq == "Tiger1":
        Initial_frame = 6
    else:
        Initial_frame = 1

    img_list = os.listdir(img_dir)
    img_list.sort()
    img_list = [
        os.path.join(img_dir, x)
        for i, x in enumerate(img_list)
        if i >= Initial_frame - 1
    ]

    with open(gt_path, "r") as myfile:
        temp = myfile.read().replace("\n", "")
    if "," in temp:
        gt = np.loadtxt(gt_path, delimiter=",")
    else:
        gt = np.loadtxt(gt_path)

    if seq == "David":
        init_bbox = gt[0]
    else:
        gt = gt[Initial_frame - 1 :, :]
        init_bbox = gt[0]

    # gt = np.loadtxt(os.path.join(seq_home, seq, 'groundtruth_rect.txt'), delimiter=',')

    assert len(img_list) == len(gt), "Lengths do not match!!"

    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    # img_list = [os.path.join(seq_home, seq, 'img', img) for img in img_list]
    data[seq] = {"images": img_list, "gt": gt}

# Save db
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)
with open(output_path, "wb") as fp:
    pickle.dump(data, fp)
