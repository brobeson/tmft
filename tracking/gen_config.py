import os
import json
import numpy as np


def gen_config(args):

    if args.seq != "":
        # generate config from a sequence name

        seq_home = os.path.expanduser("~/Videos/otb/")
        result_home = "results"

        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name, "img")
        gt_path = os.path.join(seq_home, seq_name, "groundtruth_rect.txt")
        if seq_name == "David":
            Initial_frame = 300
        elif seq_name == "Tiger1":
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

        if seq_name == "David":
            init_bbox = gt[0]
        else:
            gt = gt[Initial_frame - 1 :, :]
            init_bbox = gt[0]

        ############################################
        if gt.shape[1] == 8:
            x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
            init_bbox = gt[0]

        #############################################

        result_dir = os.path.join(result_home, "AD-MDNet")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        savefig_dir = os.path.join(result_dir, "figs")
        result_path = os.path.join(result_dir, (seq_name + "_ALVIDMDNet" + ".json"))
        ### vot test
        vot_result_dir = os.path.join(
            result_dir, "baseline", seq_name[0].lower() + seq_name[1:]
        )
        if not os.path.exists(vot_result_dir):
            os.makedirs(vot_result_dir)
        vot_result_path = os.path.join(
            vot_result_dir, (seq_name[0].lower() + seq_name[1:] + "_001" + ".txt")
        )
        dummy = 1
    elif args.json != "":
        # load config from a json file

        param = json.load(open(args.json, "r"))
        seq_name = param["seq_name"]
        img_list = param["img_list"]
        init_bbox = param["init_bbox"]
        savefig_dir = param["savefig_dir"]
        result_path = param["result_path"]
        vot_result_path = ""
        gt = None

    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ""

    return (
        img_list,
        init_bbox,
        gt,
        savefig_dir,
        args.display,
        result_path,
        vot_result_path,
        seq_name,
    )
