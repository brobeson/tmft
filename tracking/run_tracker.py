import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.cuda
import torch.utils.data as data
import torch.optim as optim
import math

sys.path.insert(0, ".")
from modules.model import (
    MDNet,
    BCELoss,
    DomainAdaptationNetwork,
    make_optimizer,
)
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
import networks.domain_adaptation_schedules
from tracking.data_prov import RegionExtractor
from tracking.bbreg import BBRegressor
from tracking.gen_config import gen_config
import modules.training


if os.path.exists("tracking/options.yaml"):
    with open("tracking/options.yaml") as yaml_file:
        opts = yaml.safe_load(yaml_file)


def forward_samples(model, image, samples, out_layer="conv3"):
    model.eval()
    extractor = RegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts["use_gpu"]:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats


def train(
    model,
    model_Adnet,
    Grd_reveres_layer,
    criterion,
    criterion_Adnet,
    optimizer,
    pos_feats,
    neg_feats,
    maxiter,
    in_layer="fc4",
):
    batch_pos = opts["batch_pos"]
    batch_neg = opts["batch_neg"]
    batch_test = opts["batch_test"]
    # properties = torch.cuda.get_device_properties(torch.cuda.device)
    # tensor_size = properties.total_memory / batch_test
    batch_neg_cand = max(opts["batch_neg_cand"], batch_neg)
    if model_Adnet is not None:
        # batch_pos =  int(opts['batch_pos']/2)
        NumPosData = pos_feats.size(0)
        NumSrcPosData = int(NumPosData / 2)
        pos_featsaAll = pos_feats
        pos_feats = pos_featsaAll[0:NumSrcPosData, :]
        pos_Tar_feats = pos_featsaAll[NumSrcPosData:NumPosData, :]

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while len(pos_idx) < batch_pos * maxiter:
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while len(neg_idx) < batch_neg_cand * maxiter:
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0
    model.train()
    if model_Adnet is not None:
        model_Adnet.train()
        networks.domain_adaptation_schedules.GradientReverseLayer.schedule.reset()

    loss1total = []
    loss2total = []

    for i in range(maxiter):
        optimizer.zero_grad()
        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat(
                        (neg_cand_score, score.detach()[:, 1].clone()), 0
                    )

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        loss2 = 0
        # loss_g_2 = 0
        if model_Adnet is not None:
            batch_Tar_pos_feats = pos_Tar_feats[pos_cur_idx]

            pos_Att_src = model_Adnet(batch_pos_feats)
            pos_Att_tar = model_Adnet(batch_Tar_pos_feats)

            # mask_src = -(pos_Att_src * torch.log(pos_Att_src + 1e-10) + (1 - pos_Att_src) * torch.log(1 - pos_Att_src + 1e-10))
            # mask_src = 2 - mask_src

            # mask_tar = -(pos_Att_tar * torch.log(pos_Att_tar + 1e-10) + (1 - pos_Att_tar) * torch.log(1 - pos_Att_tar + 1e-10))
            # mask_tar = 2 - mask_tar

            mask_src = pos_Att_src
            mask_tar = pos_Att_tar
            ############################################################################
            batch_Tar_pos_feats = batch_Tar_pos_feats * mask_tar.view(
                -1, 1, 3, 3
            ).repeat(1, 512, 1, 1).view(batch_Tar_pos_feats.shape[0], -1)
            batch_pos_feats = batch_pos_feats * mask_src.view(-1, 1, 3, 3).repeat(
                1, 512, 1, 1
            ).view(batch_pos_feats.shape[0], -1)
            ############################################################################

            # criterion_g = torch.nn.MSELoss(reduction='mean')
            # loss_g_2 = criterion_g(pos_Att_src.float(), pos_Att_tar.cuda().float())
            results = torch.cat((pos_Att_src, pos_Att_tar), 0)
            loss2 = criterion_Adnet(
                results, torch.ones_like(results, device=results.device)
            )
            loss2 = loss2 * opts["loss_factor"]
            loss2total.append(loss2.data)
            # batch_pos_feats = batch_pos_feats * mask_asdn_src.cuda()
            batch_pos_feats = torch.cat((batch_pos_feats, batch_Tar_pos_feats), 0)
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)
        # optimize
        loss1 = criterion(pos_score, neg_score)
        loss = loss1 + loss2  # + loss_g_2 * opts['loss_factor2']
        loss1total.append(loss1.data)

        loss.backward()
        if "grad_clip" in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts["grad_clip"])
            if model_Adnet is not None:
                # BUG grad_clip2 may not be in the options.
                torch.nn.utils.clip_grad_norm_(
                    model_Adnet.parameters(), opts["grad_clip2"]
                )
        optimizer.step()
        networks.domain_adaptation_schedules.GradientReverseLayer.schedule.step()

    loss1total = torch.stack(loss1total)
    loss1total = loss1total.cpu().numpy()

    if model_Adnet is not None:
        loss2total = torch.stack(loss2total)
        loss2total = loss2total.cpu().numpy()
        # plt.figure()
        # plt.plot(loss2total)
        # plt.figure()
        # plt.plot(loss1total)
        # plt.show()


def run_mdnet(
    img_list, init_bbox, gt=None, savefig_dir="", display=False, sequence_name=None
):
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1

    # Init model
    model = MDNet(opts["model_path"])
    model_Adnet = DomainAdaptationNetwork(
        opts["grl"], opts["grl_direction"], **opts[opts["grl"]]
    )
    # Grd_reverse_layer = networks.domain_adaptation_schedules.PadaScheduler((0.0, 1.0))
    if opts["use_gpu"]:
        model = model.cuda()
        model_Adnet = model_Adnet.cuda()
    # Init criterion and optimizer
    model_Adnet.train(True)
    criterion = BCELoss()
    # criterion_Adnet = DomainLoss()
    criterion_Adnet = torch.nn.BCELoss(reduction="sum")
    model.set_learnable_params(opts["ft_layers"])
    init_optimizer = make_optimizer(
        model, model_Adnet, opts["lr_init"], opts["lr_mult"]
    )
    update_optimizer = make_optimizer(
        model, model_Adnet, opts["lr_update"], opts["lr_mult"]
    )
    # init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    # update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert("RGB")

    # Draw pos/neg samples
    pos_examples = SampleGenerator(
        "gaussian", image.size, opts["trans_pos"], opts["scale_pos"]
    )(target_bbox, opts["n_pos_init"], opts["overlap_pos_init"])

    neg_examples = np.concatenate(
        [
            SampleGenerator(
                "uniform", image.size, opts["trans_neg_init"], opts["scale_neg_init"]
            )(target_bbox, int(opts["n_neg_init"] * 0.5), opts["overlap_neg_init"]),
            SampleGenerator("whole", image.size)(
                target_bbox, int(opts["n_neg_init"] * 0.5), opts["overlap_neg_init"]
            ),
        ]
    )
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)

    # Initial training
    train(
        model,
        None,
        # Grd_reverse_layer,
        None,
        criterion,
        criterion_Adnet,
        init_optimizer,
        pos_feats,
        neg_feats,
        opts["maxiter_init"],
    )
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()

    # Train bbox regressor
    bbreg_examples = SampleGenerator(
        "uniform",
        image.size,
        opts["trans_bbreg"],
        opts["scale_bbreg"],
        opts["aspect_bbreg"],
    )(target_bbox, opts["n_bbreg"], opts["overlap_bbreg"])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator(
        "gaussian", image.size, opts["trans"], opts["scale"]
    )
    pos_generator = SampleGenerator(
        "gaussian", image.size, opts["trans_pos"], opts["scale_pos"]
    )
    neg_generator = SampleGenerator(
        "uniform", image.size, opts["trans_neg"], opts["scale_neg"]
    )

    # Init pos/neg features for update
    neg_examples = neg_generator(
        target_bbox, opts["n_neg_update"], opts["overlap_neg_init"]
    )
    neg_feats = forward_samples(model, image, neg_examples)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ""
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect="auto")

        if gt is not None:
            gt_rect = plt.Rectangle(
                tuple(gt[0, :2]),
                gt[0, 2],
                gt[0, 3],
                linewidth=3,
                edgecolor="#00ff00",
                zorder=1,
                fill=False,
            )
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(
            tuple(result_bb[0, :2]),
            result_bb[0, 2],
            result_bb[0, 3],
            linewidth=3,
            edgecolor="#ff0000",
            zorder=1,
            fill=False,
        )
        ax.add_patch(rect)

        if display:
            plt.pause(0.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, "0000.jpg"), dpi=dpi)

    # Main loop
    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert("RGB")

        # Estimate target bbox
        samples = sample_generator(target_bbox, opts["n_samples"])
        sample_scores = forward_samples(model, image, samples, out_layer="fc6")

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts["trans"])
        else:
            sample_generator.expand_trans(opts["trans_limit"])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(
                target_bbox, opts["n_pos_update"], opts["overlap_pos_update"]
            )
            pos_feats = forward_samples(model, image, pos_examples)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts["n_frames_long"]:
                del pos_feats_all[0]

            neg_examples = neg_generator(
                target_bbox, opts["n_neg_update"], opts["overlap_neg_update"]
            )
            neg_feats = forward_samples(model, image, neg_examples)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts["n_frames_short"]:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts["n_frames_short"], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(
                model,
                None,
                # Grd_reverse_layer,
                None,
                criterion,
                criterion_Adnet,
                update_optimizer,
                pos_data,
                neg_data,
                opts["maxiter_update2"],
            )

        # Long term update
        elif i % opts["long_interval"] == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(
                model,
                model_Adnet,
                # Grd_reverse_layer,
                None,
                criterion,
                criterion_Adnet,
                update_optimizer,
                pos_data,
                neg_data,
                opts["maxiter_update"],
            )

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(0.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, "{:04d}.jpg".format(i)), dpi=dpi)

        if gt is None:
            pass
            print(
                "Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}".format(
                    i, len(img_list), target_score, spf
                )
            )
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print(
                "Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}".format(
                    i, len(img_list), overlap[i], target_score, spf
                )
            )

    if gt is not None:
        print("meanIOU: {:.3f}".format(overlap.mean()))
    fps = len(img_list) / spf_total
    plt.close()
    return result, result_bb, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seq", default="", help="input seq")
    parser.add_argument("-j", "--json", default="", help="input json")
    parser.add_argument("-f", "--savefig", default=False, action="store_true")
    parser.add_argument("-d", "--display", default=False, action="store_true")

    args = parser.parse_args()
    assert args.seq != "" or args.json != ""

    if "random_seed" in opts:
        np.random.seed(opts["random_seed"])
        torch.manual_seed(opts["random_seed"])
    # Generate sequence config
    (
        img_list,
        init_bbox,
        gt,
        savefig_dir,
        display,
        result_path,
        votresultpath,
        seq_name,
    ) = gen_config(args)
    print(seq_name)

    # Run tracker
    result, result_bb, fps = run_mdnet(
        img_list,
        init_bbox,
        gt=gt,
        savefig_dir=savefig_dir,
        display=display,
        sequence_name=seq_name,
    )

    # Save result
    res = {}
    res["res"] = result_bb.round().tolist()
    res["type"] = "rect"
    res["fps"] = fps
    json.dump(res, open(result_path, "w"), indent=2)

    # temp = result_bb.round().tolist()
    # temp[0] = [1]
    # with open(votresultpath, "w") as filehandle:
    #     for listitem in temp:
    #         filehandle.write("%s\n" % str(listitem)[1:-1])
