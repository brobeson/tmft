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
from tracking.data_prov import RegionExtractor
from tracking.bbreg import BBRegressor
from tracking.gen_config import gen_config
from got10k.trackers import Tracker

opts = yaml.safe_load(open("tracking/options.yaml", "r"))


def _fix_positive_samples(samples: np.ndarray, n: int, box: np.ndarray) -> np.ndarray:
    """
    Ensure a set of positive samples is valid.

    :param numpy.ndarray samples: The positive samples to fix if invalid.
    :param int n: The number of samples to create if fixing is needed.
    :param numpy.ndarray box: The target bounding box used if fixing is needed.
    :return: The original samples if no fixing is required, or the fixed samples if fixing is
        required.
    :rtype: np.ndarray
    """
    if samples.shape[0] > 0:
        return samples
    return np.tile(box, [n, 1])


class ADMDNet(Tracker):
    def __init__(self, name: str, configuration: dict):
        super().__init__(name=name)
        self.configuration = configuration

    # init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    # update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    def init(self, image, box):
        return self.initialize(image, box)

    def initialize(self, image, box):
        print("Initializing the tracker... ", end="", flush=True)
        with open("tracking/options.yaml", "r") as yaml_file:
            opts = yaml.safe_load(yaml_file)
        opts.update(self.configuration)
        # np.random.seed(0)
        # torch.manual_seed(0)
        self.display = False
        self.model = MDNet(opts["model_path"])

        self.model_Adnet = DomainAdaptationNetwork(
            opts["grl"], opts["grl_direction"], **opts[opts["grl"]]
        )
        if opts["use_gpu"]:
            self.model = self.model.cuda()
            self.model_Adnet = self.model_Adnet.cuda()
        # Init criterion and optimizer
        # model_Adnet.train(True)
        self.criterion = BCELoss()
        self.criterion_Adnet = torch.nn.BCELoss(reduction="sum")
        self.model.set_learnable_params(opts["ft_layers"])
        self.init_optimizer = make_optimizer(
            self.model, self.model_Adnet, opts["lr_init"], opts["lr_mult"]
        )
        self.update_optimizer = make_optimizer(
            self.model, self.model_Adnet, opts["lr_update"], opts["lr_mult"]
        )
        image = image.convert("RGB")
        # image = Image.open(image.filename).convert('RGB')

        # target_bbox = np.array([
        #     box[1] - 1 + (box[3] - 1) / 2,
        #     box[0] - 1 + (box[2] - 1) / 2,
        #     box[3], box[2]], dtype=np.float32)
        target_bbox = box
        self.box = target_bbox

        pos_examples = _fix_positive_samples(
            SampleGenerator(
                "gaussian", image.size, opts["trans_pos"], opts["scale_pos"]
            )(target_bbox, opts["n_pos_init"], opts["overlap_pos_init"]),
            opts["n_pos_init"],
            target_bbox,
        )

        neg_examples = np.concatenate(
            [
                SampleGenerator(
                    "uniform",
                    image.size,
                    opts["trans_neg_init"],
                    opts["scale_neg_init"],
                )(target_bbox, int(opts["n_neg_init"] * 0.5), opts["overlap_neg_init"]),
                SampleGenerator("whole", image.size)(
                    target_bbox, int(opts["n_neg_init"] * 0.5), opts["overlap_neg_init"]
                ),
            ]
        )
        neg_examples = np.random.permutation(neg_examples)

        pos_feats = self.forward_samples(self.model, image, pos_examples)
        neg_feats = self.forward_samples(self.model, image, neg_examples)

        # Initial training
        self.train(
            False, self.init_optimizer, pos_feats, neg_feats, opts["maxiter_init"]
        )
        del self.init_optimizer, neg_feats
        torch.cuda.empty_cache()

        # Train bbox regressor
        bbreg_examples = _fix_positive_samples(
            SampleGenerator(
                "uniform",
                image.size,
                opts["trans_bbreg"],
                opts["scale_bbreg"],
                opts["aspect_bbreg"],
            )(target_bbox, opts["n_bbreg"], opts["overlap_bbreg"]),
            opts["n_bbreg"],
            target_bbox,
        )
        bbreg_feats = self.forward_samples(self.model, image, bbreg_examples)
        self.bbreg = BBRegressor(image.size)
        self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
        del bbreg_feats
        torch.cuda.empty_cache()

        # Init sample generators for update
        self.sample_generator = SampleGenerator(
            "gaussian", image.size, opts["trans"], opts["scale"]
        )
        self.pos_generator = SampleGenerator(
            "gaussian", image.size, opts["trans_pos"], opts["scale_pos"]
        )
        self.neg_generator = SampleGenerator(
            "uniform", image.size, opts["trans_neg"], opts["scale_neg"]
        )

        # Init pos/neg features for update
        neg_examples = self.neg_generator(
            target_bbox, opts["n_neg_update"], opts["overlap_neg_init"]
        )
        neg_feats = self.forward_samples(self.model, image, neg_examples)
        self.pos_feats_all = [pos_feats]
        self.neg_feats_all = [neg_feats]
        self.numiter = 0
        print("done", flush=True)

    def update(self, image):
        self.find_target(image)

    def find_target(self, image):
        self.numiter = self.numiter + 1
        print("\rTracking frame", self.numiter, end="", file=sys.stdout, flush=True)
        image = image.convert("RGB")
        samples = self.sample_generator(self.box, opts["n_samples"])
        sample_scores = self.forward_samples(
            self.model, image, samples, out_layer="fc6"
        )

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        if success:
            self.sample_generator.set_trans(opts["trans"])
        else:
            self.sample_generator.expand_trans(opts["trans_limit"])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = self.forward_samples(self.model, image, bbreg_samples)
            bbreg_samples = self.bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        # result[i] = target_bbox
        # result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = _fix_positive_samples(
                self.pos_generator(
                    target_bbox, opts["n_pos_update"], opts["overlap_pos_update"]
                ),
                opts["n_pos_update"],
                target_bbox,
            )
            pos_feats = self.forward_samples(self.model, image, pos_examples)
            self.pos_feats_all.append(pos_feats)
            if len(self.pos_feats_all) > opts["n_frames_long"]:
                del self.pos_feats_all[0]

            neg_examples = self.neg_generator(
                target_bbox, opts["n_neg_update"], opts["overlap_neg_update"]
            )
            neg_feats = self.forward_samples(self.model, image, neg_examples)
            self.neg_feats_all.append(neg_feats)
            if len(self.neg_feats_all) > opts["n_frames_short"]:
                del self.neg_feats_all[0]

        # Short term update
        if not success:

            nframes = min(opts["n_frames_short"], len(self.pos_feats_all))
            pos_data = torch.cat(self.pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            self.train(
                False,
                self.update_optimizer,
                pos_data,
                neg_data,
                opts["maxiter_update2"],
            )

        # Long term update

        elif self.numiter % opts["long_interval"] == 0:

            pos_data = torch.cat(self.pos_feats_all, 0)
            neg_data = torch.cat(self.neg_feats_all, 0)
            self.train(
                True, self.update_optimizer, pos_data, neg_data, opts["maxiter_update"]
            )

        torch.cuda.empty_cache()
        self.box = bbreg_bbox

        return self.box

    def forward_samples(self, model, image, samples, out_layer="conv3"):

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
        self, LearnPermit, optimizer, pos_feats, neg_feats, maxiter, in_layer="fc4"
    ):
        batch_pos = opts["batch_pos"]
        batch_neg = opts["batch_neg"]
        batch_test = opts["batch_test"]
        batch_neg_cand = max(opts["batch_neg_cand"], batch_neg)
        if LearnPermit:
            # batch_pos =  int(opts['batch_pos']/2)
            NumPosData = pos_feats.size(0)
            NumSrcPosData = int(NumPosData / 2)
            pos_featsaAll = pos_feats
            pos_feats = pos_featsaAll[0:NumSrcPosData, :]
            pos_Tar_feats = pos_featsaAll[NumSrcPosData:NumPosData, :]

        pos_idx = np.random.permutation(pos_feats.size(0))
        neg_idx = np.random.permutation(neg_feats.size(0))
        while len(pos_idx) < batch_pos * maxiter:
            pos_idx = np.concatenate(
                [pos_idx, np.random.permutation(pos_feats.size(0))]
            )
        while len(neg_idx) < batch_neg_cand * maxiter:
            neg_idx = np.concatenate(
                [neg_idx, np.random.permutation(neg_feats.size(0))]
            )
        pos_pointer = 0
        neg_pointer = 0
        self.model.train()
        if LearnPermit:
            self.model_Adnet.train()

        # loss1total = []
        # loss2total = []

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
                self.model.eval()
                for start in range(0, batch_neg_cand, batch_test):
                    end = min(start + batch_test, batch_neg_cand)
                    with torch.no_grad():
                        score = self.model(
                            batch_neg_feats[start:end], in_layer=in_layer
                        )
                    if start == 0:
                        neg_cand_score = score.detach()[:, 1].clone()
                    else:
                        neg_cand_score = torch.cat(
                            (neg_cand_score, score.detach()[:, 1].clone()), 0
                        )

                _, top_idx = neg_cand_score.topk(batch_neg)
                batch_neg_feats = batch_neg_feats[top_idx]
                self.model.train()

            # forward
            loss2 = 0
            # loss_g_2 = 0
            if LearnPermit:
                batch_Tar_pos_feats = pos_Tar_feats[pos_cur_idx]

                pos_Att_src = self.model_Adnet(batch_pos_feats)
                pos_Att_tar = self.model_Adnet(batch_Tar_pos_feats)

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
                loss2 = (
                    self.criterion_Adnet(
                        results, torch.ones_like(results, device=results.device)
                    )
                    * opts["loss_factor"]
                )
                # loss2total.append(loss2.data)
                # batch_pos_feats = batch_pos_feats * mask_asdn_src.cuda()
                batch_pos_feats = torch.cat((batch_pos_feats, batch_Tar_pos_feats), 0)
            pos_score = self.model(batch_pos_feats, in_layer=in_layer)
            neg_score = self.model(batch_neg_feats, in_layer=in_layer)
            # optimize
            loss1 = self.criterion(pos_score, neg_score)
            loss = loss1 + loss2  # + loss_g_2 * opts['loss_factor2']
            # loss1total.append(loss1.data)

            loss.backward()
            if "grad_clip" in opts:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), opts["grad_clip"]
                )
                if LearnPermit:
                    # pass
                    torch.nn.utils.clip_grad_norm_(
                        self.model_Adnet.parameters(), opts["grad_clip2"]
                    )
            optimizer.step()

        # loss1total = torch.stack(loss1total)
        # loss1total = loss1total.cpu().numpy()
