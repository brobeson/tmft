"""
Provides a class for training a TMFT tracker.

Copyright brobeson
"""

import numpy
import torch


class Trainer:
    """
    Train a TMFT tracker.

    Arguments:
        configuration (dict): The Trainer configuration.
    """

    def __init__(self, configuration: dict) -> None:
        self.positive_batch_size = configuration["self.positive_batch_size"]
        self.negative_batch_size = configuration["self.negative_batch_size"]
        self.negative_batch_candidates = max(
            configuration["batch_neg_cand"], self.negative_batch_size
        )
        # properties = torch.cuda.get_device_properties(torch.cuda.device)
        # tensor_size = properties.total_memory / self.batch_test
        self.batch_test = configuration["self.batch_test"]
        self.positive_features = None
        self.negative_features = None

    def train(self, da_model) -> None:
        """
        Train a TMFT model.

        Arguments:
            da_model: The domain adaptation neural network. Use ``None`` to only train the CNN.
        """
        if da_model is not None:
            NumSrcPosData = int(self.positive_features.size(0) / 2)
            pos_featsaAll = pos_feats
            pos_feats = pos_featsaAll[0:NumSrcPosData, :]
            pos_Tar_feats = pos_featsaAll[
                NumSrcPosData : self.positive_features.size(0), :
            ]

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
        model.train()
        if da_model is not None:
            da_model.train()
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
            if da_model is not None:
                batch_Tar_pos_feats = pos_Tar_feats[pos_cur_idx]

                pos_Att_src = da_model(batch_pos_feats)
                pos_Att_tar = da_model(batch_Tar_pos_feats)

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
                if da_model is not None:
                    # BUG grad_clip2 may not be in the options.
                    torch.nn.utils.clip_grad_norm_(
                        da_model.parameters(), opts["grad_clip2"]
                    )
            optimizer.step()
            networks.domain_adaptation_schedules.GradientReverseLayer.schedule.step()

        loss1total = torch.stack(loss1total)
        loss1total = loss1total.cpu().numpy()

        if da_model is not None:
            loss2total = torch.stack(loss2total)
            loss2total = loss2total.cpu().numpy()
            # plt.figure()
            # plt.plot(loss2total)
            # plt.figure()
            # plt.plot(loss1total)
            # plt.show()


class Miner:
    def __init__(self, batch_test: int) -> None:
        self.batch_test = batch_test

    def hard_mine_data(
        self,
        model,
        batch_size: int,
        candidate_count: int,
        features,
        in_layer,
    ) -> torch.Tensor:
        if candidate_count <= batch_size:
            return features
        model.eval()
        for start in range(0, candidate_count, self.batch_test):
            end = min(start + self.batch_test, candidate_count)
            with torch.no_grad():
                score = model(features[start:end], in_layer=in_layer)
            if start == 0:
                neg_cand_score = score.detach()[:, 1].clone()
            else:
                neg_cand_score = torch.cat(
                    (neg_cand_score, score.detach()[:, 1].clone()), 0
                )
        _, top_idx = neg_cand_score.topk(batch_size)
        model.train()
        return features[top_idx]
