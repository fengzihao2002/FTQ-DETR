# ------------------------------------------------------------------------
# H-DETR
# Copyright (c) 2022 Peking University & Microsoft Research Asia. All Rights Reserved.
# Licensed under the MIT-style license found in the LICENSE file in the root directory
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    accuracy,
    get_world_size,
    interpolate,
    is_dist_avail_and_initialized,
    inverse_sigmoid,
)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (
    DETRsegm,
    PostProcessPanoptic,
    PostProcessSegm,
    dice_loss,
    sigmoid_focal_loss,
)
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class FineGrainedBoundaryCompensation(nn.Module):
    """Fine-Grained Boundary Compensation (FGBC).

    It modifies only the projected multi-scale feature maps before the
    transformer encoder. For each selected high-resolution level, it extracts
    a local high-frequency residual, gates it with content-aware weights, and
    feeds the result back through a bounded residual path.
    """

    def __init__(self, d_model=256, levels="0,1", alpha=0.10, kernel_size=3, warmup_epochs=2):
        super().__init__()
        if isinstance(levels, str):
            levels = [int(x.strip()) for x in levels.split(',') if x.strip() != '']
        self.levels = set(int(x) for x in levels)
        self.kernel_size = int(kernel_size)
        self.warmup_epochs = int(warmup_epochs)
        padding = self.kernel_size // 2

        self.detail_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(32, d_model),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.ones(16) * float(alpha))
        self.avg_pool = nn.AvgPool2d(self.kernel_size, stride=1, padding=padding)

    def _warmup_scale(self, current_epoch):
        if self.warmup_epochs <= 0 or current_epoch is None:
            return 1.0
        return min(1.0, max(0.0, float(current_epoch) / float(self.warmup_epochs)))

    def forward(self, srcs, masks=None, current_epoch=None):
        if not self.levels:
            return srcs
        scale = self._warmup_scale(current_epoch)
        if scale <= 0:
            return srcs

        outs = []
        for lvl, x in enumerate(srcs):
            if lvl not in self.levels:
                outs.append(x)
                continue
            low = self.avg_pool(x)
            detail = x - low
            gate = self.gate(torch.cat([x, detail], dim=1))
            comp = self.detail_proj(gate * detail)
            alpha_l = torch.tanh(self.alpha[lvl]).to(dtype=x.dtype, device=x.device)
            y = x + scale * alpha_l * comp
            if masks is not None and lvl < len(masks) and masks[lvl] is not None:
                y = y.masked_fill(masks[lvl].unsqueeze(1), 0.0)
            outs.append(y)
        return outs

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        num_queries_one2one=300,
        num_queries_one2many=0,
        mixed_selection=False,
        use_quality_branch=False,
        use_fgbc=False,
        fgbc_levels="0,1",
        fgbc_alpha=0.10,
        fgbc_kernel=3,
        fgbc_warmup_epochs=2,
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
            num_queries_one2one: number of object queries for one-to-one matching part
            num_queries_one2many: number of object queries for one-to-many matching part
            mixed_selection: a trick for Deformable DETR two stage

        """
        super().__init__()
        num_queries = num_queries_one2one + num_queries_one2many
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.use_fgbc = use_fgbc
        self.current_epoch = fgbc_warmup_epochs
        self.fgbc_warmup_epochs = fgbc_warmup_epochs
        if self.use_fgbc:
            self.fgbc = FineGrainedBoundaryCompensation(
                d_model=hidden_dim,
                levels=fgbc_levels,
                alpha=fgbc_alpha,
                kernel_size=fgbc_kernel,
                warmup_epochs=fgbc_warmup_epochs,
            )
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.quality_embed = MLP(hidden_dim, hidden_dim, 1, 2)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        elif mixed_selection:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.xavier_uniform_(self.quality_embed.layers[0].weight)
        nn.init.constant_(self.quality_embed.layers[0].bias, 0)
        nn.init.constant_(self.quality_embed.layers[-1].weight, 0)
        nn.init.constant_(self.quality_embed.layers[-1].bias, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (
            (transformer.decoder.num_layers + 1)
            if two_stage
            else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.num_queries_one2one = num_queries_one2one
        self.mixed_selection = mixed_selection
        self.use_quality_branch = use_quality_branch

    def set_current_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.use_fgbc:
            srcs = self.fgbc(srcs, masks, current_epoch=getattr(self, "current_epoch", None))

        query_embeds = None
        if not self.two_stage or self.mixed_selection:
            query_embeds = self.query_embed.weight[0 : self.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = (
            torch.zeros([self.num_queries, self.num_queries,]).bool().to(src.device)
        )
        self_attn_mask[self.num_queries_one2one :, 0 : self.num_queries_one2one,] = True
        self_attn_mask[0 : self.num_queries_one2one, self.num_queries_one2one :,] = True

        (
            hs_cls,
            hs_loc,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(hs_cls.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs_cls[lvl])
            tmp = self.bbox_embed[lvl](hs_loc[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(
                outputs_class[:, 0 : self.num_queries_one2one]
            )
            outputs_classes_one2many.append(
                outputs_class[:, self.num_queries_one2one :]
            )
            outputs_coords_one2one.append(
                outputs_coord[:, 0 : self.num_queries_one2one]
            )
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one :])
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        if self.use_quality_branch:
            outputs_quality_one2one = self.quality_embed(
                hs_loc[-1][:, 0 : self.num_queries_one2one]
            ).squeeze(-1)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }
        if self.use_quality_branch:
            out["pred_quality"] = outputs_quality_one2one
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one
            )
            out["aux_outputs_one2many"] = self._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many
            )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
            self,
            num_classes,
            matcher,
            weight_dict,
            losses,
            focal_alpha=0.25,
            use_qa_cls=False,
            qa_iou_power=1.0,
            qa_skip_enc=False,
            use_quality_branch=False,
            use_quality_ranking=False,
            quality_rank_margin=0.1,
            quality_rank_topk=3,
            quality_rank_iou_thr=0.1,
            quality_rank_center_thr=0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.use_qa_cls = use_qa_cls
        self.qa_iou_power = qa_iou_power
        self.qa_skip_enc = qa_skip_enc
        self.use_quality_branch = use_quality_branch
        self.use_quality_ranking = use_quality_ranking
        self.quality_rank_margin = quality_rank_margin
        self.quality_rank_topk = quality_rank_topk
        self.quality_rank_iou_thr = quality_rank_iou_thr
        self.quality_rank_center_thr = quality_rank_center_thr

    def loss_labels_hard(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def get_matched_quality_targets(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        if target_classes_o.numel() == 0:
            return idx, target_classes_o, None

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        iou_mat, _ = box_ops.box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        )
        quality = torch.diag(iou_mat).detach().clamp(min=0.0, max=1.0)
        quality = quality.pow(self.qa_iou_power).to(outputs["pred_logits"].dtype)

        return idx, target_classes_o, quality

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        if self.use_qa_cls and len(target_classes_o) > 0:
            _, _, quality = self.get_matched_quality_targets(outputs, targets, indices)
            target_classes_onehot[idx[0], idx[1], target_classes_o] = quality

        loss_ce = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes,
                    alpha=self.focal_alpha,
                    gamma=2,
                )
                * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def loss_quality(self, outputs, targets, indices, num_boxes):
        if (not self.use_quality_branch) or ("pred_quality" not in outputs):
            zero = outputs["pred_logits"].sum() * 0.0
            return {"loss_quality": zero}

        idx, target_classes_o, quality = self.get_matched_quality_targets(
            outputs, targets, indices
        )

        if quality is None or target_classes_o.numel() == 0:
            zero = outputs["pred_quality"].sum() * 0.0
            return {"loss_quality": zero}

        pred_quality = outputs["pred_quality"][idx].sigmoid()
        loss_quality = F.smooth_l1_loss(pred_quality, quality, reduction="sum") / num_boxes

        return {"loss_quality": loss_quality}

    def loss_quality_rank(self, outputs, targets, indices, num_boxes):
        if (
            (not self.use_quality_ranking)
            or (not self.use_quality_branch)
            or ("pred_quality" not in outputs)
        ):
            zero = outputs["pred_logits"].sum() * 0.0
            return {"loss_quality_rank": zero}

        pred_quality = outputs["pred_quality"].sigmoid()
        pred_logits_detached = outputs["pred_logits"].detach().sigmoid()
        pred_boxes_detached = outputs["pred_boxes"].detach()

        total_loss = pred_quality.sum() * 0.0
        pair_count = 0
        num_queries = outputs["pred_logits"].shape[1]
        all_query_ids = torch.arange(num_queries, device=pred_quality.device)

        for batch_i, (src_ids, tgt_ids) in enumerate(indices):
            if src_ids.numel() == 0:
                continue

            matched_mask = torch.zeros(num_queries, dtype=torch.bool, device=pred_quality.device)
            matched_mask[src_ids] = True
            unmatched_ids = all_query_ids[~matched_mask]
            if unmatched_ids.numel() == 0:
                continue

            unmatched_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_detached[batch_i, unmatched_ids])
            unmatched_centers = pred_boxes_detached[batch_i, unmatched_ids, :2]

            for src_id, tgt_id in zip(src_ids.tolist(), tgt_ids.tolist()):
                gt_label = targets[batch_i]["labels"][tgt_id]
                gt_box = targets[batch_i]["boxes"][tgt_id].unsqueeze(0)
                gt_box_xyxy = box_ops.box_cxcywh_to_xyxy(gt_box)
                gt_center = gt_box[0, :2]

                ious, _ = box_ops.box_iou(unmatched_boxes_xyxy, gt_box_xyxy)
                ious = ious.squeeze(1)
                center_dist = torch.norm(unmatched_centers - gt_center.unsqueeze(0), dim=1)
                cls_scores = pred_logits_detached[batch_i, unmatched_ids, gt_label]

                local_mask = (ious > self.quality_rank_iou_thr) | (center_dist < self.quality_rank_center_thr)
                candidate_ids = unmatched_ids[local_mask]
                candidate_scores = cls_scores[local_mask]

                if candidate_ids.numel() == 0:
                    candidate_ids = unmatched_ids
                    candidate_scores = cls_scores

                topk = min(self.quality_rank_topk, candidate_ids.numel())
                if topk <= 0:
                    continue

                _, order = torch.topk(candidate_scores, k=topk, dim=0)
                hard_neg_ids = candidate_ids[order]

                pos_q = pred_quality[batch_i, src_id]
                neg_q = pred_quality[batch_i, hard_neg_ids]
                total_loss = total_loss + F.relu(self.quality_rank_margin - (pos_q - neg_q)).sum()
                pair_count += hard_neg_ids.numel()

        if pair_count == 0:
            return {"loss_quality_rank": pred_quality.sum() * 0.0}

        return {"loss_quality_rank": total_loss / pair_count}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "quality": self.loss_quality,
            "quality_rank": self.loss_quality_rank,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            if loss in {"quality", "quality_rank"} and "pred_quality" not in outputs:
                continue
            kwargs = {}
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in {"masks", "quality", "quality_rank"}:
                        continue
                    kwargs = {}
                    if loss == "labels":
                        kwargs["log"] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in {"masks", "quality", "quality_rank"}:
                    continue
                kwargs = {}
                if loss == "labels":
                    kwargs["log"] = False

                if loss == "labels" and self.use_qa_cls and self.qa_skip_enc:
                    l_dict = self.loss_labels_hard(
                        enc_outputs, bin_targets, indices, num_boxes, **kwargs
                    )
                else:
                    l_dict = self.get_loss(
                        loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs
                    )

                l_dict = {k + f"_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, topk=100, use_quality_rescore=False, quality_score_alpha=1.0, quality_score_beta=0.5):
        super().__init__()
        self.topk = topk
        self.use_quality_rescore = use_quality_rescore
        self.quality_score_alpha = quality_score_alpha
        self.quality_score_beta = quality_score_beta
        print("topk for eval:", self.topk)

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        if self.use_quality_rescore and ("pred_quality" in outputs):
            quality = outputs["pred_quality"].sigmoid().unsqueeze(-1)
            if self.quality_score_alpha != 1.0:
                prob = prob.pow(self.quality_score_alpha)
            prob = prob * quality.pow(self.quality_score_beta)

        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.topk, dim=1
        )
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.use_quality_ranking and not args.use_quality_branch:
        raise ValueError("--use_quality_ranking requires --use_quality_branch")

    num_classes = args.num_classes
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        num_queries_one2one=args.num_queries_one2one,
        num_queries_one2many=args.num_queries_one2many,
        mixed_selection=args.mixed_selection,
        use_quality_branch=args.use_quality_branch,
        use_fgbc=args.use_fgbc,
        fgbc_levels=args.fgbc_levels,
        fgbc_alpha=args.fgbc_alpha,
        fgbc_kernel=args.fgbc_kernel,
        fgbc_warmup_epochs=args.fgbc_warmup_epochs,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.cls_loss_coef, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f"_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    new_dict = dict()
    for key, value in weight_dict.items():
        new_dict[key] = value
        new_dict[key + "_one2many"] = value
    weight_dict = new_dict

    if args.use_quality_branch:
        weight_dict["loss_quality"] = args.quality_loss_coef
    if args.use_quality_ranking:
        weight_dict["loss_quality_rank"] = args.quality_rank_loss_coef

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    if args.use_quality_branch:
        losses += ["quality"]
    if args.use_quality_ranking:
        losses += ["quality_rank"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(
        num_classes,
        matcher,
        weight_dict,
        losses,
        focal_alpha=args.focal_alpha,
        use_qa_cls=args.use_qa_cls,
        qa_iou_power=args.qa_iou_power,
        qa_skip_enc=args.qa_skip_enc,
        use_quality_branch=args.use_quality_branch,
        use_quality_ranking=args.use_quality_ranking,
        quality_rank_margin=args.quality_rank_margin,
        quality_rank_topk=args.quality_rank_topk,
        quality_rank_iou_thr=args.quality_rank_iou_thr,
        quality_rank_center_thr=args.quality_rank_center_thr,
    )
    criterion.to(device)
    postprocessors = {
        "bbox": PostProcess(
            topk=args.topk,
            use_quality_rescore=args.use_quality_rescore,
            quality_score_alpha=args.quality_score_alpha,
            quality_score_beta=args.quality_score_beta,
        )
    }
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85
            )

    return model, criterion, postprocessors
