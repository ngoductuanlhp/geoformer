import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from util.config import cfg

from model.matcher import HungarianMatcher


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = 1
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss

def compute_dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    # inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / (num_boxes + 1e-6)

def compute_sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / (num_boxes + 1e-6)

def compute_score_loss(inputs, targets, num_boxes):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    # prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return ce_loss.mean(1).sum() / (num_boxes + 1e-6)


class InstCriterion(nn.Module):
    def __init__(self):
        super(InstCriterion, self).__init__()
        # self.semantic_criterion = nn.BCEWithLogitsLoss()
        # self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)
        # self.similarity_criterion = nn.BCEWithLogitsLoss()
        # similarity_criterion = FocalLossV1()
        # semantic_criterion = FocalLossV1(gamma=2, alpha=0.25)
        self.score_criterion = nn.BCELoss(reduction='none')
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)


    def forward(self, model_outputs, batch_inputs, epoch):
        # '''semantic loss'''
        semantic_scores = model_outputs['semantic_scores']
        semantic_labels = batch_inputs['labels']
        instance_labels = batch_inputs['instance_labels']

        pt_offsets = model_outputs['pt_offsets']
        instance_info = batch_inputs['instance_infos']
        coords = batch_inputs['locs_float']
        

        loss_dict_out = {}
        loss = torch.tensor(0.0, requires_grad=True).to(semantic_scores.device)

        semantic_loss = self.semantic_criterion(semantic_scores, semantic_labels)
        
        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid = (instance_labels != cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)


        loss += semantic_loss + offset_norm_loss + offset_dir_loss

        loss_dict_out['sem_loss'] = (semantic_loss.item(), semantic_labels.shape[0])
        loss_dict_out['offset_norm_loss'] = (offset_norm_loss.item(), valid.sum())
        loss_dict_out['offset_dir_loss'] = (offset_dir_loss.item(), valid.sum())

        dice_loss = offset_dir_loss.new_tensor(0.0)
        
        if (epoch > cfg.prepare_epochs):
            
            mask_logits     = model_outputs['mask_logits']
            object_idxs     = model_outputs['object_idxs']

            proposals_idx_shift     = model_outputs['proposals_idx_shift']
            proposals_offset_shift  = model_outputs['proposals_offset_shift']

            instance_masked = instance_labels[object_idxs]
            semantic_masked = semantic_labels[object_idxs]

            inst_batch_ids      = model_outputs['inst_batch_ids']
            batch_ids           = model_outputs['batch_idxs']
            batch_offsets       = model_outputs['batch_offsets']
            inst_pred_seg_label = model_outputs['inst_pred_seg_label']

            inst_num = inst_batch_ids.size(0)
            inst_gt_mask = torch.zeros_like(mask_logits)
            weights = torch.zeros_like(mask_logits)

            assert inst_pred_seg_label.size(0) == inst_num
            valid_num_dice_loss = 0

            num_pos = 0

            for n in range(inst_num):
                start = proposals_offset_shift[n].item()
                end = proposals_offset_shift[n+1].item()
                ins_ids_n = proposals_idx_shift[start:end, 1]
                ins_label_n = torch.mode(instance_labels[ins_ids_n.long()])[0].item()
                seg_label_n_pred = inst_pred_seg_label[n].item()
                inst_n_batch_id = int(inst_batch_ids[n].item())
                weights[n, batch_ids==inst_n_batch_id] = 1


                if ins_label_n < 0 or seg_label_n_pred <= 3:
                    continue
                weights[n,semantic_masked!=seg_label_n_pred] = 0

                batch_start = batch_offsets[inst_n_batch_id].item()
                batch_end = batch_offsets[inst_n_batch_id+1].item()
                cover_percent = (instance_masked[batch_ids==inst_n_batch_id] == ins_label_n).float().sum() /\
                                (instance_labels[batch_start:batch_end] == ins_label_n).float().sum()

                if cover_percent > 0.3:
                    inst_gt_mask[n] = (instance_masked ==ins_label_n)
                    num_pos += 1
                valid_id = (weights[n]>0).nonzero().squeeze(dim=1)
                if valid_id.size(0) >0:
                    inst_gt = inst_gt_mask[n][valid_id]
                    inst_pred = torch.sigmoid(mask_logits[n])[valid_id]
                    dice_loss += dice_coefficient(inst_pred, inst_gt).mean()
                    valid_num_dice_loss += 1

            # print(torch.count_nonzero(inst_gt_mask), weights.sum())
            score_loss = self.score_criterion(torch.sigmoid(mask_logits.view(-1)), inst_gt_mask.view(-1))
            score_loss = (score_loss* weights.view(-1)).sum() / (weights.sum() + 1e-6)
            score_loss = score_loss.mean()
            loss_dict_out['score_loss'] = (score_loss, proposals_offset_shift.size(0)-1)
            loss += score_loss

            dice_loss = dice_loss / (valid_num_dice_loss+  + 1e-6)
            loss += dice_loss
            loss_dict_out['dice_loss'] = (dice_loss, 1)

        loss_dict_out['loss'] = (loss.item(), semantic_labels.shape[0])
        return loss, loss_dict_out
