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


class FocalLossV1(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, label):
        pred = pred.sigmoid()
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        alpha_factor = torch.ones(pred.shape).cuda() * self.alpha
        alpha_factor = torch.where(torch.eq(label, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(label, 1.), 1. - pred, pred)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        bce = -(label * torch.log(pred) + (1.0 - label) * torch.log(1.0 - pred))

        # cls_loss = focal_weight * torch.pow(bce, gamma)
        # num_positive = (gt_mask==1).sum().float()
        cls_loss = (focal_weight * bce)
        # cls_loss = cls_loss.sum()
        cls_loss = cls_loss.sum() / (label.shape[0] + 1e-6)
        
        return cls_loss


class InstSetCriterion(nn.Module):
    def __init__(self):
        super(InstSetCriterion, self).__init__()

        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)

        self.score_criterion = nn.BCELoss(reduction='none').cuda()

        self.batch_size = cfg.batch_size
        self.n_queries = cfg.n_query_points

        self.matcher = HungarianMatcher(self.batch_size, self.n_queries)

        self.loss_weight = {
            'dice_loss': 1,
            'focal_loss': 1,
            'cls_loss': 1,
        }

    def single_layer_loss(self, mask_prediction, instance_masked, semantic_masked, batch_ids):
        loss = torch.tensor(0.0, requires_grad=True).to(instance_masked.device)
        loss_dict = {}

        mask_logits_list  = mask_prediction['mask_logits'] # list of n_queries x N_mask
        cls_logits   = mask_prediction['cls_logits'] # batch x n_queries x n_classes

        # obj_logits   = 1 - torch.mean(cls_logits[..., 0:4], dim=2) # batch x n_queries

        for k in self.loss_weight:
            loss_dict[k] = torch.tensor(0.0, requires_grad=True).to(cls_logits.device)

        # pred_inds_list, inst_masks_gt_list, sem_cls_gt_list = self.matcher.forward_seg(mask_logit, cls_logits, obj_logit, instance_masked, semantic_masked, batch_ids)
        # per_query_gt_inds, query_matched_mask = None, None
        num_gt = 0 
        for batch in range(self.batch_size):
            mask_logit_b = mask_logits_list[batch]
            cls_logit_b = cls_logits[batch] # n_queries x n_classes
            # obj_logit_b = obj_logits[batch]
            instance_masked_b = instance_masked[batch_ids==batch]
            semantic_masked_b = semantic_masked[batch_ids==batch]

            # print('mask_logit_b', mask_logit_b.shape, cls_logit_b.shape)
            # print('instance_masked_b', instance_masked_b.shape)

            if mask_logit_b == None:
                continue
            
            pred_inds, inst_mask_gt, sem_cls_gt = self.matcher.forward_seg_single(mask_logit_b, cls_logit_b, 
                                                    instance_masked_b, semantic_masked_b)
            if pred_inds is None:
                continue
            mask_logit_pred = mask_logit_b[pred_inds]

            num_gt_batch = len(pred_inds)
            num_gt += num_gt_batch
             
            loss_dict['dice_loss'] += compute_dice_loss(mask_logit_pred, inst_mask_gt, num_gt_batch)
            # loss_dict['focal_loss'] += compute_sigmoid_focal_loss(mask_logit_pred, inst_mask_gt, num_gt_batch)
            loss_dict['focal_loss'] += compute_score_loss(mask_logit_pred, inst_mask_gt, num_gt_batch)
            cls_label = torch.zeros((self.n_queries)).to(cls_logits.device)
            cls_label[pred_inds] = sem_cls_gt

            loss_dict['cls_loss'] += F.cross_entropy(
                cls_logit_b,
                cls_label.long(),
                reduction="mean",
            )

        for k in self.loss_weight:
            loss_dict[k] = loss_dict[k] * self.loss_weight[k] / self.batch_size
            loss += loss_dict[k]

        return loss, loss_dict, num_gt

    def forward(self, model_outputs, batch_inputs, epoch):

        # '''semantic loss'''
        # semantic_scores = model_outputs['semantic_scores']
        semantic_scores = model_outputs['semantic_scores']
        semantic_labels = batch_inputs['labels']
        instance_labels = batch_inputs['instance_labels']

        '''offset loss'''
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


        if epoch <= cfg.prepare_epochs:
            loss_dict_out['sem_loss'] = (semantic_loss.item(), semantic_labels.shape[0])
            loss_dict_out['offset_norm_loss'] = (offset_norm_loss.item(), valid.sum())
            loss_dict_out['offset_dir_loss'] = (offset_dir_loss.item(), valid.sum())
            loss_dict_out['loss'] = (loss.item(), semantic_labels.shape[0])
            return loss, loss_dict_out

        mask_predictions = model_outputs['mask_predictions']
        fg_idxs     = model_outputs['fg_idxs']
        # num_insts   = model_outputs['num_insts']
        
        instance_masked = instance_labels[fg_idxs]
        semantic_masked = semantic_labels[fg_idxs]

        batch_ids           = model_outputs['batch_idxs']

        

        ''' Main loss '''
        main_loss, loss_dict, num_gt = self.single_layer_loss(mask_predictions[-1], instance_masked, semantic_masked, batch_ids)

        loss += main_loss

        ''' Auxilary loss '''
        for l in range(cfg.dec_nlayers-1):
            interm_loss, _, _ = self.single_layer_loss(mask_predictions[l], instance_masked, semantic_masked, batch_ids)
            loss += interm_loss
            
        loss_dict_out['focal_loss'] = (loss_dict['focal_loss'].item(), num_gt)
        loss_dict_out['dice_loss'] = (loss_dict['dice_loss'].item(), num_gt)
        loss_dict_out['cls_loss'] = (loss_dict['cls_loss'].item(), self.n_queries)
        loss_dict_out['sem_loss'] = (semantic_loss.item(), semantic_labels.shape[0])
        loss_dict_out['offset_norm_loss'] = (offset_norm_loss.item(), valid.sum())
        loss_dict_out['offset_dir_loss'] = (offset_dir_loss.item(), valid.sum())
        loss_dict_out['loss'] = (loss.item(), semantic_labels.shape[0])

        return loss, loss_dict_out



class InstCriterion(nn.Module):
    def __init__(self):
        super(InstCriterion, self).__init__()
        # self.semantic_criterion = nn.BCEWithLogitsLoss()
        # self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label)
        # self.similarity_criterion = nn.BCEWithLogitsLoss()
        # similarity_criterion = FocalLossV1()
        # semantic_criterion = FocalLossV1(gamma=2, alpha=0.25)
        self.score_criterion = nn.BCELoss(reduction='none')

    def forward(self, model_outputs, batch_inputs):
        loss_dict = {}

        # '''semantic loss'''
        # semantic_scores = model_outputs['semantic_scores']
        semantic_labels = batch_inputs['labels']
        instance_labels = batch_inputs['instance_labels']

        mask_logits = model_outputs['mask_logits']
        fg_idxs     = model_outputs['fg_idxs']
        num_insts   = model_outputs['num_insts']
        
        instance_masked = instance_labels[fg_idxs]
        semantic_masked = semantic_labels[fg_idxs]

        batch_ids           = model_outputs['batch_idxs']
        batch_offsets       = batch_inputs['offsets']
        fps_sampling_inds   = model_outputs['fps_sampling_inds'] 

        loss = torch.tensor([0.0], requires_grad=True).to(instance_labels.device)
        

        sampling_ins_label = instance_labels[fps_sampling_inds.long()]
        sampling_seg_label = semantic_labels[fps_sampling_inds.long()]

        cover_percents = []
        for n in range(num_insts):
            inst_n_batch_id = n // cfg.n_query_points
            batch_start = batch_offsets[inst_n_batch_id].item()
            batch_end = batch_offsets[inst_n_batch_id+1].item()
            cover_percent = (instance_masked[batch_ids==inst_n_batch_id] == sampling_ins_label[n]).float().sum() /\
                            (instance_labels[batch_start:batch_end] == sampling_ins_label[n]).float().sum()

            cover_percents.append(cover_percent)

            
            # if cover_percent < 0.3:
            #     continue
        cover_percents = torch.tensor(cover_percents).float().cuda()

        positive_inds = torch.nonzero((sampling_ins_label >= 0) & (sampling_seg_label >= 4) & (cover_percents >= 0.3)).long()
        num_positive_inds = len(positive_inds)
        n_mask = mask_logits[-1].shape[-1]

        inst_gt_masks = torch.zeros((num_positive_inds, n_mask)).cuda()
        weights = torch.zeros((num_positive_inds, n_mask)).cuda()

        for i, ind in enumerate(positive_inds):
            n_batch_id = ind // cfg.n_query_points
            inst_gt_masks[i,:] = (instance_masked == sampling_ins_label[ind])
            weights[i, batch_ids==n_batch_id] = 1
            weights[i, semantic_masked!=sampling_seg_label[ind]] = 0

        

        for l in range(cfg.dec_nlayers):
            # FIXME lightweight
            if l < cfg.dec_nlayers-4:
                continue

            mask_logit = mask_logits[l]

            # inst_gt_masks       = []
            # weights             = []
            # mask_select_inds    = []
            # inst_gt_mask = torch.zeros_like(mask_logit)
            # weights = torch.zeros_like(mask_logit)

            # n_mask = mask_logit.shape[-1]

            # assert inst_pred_seg_label.size(0) == num_insts
            valid_num_dice_loss = 0

            dice_loss = loss.new_tensor(0.0)
            for i, ind in enumerate(positive_inds):
                valid_id = torch.nonzero(weights[i] > 0).squeeze(dim=1)
                if valid_id.size(0) >0:
                    inst_gt = inst_gt_masks[i, valid_id]
                    # print("DEBUG", torch.sigmoid(mask_logit[ind]).shape, valid_id.shape)
                    inst_pred = torch.sigmoid(mask_logit[ind, valid_id])
                    # print(inst_pred.shape, inst_gt.shape)
                    dice_loss += dice_coefficient(inst_pred, inst_gt).mean()
                    valid_num_dice_loss += 1

            
            if num_positive_inds == 0:
                score_loss = 0
                dice_loss = 0
                num_insts = 0
            else:

                mask_logit_select = mask_logit[positive_inds, :]
                # inst_gt_masks = torch.stack(inst_gt_masks).float()
                # weights = torch.stack(weights).float()
                # print(torch.count_nonzero(inst_gt_mask), weights.sum())
                score_loss = self.score_criterion(torch.sigmoid(mask_logit_select.view(-1)), inst_gt_masks.view(-1))
                score_loss = (score_loss* weights.view(-1)).sum() / (weights.sum() + 1e-6)
                score_loss = score_loss.mean()
                
                loss += (cfg.loss_weight[3] * score_loss)

                dice_loss = dice_loss / (valid_num_dice_loss + 1e-6)
                loss += dice_loss
            
            if l == cfg.dec_nlayers -1:
                print('num_positive {}/{}'.format(num_positive_inds, num_insts))
                loss_dict['score_loss'] = (score_loss, num_insts)
                loss_dict['dice_loss'] = (dice_loss, 1)

        loss_dict['loss'] = (loss.item(), semantic_labels.shape[0])
        return loss, loss_dict
