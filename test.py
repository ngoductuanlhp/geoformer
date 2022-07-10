from logging import Logger
import torch
import time
import numpy as np
import random
import os

from util.config import cfg
cfg.task = 'test'
from util.log import create_logger
import util.utils as utils
import util.eval as eval

import os.path as osp
from checkpoint import strip_prefix_if_present, align_and_update_state_dicts
from checkpoint import checkpoint

from model.geoformer.geoformer import GeoFormer
from datasets.scannetv2_inst import InstDataset

FOLD0 = [2,3,4,7,9,11,12,13,18]
FOLD1 = [5,6,8,10,14,15,16,17,19]
BENCHMARK_SEMANTIC_LABELS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


def init():
    global result_dir
    result_dir = cfg.exp_path
    os.makedirs(cfg.exp_path, exist_ok=True)

    global logger
    logger = create_logger()
    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def do_test(model, dataloader, cur_epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')


    model.eval()
    net_device = next(model.parameters()).device

    num_test_scenes = len(dataloader)

    with torch.no_grad():
        matches = {}
        for i, batch_input in enumerate(dataloader):
            N = batch_input['feats'].shape[0]
            # test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:12]
            test_scene_name = batch_input['test_scene_name'][0]
            torch.cuda.empty_cache()

            for key in batch_input:
                if torch.is_tensor(batch_input[key]):
                    batch_input[key] = batch_input[key].to(net_device)

            outputs = model(batch_input, cur_epoch, training=False)

            if (cur_epoch > cfg.prepare_epochs):
                if 'proposal_scores' not in outputs.keys():
                    continue

                cls_final, scores_final, masks_final = outputs['proposal_scores']   # (nProposal, 1) float, cuda
                if(isinstance(cls_final, list)):
                    continue

                

                SEMANTIC_FOLD = FOLD1 if cfg.cvfold == 1 else FOLD0

                # print(semantic_test)
                temp = torch.tensor(SEMANTIC_FOLD, device=scores_final.device)[cls_final-4]
                # temp = torch.tensor(semantic_test, device=scores_pred.device)[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()] - 4]
                semantic_id = torch.tensor(BENCHMARK_SEMANTIC_LABELS, device=scores_final.device)[temp] # (nProposal), long


                ##### nms
                if semantic_id.shape[0] == 0:
                    pick_idxs = np.empty(0)
                else:
                    proposals_pred_f = masks_final.float()  # (nProposal, N), float, cuda
                    intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                    proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                    pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_final.cpu().numpy(), cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
                clusters = masks_final[pick_idxs]
                cluster_scores = scores_final[pick_idxs]
                cluster_semantic_id = semantic_id[pick_idxs]

                nclusters = clusters.shape[0]

                ##### prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = cluster_scores.cpu().numpy()
                    pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                    pred_info['mask'] = clusters.cpu().numpy()
                    gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt

            ##### print
            logger.info("instance iter: {}/{} point_num: {} ncluster: {}".format(batch_input['id'][0] + 1, num_test_scenes, N, nclusters))
            # torch.cuda.empty_cache()

        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            eval.print_results(avgs, logger)


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


if __name__ == '__main__':
    init()

    ##### model
    logger.info('=> creating model ...')
    model = GeoFormer()
    model = model.cuda(0)

    # logger.info(model)
    logger.info('# parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))



    checkpoint_fn = cfg.resume
    if os.path.isfile(checkpoint_fn):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
        state = torch.load(checkpoint_fn)
        # cur_epoch = state['epoch'] + 1
        model_state_dict = model.state_dict()
        loaded_state_dict = strip_prefix_if_present(state['state_dict'], prefix="module.")
        align_and_update_state_dicts(model_state_dict, loaded_state_dict)
        model.load_state_dict(model_state_dict)
        # model.load_state_dict(state['state_dict'])

        # logger.info("=> loaded checkpoint '{}' (cur_epoch {})".format(checkpoint_fn, cur_epoch))
    else:
        raise RuntimeError

    dataset = InstDataset(split_set='val')
    test_loader = dataset.testLoader()

    cur_epoch = 300
    ##### evaluate
    do_test(
        model,
        test_loader,
        cur_epoch
    )
