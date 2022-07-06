
import torch
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np
import glob

import sys

sys.path.append( './' )

# from checkpoint import align_and_update_state_dicts

from util.config import cfg
from util.log import create_logger
import util.utils as utils

from checkpoint import strip_prefix_if_present, align_and_update_state_dicts
from checkpoint import checkpoint

import datetime
import math

from model.geoformer.geoformer_fs import GeoFormerFS
from datasets.scannetv2_fs_inst import FSInstDataset

from util.dataloader_util import synchronize, get_rank

def compute_learning_rate(curr_epoch_normalized, max_epochs):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (cfg.warm_lr_epochs / max_epochs)
        and cfg.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = cfg.warm_lr + curr_epoch_normalized * max_epochs * (
            (cfg.base_lr - cfg.warm_lr) / cfg.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = cfg.final_lr + 0.5 * (cfg.base_lr - cfg.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(optimizer, curr_epoch, max_epochs):
    curr_lr = compute_learning_rate(curr_epoch, max_epochs)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr

def init():
    # os.makedirs(cfg.exp_path, exist_ok=True)
    # log the config
    global logger
    logger = create_logger()
    logger.info(cfg)
    # summary writer
    # global writer
    # writer = SummaryWriter(cfg.exp_path)

def extract(
        train_loader, 
        model, 
        fold=0
    ):

    data_time = utils.AverageMeter()

    model.eval()
    net_device = next(model.parameters()).device
    
    end_time = time.time()

    print('len dataloader', len(train_loader))
    for iteration, batch in enumerate(train_loader):

        data_time.update(time.time() - end_time)
        torch.cuda.empty_cache()

        support_dict, query_dict, scene_infos = batch

        for key in query_dict:
            if torch.is_tensor(query_dict[key]):
                query_dict[key] = query_dict[key].to(net_device)

        _ = model.forward_extract(query_dict, scene_infos, fold=fold)

    import pickle
    if fold == 0:
        save_path = 'data/scannetv2/scene_graph_info_train.pkl'
    else:
        save_path = 'data/scannetv2/scene_graph_info_test.pkl'

    with open(save_path, 'wb') as handle:
        pickle.dump(model.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Total saved scenes:', len(model.save_dict.keys()))
    print('Save to', save_path)

if __name__ == '__main__':
    ##### init
    init()

     ###
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    local_rank = 0
    torch.cuda.set_device(local_rank)
    np.random.seed(cfg.manual_seed + get_rank())
    torch.manual_seed(cfg.manual_seed + get_rank())
    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    if use_cuda:
        torch.cuda.manual_seed_all(cfg.manual_seed + get_rank())

    ##### model
    logger.info('=> creating model ...')

    model = GeoFormerFS()
    model = model.cuda()

    

    if cfg.resume:
        checkpoint_fn = cfg.resume
        if os.path.isfile(checkpoint_fn):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_fn))
            state = torch.load(checkpoint_fn, map_location=torch.device("cpu"))

            model_state_dict = model.state_dict()
            loaded_state_dict = strip_prefix_if_present(state['state_dict'], prefix="module.")
            align_and_update_state_dicts(model_state_dict, loaded_state_dict)
            model.load_state_dict(model_state_dict)

        else:
            raise ValueError("=> no checkpoint found at '{}'".format(checkpoint_fn))

    # model.create_support_set_aggregator()
    print("Extract training graph")
    dataset = FSInstDataset(split_set='train')
    loader = dataset.trainLoader_debug()

    extract(
        loader, 
        model
    )

    print("Extract Testing graph")
    dataset = FSInstDataset(split_set='val')
    loader = dataset.trainLoader_debug()

    extract(
        loader, 
        model,
        fold=1
    )
