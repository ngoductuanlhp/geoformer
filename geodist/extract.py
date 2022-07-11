
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

# from model.geoformer.geoformer_fs import GeoFormerFS
from model.geoformer.geoformer_fs_maskaggregate import GeoFormerFS
from datasets.scannetv2_fs_inst import FSInstDataset

from util.dataloader_util import synchronize, get_rank


def init():
    global logger
    logger = create_logger()
    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)

def extract(
        train_loader, 
        model, 
        fold=0
    ):


    model.eval()
    net_device = next(model.parameters()).device
    

    print('len dataloader', len(train_loader))
    global_shape = np.array([0,0,0])
    with torch.no_grad():
        for iteration, batch in enumerate(train_loader):

            torch.cuda.empty_cache()

            support_dict, query_dict, scene_infos = batch

            for key in query_dict:
                if torch.is_tensor(query_dict[key]):
                    query_dict[key] = query_dict[key].to(net_device)

            _ = model.forward_extract(query_dict, scene_infos, fold=fold)

            spatial_shape= query_dict['spatial_shape']
            print(spatial_shape)
            # global_shape = np.maximum(global_shape, spatial_shape)
    print(global_shape)
    import pickle
    if fold == 0:
        save_path = 'data/scannetv2/geoformer_scene_info_train.pkl'
    else:
        save_path = 'data/scannetv2/geoformer_scene_info_val.pkl'

    with open(save_path, 'wb') as handle:
        pickle.dump(model.save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Total saved scenes:', len(model.save_dict.keys()), len(train_loader))
    print('Save to', save_path)

    model.save_dict = {}

if __name__ == '__main__':
    ##### init
    init()

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
        model,
        fold=0
    )

    print("Extract Testing graph")
    dataset = FSInstDataset(split_set='val')
    loader = dataset.trainLoader_debug()

    extract(
        loader, 
        model,
        fold=1
    )
