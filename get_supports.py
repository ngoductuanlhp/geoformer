import os, sys, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
from torch.utils.data import DataLoader
sys.path.append('../')

from util.config import cfg
from lib.pointgroup_ops.functions import pointgroup_ops

import random
import pickle
from torch.utils.data import Dataset as TorchDataset
from datasets.scannetv2_fs_inst import FSInstDataset
random.seed(1111)

FOLD0 = [2,3,4,7,9,11,12,13,18]
FOLD1 = [5,6,8,10,14,15,16,17,19]
FOLD2 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

FOLD = {
    0: FOLD0,
    1: FOLD1, 
    2: sorted(FOLD0 + FOLD1)
}


def main():
    class2instances_file = os.path.join('/home/ubuntu/fewshot3d_ws/geoformer/data/scannetv2/class2instances.pkl')
    with open(class2instances_file, 'rb') as f:
        class2instances = pickle.load(f)

    support_set_file = os.path.join('/home/ubuntu/fewshot3d_ws/geoformer/data/scannetv2/support_sets/fullscene_fold1_1shot_10sets.pkl')
    with open(support_set_file, 'rb') as f:
        support_sets = pickle.load(f)

    dataset = FSInstDataset('val')

    exist_tuples = []
    for s in support_sets:
        for k in s.keys():
            for it in s[k]:
                exist_tuples.append(it[0]+'_'+str(it[1]))


    support_set_new = []

    for i in range(3):
        support_set = {cls:[] for cls in [5,6,8,10,14,15,16,17,19]}
        for cls in [5,6,8,10,14,15,16,17,19]:
            for i in range(1):
                while True:
                    support_tuple       = random.choice(class2instances[cls])
                    support_scene_name, support_instance_id = support_tuple[0], support_tuple[1]

                    # support_xyz_middle, support_xyz_scaled, support_rgb, support_label, support_instance_label \
                    #     = self.load_single_block(support_scene_name, support_instance_id, aug=False,permutate=False,val=True)
                    support_xyz_middle, support_xyz_scaled, support_rgb, support_label, support_instance_label \
                        = dataset.load_single(support_scene_name, aug=False, permutate=False, val=True)
                    support_mask = (support_instance_label == support_instance_id).astype(int)

                    ratio = np.count_nonzero(support_mask) / support_xyz_middle.shape[0]
                    # print("Sup conditions: ", np.count_nonzero(support_mask), ratio)
                    # if np.count_nonzero(support_mask) >= 1000 and ratio >= 0.05:
                    if np.count_nonzero(support_mask) >= 1000 and ratio >= 0.05 and (support_scene_name+'_'+str(support_instance_id)) not in exist_tuples:
                        print("Pick {}, {}".format(support_scene_name, support_instance_id))
                        break
                support_set[cls].append(support_tuple)
        support_set_new.append(support_set)

    support_sets[1] = support_set_new[0]
    support_sets[7] = support_set_new[1]
    support_sets[8] = support_set_new[2]

    with open('/home/ubuntu/fewshot3d_ws/geoformer/data/scannetv2/support_sets/fullscene_fold1_1shot_10sets3.pkl', 'wb') as f:
        pickle.dump(support_sets, f, pickle.HIGHEST_PROTOCOL)
main()