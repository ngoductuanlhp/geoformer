
import sys
sys.path.append('./')

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import pickle
import faiss                     # make faiss available
import faiss.contrib.torch_utils
from numba import njit, prange
from numba import types
from numba.extending import overload
import time
from tqdm import tqdm


def unique_with_inds(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)

class ShortestObj(object):
    def __init__(self):
        # faiss_cfg = faiss.GpuIndexFlatConfig()
        # faiss_cfg.useFloat16 = False
        # faiss_cfg.device = 0

        # # self.geo_knn = faiss.index_cpu_to_gpu(self.knn_res, 0, faiss.IndexFlatL2(3))
        # self.geo_knn = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 3, faiss_cfg)
        self.res = faiss.StandardGpuResources()
        # self.res.noTempMemory()
        # # index_flat = faiss.IndexFlatL2(3)  # build a flat (CPU) index
        # # self.geo_knn = faiss.index_cpu_to_gpu(self.res, 0, faiss.IndexFlatL2(3))
        self.geo_knn = faiss.GpuIndexFlatL2(self.res, 3)
        pass

    def knn(self, locs_float_, query_inds, max_step=16, neighbor=32, radius=0.1):
        

        start_time = time.time()

        self.geo_knn.add(locs_float_)
        distances_arr, indices_arr = self.geo_knn.search(locs_float_, neighbor)
        distances_arr = torch.sqrt(distances_arr)
        # distances_arr, indices_arr = faiss.knn_gpu(self.res, locs_float_, locs_float_, neighbor)
        self.geo_knn.reset()
        return distances_arr, indices_arr

    def shortest_path(self, locs_float_, query_inds, distances_arr, indices_arr, max_step=32, neighbor=32, radius=0.1):
        query_locs_ = locs_float_[query_inds]
        # quit()
        distances_arr = distances_arr.cuda()
        indices_arr = indices_arr.cuda()


        geo_dist = torch.zeros((query_locs_.shape[0], locs_float_.shape[0]), dtype=torch.float, device=indices_arr.device)-1
        visited = torch.zeros((query_locs_.shape[0], locs_float_.shape[0]), dtype=torch.bool, device=indices_arr.device)
        
        # print('locs_float_', locs_float_[0])
        for q in (range(query_inds.shape[0])):
            # print('debug', query_locs_b)
            D_geo, I_geo = distances_arr[query_inds[q]], indices_arr[query_inds[q]]


            indices, distances = I_geo[1:].reshape(-1), D_geo[1:].reshape(-1)

            # print(distances)
            cond = ((distances <= radius) & (indices >= 0)).bool()


            # breakpoint()
            distances = distances[cond]
            indices = indices[cond]

            for it in range(max_step):
                # breakpoint()

                indices_unique, corres_inds = unique_with_inds(indices)
                distances_uniques = distances[corres_inds]

                inds = torch.nonzero((visited[q, indices_unique]==False)).view(-1)

                if len(inds) < 4:
                    break
                indices_unique = indices_unique[inds]
                distances_uniques = distances_uniques[inds]

                geo_dist[q, indices_unique] = distances_uniques
                visited[q, indices_unique] = True

                D_geo, I_geo = distances_arr[indices_unique][:, 1:], indices_arr[indices_unique][:, 1:]
                # D_geo, I_geo = self.geo_knn.search(locs_float_[indices_unique], neighbor)
                # D_geo = torch.sqrt(D_geo)

                D_geo_cumsum = D_geo + distances_uniques.unsqueeze(-1)

                indices, distances_local, distances_global = I_geo.reshape(-1), D_geo.reshape(-1), D_geo_cumsum.reshape(-1)
                cond = (distances_local <= radius) & (indices >= 0)
                distances = distances_global[cond]
                indices = indices[cond]
        
        # end_time = time.time()
        # print('time', end_time - start_time)

        # debug_dist = torch.sum((geo_dist>-1), dim=1)
        # print(debug_dist)
        # # print(torch.mean(geo_dist_b[geo_dist_b>0]), torch.count_nonzero(geo_dist_b), torch.numel(geo_dist_b))
        # # geo_dist_b = geo_dist_b * 2
        # geo_dist[geo_dist<0] = 1
        # geo_dist[geo_dist>=5] = 1

        geo_dist = geo_dist.cpu().numpy()
        return geo_dist

# def main():

#     shortestObj = ShortestObj()

#     with open('data/scannetv2/geoformer_scene_info_train.pkl', 'rb') as f:
#         geoformer_scene_info_train = pickle.load(f)

#     with open('data/scannetv2/geoformer_knn_train.pkl', 'rb') as f:
#         geoformer_knn_train = pickle.load(f)

#     for scene_name in tqdm(list(geoformer_scene_info_train.keys())[:10]):
#         scene_dict = geoformer_scene_info_train[scene_name]
#         knn = geoformer_knn_train[scene_name]
#         locs_float_ = torch.from_numpy(scene_dict['locs_float_'])
#         pre_enc_inds = torch.from_numpy(scene_dict['pre_enc_inds_arr'][0]).long()
#         query_inds = pre_enc_inds[:128] # first 128 indices

#         distances_arr = knn['distances_arr']
#         indices_arr = knn['indices_arr']
#         geo_dist = shortestObj.shortest_path(locs_float_, query_inds, distances_arr, indices_arr)

#     # save_dict = {}

#     # for scene_name in tqdm(geoformer_scene_info_train.keys()):
#     #     scene_dict = geoformer_scene_info_train[scene_name]

#     #     locs_float_ = torch.from_numpy(scene_dict['locs_float_'])
#     #     pre_enc_inds = torch.from_numpy(scene_dict['pre_enc_inds_arr'][0]).long()
#     #     query_inds = pre_enc_inds[:128] # first 128 indices

#     #     distances_arr, indices_arr = shortestObj.knn(locs_float_, query_inds)
#     #     # print(distances_arr)
#     #     save_dict[scene_name] = {
#     #         'distances_arr': distances_arr,
#     #         'indices_arr': indices_arr,
#     #     }
#     # with open('data/scannetv2/geoformer_knn_train.pkl', 'wb') as handle:
#     #     pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#     # geo_dist = geo_dist.cpu().numpy()
#     # np.save('scene0010_00', geo_dist)
# main()