
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


        start_time = time.time()
        geo_dist = torch.zeros((query_locs_.shape[0], locs_float_.shape[0]), dtype=torch.float, device=indices_arr.device)-1
        visited = torch.zeros((query_locs_.shape[0], locs_float_.shape[0]), dtype=torch.bool, device=indices_arr.device)
        
        # print('locs_float_', locs_float_[0])
        for q in (range(query_locs_.shape[0])):
            geo_dist[q, query_inds[q]] = 0.0
            visited[q, query_inds[q]] = True

            D_geo, I_geo = distances_arr[query_inds[q]], indices_arr[query_inds[q]]


            indices, distances = I_geo[1:].reshape(-1), D_geo[1:].reshape(-1)

            cond = ((distances <= radius) & (indices >= 0)).bool()

            distances = distances[cond]
            indices = indices[cond]

            for it in range(max_step):
                indices_unique, corres_inds = unique_with_inds(indices)
                distances_uniques = distances[corres_inds]
                # print('indices_unique', indices_unique.shape)
                # indices_unique = indices
                # distances_uniques = distances

                inds = torch.nonzero((visited[q, indices_unique]==False)).view(-1)

                if len(inds) < 4:
                    break
                indices_unique = indices_unique[inds]
                distances_uniques = distances_uniques[inds]

                geo_dist[q, indices_unique] = distances_uniques
                visited[q, indices_unique] = True

                D_geo, I_geo = distances_arr[indices_unique][:, 1:], indices_arr[indices_unique][:, 1:]

                D_geo_cumsum = D_geo + distances_uniques.unsqueeze(-1)

                indices, distances_local, distances_global = I_geo.reshape(-1), D_geo.reshape(-1), D_geo_cumsum.reshape(-1)
                cond = (distances_local <= radius) & (indices >= 0)
                distances = distances_global[cond]
                indices = indices[cond]
        


        end_time = time.time()

        print('time', end_time - start_time)
        geo_dist = geo_dist.cpu().numpy()
        return geo_dist

    def shortest_path_chunk(self, locs_float_, query_inds, distances_arr, indices_arr, max_step=48, neighbor=32, radius=0.5):
        query_locs_ = locs_float_[query_inds]
        # quit()
        distances_arr = distances_arr[:, 1:].cuda()
        indices_arr = indices_arr[:, 1:].cuda()

        n_queries = query_locs_.shape[0]
        n_points = locs_float_.shape[0]

        start_time = time.time()

        geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float, device=indices_arr.device)-1
        visited = torch.zeros((n_queries, n_points), dtype=torch.bool, device=indices_arr.device)
        
        arange_tensor = torch.arange(0, n_queries, dtype=torch.long, device=locs_float_.device)

        geo_dist[arange_tensor, query_inds] = 0.0
        visited[arange_tensor, query_inds] = True
            

        distances, indices = distances_arr[query_inds], indices_arr[query_inds] # N_queries x n_neighbors

        cond = (distances <= radius) & (indices >= 0) # N_queries x n_neighbors

        queries_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True) # n_temp
        points_inds = indices[queries_inds, neighbors_inds]  # n_temp
        points_distances = distances[queries_inds, neighbors_inds]  # n_temp

        geo_dist[queries_inds, points_inds] = points_distances
        visited[queries_inds, points_inds] = True


        for it in range(max_step):
            # print('points_inds', points_inds.shape)
            stack_pointquery_inds = torch.stack([points_inds, queries_inds], dim=0)

            # print('stack_pointquery_inds', stack_pointquery_inds.shape)
            indices_unique, corres_inds = unique_with_inds(stack_pointquery_inds)


            # print('corres_inds', corres_inds.shape)

            points_inds = points_inds[corres_inds]
            queries_inds = queries_inds[corres_inds]
            points_distances = points_distances[corres_inds]


            distances_new, indices_new = distances_arr[points_inds], indices_arr[points_inds] # n_temp x n_neighbors
            distances_new_cumsum = distances_new + points_distances[:, None] # n_temp x n_neighbors

            queries_inds = queries_inds[:, None].repeat(1, neighbor-1) # n_temp x n_neighbors

            visited_cond = visited[queries_inds.flatten(), indices_new.flatten()].reshape(*distances_new.shape)
            cond = (distances_new <= radius) & (indices_new >= 0) & (visited_cond == False)# n_temp x n_neighbors

            # print(cond.shape)
            temp_inds, neighbors_inds = torch.nonzero(cond, as_tuple=True) # n_temp2
            points_inds = indices_new[temp_inds, neighbors_inds]  # n_temp2
            points_distances = distances_new_cumsum[temp_inds, neighbors_inds]  # n_temp2
            queries_inds2 = queries_inds[temp_inds, neighbors_inds]  # n_temp2

            geo_dist[queries_inds2, points_inds] = points_distances
            visited[queries_inds2, points_inds] = True

            queries_inds = queries_inds2
        


        end_time = time.time()

        print('time', end_time - start_time)
        geo_dist = geo_dist.cpu().numpy()
        return geo_dist

# def main():

#     shortestObj = ShortestObj()

#     with open('data/scannetv2/geoformer_scene_info_train.pkl', 'rb') as f:
#         geoformer_scene_info_train = pickle.load(f)

#     with open('data/scannetv2/geoformer_scene_info_val.pkl', 'rb') as f:
#         geoformer_scene_info_val = pickle.load(f)

#     # with open('data/scannetv2/geoformer_knn_train.pkl', 'rb') as f:
#     #     geoformer_knn_train = pickle.load(f)

#     # for scene_name in tqdm(list(geoformer_scene_info_train.keys())[:10]):
#     #     scene_dict = geoformer_scene_info_train[scene_name]
#     #     knn = geoformer_knn_train[scene_name]
#     #     locs_float_ = torch.from_numpy(scene_dict['locs_float_'])
#     #     pre_enc_inds = torch.from_numpy(scene_dict['pre_enc_inds_arr'][0]).long()
#     #     query_inds = pre_enc_inds[:128] # first 128 indices

#     #     distances_arr = knn['distances_arr']
#     #     indices_arr = knn['indices_arr']
#     #     geo_dist = shortestObj.shortest_path(locs_float_, query_inds, distances_arr, indices_arr)

#     save_dict = {}

#     for scene_name in tqdm(geoformer_scene_info_val.keys()):
#         # if scene_name in geoformer_scene_info_train.keys():
#         #     continue
#         scene_dict = geoformer_scene_info_val[scene_name]

#         locs_float_ = torch.from_numpy(scene_dict['locs_float_'])
#         pre_enc_inds = torch.from_numpy(scene_dict['pre_enc_inds_arr'][0]).long()
#         query_inds = pre_enc_inds[:128] # first 128 indices

#         distances_arr, indices_arr = shortestObj.knn(locs_float_, query_inds)
#         # print(distances_arr)
#         save_dict[scene_name] = {
#             'distances_arr': distances_arr,
#             'indices_arr': indices_arr,
#         }
#     print(len(list(save_dict.keys())))
#     with open('data/scannetv2/geoformer_knn_val.pkl', 'wb') as handle:
#         pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#     # geo_dist = geo_dist.cpu().numpy()
#     # np.save('scene0010_00', geo_dist)
# main()