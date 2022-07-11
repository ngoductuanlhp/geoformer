import torch
import torch.nn as nn
import spconv as spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
from torch.nn import functional as F
from util import utils
import numpy as np

import pickle
import faiss                     # make faiss available
import faiss.contrib.torch_utils
from numba import njit, prange
from numba import types
from numba.extending import overload

def unique_with_inds(x, dim=-1):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


def find_knn(gpu_index, locs, neighbor=32):
    n_points = locs.shape[0]
    # Search with torch GPU using pre-allocated arrays
    new_d_torch_gpu = torch.zeros(n_points, neighbor, device=locs.device, dtype=torch.float32)
    new_i_torch_gpu = torch.zeros(n_points, neighbor, device=locs.device, dtype=torch.int64)

    gpu_index.add(locs)

    gpu_index.search(locs, neighbor, new_d_torch_gpu, new_i_torch_gpu)
    gpu_index.reset()
    new_d_torch_gpu = torch.sqrt(new_d_torch_gpu)

    return new_d_torch_gpu, new_i_torch_gpu

def find_shortest_path(gpu_index, pre_enc_inds, locs_float_, batch_offset_, max_step=10, neighbor=32, radius=0.1, n_queries=128):
    
    batch_size = pre_enc_inds.shape[0]
    geo_dists = []
    for b in range(batch_size):
        start = batch_offset_[b]
        end = batch_offset_[b+1]

        query_inds = pre_enc_inds[b][:n_queries]
        locs_float_b = locs_float_[start:end]

        n_points = end - start

        new_d_torch_gpu, new_i_torch_gpu = find_knn(gpu_index, locs_float_b, neighbor=neighbor)

        geo_dist = torch.zeros((n_queries, n_points), dtype=torch.float32, device=locs_float_.device)-1
        visited = torch.zeros((n_queries, n_points), dtype=torch.bool, device=locs_float_.device)
        
        for q in range(n_queries):
            D_geo, I_geo = new_d_torch_gpu[query_inds[q]], new_i_torch_gpu[query_inds[q]]


            indices, distances = I_geo[1:].reshape(-1), D_geo[1:].reshape(-1)

            cond = ((distances <= radius) & (indices >= 0)).bool()


            distances = distances[cond]
            indices = indices[cond]

            for it in range(max_step):

                indices_unique, corres_inds = unique_with_inds(indices)
                distances_uniques = distances[corres_inds]

                inds = torch.nonzero((visited[q, indices_unique]==False)).view(-1)

                if len(inds) < neighbor//2:
                    break
                indices_unique = indices_unique[inds]
                distances_uniques = distances_uniques[inds]

                geo_dist[q, indices_unique] = distances_uniques
                visited[q, indices_unique] = True

                D_geo, I_geo = new_d_torch_gpu[indices_unique][:, 1:], new_i_torch_gpu[indices_unique][:, 1:]

                D_geo_cumsum = D_geo + distances_uniques.unsqueeze(-1)

                indices, distances_local, distances_global = I_geo.reshape(-1), D_geo.reshape(-1), D_geo_cumsum.reshape(-1)
                cond = (distances_local <= radius) & (indices >= 0)
                distances = distances_global[cond]
                indices = indices[cond]
        geo_dists.append(geo_dist)
    return geo_dists