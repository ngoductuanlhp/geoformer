import pickle
import numpy as np

import sys
sys.path.append( './' )
import faiss                     # make faiss available
import faiss.contrib.torch_utils
from tqdm import tqdm
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--data_root', type=str, default="data")
    parser.add_argument('--dataset', type=str, default="scannetv2")
    parser.add_argument('--info_path', type=str, default="scene_graph_info_train.pkl")
    parser.add_argument('--edge_out_path', type=str, default="edge_dict_train.pkl")

    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--r', type=float, default=0.1)
    args = parser.parse_args()

    with open(os.path.join(args.data_root, args.dataset, args.info_path), 'rb') as handle:
        scene_info = pickle.load(handle)

    res = faiss.StandardGpuResources()  # use a single GPU
    index_flat = faiss.IndexFlatL2(3)  # build a flat (CPU) index

    # make it a flat GPU index

    edges_dict = {}

    k = args.k
    radius = args.r

    for s in tqdm(scene_info.keys()):
        print("process:", s)
        scene = scene_info[s]

        xb = np.ascontiguousarray(scene['locs_float_'])
        xq = np.ascontiguousarray(scene['query_locs'])

        n_queries = xq.shape[0]

        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(xb)         # add vectors to the index

                                # we want to see 4 nearest neighbors
        D, I = gpu_index_flat.search(xq, 2)  # actual search
        D = D[:, 1:]
        I = I[:, 1:]
        D = np.sqrt(D)
        D[D >= radius] = 100

        query_context_edge = np.concatenate([np.arange(n_queries)[:, None], I+n_queries, D], axis=-1)

        D2, I2 = gpu_index_flat.search(xb, k+1)  # actual search
        D2 = D2[:, 1:]
        I2 = I2[:, 1:] # skip the nearest itself
        D2 = np.sqrt(D2)

        edges = []
        for i in range(I2.shape[0]):
            radius_mask = (D2[i] <= radius)
            count_valid = np.count_nonzero(radius_mask)
            if count_valid == 0:
                temp = np.array([[i+n_queries, I2[i,0], 100]])
            else:

                temp = np.zeros((count_valid,3))
                temp[:, 0] = i + n_queries
                temp[:, 1] = I2[i][radius_mask] + n_queries
                temp[:, 2] = D2[i][radius_mask]

            edges.append(temp)
        
        edges = np.concatenate(edges, axis=0)
        edges = np.concatenate([query_context_edge, edges])
        edges_dict[s] = edges
        # break
    
    saved_path = os.path.join(args.data_root, args.dataset, args.edge_out_path)
    with open(saved_path, 'wb') as handle:
        pickle.dump(edges_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Graph save to', saved_path)

if __name__ == '__main__':
    main()