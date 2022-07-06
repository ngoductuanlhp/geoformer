import cudf
import cugraph

import sys
sys.path.append( './' )

import numpy as np
import pickle

from tqdm import tqdm
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--data_root', type=str, default="data")
    parser.add_argument('--dataset', type=str, default="scannetv2")
    parser.add_argument('--info_path', type=str, default="scene_graph_info_train")
    parser.add_argument('--edge_in_path', type=str, default="edge_dict_train")
    parser.add_argument('--geodist_out_path', type=str, default="geo_dist_train")
    

    args = parser.parse_args()

    geodist_dir = os.path.join(args.data_root, args.dataset, args.geodist_out_path)
    os.makedirs(geodist_dir, exist_ok=True)

    with open(args.edge_in_path, 'rb') as handle:
        edges_dict = pickle.load(handle)
    
    with open(args.info_path, 'rb') as handle:
        scene_info = pickle.load(handle)


    geo_files = os.listdir(geodist_dir)
    geo_keys = [f.split('.')[0] for f in geo_files]

    list_key = sorted(edges_dict.keys())[0:100]
    for s in list_key:
        if s in geo_keys:
            continue
        print("process:", s)
        edges = edges_dict[s]
        scene = scene_info[s]

        xq = np.ascontiguousarray(scene['query_locs'])

        n_queries = xq.shape[0]

        edges = edges.reshape(-1, 3)
        gdf = cudf.DataFrame()
        # print(edges.shape)
        gdf['src'] = edges[:, 0]
        gdf['dst'] = edges[:, 1]
        gdf["data"] = edges[:, 2]

        num_vertice = len(np.unique(edges[:, 0:2]))
        # st = time.time()
        G = cugraph.Graph()
        G.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='data')


        # print(edges.shape)

        # geo_distance = np.zeros((128, num_vertice-128)) + 10000
        geo_dict = {}
        for i in range(0, n_queries):
            df = cugraph.sssp(G, i)

            # print('num_vertices', num_vertices)
            df = cugraph.filter_unreachable(df)
            # print(df)
            # quit()
            dis = np.array(df['distance'].to_array())
            vertex = np.array(df['vertex'].to_array())
            # num_vertices = len(vertex)
            # 
            mask = (vertex >= n_queries)
            vertex = vertex[mask] - n_queries
            dis = dis[mask]

            # geo_dict = 
            geo_dict[i] = {
                'vertex': vertex,
                'dis': dis,
            }
        
        save_scene_path = os.path.join(geodist_dir, s+'.npy')
        with open(save_scene_path, 'wb') as handle:
            pickle.dump(geo_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Saved folder', save_scene_path)