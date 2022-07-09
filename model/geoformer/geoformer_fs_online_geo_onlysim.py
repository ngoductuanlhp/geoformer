import torch
import torch.nn as nn
import spconv as spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')
from torch.nn import functional as F
from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
import numpy as np
from model.transformer import TransformerEncoder
from util.config import cfg

from model.geoformer.geoformer_modules import ResidualBlock, VGGBlock, UBlock, conv_with_kaiming_uniform

from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetSAModuleVotesSeparate
from lib.pointnet2.pointnet2_utils import furthest_point_sample

from model.pos_embedding import PositionEmbeddingCoordsSine
from model.helper import unique_with_inds

from util.config import cfg
import time


from model.helper import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT,
                            get_clones, GenericMLP, BatchNormDim1Swap)

from model.transformer_detr import TransformerDecoder, TransformerDecoderLayer

import pickle
import faiss                     # make faiss available
import faiss.contrib.torch_utils
from numba import njit, prange
from numba import types
from numba.extending import overload


@njit(parallel=True)
def shortest_path(
    query_inds: np.ndarray,
    distances_arr: np.ndarray,
    indices_arr: np.ndarray,
    max_step: int = 16,
    radius: float = 0.01
):  
    n_points = distances_arr.shape[1]
    geo_dist_b = np.full((query_inds.shape[0], n_points), -1, dtype=np.float32)
    visited_b = np.full((query_inds.shape[0], n_points), 0, dtype=np.int32)
    for q in prange(query_inds.shape[0]):
        visited_b[q, query_inds[q]] += 1
        geo_dist_b[q, query_inds[q]] = 0.0
        distances = distances_arr[query_inds[q]][1:]
        indices = indices_arr[query_inds[q]][1:]
        cond = (distances <= radius) & (indices >= 0)
        distances = distances[cond]
        indices = indices[cond]

        for it in range(max_step):
            # breakpoint()
            indices_unique, corres_inds = np.unique(indices, return_index=True)
            distances_uniques = distances[corres_inds]

            inds = np.nonzero((visited_b[q, indices_unique] < 1)).view(-1)

            if len(inds) < 4:
                break
            indices_unique = indices_unique[inds]
            distances_uniques = distances_uniques[inds]

            geo_dist_b[q, indices_unique] = distances_uniques
            visited_b[q, indices_unique] = True

            D_geo = distances_arr[indices_unique]
            I_geo = indices_arr[indices_unique]

            D_geo_cumsum = D_geo + distances_uniques.unsqueeze(-1)

            indices, distances_local, distances_global = I_geo.reshape(-1), D_geo.reshape(-1), D_geo_cumsum.reshape(-1)
            cond = (distances_local <= radius)  & (indices >= 0)
            distances = distances_global[cond]
            indices = indices[cond]

    return geo_dist_b

class GeoFormerFS(nn.Module):
    def __init__(self):
        super().__init__()

        input_c = cfg.input_channel
        m = cfg.m

        classes = cfg.classes

        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.prepare_epochs = cfg.prepare_epochs

        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, use_backbone_transformer=cfg.use_backbone_transformer, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        #### semantic segmentation
        # self.linear = nn.Linear(m, classes) # bias(default): True
        self.semantic = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU(),
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.semantic_linear = nn.Linear(m, classes, bias=True)

        ################################
        ################################
        ################################
        ### for instance embedding
        self.output_dim = 16
        # self.output_dim = cfg.dec_dim
        self.mask_conv_num = 3
        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        mask_tower = []
        for i in range(self.mask_conv_num):
            mask_tower.append(conv_block(m, m))
        mask_tower.append(nn.Conv1d(
            m,  self.output_dim, 1
        ))
        self.add_module('mask_tower', nn.Sequential(*mask_tower))

        ### convolution before the condinst take place (convolution num before the generated parameters take place)
        before_embedding_conv_num = 1
        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        before_embedding_tower = []
        for i in range(before_embedding_conv_num-1):
            before_embedding_tower.append(conv_block(cfg.dec_dim, cfg.dec_dim))
        before_embedding_tower.append(conv_block(cfg.dec_dim, self.output_dim))
        self.add_module("before_embedding_tower", nn.Sequential(*before_embedding_tower))

        ### cond inst generate parameters for
        USE_COORDS = True
        self.use_coords = USE_COORDS
        self.embedding_conv_num = 2
        weight_nums = []
        bias_nums = []
        for i in range(self.embedding_conv_num):
            if i ==0:
                if USE_COORDS:
                    weight_nums.append((self.output_dim+3) * self.output_dim)
                else:
                    weight_nums.append(self.output_dim * self.output_dim)
                bias_nums.append(self.output_dim)
            elif i == self.embedding_conv_num-1:
                weight_nums.append(self.output_dim)
                bias_nums.append(1)
            else:
                weight_nums.append(self.output_dim*self.output_dim)
                bias_nums.append(self.output_dim)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = nn.Conv1d(self.output_dim, self.num_gen_params, kernel_size=1)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        ''' Set aggregate '''
        set_aggregate_dim_out = 2* m
        mlp_dims = [m, 2*m, 2*m, set_aggregate_dim_out]
        self.set_aggregator = PointnetSAModuleVotesSeparate(
            radius=0.2,
            nsample=64,
            npoint=cfg.n_decode_point,
            mlp=mlp_dims,
            normalize_xyz=True,
        )

        ''' Position embedding '''
        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=cfg.dec_dim, pos_type="fourier", normalize=True
        )

        ''' DETR-Decoder '''
        decoder_layer = TransformerDecoderLayer(
            d_model=cfg.dec_dim,
            nhead=cfg.dec_nhead,
            dim_feedforward=cfg.dec_ffn_dim,
            dropout=cfg.dec_dropout,
            normalize_before=True,
            use_rel=cfg.use_rel,
        )

        self.decoder = TransformerDecoder(
            decoder_layer, num_layers=cfg.dec_nlayers, return_intermediate=True
        )

        self.query_projection = GenericMLP(
            input_dim=cfg.dec_dim,
            hidden_dims=[cfg.dec_dim],
            output_dim=cfg.dec_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        self.encoder_to_decoder_projection = GenericMLP(
            input_dim=set_aggregate_dim_out,
            hidden_dims=[set_aggregate_dim_out],
            output_dim=cfg.dec_dim,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_use_activation=True,
            output_use_norm=True,
            output_use_bias=False,
        )
        self.similarity_net = nn.Sequential(
            nn.Linear(3 * set_aggregate_dim_out, 3*set_aggregate_dim_out, bias=True),
            norm_fn(3*set_aggregate_dim_out),
            nn.ReLU(),
            nn.Linear(3*set_aggregate_dim_out, 3*set_aggregate_dim_out, bias=True),
            norm_fn(3*set_aggregate_dim_out),
            nn.ReLU(),
            nn.Linear(3*set_aggregate_dim_out, 1, bias=True)
        )
        # self.detr_sem_head = GenericMLP(
        #     input_dim=cfg.dec_dim,
        #     hidden_dims=[cfg.dec_dim, cfg.dec_dim],
        #     norm_fn_name="bn1d",
        #     activation="relu",
        #     use_conv=True,
        #     output_dim=classes
        # )

        self.init_knn()

        self.apply(self.set_bn_init)

        self.threshold_ins = cfg.threshold_ins
        self.min_pts_num = cfg.min_pts_num
        #### fix parameter
        self.module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'semantic': self.semantic, 'semantic_linear': self.semantic_linear,
                      'mask_tower': self.mask_tower,
                      'set_aggregator': self.set_aggregator,
                      'pos_embedding': self.pos_embedding, 'encoder_to_decoder_projection': self.encoder_to_decoder_projection,
                      'query_projection': self.query_projection, 'decoder': self.decoder,
                      'before_embedding_tower': self.before_embedding_tower, 'controller': self.controller,
                      'similarity_net': self.similarity_net}

        for m in self.fix_module:
            mod = self.module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        
        self.save_dict = {}

    def init_knn(self):
        faiss_cfg = faiss.GpuIndexFlatConfig()
        faiss_cfg.useFloat16 = False
        faiss_cfg.device = 0

        # self.knn_res = faiss.StandardGpuResources()
        # self.geo_knn = faiss.index_cpu_to_gpu(self.knn_res, 0, faiss.IndexFlatL2(3))
        self.geo_knn = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 3, faiss_cfg)

    def train(self, mode=True):
        super().train(mode)
        for mod in self.fix_module:
            mod = getattr(self, mod)
            for m in mod.modules():
                m.eval()

    def set_eval(self):
        for m in self.fix_module:
            self.module_map[m] = self.module_map[m].eval()

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm1d') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def generate_proposal(self, geo_dist_arr, mask_logits, similarity_score_filter, fg_idxs, batch_offsets, threshold=0.5, min_pts_num=50):
        # batch = mask_logits.shape[0]
        batch = len(mask_logits)
        n_queries = mask_logits[0].shape[0]
        n_inst = batch * n_queries
        proposal_len = []
        proposal_len.append(0)
        proposal_idx = []
        seg_preds = []
        num = 0
        scores = []


        for b in range(batch):
            start   = batch_offsets[b]
            end     = batch_offsets[b+1]
            mask_logit_b = mask_logits[b].sigmoid()
            geo_dist_b = geo_dist_arr[b].cuda()
            for n in range(n_queries):
                
                # cond = (mask_logit_b[n] > threshold).float() + (geo_dist_b[n] < 0.1).float()
                cond = (mask_logit_b[n] > threshold).float()
                proposal_id_n = cond.nonzero().squeeze(dim=1)

                # ANCHOR fewshot
                if proposal_id_n.size(0) < min_pts_num:
                    continue
                    
                score = mask_logit_b[n][proposal_id_n].mean() * torch.pow(similarity_score_filter[b,n], 0.24)
                proposal_id_n = proposal_id_n + start
                
                proposal_id_n = fg_idxs[proposal_id_n.long()].unsqueeze(dim=1)
                # id_proposal_id_n = torch.cat([proposal_id_n, torch.ones_like(proposal_id_n)*b], dim=1)
                id_proposal_id_n = torch.cat([torch.ones_like(proposal_id_n)*num, proposal_id_n], dim=1)
                num += 1
                tmp = proposal_len[-1]
                proposal_len.append(proposal_id_n.size(0)+tmp)
                proposal_idx.append(id_proposal_id_n)
                scores.append(score)
                seg_preds.append(0)

        if len(proposal_idx) == 0:
            return proposal_idx, proposal_len, scores, seg_preds
        proposal_idx = torch.cat(proposal_idx, dim=0)
        proposal_len = torch.from_numpy(np.array(proposal_len)).cuda()
        scores = torch.stack(scores)
        seg_preds = torch.tensor(seg_preds).cuda()

        return proposal_idx, proposal_len, scores, seg_preds

    def random_point_sample(self, batch_offsets, npoint):
        batch_size = batch_offsets.shape[0] - 1
        
        batch_points = (batch_offsets[1:] - batch_offsets[:-1])
        
        sampling_indices = [torch.tensor(np.random.choice(batch_points[i].item(), npoint, replace=(npoint>batch_points[i])), dtype=torch.int).cuda() + batch_offsets[i]
                             for i in range(batch_size)]
        sampling_indices = torch.cat(sampling_indices)
        return sampling_indices

    def random_point_sample_b(self, batch_points, npoint):
        
        sampling_indices = torch.tensor(np.random.choice(batch_points, npoint, replace=False), dtype=torch.int).cuda()
        return sampling_indices

    def sample_query_embedding(self, xyz, pc_dims, n_quries):
        query_sampling_inds = furthest_point_sample(xyz, n_quries).long()

        query_locs = [torch.gather(xyz[..., x], 1, query_sampling_inds) for x in range(3)]
        query_locs = torch.stack(query_locs)
        query_locs = query_locs.permute(1, 2, 0)

        query_embedding_pos = self.pos_embedding(query_locs, input_range=pc_dims)
        query_embedding_pos = self.query_projection(query_embedding_pos.float())
        return query_locs, query_embedding_pos, query_sampling_inds


    def parse_dynamic_params(self, params, out_channels):
        assert params.dim()==2
        assert len(self.weight_nums) == len(self.bias_nums)
        assert params.size(1) == sum(self.weight_nums) + sum(self.bias_nums)

        num_instances = params.size(0)
        num_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(params, self.weight_nums+self.bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_instances*out_channels, -1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances*out_channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_instances, -1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances)

        return weight_splits, bias_splits


    def mask_heads_forward(self, geo_dist, mask_features, weights, biases, num_insts, coords_, fps_sampling_coords, use_coords=True):
        assert mask_features.dim() == 3
        n_layers = len(weights)
        c = mask_features.size(1)
        n_mask = mask_features.size(0)
        x = mask_features.permute(2,1,0).repeat(num_insts, 1, 1) ### num_inst * c * N_mask

        # geo_dist = geo_dist.cuda()


        relative_coords = fps_sampling_coords[:, None, :] - coords_[None, :,:]
        relative_coords = relative_coords.permute(0,2,1)

        relative_coords_geo = geo_dist.unsqueeze(-1).repeat(1,1,3)  # N_inst * N_mask * 3
        # relative_coords_geo = geo_dist.unsqueeze(-1)
        relative_coords_geo = relative_coords_geo.permute(0,2,1)

        if use_coords:
            # if relative_coords.shape[2] > x.shape[2]:
            #     relative_coords = relative_coords[..., :x.shape[2]]
            # elif relative_coords.shape[2] < x.shape[2]:
            #     res = x.shape[2] - relative_coords.shape[2]
            #     relative_coords = torch.cat([relative_coords, torch.ones((relative_coords.shape[0], relative_coords.shape[1], res)).float().to(relative_coords.device)],dim=2)
            x = torch.cat([relative_coords, x], dim=1) ### num_inst * (3+c) * N_mask
            # x = torch.cat([relative_coords_geo, x], dim=1) ### num_inst * (3+c) * N_mask

        x = x.reshape(1, -1, n_mask) ### 1 * (num_inst*c') * Nmask
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)

        return x
    

    def get_mask_prediction(self, geo_dist_arr, param_kernels, mask_features, locs_float_, fps_sampling_locs, batch_offsets_):
        # param_kernels = param_kernels.permute(0, 2, 1, 3) # num_layers x batch x n_queries x channel
        num_layers, n_queries, batch, channel = (
            param_kernels.shape[0],
            param_kernels.shape[1],
            param_kernels.shape[2],
            param_kernels.shape[3],
        )

        outputs = []
        n_inst_per_layer = batch * n_queries
        for l in range(num_layers):

            param_kernel = param_kernels[l] # n_queries x batch x channel
            # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
            param_kernel2 = param_kernel.transpose(0,1).flatten(0,1) # (batch * n_queries) * channel
            before_embedding_feature    = self.before_embedding_tower(torch.unsqueeze(param_kernel2, dim=2))
            controllers                  = self.controller(before_embedding_feature).squeeze(dim=2)

            controllers  = controllers.reshape(batch, n_queries, -1)

            mask_logits_list = []
            for b in range(batch):
                start = batch_offsets_[b]
                end = batch_offsets_[b+1]

                if end - start == 0:
                    mask_logits_list.append(None)
                    continue

                controller      = controllers[b] # n_queries x channel
                weights, biases = self.parse_dynamic_params(controller, self.output_dim)

                mask_feature_b = mask_features[start:end, :]
                locs_float_b   = locs_float_[start:end, :]
                fps_sampling_locs_b = fps_sampling_locs[b]

                geo_dist = geo_dist_arr[b]

                mask_logits         = self.mask_heads_forward(geo_dist, mask_feature_b, weights, biases, n_queries, locs_float_b, 
                                                            fps_sampling_locs_b, use_coords=self.use_coords)
                
                mask_logits     = mask_logits.squeeze(dim=0) # (n_queries) x N_mask
                mask_logits_list.append(mask_logits)
                
            output = {'mask_logits': mask_logits_list}
            outputs.append(output)
        return outputs

    def preprocess_input(self, batch_input, batch_size):
        voxel_coords = batch_input['voxel_locs']              # (M, 1 + 3), long, cuda
        v2p_map = batch_input['v2p_map']                     # (M, 1 + maxActive), int, cuda
        locs_float = batch_input['locs_float']              # (N, 3), float32, cuda
        feats = batch_input['feats']                        # (N, C), float32, cuda
        spatial_shape = batch_input['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, locs_float), 1).float()

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda
        sparse_input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        return sparse_input

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        N, C = xyz.shape
        # print("DEBUG", N)
        centroids = torch.zeros(npoint, dtype=torch.long).to(device)
        distance = torch.ones(N).to(device) * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[ farthest, :].view(1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids


    

    
    def calculate_geo_dist(self, locs_float_, batch_offsets_, query_locs, query_inds, max_step=6, neighbor=20, radius=0.1):
        batch_size = query_locs.shape[0]

        geo_dist = []
        for b in range(batch_size):
            start = batch_offsets_[b]
            end = batch_offsets_[b+1]

            locs_float_b = locs_float_[start:end, :]
            query_locs_b = query_locs[b]


            if len(query_locs_b) == 0:
                continue
            geo_dist_b = torch.zeros((query_locs_b.shape[0], locs_float_b.shape[0]), dtype=torch.float, device=locs_float_.device)-1
                # visited_b = torch.zeros((query_locs_b.shape[0], locs_float_b.shape[0]), dtype=torch.bool, device=locs_float_.device)


                # self.geo_knn.add(locs_float_b)
                # distances_arr, indices_arr = self.geo_knn.search(locs_float_b, neighbor)
                # distances_arr = torch.sqrt(distances_arr)
                # self.geo_knn.reset()
                # # print('locs_float_', locs_float_[0])
                # for q in range(query_locs_b.shape[0]):
                #     # print('debug', query_locs_b)
                #     D_geo, I_geo = distances_arr[query_inds[b,q]], indices_arr[query_inds[b,q]]
                #     indices, distances = I_geo.reshape(-1), D_geo.reshape(-1)
                #     cond = (distances <= radius) & (indices >= 0)
                #     distances = distances[cond]
                #     indices = indices[cond]

                #     for it in range(max_step):
                #         # breakpoint()
                #         indices_unique, corres_inds = unique_with_inds(indices)
                #         distances_uniques = distances[corres_inds]

                #         inds = torch.nonzero((visited_b[q, indices_unique]==False)).view(-1)

                #         if len(inds) < 4:
                #             break
                #         indices_unique = indices_unique[inds]
                #         distances_uniques = distances_uniques[inds]

                #         geo_dist_b[q, indices_unique] = distances_uniques
                #         visited_b[q, indices_unique] = True

                #         D_geo, I_geo = distances_arr[indices_unique], indices_arr[indices_unique]
                #         # D_geo, I_geo = self.geo_knn.search(locs_float_[indices_unique], neighbor)
                #         # D_geo = torch.sqrt(D_geo)

                #         D_geo_cumsum = D_geo + distances_uniques.unsqueeze(-1)

                #         indices, distances_local, distances_global = I_geo.reshape(-1), D_geo.reshape(-1), D_geo_cumsum.reshape(-1)
                #         cond = (distances_local <= radius) & (indices >= 0)
                #         distances = distances_global[cond]
                #         indices = indices[cond]
                
                # # print(torch.mean(geo_dist_b[geo_dist_b>0]), torch.count_nonzero(geo_dist_b), torch.numel(geo_dist_b))
                # # geo_dist_b = geo_dist_b * 2
                # geo_dist_b[geo_dist_b<0] = 1
                # geo_dist_b[geo_dist_b>=5] = 1
            # geo_dist_b = geo_dist_b/10
            geo_dist.append(geo_dist_b)

        return geo_dist


    def process_support(self, batch_input, training=True):
        batch_size      = cfg.batch_size if training else 1
        batch_idxs      = batch_input['locs'][:, 0].int()
        p2v_map         = batch_input['p2v_map']
        locs_float      = batch_input['locs_float']
        batch_offsets   = batch_input['batch_offsets']
        support_mask    = batch_input['support_masks']

        # pc_dims = [
        #     batch_input["pc_maxs"],
        #     batch_input["pc_mins"],
        # ]
        with torch.no_grad():
            sparse_input = self.preprocess_input(batch_input, batch_size)

            ''' Backbone net '''
            output = self.input_conv(sparse_input)
            output = self.unet(output)
            output = self.output_layer(output)
            output_feats = output.features[p2v_map.long()]
            output_feats = output_feats.contiguous()

            mask_indices    = torch.nonzero(support_mask==1).view(-1)
            output_feats_   = output_feats[mask_indices]
            locs_float_     = locs_float[mask_indices]

            batch_idxs_     = batch_idxs[mask_indices]
            batch_offsets_  = utils.get_batch_offsets(batch_idxs_, batch_size)

            support_embeddings = []
            for b in range(batch_size):
                start = batch_offsets_[b]
                end = batch_offsets_[b+1]

                locs_float_b = locs_float_[start:end, :].unsqueeze(0)
                output_feats_b = output_feats_[start:end, :].unsqueeze(0)
                # support_embedding = torch.mean(output_feats_b, dim=1) # channel

                context_locs_b, grouped_features, grouped_xyz, pre_enc_inds = self.set_aggregator.group_points(locs_float_b.contiguous(), 
                                                                    output_feats_b.transpose(1,2).contiguous(), npoint_new=32)
                context_feats_b = self.set_aggregator.mlp(grouped_features, grouped_xyz, pooling='avg')
                context_feats_b = context_feats_b.transpose(1,2) # 1 x n_point x channel

                support_embedding = torch.mean(context_feats_b, dim=1) # channel
                support_embeddings.append(support_embedding)
            support_embeddings = torch.cat(support_embeddings) # batch x channel
            return support_embeddings

    def forward(self, support_dict, scene_dict, scene_infos, training=True,remember=False,support_embeddings=None):
        outputs = {}
        batch_size = cfg.batch_size if training else 1

        if support_embeddings is None:
            support_embeddings = self.process_support(support_dict, training) # batch x channel

        outputs     = {}
        batch_size  = cfg.batch_size if training else 1
        batch_idxs  = scene_dict['locs'][:, 0].int()
        p2v_map     = scene_dict['p2v_map']
        locs_float  = scene_dict['locs_float']
        batch_offsets = scene_dict['batch_offsets']

        pc_dims = [
            scene_dict["pc_mins"],
            scene_dict["pc_maxs"],
        ]

        N_points = locs_float.shape[0]

        if remember:
            (context_locs, context_feats, 
            fg_idxs, batch_offsets, 
            output_feats_,batch_idxs_, locs_float_, batch_offsets_, semantic_preds_, semantic_scores,
            query_sampling_inds,
            query_locs, query_embedding_pos, 
                                    geo_dist_arr) = self.cache_data
            outputs['semantic_scores'] = semantic_scores
        else:
            context_backbone = torch.no_grad if 'unet' in self.fix_module else torch.enable_grad
            with context_backbone():
                sparse_input = self.preprocess_input(scene_dict, batch_size)

                ''' Backbone net '''
                output = self.input_conv(sparse_input)
                output = self.unet(output)
                output = self.output_layer(output)
                output_feats = output.features[p2v_map.long()]
                output_feats = output_feats.contiguous()

                ''' Semantic head'''
                semantic_feats  = self.semantic(output_feats)
                semantic_scores = self.semantic_linear(semantic_feats)   # (N, nClass), float
                semantic_preds  = semantic_scores.max(1)[1]    # (N), long

                outputs['semantic_scores'] = semantic_scores

                if cfg.train_fold == cfg.cvfold:
                    fg_condition = semantic_preds >= 4
                else:
                    fg_condition = semantic_preds == 3
                    
                fg_idxs         = torch.nonzero(fg_condition).view(-1)

                batch_idxs_     = batch_idxs[fg_idxs]
                batch_offsets_  = utils.get_batch_offsets(batch_idxs_, batch_size)
                locs_float_     = locs_float[fg_idxs]
                output_feats_   = output_feats[fg_idxs]
                semantic_preds_ = semantic_preds[fg_idxs]

            context_aggregator = torch.no_grad if 'set_aggregator' in self.fix_module else torch.enable_grad
            with context_aggregator():
                # query_sampling_inds_arr = []
                # pre_enc_inds_arr = []
                # for b in range(batch_size):
                #     s_dict = scene_dict['scene_graph_info'][b]

                    
                #     query_sampling_inds = torch.from_numpy(s_dict['query_sampling_inds']).cuda()
                #     pre_enc_inds = torch.from_numpy(s_dict['pre_enc_inds_arr']).cuda()

                #     query_sampling_inds_arr.append(query_sampling_inds)
                #     pre_enc_inds_arr.append(pre_enc_inds)
                # query_sampling_inds_arr = torch.stack(query_sampling_inds_arr)
                
                # 1-dim: n_queries*batch_size
                context_locs = []
                context_feats = []
                for b in range(batch_size):
                    start = batch_offsets_[b]
                    end = batch_offsets_[b+1]
                    locs_float_b = locs_float_[start:end, :]
                    output_feats_b = output_feats_[start:end, :]
                    batch_points = (end - start).cpu().item()

                    if batch_points == 0:
                        outputs['proposal_scores']  = None
                        return outputs

                    locs_float_b = locs_float_b.unsqueeze(0)
                    output_feats_b = output_feats_b.unsqueeze(0)

                    context_locs_b, grouped_features, grouped_xyz, pre_enc_inds = self.set_aggregator.group_points(locs_float_b.contiguous(), 
                                                                    output_feats_b.transpose(1,2).contiguous())
                    context_feats_b = self.set_aggregator.mlp(grouped_features, grouped_xyz)
                    context_feats_b = context_feats_b.transpose(1,2)

                    context_locs.append(context_locs_b)
                    context_feats.append(context_feats_b)

                context_locs = torch.cat(context_locs)
                context_feats = torch.cat(context_feats) # batch x npoint x channel

                
                

                query_locs, query_embedding_pos, query_sampling_inds = self.sample_query_embedding(context_locs, pc_dims, cfg.n_query_points)

                geo_dist_arr = self.calculate_geo_dist(locs_float_, batch_offsets_, query_locs, query_sampling_inds)
                # geo_dist_arr = scene_dict['geo_dists']

                self.cache_data = (context_locs, context_feats, 
                                    fg_idxs, batch_offsets, 
                                    output_feats_,batch_idxs_, locs_float_, batch_offsets_, semantic_preds_, semantic_scores, query_sampling_inds,
                                    query_locs, query_embedding_pos, 
                                    geo_dist_arr)

        context_mask_tower = torch.no_grad if 'mask_tower' in self.fix_module else torch.enable_grad
        with context_mask_tower():
            mask_features_   = self.mask_tower(torch.unsqueeze(output_feats_, dim=2).permute(2,1,0)).permute(2,1,0)
        
        ''' channel-wise correlate '''
        channel_wise_tensor = context_feats * support_embeddings.unsqueeze(1).repeat(1,cfg.n_decode_point,1)
        subtraction_tensor = context_feats - support_embeddings.unsqueeze(1).repeat(1,cfg.n_decode_point,1)
        aggregation_tensor = torch.cat([channel_wise_tensor, subtraction_tensor, context_feats], dim=2) # batch * n_sampling *(3*channel)
        
        aggregation_tensor_sampled  = [torch.gather(aggregation_tensor[..., x], 1, query_sampling_inds) for x in range(aggregation_tensor.shape[-1])]
        aggregation_tensor_sampled  = torch.stack(aggregation_tensor_sampled) # channel x batch x n_sampling
        aggregation_tensor_sampled  = aggregation_tensor_sampled.permute(1,2,0) # batch x n_sampling x channel

        similarity_score = self.similarity_net(aggregation_tensor_sampled.flatten(0,1)).squeeze(-1).reshape(batch_size, aggregation_tensor_sampled.shape[1]) # batch  x n_sampling
        
        if not training:
            fps_sampling_inds3 = []
            similarity_score_filter = []
            for b in range(batch_size):
                scene_candidate_inds = torch.nonzero((similarity_score[b,...].sigmoid() >= cfg.similarity_thresh))
                # print(scene_candidate_inds)
                fps_sampling_b = query_sampling_inds[b][scene_candidate_inds.long()].squeeze(-1)
                # print('fps_sampling_b', fps_sampling_b.shape)
                fps_sampling_inds3.append(fps_sampling_b)
                similarity_score_filter.append(similarity_score[b, scene_candidate_inds].sigmoid().squeeze(-1))
            query_sampling_inds = torch.stack(fps_sampling_inds3)
            similarity_score_filter = torch.stack(similarity_score_filter)
            # print('before/after: ', similarity_score.shape, similarity_score_filter.shape)
            if query_sampling_inds.shape[1] == 0:
                outputs['proposal_scores']  = None
                return outputs

            
            query_locs = [torch.gather(query_locs[..., x], 1, query_sampling_inds) for x in range(3)]
            query_locs = torch.stack(query_locs)
            query_locs = query_locs.permute(1, 2, 0)
            
            query_embedding_pos_T = query_embedding_pos.transpose(1,2)
            query_embedding_pos_T = [torch.gather(query_embedding_pos_T[..., x], 1, query_sampling_inds) for x in range(query_embedding_pos_T.shape[-1])]
            query_embedding_pos_T = torch.stack(query_embedding_pos_T)
            query_embedding_pos = query_embedding_pos_T.permute(1, 0, 2)

            geo_dist_arr_filtered = []
            for b in range(batch_size):
                query_sampling_inds_b = query_sampling_inds[b]
                geo_dist= geo_dist_arr[b]

                geo_dist = geo_dist[query_sampling_inds_b, :]
                geo_dist_arr_filtered.append(geo_dist)
            geo_dist_arr = geo_dist_arr_filtered

            
        context_embedding_pos = self.pos_embedding(context_locs, input_range=pc_dims)

        # context_feats = self.encoder_to_decoder_projection(
        #     aggregation_tensor.permute(0, 2, 1)
        # ) # batch x channel x npoints
        
        context_feats = self.encoder_to_decoder_projection(
            context_feats.permute(0, 2, 1)
        ) # batch x channel x npoints

        ''' Init dec_inputs by query features '''
        context_feats_T = context_feats.transpose(1,2) # batch x npoints x channel 
        dec_inputs      = [torch.gather(context_feats_T[..., x], 1, query_sampling_inds) for x in range(context_feats_T.shape[-1])]
        dec_inputs      = torch.stack(dec_inputs) # channel x batch x npoints
        dec_inputs      = dec_inputs.permute(2,1,0) # npoints x batch x channel

        # decoder expects: npoints x batch x channel
        context_embedding_pos   = context_embedding_pos.permute(2, 0, 1)
        context_feats           = context_feats.permute(2, 0, 1)
        query_embedding_pos     = query_embedding_pos.permute(2, 0, 1)

        # Encode relative pos
        relative_coords = torch.abs(query_locs[:,:,None,:] - context_locs[:,None,:,:])
        n_queries, n_contexts = relative_coords.shape[1], relative_coords.shape[2]
        relative_embbeding_pos = self.pos_embedding(relative_coords.reshape(batch_size, n_queries*n_contexts, -1), input_range=pc_dims).reshape(batch_size, -1, n_queries, n_contexts,)
        relative_embbeding_pos   = relative_embbeding_pos.permute(2,3,0,1)

        # num_layers x n_queries x batch x channel
        dec_outputs = self.decoder(
            tgt=dec_inputs, 
            memory=context_feats, 
            pos=context_embedding_pos, 
            query_pos=query_embedding_pos,
            relative_pos=relative_embbeding_pos
        )

        if not training:
            dec_outputs = dec_outputs[-1:,...]

        if len(fg_idxs) == 0:
            # outputs['mask_logits']  = None
            outputs['proposal_scores'] = None
            return outputs

        mask_predictions = self.get_mask_prediction(geo_dist_arr, dec_outputs, mask_features_, locs_float_, query_locs, batch_offsets_)

        
        if training:
            outputs['fg_idxs']              = fg_idxs
            outputs['num_insts']            = cfg.n_query_points * batch_size
            outputs['batch_idxs']           = batch_idxs_
            outputs['query_sampling_inds']  = query_sampling_inds
            outputs['simnet']               = similarity_score
            outputs['mask_predictions']     = mask_predictions
        
        if not training:
            mask_prediction_last_layer = mask_predictions[-1]
            mask_logit_final = mask_prediction_last_layer['mask_logits'] #.reshape(batch_size, cfg.n_query_points, -1) # batch x n_queries x N_mask
            # cls_logit_final = mask_prediction_last_layer['cls_logits'] #.reshape(batch_size, cfg.n_query_points, -1) # batch x n_queries x n_classes
            proposal_idx, proposal_len, scores, seg_preds = self.generate_proposal(geo_dist_arr, mask_logit_final, similarity_score_filter, fg_idxs, batch_offsets, 
                                                threshold=0.2, min_pts_num=100)
            outputs['proposal_scores'] = (scores, proposal_idx, proposal_len, seg_preds)
        return outputs

    def forward_extract(self, scene_dict, scene_infos, fold=0):
        outputs = {}

        outputs     = {}
        batch_idxs  = scene_dict['locs'][:, 0].int()
        p2v_map     = scene_dict['p2v_map']
        locs_float  = scene_dict['locs_float']
        batch_offsets = scene_dict['batch_offsets']


        batch_size = len(batch_offsets) - 1

        pc_dims = [
            scene_dict["pc_mins"],
            scene_dict["pc_maxs"],
        ]

        N_points = locs_float.shape[0]

        sparse_input = self.preprocess_input(scene_dict, batch_size)

        ''' Backbone net '''
        output = self.input_conv(sparse_input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[p2v_map.long()]
        output_feats = output_feats.contiguous()


        ''' Semantic head'''
        semantic_feats  = self.semantic(output_feats)
        semantic_scores = self.semantic_linear(semantic_feats)   # (N, nClass), float
        semantic_preds  = semantic_scores.max(1)[1]    # (N), long

        outputs['semantic_scores'] = semantic_scores

        if fold == 0:
            fg_condition = semantic_preds >= 4
        else:
            fg_condition = semantic_preds == 3
            
        fg_idxs         = torch.nonzero(fg_condition).view(-1)

        batch_idxs_     = batch_idxs[fg_idxs]
        batch_offsets_  = utils.get_batch_offsets(batch_idxs_, batch_size)
        locs_float_     = locs_float[fg_idxs]
        output_feats_   = output_feats[fg_idxs]
        semantic_preds_ = semantic_preds[fg_idxs]


        # 1-dim: n_queries*batch_size
        context_locs = []
        context_feats = []
        pre_enc_inds_arr = []
        for b in range(batch_size):
            start = batch_offsets_[b]
            end = batch_offsets_[b+1]
            locs_float_b = locs_float_[start:end, :]
            output_feats_b = output_feats_[start:end, :]
            batch_points = (end - start).cpu().item()

            if batch_points == 0:
                print('bug')

                outputs['mask_predictions']  = None
                outputs['proposal_scores']  = None
                return outputs


            locs_float_b = locs_float_b.unsqueeze(0)
            output_feats_b = output_feats_b.unsqueeze(0)

            context_locs_b, grouped_features, grouped_xyz, pre_enc_inds = self.set_aggregator.group_points(locs_float_b.contiguous(), 
                                                            output_feats_b.transpose(1,2).contiguous())
            context_feats_b = self.set_aggregator.mlp(grouped_features, grouped_xyz)
            # query_xyz1_b, output_feats1_b, pre_enc_inds = self.set_aggregator(locs_float_b.contiguous(), 
            #                                                 output_feats_b.transpose(1,2).contiguous())
            context_feats_b = context_feats_b.transpose(1,2)

            context_locs.append(context_locs_b)
            context_feats.append(context_feats_b)
            pre_enc_inds_arr.append(pre_enc_inds)


        context_locs = torch.cat(context_locs)
        context_feats = torch.cat(context_feats) # batch x npoint x channel


        query_locs, query_embedding_pos, query_sampling_inds = self.sample_query_embedding(context_locs, pc_dims, cfg.n_query_points)

        for b in range(batch_size):
            start = batch_offsets_[b]
            end = batch_offsets_[b+1]
            s_info = scene_infos[b]['query_scene']

            s_dict = {
                'query_sampling_inds': query_sampling_inds[b].detach().cpu().numpy(),
                'pre_enc_inds_arr': pre_enc_inds_arr[b].detach().cpu().numpy(),
                'query_locs': query_locs[b].detach().cpu().numpy(),
                'locs_float_': locs_float_[start:end, :].detach().cpu().numpy()

            }
            print("save", s_info, len(self.save_dict.keys()))
            self.save_dict[s_info] = s_dict

        return None