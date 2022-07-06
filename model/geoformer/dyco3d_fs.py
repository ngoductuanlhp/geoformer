import torch
import torch.nn as nn
import spconv
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
import time
from model.geoformer.geoformer_modules import ResidualBlock, VGGBlock, UBlock, conv_with_kaiming_uniform

class DyCo3dFS(nn.Module):
    def __init__(self):
        super().__init__()

        input_c = cfg.input_channel
        m = cfg.m

        # FIXME check classes
        classes = cfg.classes

        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs

        # self.pretrain_path = cfg.pretrain_path
        # self.pretrain_module = cfg.pretrain_module
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
            before_embedding_tower.append(conv_block(m, m))
        before_embedding_tower.append(conv_block(m, self.output_dim))
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



        #### offset
        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### score branch
        self.score_unet = UBlock([m, 2*m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        self.apply(self.set_bn_init)

        self.threshold_ins = cfg.threshold_ins
        self.min_pts_num = cfg.min_pts_num
        #### fix parameter
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'semantic': self.semantic, 'semantic_linear': self.semantic_linear, 
                      'offset': self.offset, 'offset_linear': self.offset_linear,
                      'score_unet': self.score_unet, 'score_outputlayer': self.score_outputlayer,
                      'mask_tower': self.mask_tower}

        # ANCHOR Freeze backbone
        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

        self.cache_data1 = None
        self.cache_data2 = None


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def get_features(self, inp, inp_map):
        # ANCHOR feature extraction
        output = self.input_conv(inp)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[inp_map.long()]
        output_feats = output_feats.contiguous()
        return output_feats


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean0 = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean0, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map, clusters_coords_mean0

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



    def mask_heads_forward(self, mask_features, weights, biases, inst_batch_id, clusters_coords_mean, coords_, use_coords=True):
        num_insts = inst_batch_id.size(0)
        assert mask_features.dim() == 3
        n_layers = len(weights)
        c = mask_features.size(1)
        n_mask = mask_features.size(0)
        x = mask_features.permute(2,1,0).repeat(num_insts, 1, 1) ### num_inst * c * N_mask

        relative_coords = clusters_coords_mean.reshape(-1, 1, 3) - coords_.reshape(1, -1, 3) ### N_inst * N_mask * 3
        relative_coords = relative_coords.permute(0,2,1) ### num_inst * 3 * n_mask

        if use_coords:
            x = torch.cat([relative_coords, x], dim=1) ### num_inst * (3+c) * N_mask

        x = x.reshape(1, -1, n_mask) ### 1 * (num_inst*c') * Nmask
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)

        return x



    def get_instance_batch_id(self, batch_ids, inst_idx, inst_offsets):
        inst_num = inst_offsets.size(0) - 1
        inst_batch_id = torch.zeros(inst_num).int().cuda()
        for i in range(inst_num):
            start = inst_offsets[i].item()
            end = inst_offsets[i+1].item()
            pts_ids = inst_idx[start:end, 1]
            inst_batch_id[i] = batch_ids[pts_ids[0].long()]
            if batch_ids[pts_ids.long()].unique().size(0) > 1:
                assert RuntimeError
        return inst_batch_id

    def generate_proposal(self, mask_logits, batch_id, threshold, seg_pred, inst_pred_seg_label=None, min_pts_num=50):
        n_inst = mask_logits.size(0)
        proposal_len = []
        proposal_len.append(0)
        proposal_idx = []
        num = 0
        scores = []
        for n in range(n_inst):
            # proposal_id_n = ((mask_logits[n] > threshold) & (seg_pred == inst_pred_seg_label[n].item())).nonzero().squeeze(dim=1)
            proposal_id_n = (mask_logits[n] > threshold).nonzero().squeeze(dim=1)
            score = mask_logits[n][proposal_id_n].mean()
            # seg_label = inst_pred_seg_label[n]
            if proposal_id_n.size(0) < min_pts_num:
                continue
            proposal_id_n = batch_id[proposal_id_n.long()].unsqueeze(dim=1)
            id_proposal_id_n = torch.cat([torch.ones_like(proposal_id_n)*num, proposal_id_n], dim=1)
            num += 1
            tmp = proposal_len[-1]
            proposal_len.append(proposal_id_n.size(0)+tmp)
            proposal_idx.append(id_proposal_id_n)
            scores.append(score)

        if len(proposal_idx) == 0:
            return proposal_idx, proposal_len, scores
        proposal_idx = torch.cat(proposal_idx, dim=0)
        proposal_len = torch.from_numpy(np.array(proposal_len)).cuda()
        scores = torch.stack(scores)
        # scores = torch.from_numpy(np.array(scores, dtype=np.float32)).cuda()
        return proposal_idx, proposal_len, scores


    def get_instance_seg_pred_label(self, semantic_label, proposals_idx, proposals_shift):
        instance_num = proposals_shift.size(0) - 1
        seg_labels = []
        for n in range(instance_num):
            start = proposals_shift[n].item()
            end = proposals_shift[n+1].item()
            ins_ids_n = proposals_idx[start:end, 1]
            seg_label_n = torch.mode(semantic_label[ins_ids_n.long()])[0].item()
            seg_labels.append(seg_label_n)

        return torch.from_numpy(np.array(seg_labels, dtype=np.int32)).cuda()

    def process_support(self, support_dict, training=True):
        batch_size      = cfg.batch_size if training else 1
        with torch.no_grad():
        # self.eval()
        # p2v: point i -> voxel jth
        # vp2: voxel j -> list of activate point
            v2p_map = support_dict['v2p_map'].cuda()
            p2v_map = support_dict['p2v_map'].cuda()
            voxel_locs = support_dict['voxel_locs'].cuda()

            coords_float = support_dict['locs_float'].cuda()              # (N, 3), float32, cuda
            feats = support_dict['feats'].cuda()                          # (N, C), float32, cuda
            support_mask = support_dict['support_masks'].cuda()                        # (N), long, cuda
            batch_offsets = support_dict['batch_offsets'].cuda()  
            mask_offsets = support_dict['mask_offsets'].cuda()             # (B + 1), int, cuda
            spatial_shape = support_dict['spatial_shape']

            
            if cfg.use_coords:
                feats = torch.cat((feats, coords_float), 1)


            voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

            input_ = spconv.SparseConvTensor(voxel_feats, voxel_locs.int(), spatial_shape, batch_size)

            # print(input_.spatial_shape)
            support_feats = self.get_features(input_, p2v_map)

            # masking
            # print(support_feats.shape, support_mask.shape)
            # print("Support mask", torch.unique(support_mask), torch.count_nonzero(support_mask), support_mask.shape)
            # support_feats = support_feats * support_mask.unsqueeze(-1)
            mask_indices = torch.nonzero(support_mask==1).view(-1)
            support_feats = support_feats[mask_indices]

            # average pooling
            support_embeddings = pointgroup_ops.sec_mean(support_feats, mask_offsets)
                # self.train()
            # print("Support", support_embeddings.shape)
            return support_embeddings

    def filter_clusters(self, query_feats, support_embeddings,  cluster_batch_idxs, cluster_embeddings, proposals_idx_shift, proposals_offset_shift, batch_idxs):
        # print("DEBUG", query_feats.shape, support_embeddings.shape)
        # print("Num clusters", proposals_offset_shift.shape[0] -1)
        # print("num of cluster id", proposals_idx_shift[:, 1].long())
        # print(proposals_idx_shift.shape)
        proposals_idx_shift = proposals_idx_shift.cuda()
        proposals_offset_shift = proposals_offset_shift.cuda()
        # support_embeddings = support_embeddings.unsqueeze(0)
        broadcasted_support_embeddings = torch.index_select(support_embeddings, 0, cluster_batch_idxs.long())
        # print('cluster_embeddings', cluster_embeddings.shape, broadcasted_support_embeddings.shape)
        cosine_similarities = F.cosine_similarity(cluster_embeddings, broadcasted_support_embeddings)
        # support_embeddings = support_embeddings.repeat(cluster_embeddings.shape[0], 1)
        # cosine_similarities = F.cosine_similarity(cluster_embeddings, support_embeddings)
        # print('Cosine', torch.max(cosine_similarities), torch.min(cosine_similarities))
        # print("shape", cosine_similarities.shape, cluster_embeddings.shape)
        # print('cosine_similarities', torch.isinf(cosine_similarities[0]))
        cluster_idx = torch.nonzero((cosine_similarities >= cfg.cosine_thresh).long())
        # print("Filter cluster: old = {}, new = {}".format(proposals_offset_shift.shape[0] - 1, cluster_idx.shape[0]))
        cluster_idx = cluster_idx.squeeze(-1)
        if cluster_idx.shape[0] == 0:
            return False, None, None, None
        proposals_idx_shift2 = []
        proposals_offset_shift2 = [0]
        num = 0
        for idx in cluster_idx:
            indices = proposals_idx_shift[proposals_offset_shift[idx]:proposals_offset_shift[idx+1],1].unsqueeze(-1)
            proposals_idx_shift2.append(torch.cat([torch.LongTensor(indices.shape[0], 1).fill_(num).cuda(), indices], 1))
            proposals_offset_shift2.append(proposals_offset_shift2[-1] + proposals_offset_shift[idx+1]-proposals_offset_shift[idx])
            num += 1
        
        proposals_idx_shift2 = torch.cat(proposals_idx_shift2, 0)
        proposals_offset_shift2 = torch.tensor(proposals_offset_shift2, dtype=torch.int) 
        # print(proposals_idx_shift2.shape, proposals_offset_shift2.shape, proposals_offset_shift2[-1])
        return True, proposals_idx_shift2.cpu(), proposals_offset_shift2.cpu(), cluster_idx
        

    def process_query(self, query_dict, batch_size):
        # with torch.no_grad():
        v2p_map = query_dict['v2p_map'].cuda()
        p2v_map = query_dict['p2v_map'].cuda()
        voxel_locs = query_dict['voxel_locs'].cuda()

        coords = query_dict['locs'].cuda() 
        coords_float = query_dict['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = query_dict['feats'].cuda()                          # (N, C), float32, cuda
        batch_offsets = query_dict['batch_offsets'].cuda()                # (B + 1), int, cuda
        spatial_shape = query_dict['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_locs.int(), spatial_shape, batch_size)
        
        query_feats = self.get_features(input_, p2v_map)

        return query_feats, coords, coords_float, batch_offsets

    def set_aggregation(self, cluster_idxs, cluster_offsets, semantic_pred, shifted_coords):
        # cluster_idxs Nx2: 0 for cluster id, 1 for point id
        # cluster_offsets: num_cluster + 1
        # print("Check shape", semantic_pred.shape, shifted_coords.shape)
        primary_inst_thresh_num = 50

        primary_insts = []
        fragments = []

        primary_inst_list = []
        prim_num = 0
        for idx in range(len(cluster_offsets)-1):
            num_points = cluster_offsets[idx+1] - cluster_offsets[idx]

            point_idxs = cluster_idxs[cluster_offsets[idx]:cluster_offsets[idx+1], 1]
            # print('point_idxs', point_idxs)
            center = torch.mean(shifted_coords[point_idxs.long(),:], dim=0)
            semantic = torch.mode(semantic_pred[point_idxs.long()])[0].item()

            if num_points >= primary_inst_thresh_num:
                r_size = 0.01 * torch.sqrt(torch.FloatTensor([num_points]))
                # print('r_size', r_size)
                r_set = max(r_size, 0.1)
                prim_inst = (prim_num, center, semantic, r_set)
                primary_insts.append(prim_inst)
                primary_inst_list.append(torch.cat([torch.LongTensor(num_points, 1).fill_(prim_num), point_idxs.unsqueeze(-1)], axis=1))

                prim_num += 1
            else:
                frag = (center, semantic, point_idxs)
                fragments.append(frag)

        # print("num fragments", len(fragments))
        for i, frag in enumerate(fragments):
            prim_idx = -1
            min_dis = 1000000
            r_set_prim = -1

            frag_center, frag_semantic, point_idxs = frag
            for j, prim_inst in enumerate(primary_insts):
                
                prim_inst_idx, prim_inst_center, prim_inst_semantic, prim_inst_r_set = prim_inst
                current_dist = torch.dist(frag_center, prim_inst_center, 2).cpu()
                if frag_semantic == prim_inst_semantic and current_dist < min_dis:
                # if current_dist < min_dis:
                    prim_idx = prim_inst_idx
                    min_dis = current_dist
                    r_set_prim = prim_inst_r_set
            
            # union
            if r_set_prim != -1 and min_dis < r_set_prim:
                # print("Union ", min_dis, r_set_prim)
                prim_inst = primary_inst_list[prim_idx]
                frag_inst = torch.cat([torch.LongTensor(point_idxs.shape[0], 1).fill_(prim_idx), point_idxs.unsqueeze(-1)], axis=1)
                prim_inst = torch.cat([prim_inst, frag_inst], axis=0)

                primary_inst_list[prim_idx] = prim_inst

        cluster_offsets_new = [0]
        for idx in range(len(primary_inst_list)):
            prim_inst = primary_inst_list[idx]
            cluster_offsets_new.append(cluster_offsets_new[-1] + prim_inst.shape[0])

        cluster_idxs_new = torch.cat(primary_inst_list, axis=0)
        cluster_offsets_new = torch.IntTensor(cluster_offsets_new)
        print('Set aggregation: before: {} | after: {}'.format(cluster_offsets.shape[0]-1, cluster_offsets_new.shape[0]-1))
        return cluster_idxs_new, cluster_offsets_new


    def train_similarity(self, query_feats, support_embeddings, batch_idxs, query_semantic_labels, query_instance_labels, support_classes):
        unique_insts = torch.unique(query_instance_labels, sorted=True).tolist()[1:] # ignore -100
        n_instance = len(unique_insts)
        instance_batch_idxs = torch.LongTensor(n_instance).cuda()

        inst_embeddings = []
        seg_preds = []
        for i, inst in enumerate(unique_insts):
            inst_idxs = torch.nonzero(query_instance_labels == inst)
            # num_idxs = torch.count_nonzero(query_instance_labels == inst)
            inst_embed = torch.mean(query_feats[inst_idxs], dim=0, keepdim=False) # 1x16

            inst_embeddings.append(inst_embed)

            batch_id = batch_idxs[inst_idxs[0].long()]
            instance_batch_idxs[i] = batch_id
            
            seg_preds.append(query_semantic_labels[inst_idxs[0].long()])

        seg_preds = torch.FloatTensor(seg_preds).cuda()

        inst_embeddings = torch.cat(inst_embeddings, dim=0).cuda()

        # print('Debug', inst_embeddings.shape)
        broadcasted_support_embeddings = torch.index_select(support_embeddings, 0, instance_batch_idxs.long())

        channel_wise_tensor = inst_embeddings * broadcasted_support_embeddings
        subtraction_tensor = inst_embeddings - broadcasted_support_embeddings

        aggregation_tensor = torch.cat([channel_wise_tensor, subtraction_tensor, inst_embeddings], dim=1)
        similarity_score = self.similarity_net(aggregation_tensor).squeeze(-1)

        seg_labels = torch.index_select(support_classes, 0, instance_batch_idxs.long())
        # print('support_classes', support_classes)

        similarity_mask = (seg_preds == seg_labels).float()
        similarity_loss = (similarity_score, similarity_mask)

        print("DEBUG: num inst {}, num positive: {}".format(n_instance, torch.count_nonzero(similarity_mask)))
        # simi_pred = similarity_score.detach().sigmoid()
        # print("DEBUG similarity", torch.count_nonzero(simi_pred >= 0.5), simi_pred.shape[0])
        return True, None, None, similarity_loss


        

    def cal_similarity(self, query_feats, support_embeddings, cluster_batch_idxs, cluster_embeddings, proposals_idx_shift, proposals_offset_shift, batch_idxs, query_semantic_labels, support_classes, training=False):
        proposals_idx_shift = proposals_idx_shift.cuda()
        proposals_offset_shift = proposals_offset_shift.cuda() # N_cluster + 1

        # cluster_feats = query_feats[proposals_idx_shift[:, 1].long()]
        # cluster_embeddings = pointgroup_ops.sec_mean(cluster_feats, proposals_offset_shift) # N_cluster x 16

        # cluster_batch_idxs = torch.LongTensor(cluster_embeddings.shape[0]).cuda()
        # for i in range(proposals_offset_shift.shape[0]-1):
        #     batch_id = batch_idxs[proposals_idx_shift[proposals_offset_shift[i].long(), 1].long()]
        #     # cluster_batch_idxs[proposals_offset_shift[i]:proposals_offset_shift[i+1]] = batch_id
        #     cluster_batch_idxs[i] = batch_id

        broadcasted_support_embeddings = torch.index_select(support_embeddings, 0, cluster_batch_idxs.long())

        # print("DEBUG", cluster_embeddings.shape, broadcasted_support_embeddings.shape)
        channel_wise_tensor = cluster_embeddings * broadcasted_support_embeddings
        subtraction_tensor = cluster_embeddings - broadcasted_support_embeddings

        aggregation_tensor = torch.cat([channel_wise_tensor, subtraction_tensor, cluster_embeddings], dim=1)
        similarity_score = self.similarity_net(aggregation_tensor).squeeze(-1)

        if training:
            seg_labels = torch.index_select(support_classes, 0, cluster_batch_idxs.long())
            # similarity_mask = torch.zeros_like(seg_labels, dtype=torch.float32).cuda()
            seg_preds = []
            for n in range(proposals_offset_shift.shape[0]-1):
                start = proposals_offset_shift[n].item()
                end = proposals_offset_shift[n+1].item()
                ins_ids_n = proposals_idx_shift[start:end, 1]
                # semantic = query_semantic_labels[ins_ids_n.long()]
                # semantic_unique = torch.unique(semantic)
                # if seg_labels[n] in semantic_unique:
                #     ratio = torch.count_nonzero(semantic==seg_labels[n]) / semantic.shape[0]
                #     if ratio >= 0.3:
                #         similarity_mask[n] = 1

                seg_pred_n = torch.mode(query_semantic_labels[ins_ids_n.long()])[0].item()
                seg_preds.append(seg_pred_n)

            seg_preds = torch.FloatTensor(seg_preds).cuda()
            
            similarity_mask = (seg_preds == seg_labels).float()
            similarity_loss = (similarity_score, similarity_mask)
            print("DEBUG: num cluster {}, num positive: {}".format(proposals_offset_shift.shape[0]-1, torch.count_nonzero(similarity_mask)))

            # simi_pred = similarity_score.detach().sigmoid()
            # print("DEBUG similarity", torch.count_nonzero(simi_pred >= 0.5), simi_pred.shape[0])
            return True, None, None, None, similarity_loss

        

        cluster_idx = torch.nonzero((similarity_score.sigmoid() >= cfg.similarity_thresh).long())
        # print("DEBUG similarity", torch.count_nonzero(simi_pred >= 0.5), simi_pred.shape[0])
        # print("Filter cluster: old = {}, new = {}".format(proposals_offset_shift.shape[0] - 1, cluster_idx.shape[0]))
        cluster_idx = cluster_idx.squeeze(-1)
        if cluster_idx.shape[0] == 0:
            return False, None, None, None, None
        proposals_idx_shift2 = []
        proposals_offset_shift2 = [0]
        num = 0
        
        for idx in cluster_idx:
            indices = proposals_idx_shift[proposals_offset_shift[idx]:proposals_offset_shift[idx+1],1].unsqueeze(-1)
            proposals_idx_shift2.append(torch.cat([torch.LongTensor(indices.shape[0], 1).fill_(num).cuda(), indices], 1))
            proposals_offset_shift2.append(proposals_offset_shift2[-1] + proposals_offset_shift[idx+1]-proposals_offset_shift[idx])
            num += 1
        
        proposals_idx_shift2 = torch.cat(proposals_idx_shift2, 0)
        proposals_offset_shift2 = torch.tensor(proposals_offset_shift2, dtype=torch.int) 
        # print(proposals_idx_shift2.shape, proposals_offset_shift2.shape, proposals_offset_shift2[-1])

        
        return True, proposals_idx_shift2.cpu(), proposals_offset_shift2.cpu(), cluster_idx, None



    def forward(self, support_dict, query_dict, epoch=200, ins_sample_num=70, training=True, remember=False, support_embeddings=None):
        outputs = {}
        batch_size = cfg.batch_size if training else 1

        if remember:
            query_feats, coords, coords_float, batch_offsets, batch_idxs, semantic_scores, pt_offsets, semantic_preds = self.cache_data1
        else:
            query_feats, coords, coords_float, batch_offsets = self.process_query(query_dict, batch_size)
            batch_idxs = coords[:, 0].int()
            semantic_feats = self.semantic(query_feats)
            semantic_scores = self.semantic_linear(semantic_feats)   # (N, nClass), float
            semantic_preds = semantic_scores.sigmoid().max(1)[1]
            # semantic_preds = (semantic_scores.sigmoid() >= 0).long()
            #### offset
            pt_offsets_feats = self.offset(query_feats)
            pt_offsets = self.offset_linear(pt_offsets_feats)   # (N, 3), float32
            self.cache_data1 = (query_feats, coords, coords_float, batch_offsets, batch_idxs, semantic_scores, pt_offsets, semantic_preds)


        if support_embeddings is None:
            # print("process suport")
            support_embeddings = self.process_support(support_dict, training=training)
        else:
            support_embeddings = support_embeddings.to(query_feats.device)
        
        outputs['semantic_scores'] = semantic_scores
        outputs['pt_offsets'] = pt_offsets

        if training:
            instance_labels = query_dict['instance_labels'].cuda() 
            instance_info = query_dict['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
            instance_pointnum = query_dict['instance_pointnum'].cuda()  # (total_nInst), int, cuda
            semantic_labels = query_dict['labels'].cuda()

            outputs['semantic_scores'] = (semantic_scores, semantic_labels)
            outputs['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)

        if(epoch > self.prepare_epochs) or not training:
            # FIXME get candidate test class
            if cfg.train_fold == cfg.cvfold:
                conditions = semantic_preds >= 4
            else:
                conditions = semantic_preds >= 3
            object_idxs         = torch.nonzero(conditions).view(-1)
            
            if object_idxs.shape[0] == 0:
                outputs['proposal_scores'] = None
                outputs['mask_logits'] = None
                return outputs
            # print(object_idxs)
            batch_idxs_ = batch_idxs[object_idxs]
            # print(batch_idxs_.shape, batch_idxs.shape)
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            semantic_preds_ = semantic_preds[object_idxs]

            # FIXME get all
            # semantic_preds_ = torch.ones_like(semantic_preds_).cuda() * 3
            semantic_preds_cpu = semantic_preds_.int().cpu()

            if remember:
                proposals_idx_shift, proposals_offset_shift, cluster_embeddings, cluster_batch_idxs = self.cache_data2
                # proposals_idx_shift, proposals_offset_shift, cluster_batch_idxs, cluster_embeddings, params_feats, clusters_coords_mean = self.cache_data2
                # proposals_idx_shift, proposals_offset_shift, cluster_batch_idxs, cluster_embeddings, inst_batch_id, mask_logits = self.cache_data2
            else:
                idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
                proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                
                proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()

                cluster_feats = query_feats[proposals_idx_shift[:, 1].long().cuda()]
                cluster_embeddings = pointgroup_ops.sec_mean(cluster_feats, proposals_offset_shift.cuda()) # N_cluster x 16

                cluster_batch_idxs = torch.LongTensor(cluster_embeddings.shape[0]).cuda()
                for i in range(proposals_offset_shift.shape[0]-1):
                    batch_id = batch_idxs[proposals_idx_shift[proposals_offset_shift[i].long().cuda(), 1].long()]
                    cluster_batch_idxs[i] = batch_id

                self.cache_data2 = (proposals_idx_shift, proposals_offset_shift, cluster_embeddings, cluster_batch_idxs)

            # ANCHOR filter out cluster
            filter_valid, proposals_idx_shift, proposals_offset_shift, cluster_idx = self.filter_clusters(query_feats, support_embeddings, cluster_batch_idxs, cluster_embeddings, proposals_idx_shift, proposals_offset_shift, batch_idxs)

            if not filter_valid or proposals_offset_shift.shape[0] <= 2:
                outputs['object_idxs'] = object_idxs

                outputs['proposals_offset_shift'] = proposals_offset_shift
                outputs['proposals_idx_shift'] = proposals_idx_shift
                # ret['inst_batch_id'] = inst_batch_id
                outputs['batch_idxs'] = batch_idxs_
                outputs['proposal_scores'] = None
                outputs['cluster_filter']  = []
                outputs['score_bug'] = True
                outputs['mask_logits'] = None

                return outputs
            
            outputs['score_bug'] = False

            # ANCHOR set aggregation:
            # proposals_idx_shift, proposals_offset_shift = self.set_aggregation(proposals_idx_shift, proposals_offset_shift, semantic_preds, coords_float + pt_offsets)
                

            # FIXME DEBUG
            # cluster_filter = torch.LongTensor(query_feats.shape[0], 1).fill_(-100)
            # for idx in range(proposals_offset_shift.shape[0]-1):
            #     cluster_filter[proposals_idx_shift[proposals_offset_shift[idx]:proposals_offset_shift[idx+1],1].long()] = idx + 1
            # ret['cluster_filter'] = cluster_filter.cpu()

            input_feats, inp_map, clusters_coords_mean = self.clusters_voxelization(proposals_idx_shift, proposals_offset_shift, query_feats, coords_float, self.score_fullscale, self.score_scale, self.mode)

            # ### to generate weights
            params = self.score_unet(input_feats)
            params = self.score_outputlayer(params)
            params_feats = params.features[inp_map.long()] # (sumNPoint, C)
            params_feats = pointgroup_ops.roipool(params_feats, proposals_offset_shift.cuda())  # (nProposal, C)
            # print('cluster_idx', cluster_idx.shape)
            # params_feats            = torch.index_select(params_feats, 0, cluster_idx)
            # clusters_coords_mean    = torch.index_select(clusters_coords_mean, 0, cluster_idx)
            if len(params_feats) > ins_sample_num and ins_sample_num >0:
                params_feats = params_feats[:ins_sample_num]
                proposals_offset_shift = proposals_offset_shift[:ins_sample_num+1]
                clusters_coords_mean = clusters_coords_mean[:ins_sample_num]

            inst_batch_id = self.get_instance_batch_id(batch_idxs, proposals_idx_shift, proposals_offset_shift)
            # # inst_pred_seg_label = self.get_instance_seg_pred_label(semantic_preds, proposals_idx_shift, proposals_offset_shift)


            before_embedding_feature = self.before_embedding_tower(torch.unsqueeze(params_feats, dim=2))
            controller = self.controller(before_embedding_feature).squeeze(dim=2)

            weights, biases = self.parse_dynamic_params(controller, self.output_dim)

            mask_features = self.mask_tower(torch.unsqueeze(query_feats, dim=2).permute(2,1,0)).permute(2,1,0)
            mask_features = mask_features[object_idxs]

            # n_inst = len(params_feats)
            mask_logits = self.mask_heads_forward(mask_features, weights, biases, inst_batch_id, clusters_coords_mean, coords_, use_coords=self.use_coords)
            
            # print("DEvice", cluster_idx.device, mask_logits.device)
            # print('mask_logits', mask_logits.shape, mask_logits)
            # mask_logits            = torch.index_select(mask_logits.squeeze(dim=0), 0, cluster_idx)
            # inst_batch_id    = torch.index_select(inst_batch_id, 0, cluster_idx)

            
            n_inst = mask_logits.shape[0]
            if n_inst > ins_sample_num and ins_sample_num >0:
                mask_logits = mask_logits[:ins_sample_num]
                proposals_offset_shift = proposals_offset_shift[:ins_sample_num+1]
                clusters_coords_mean = clusters_coords_mean[:ins_sample_num]

            outputs['mask_logits'] = mask_logits.squeeze(dim=0) ### N_inst * N_mask

            outputs['object_idxs'] = object_idxs

            outputs['proposals_offset_shift'] = proposals_offset_shift
            outputs['proposals_idx_shift'] = proposals_idx_shift
            outputs['inst_batch_id'] = inst_batch_id
            outputs['batch_idxs'] = batch_idxs_
            outputs['batch_offsets'] = batch_offsets

            if not training:
                ### generate proposal idx
                proposal_idx, proposal_len, scores = self.generate_proposal(mask_logits.squeeze(dim=0).sigmoid(), object_idxs,
                                                                            threshold=0.5, seg_pred=semantic_preds_, inst_pred_seg_label=None,
                                                                            min_pts_num=50)
                seg_preds = None
                outputs['proposal_scores'] = (scores, proposal_idx, proposal_len, seg_preds)


        return outputs