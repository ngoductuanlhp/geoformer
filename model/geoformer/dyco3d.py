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

class DyCo3d(nn.Module):
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


        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, use_backbone_transformer=True, indice_key_id=1)

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

    def preprocess_input(self, batch_input):
        voxel_coords = batch_input['voxel_locs']              # (M, 1 + 3), long, cuda
        v2p_map = batch_input['v2p_map']                     # (M, 1 + maxActive), int, cuda
        locs_float = batch_input['locs_float']              # (N, 3), float32, cuda
        feats = batch_input['feats']                        # (N, C), float32, cuda
        spatial_shape = batch_input['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, locs_float), 1).float()

        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda
        sparse_input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        return sparse_input


    def forward(self, batch_input, epoch, ins_sample_num=70, training=True):
        outputs = {}
        batch_size = cfg.batch_size if training else 1

        batch_idxs  = batch_input['locs'][:, 0].int()
        p2v_map     = batch_input['p2v_map']
        locs_float  = batch_input['locs_float']
        batch_offsets = batch_input['offsets']

        sparse_input = self.preprocess_input(batch_input)

        ''' Backbone net '''
        output = self.input_conv(sparse_input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[p2v_map.long()]
        output_feats = output_feats.contiguous()

        ''' Semantic head'''
        semantic_feats = self.semantic(output_feats)
        semantic_scores = self.semantic_linear(semantic_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]    # (N), long

        ''' Offset head'''
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)   # (N, 3), float32

        outputs['semantic_scores'] = semantic_scores
        outputs['pt_offsets'] = pt_offsets

        if epoch <= self.prepare_epochs:
            return outputs
        
        # FIXME get candidate test class
        if cfg.train_fold == cfg.cvfold:
            conditions = semantic_preds >= 4
        else:
            conditions = semantic_preds == 3
        object_idxs         = torch.nonzero(conditions).view(-1)
        
        if object_idxs.shape[0] == 0:
            outputs['mask_logits'] = None
            return outputs


        batch_idxs_ = batch_idxs[object_idxs]
        batch_offsets_ = utils.get_batch_offsets(batch_idxs_, batch_size)
        coords_ = locs_float[object_idxs]
        pt_offsets_ = pt_offsets[object_idxs]
        semantic_preds_ = semantic_preds[object_idxs]
        semantic_preds_cpu = semantic_preds_.int().cpu()


        idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
        proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
        
        proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()

        if proposals_offset_shift.shape[0] <= 2:
            outputs['mask_logits'] = None
            return outputs

        input_feats, inp_map, clusters_coords_mean = self.clusters_voxelization(proposals_idx_shift, proposals_offset_shift, output_feats, locs_float, self.score_fullscale, self.score_scale, self.mode)

        # ### to generate weights
        params = self.score_unet(input_feats)
        params = self.score_outputlayer(params)
        params_feats = params.features[inp_map.long()] # (sumNPoint, C)
        params_feats = pointgroup_ops.roipool(params_feats, proposals_offset_shift.cuda())  # (nProposal, C)

        if len(params_feats) > ins_sample_num and ins_sample_num >0:
            params_feats = params_feats[:ins_sample_num]
            proposals_offset_shift = proposals_offset_shift[:ins_sample_num+1]
            clusters_coords_mean = clusters_coords_mean[:ins_sample_num]

        inst_batch_ids = self.get_instance_batch_id(batch_idxs, proposals_idx_shift, proposals_offset_shift)
        inst_pred_seg_label = self.get_instance_seg_pred_label(semantic_preds, proposals_idx_shift, proposals_offset_shift)

        before_embedding_feature = self.before_embedding_tower(torch.unsqueeze(params_feats, dim=2))
        controller = self.controller(before_embedding_feature).squeeze(dim=2)

        weights, biases = self.parse_dynamic_params(controller, self.output_dim)

        mask_features = self.mask_tower(torch.unsqueeze(output_feats, dim=2).permute(2,1,0)).permute(2,1,0)
        mask_features = mask_features[object_idxs]

        mask_logits = self.mask_heads_forward(mask_features, weights, biases, inst_batch_ids, clusters_coords_mean, coords_, use_coords=self.use_coords)
        

        
        n_inst = mask_logits.shape[0]
        if n_inst > ins_sample_num and ins_sample_num >0:
            mask_logits = mask_logits[:ins_sample_num]
            proposals_offset_shift = proposals_offset_shift[:ins_sample_num+1]
            clusters_coords_mean = clusters_coords_mean[:ins_sample_num]

        outputs['mask_logits'] = mask_logits.squeeze(dim=0) ### N_inst * N_mask

        outputs['object_idxs'] = object_idxs

        outputs['proposals_offset_shift'] = proposals_offset_shift
        outputs['proposals_idx_shift'] = proposals_idx_shift
        outputs['inst_batch_ids'] = inst_batch_ids
        outputs['batch_idxs'] = batch_idxs_
        outputs['batch_offsets'] = batch_offsets
        outputs['inst_pred_seg_label'] = inst_pred_seg_label

        if not training:
            ### generate proposal idx
            proposal_idx, proposal_len, scores = self.generate_proposal(mask_logits.squeeze(dim=0).sigmoid(), object_idxs,
                                                                        threshold=0.5, seg_pred=semantic_preds_, inst_pred_seg_label=None,
                                                                        min_pts_num=50)
            outputs['proposal_scores'] = (scores, proposal_idx, proposal_len)


        return outputs