import torch
import torch.nn as nn
# import spconv.pytorch as spconv
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
from util.config import cfg

from model.geoformer.geoformer_modules import ResidualBlock, VGGBlock, UBlock, conv_with_kaiming_uniform

from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetSAModuleVotesSeparate
from lib.pointnet2.pointnet2_utils import furthest_point_sample

from model.pos_embedding import PositionEmbeddingCoordsSine

from util.config import cfg

from model.helper import (ACTIVATION_DICT, NORM_DICT, WEIGHT_INIT_DICT,
                            get_clones, GenericMLP, BatchNormDim1Swap)

from model.transformer_detr import TransformerDecoder, TransformerDecoderLayer


class GeoFormer(nn.Module):
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

        # self.offset = nn.Sequential(
        #     nn.Linear(m, m, bias=True),
        #     norm_fn(m),
        #     nn.ReLU()
        # )
        # self.offset_linear = nn.Linear(m, 3, bias=True)

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

        self.detr_sem_head = GenericMLP(
            input_dim=cfg.dec_dim,
            hidden_dims=[cfg.dec_dim, cfg.dec_dim],
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            output_dim=classes
        )

        self.apply(self.set_bn_init)

        self.threshold_ins = cfg.threshold_ins
        self.min_pts_num = cfg.min_pts_num
        #### fix parameter
        self.module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'semantic': self.semantic, 'semantic_linear': self.semantic_linear,
                    #   'offset': self.offset, 'offset_linear': self.offset_linear,
                      'mask_tower': self.mask_tower}

        for m in self.fix_module:
            mod = self.module_map[m]
            for param in mod.parameters():
                param.requires_grad = False

    def set_eval(self):
        for m in self.fix_module:
            self.module_map[m] = self.module_map[m].eval()

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm1d') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def generate_proposal(self, mask_logits, cls_logits, fg_idxs, batch_offsets, threshold, seg_pred=None, inst_pred_seg_label=None, min_pts_num=50):
        # batch = mask_logits.shape[0]
        batch = len(mask_logits)
        n_queries = mask_logits[0].shape[0]
        n_inst = batch * n_queries
        proposal_len = []
        proposal_len.append(0)
        proposal_idx = []
        seg_preds = []
        # seg_preds_scores = []
        num = 0
        scores = []

        

        # mask_logits = mask_logits.sigmoid() # batch x n_queries x N_mask
        # print('cls_logits', cls_logits.shape)
        cls_logits_pred = cls_logits.max(2)[1] # batch x n_queries x 1 
        for b in range(batch):
            # print("DEBUG", mask_logits[b].shape)
            start   = batch_offsets[b]
            end     = batch_offsets[b+1]
            mask_logit_b = mask_logits[b].sigmoid()
            for n in range(n_queries):
                
                proposal_id_n = ((mask_logit_b[n] > threshold) & (seg_pred[start:end] == cls_logits_pred[b,n])).nonzero().squeeze(dim=1)
                # print("DEBUG", proposal_id_n.size(0), cls_logits[b,n])
                if proposal_id_n.size(0) < min_pts_num or cls_logits_pred[b,n] < 4:
                    continue
                # proposal_id_n = ((mask_logits[b,n] > threshold) & (seg_pred == inst_pred_seg_label[n].item())).nonzero().squeeze(dim=1)
                # print(n, start, end, proposal_id_n)

                score = mask_logit_b[n][proposal_id_n].mean()
                # print('proposal_id_n', proposal_id_n.shape, torch.count_nonzero(proposal_id_n))
                proposal_id_n = proposal_id_n + start
                # seg_mod = torch.mode(seg_pred[proposal_id_n.long()])[0].item()
                # seg_label = inst_pred_seg_label[n]
                
                proposal_id_n = fg_idxs[proposal_id_n.long()].unsqueeze(dim=1)
                # id_proposal_id_n = torch.cat([proposal_id_n, torch.ones_like(proposal_id_n)*b], dim=1)
                id_proposal_id_n = torch.cat([torch.ones_like(proposal_id_n)*num, proposal_id_n], dim=1)
                num += 1
                tmp = proposal_len[-1]
                proposal_len.append(proposal_id_n.size(0)+tmp)
                proposal_idx.append(id_proposal_id_n)
                scores.append(score)
                # seg_preds.append(seg_mod)
                seg_preds.append(cls_logits_pred[b,n])
                # seg_preds_scores.append(cls_logits[b,n,cls_logits_pred[b,n]])

        if len(proposal_idx) == 0:
            return proposal_idx, proposal_len, scores, seg_preds
        proposal_idx = torch.cat(proposal_idx, dim=0)
        proposal_len = torch.from_numpy(np.array(proposal_len)).cuda()
        scores = torch.stack(scores)
        seg_preds = torch.tensor(seg_preds).cuda()
        # seg_preds_scores = torch.tensor(seg_preds_scores).cuda()
        # scores = torch.from_numpy(np.array(scores, dtype=np.float32)).cuda()
        # print('proposal_len', proposal_len)
        return proposal_idx, proposal_len, scores, seg_preds

    def random_point_sample(self, batch_offsets, npoint):
        batch_size = batch_offsets.shape[0] - 1
        
        batch_points = (batch_offsets[1:] - batch_offsets[:-1])
        
        sampling_indices = [torch.tensor(np.random.choice(batch_points[i].item(), npoint, replace=(npoint>batch_points[i])), dtype=torch.int).cuda() + batch_offsets[i]
                             for i in range(batch_size)]
        sampling_indices = torch.cat(sampling_indices)
        # print('sampling_indices', torch.max(sampling_indices), torch.min(sampling_indices))
        return sampling_indices

    def random_point_sample_b(self, batch_points, npoint):
        
        sampling_indices = torch.tensor(np.random.choice(batch_points, npoint, replace=False), dtype=torch.int).cuda()
        return sampling_indices

    def sample_query_embedding(self, xyz, pc_dims, num_queries):
        fps_sampling_inds = furthest_point_sample(xyz, num_queries)
        fps_sampling_inds = fps_sampling_inds.long()
        query_xyz = [torch.gather(xyz[..., x], 1, fps_sampling_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=pc_dims)
        query_embed = self.query_projection(pos_embed.float())
        return query_xyz, query_embed, fps_sampling_inds
        # return query_xyz, fps_sampling_inds

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


    def mask_heads_forward(self, mask_features, weights, biases, num_insts, coords_, fps_sampling_coords, use_coords=True):
        assert mask_features.dim() == 3
        n_layers = len(weights)
        c = mask_features.size(1)
        n_mask = mask_features.size(0)
        x = mask_features.permute(2,1,0).repeat(num_insts, 1, 1) ### num_inst * c * N_mask

        relative_coords = fps_sampling_coords.reshape(-1, 1, 3) - coords_.reshape(1, -1, 3) ### N_inst * N_mask * 3
        relative_coords = relative_coords.permute(0,2,1) ### num_inst * 3 * n_mask
        # coords_ = coords_.reshape(1, -1, 3).repeat(num_insts, 1, 1).permute(0,2,1)
        if use_coords:
            x = torch.cat([relative_coords, x], dim=1) ### num_inst * (3+c) * N_mask

        x = x.reshape(1, -1, n_mask) ### 1 * (num_inst*c') * Nmask
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)

        return x



    

    def get_mask_prediction(self, param_kernels, mask_features, locs_float_, fps_sampling_locs, batch_offsets_):
        # param_kernels = param_kernels.permute(0, 2, 1, 3) # num_layers x batch x n_queries x channel
        num_layers, n_queries, batch, channel = (
            param_kernels.shape[0],
            param_kernels.shape[1],
            param_kernels.shape[2],
            param_kernels.shape[3],
        )
        # param_kernels = param_kernels.reshape(num_layers, batch * n_queries, channel)

        # before_embedding_feature    = self.before_embedding_tower(torch.unsqueeze(param_kernels, dim=2))
        # controller                  = self.controller(before_embedding_feature).squeeze(dim=2)

        # controller = controller.reshape(num_layers, (batch * n_queries), -1)

        outputs = []
        n_inst_per_layer = batch * n_queries
        for l in range(num_layers):

            param_kernel = param_kernels[l] # n_queries x batch x channel
            # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
            cls_logits = self.detr_sem_head(param_kernel.permute(1,2,0)).transpose(1, 2) # batch x n_queries x n_classes

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
                mask_logits         = self.mask_heads_forward(mask_feature_b, weights, biases, n_queries, locs_float_b, 
                                                            fps_sampling_locs_b, use_coords=self.use_coords)
                
                
                mask_logits     = mask_logits.squeeze(dim=0) # (n_queries) x N_mask
                mask_logits_list.append(mask_logits)

                # mask_logits     = mask_logits.reshape(batch, n_queries, -1) # batch x n_queries x N_mask
                
            output = {'cls_logits': cls_logits, 'mask_logits': mask_logits_list}
            outputs.append(output)
        return outputs

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

    def forward(self, batch_input, epoch, training=True):
        outputs = {}
        batch_size = cfg.batch_size if training else 1

        batch_idxs  = batch_input['locs'][:, 0].int()
        p2v_map     = batch_input['p2v_map']
        locs_float  = batch_input['locs_float']
        batch_offsets = batch_input['offsets']

        pc_dims = [
            batch_input["pc_maxs"],
            batch_input["pc_mins"],
        ]

        N_points = locs_float.shape[0]

        # with torch.no_grad():
        sparse_input = self.preprocess_input(batch_input)

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

        ''' Offset head'''
        # pt_offsets_feats = self.offset(output_feats)
        # pt_offsets = self.offset_linear(pt_offsets_feats)   # (N, 3), float32

        # outputs['pt_offsets'] = pt_offsets

        outputs['semantic_scores'] = semantic_scores

        if epoch <= self.prepare_epochs:
            return outputs

        mask_features   = self.mask_tower(torch.unsqueeze(output_feats, dim=2).permute(2,1,0)).permute(2,1,0)

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
    
        mask_features_  = mask_features[fg_idxs]

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
                outputs['mask_predictions']  = None
                return outputs

            if batch_points <= cfg.n_downsampling:
                npoint = batch_points  
            else:
                npoint = cfg.n_downsampling

            sampling_indices        = self.random_point_sample_b(batch_points, npoint).long()

            locs_float_b = locs_float_b[sampling_indices].unsqueeze(0)
            output_feats_b = output_feats_b[sampling_indices].unsqueeze(0)
            
            context_locs_b, grouped_features, grouped_xyz, pre_enc_inds = self.set_aggregator.group_points(locs_float_b.contiguous(), 
                                                                    output_feats_b.transpose(1,2).contiguous())
            context_feats_b = self.set_aggregator.mlp(grouped_features, grouped_xyz)
            context_feats_b = context_feats_b.transpose(1,2)

            context_locs.append(context_locs_b)
            context_feats.append(context_feats_b)
        
        context_locs = torch.cat(context_locs)
        context_feats = torch.cat(context_feats) # batch x npoint x channel

        context_embedding_pos = self.pos_embedding(context_locs, input_range=pc_dims)

        # query_embedding_xyz: batch x n_queries x 3
        query_locs, query_embedding_pos, query_sampling_inds = self.sample_query_embedding(context_locs, pc_dims, cfg.n_query_points)
        
        # original_queries_inds = fg_idxs[sampling_indices[fps_sampling_inds.flatten()[fps_sampling_inds2.flatten()]].flatten()]
        # # sampling_indices[fps_sampling_inds.flatten()[fps_sampling_inds2.flatten()]].cpu()
        # # print("TEST")
        # vis_queries_inds = torch.ones(N_points, 1) * -100
        # vis_queries_inds[original_queries_inds] = 1
        # outputs['vis_queries_inds'] = vis_queries_inds

        context_feats = self.encoder_to_decoder_projection(
            context_feats.permute(0, 2, 1)
        ) # batch x channel x npoints

        ''' Init dec_inputs by query features '''
        context_feats_T = context_feats.transpose(1,2) # batch x npoints x channel 
        dec_inputs      = [torch.gather(context_feats_T[..., x], 1, query_sampling_inds) for x in range(context_feats_T.shape[-1])]
        dec_inputs      = torch.stack(dec_inputs) # channel x batch x npoints

        # support_embeddings = support_embeddings.permute(0,2,1)
        # dec_inputs      = (dec_inputs.permute(1,2,0) * support_embeddings).permute(1,0,2)
        dec_inputs      = dec_inputs.permute(2,1,0) # npoints x batch x channel

        # decoder expects: npoints x batch x channel
        context_embedding_pos   = context_embedding_pos.permute(2, 0, 1)
        context_feats           = context_feats.permute(2, 0, 1)
        query_embedding_pos     = query_embedding_pos.permute(2, 0, 1)

        

        # num_layers x n_queries x batch x channel
        dec_outputs = self.decoder(
            dec_inputs, 
            context_feats, 
            pos=context_embedding_pos, 
            query_pos=query_embedding_pos
        )

        if not training:
            dec_outputs = dec_outputs[-1:,...]


        outputs['fg_idxs']              = fg_idxs
        outputs['num_insts']            = cfg.n_query_points * batch_size
        outputs['batch_idxs']           = batch_idxs_
        outputs['query_sampling_inds']  = query_sampling_inds

        if len(fg_idxs) == 0:
            outputs['mask_predictions'] = None
            return outputs

        mask_predictions = self.get_mask_prediction(dec_outputs, mask_features_, locs_float_, query_locs, batch_offsets_)

        outputs['mask_predictions']  = mask_predictions
        # ret['proposals']    = proposals
        

        if not training:
            mask_prediction_last_layer = mask_predictions[-1]
            mask_logit_final = mask_prediction_last_layer['mask_logits'] #.reshape(batch_size, cfg.n_query_points, -1) # batch x n_queries x N_mask
            cls_logit_final = mask_prediction_last_layer['cls_logits'] #.reshape(batch_size, cfg.n_query_points, -1) # batch x n_queries x n_classes
            # mask_logit_final = mask_logits[-1].reshape(batch_size, cfg.n_query_points, -1) # batch x n_queries x N_mask

            
            # query_sampling_inds = sampling_indices[pre_enc_inds.flatten()[query_sampling_inds.flatten().long()].long()].flatten()
            # print(query_sampling_inds)
            proposal_idx, proposal_len, scores, seg_preds = self.generate_proposal(mask_logit_final, cls_logit_final, fg_idxs, batch_offsets,
                                                threshold=0.5, seg_pred=semantic_preds_,
                                                min_pts_num=50)
            outputs['proposal_scores'] = (scores, proposal_idx, proposal_len, seg_preds)
        return outputs
