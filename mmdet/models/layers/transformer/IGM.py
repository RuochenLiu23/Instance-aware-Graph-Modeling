import copy
import argparse

import torch
from mmcv.ops import batched_nms
from torch import Tensor, nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj, OptTensor

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from .deformable_detr_layers import DeformableDetrTransformerDecoder
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid
from .IGN import IGN_Net


class DDQTransformerDecoder(DeformableDetrTransformerDecoder):
    """Query Feature Learning Network (decoder) with IGN."""

    def __init__(self, *args, **kwargs):
        self._setup_args()
        super().__init__(*args, **kwargs)

    def _setup_args(self):
        """Setup arguments for IGN_Net"""
        parser = argparse.ArgumentParser(
            description='Instance-aware Graph Network with Dynamic Graph Construction'
        )
        
        # IGN_Net required parameters
        parser.add_argument('--hidden_channels', type=int, default=256, 
                            help='Hidden layer dimension')
        parser.add_argument('--d_model', type=int, default=256, 
                            help='Model dimension')
        parser.add_argument('--d_inner', type=int, default=1024, 
                            help='SSM inner dimension')
        parser.add_argument('--dt_rank', type=int, default=64, 
                            help='Low-rank approximation of time step')
        parser.add_argument('--d_state', type=int, default=64, 
                            help='State dimension')
        parser.add_argument('--layer_num', type=int, default=2, 
                            help='Number of layers')
        parser.add_argument('--dropout', type=float, default=0.1, 
                            help='Dropout rate')
        parser.add_argument('--bias', action='store_true', default=False, 
                            help='Whether to use bias')
        parser.add_argument('--mamba_dropout', type=float, default=0.1, 
                            help='SSM module dropout')
        parser.add_argument('--knn_k', type=int, default=8, 
                            help='K value for KNN')
        
        self.args, _ = parser.parse_known_args()

    def _init_layers(self) -> None:
        """Initialize decoder layers with IGN components."""
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)
        
        # Initialize IGN_Net for feature extraction
        self.ign_net = IGN_Net(
            in_channels=self.embed_dims,
            hidden_channels=self.args.hidden_channels,
            num_classes=None,
            args=self.args
        )
        
        self.knn_k = self.args.knn_k
        
        # Attention fusion module for combining IGN and queries
        self.attention_fusion = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims // 2, 1),
            nn.Sigmoid()
        )

    def calculate_adjacency_matrix(self, features, positions):
        """
        Calculate adjacency matrix by fusing feature and position similarities
        
        Args:
            features: Query features [batch_size, num_queries, embed_dims]
            positions: Query positions (reference points) [batch_size, num_queries, 2/4]
            
        Returns:
            KNN-based adjacency matrix [batch_size, num_queries, num_queries]
        """
        batch_size, num_queries, _ = features.shape
        
        # Feature cosine similarity
        feat_norm = F.normalize(features, p=2, dim=2)
        feat_sim = torch.matmul(feat_norm, feat_norm.transpose(1, 2))
        
        # Position distance similarity
        pos_diff = positions[:, :, None] - positions[:, None, :]
        pos_dist = torch.norm(pos_diff, p=2, dim=3)
        pos_sim = torch.exp(-pos_dist / (2 * pos_dist.mean()))
        
        # Fuse similarities
        adj_matrix = 0.5 * feat_sim + 0.5 * pos_sim
        
        # KNN adjacency matrix
        knn_adj = torch.zeros_like(adj_matrix)
        for b in range(batch_size):
            for i in range(num_queries):
                _, top_k = torch.topk(adj_matrix[b, i], k=self.knn_k + 1)
                knn_adj[b, i, top_k] = 1.0
                knn_adj[b, top_k, i] = 1.0
        return knn_adj

    def select_distinct_queries(self, reference_points: Tensor, query: Tensor,
                                self_attn_mask: Tensor, layer_index):
        """Select distinct queries using NMS for deduplication"""
        num_imgs = len(reference_points)
        dis_start, num_dis = self.cache_dict['dis_query_info']
        dis_mask = self_attn_mask[:, dis_start:dis_start + num_dis,
                                  dis_start:dis_start + num_dis]
        scores = self.cache_dict['cls_branches'][layer_index](
            query[:, dis_start:dis_start + num_dis]).sigmoid().max(-1).values
        proposals = reference_points[:, dis_start:dis_start + num_dis]
        proposals = bbox_cxcywh_to_xyxy(proposals)

        attn_mask_list = []
        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            attn_mask = ~dis_mask[img_id * self.cache_dict['num_heads']][0]
            ori_index = attn_mask.nonzero().view(-1)
            _, keep_idxs = batched_nms(single_proposals[ori_index],
                                       single_scores[ori_index],
                                       torch.ones(len(ori_index)),
                                       self.cache_dict['dqs_cfg'])

            real_keep_index = ori_index[keep_idxs]
            attn_mask = torch.ones_like(dis_mask[0]).bool()
            attn_mask[real_keep_index] = False
            attn_mask[:, real_keep_index] = False
            attn_mask = attn_mask[None].repeat(self.cache_dict['num_heads'], 1, 1)
            attn_mask_list.append(attn_mask)
        attn_mask = torch.cat(attn_mask_list)
        self_attn_mask = copy.deepcopy(self_attn_mask)
        self_attn_mask[:, dis_start:dis_start + num_dis,
                       dis_start:dis_start + num_dis] = attn_mask
        self.cache_dict['distinct_query_mask'].append(~attn_mask)
        return self_attn_mask

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tensor:
        """
        Forward pass with IGN
        
        Args:
            query: Query embeddings [batch_size, num_queries, embed_dims]
            value: Value features from encoder
            key_padding_mask: Padding mask for keys
            self_attn_mask: Self-attention mask
            reference_points: Reference points for queries
            spatial_shapes: Spatial shapes of multi-scale features
            level_start_index: Start index for each feature level
            valid_ratios: Valid ratios for each feature level
            reg_branches: Regression branches for bbox refinement
            
        Returns:
            Decoder output and reference points
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        self.cache_dict['distinct_query_mask'] = []
        
        if self_attn_mask is None:
            self_attn_mask = torch.zeros((query.size(1), query.size(1)),
                                         device=query.device).bool()
        self_attn_mask = self_attn_mask[None].repeat(
            len(query) * self.cache_dict['num_heads'], 1, 1)
        
        # Decoder layer
        for layer_index, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :],
                num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(query_sine_embed)
            query = layer(
                query, query_pos=query_pos, value=value,
                key_padding_mask=key_padding_mask, self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                valid_ratios=valid_ratios, reference_points=reference_points_input, **kwargs)

            # Bbox refinement and query selection
            if not self.training:
                tmp = reg_branches[layer_index](query)
                new_reference_points = tmp + inverse_sigmoid(reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if layer_index < (len(self.layers) - 1):
                    self_attn_mask = self.select_distinct_queries(
                        reference_points, query, self_attn_mask, layer_index)
            else:
                num_dense = self.cache_dict['num_dense_queries']
                tmp = reg_branches[layer_index](query[:, :-num_dense])
                tmp_dense = self.aux_reg_branches[layer_index](query[:, -num_dense:])
                tmp = torch.cat([tmp, tmp_dense], dim=1)
                new_reference_points = tmp + inverse_sigmoid(reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if layer_index < (len(self.layers) - 1):
                    self_attn_mask = self.select_distinct_queries(
                        reference_points, query, self_attn_mask, layer_index)

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
        
        # IGN processing 
        if self.return_intermediate:
            last_query = intermediate[-1]
        else:
            last_query = query
        
        batch_size, num_queries, embed_dims = last_query.shape
        ign_outputs = []
        
        # Extract bbox centers as position features
        center_points = reference_points[:, :, :2]
        
        for b in range(batch_size):
            feat = last_query[b]
            pos = center_points[b]
            
            # Construct dynamic graph structure
            adj_matrix = self.calculate_adjacency_matrix(
                feat.unsqueeze(0), pos.unsqueeze(0)
            )[0]
            edge_index = adj_matrix.nonzero().t().contiguous()
            
            # Build sparse adjacency tensor
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], 
                                sparse_sizes=(feat.size(0), feat.size(0)))
            
            feat = feat.to(self.ign_net.lin1.weight.device)
            edge_index = edge_index.to(self.ign_net.lin1.weight.device)
            adj_t = adj_t.to(self.ign_net.lin1.weight.device)
            
            # IGN feature extraction
            h = self.ign_net(
                x=feat, 
                edge_index=edge_index,
                adj_t=adj_t,
                return_features=True
            )
            
            # Residual connection
            h = h + last_query[b]
            ign_outputs.append(h)
        
        ign_outputs = torch.stack(ign_outputs)
        
        # Attention-based feature fusion
        concat_features = torch.cat([ign_outputs, last_query], dim=-1)
        attn_weights = self.attention_fusion(concat_features)
        query = attn_weights * ign_outputs + (1 - attn_weights) * last_query

        if self.return_intermediate:
            intermediate[-1] = query
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return query, reference_points