import torch
import math
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from torch_sparse import SparseTensor
from einops import repeat, einsum


class ActionNet(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = ssm_linear(args.d_inner, 2, with_bias=True)
        self.dropout = torch.nn.Dropout(0.6)
        self.act = torch.nn.ReLU()
        self.reset_parameters()
    
    def reset_parameters(self):
        self.net.reset_parameters()

    def forward(self, x) -> Tensor:
        x = self.dropout(self.net(x))
        return x


class ssm_linear(torch.nn.Module):
    """
    Simple linear layer for SSM
    """
    def __init__(self, in_features, out_features, with_bias=False):
        super(ssm_linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = input @ self.weight
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_ssm_block(torch.nn.Module):
    """
    GCN_SSM block combining SSM architecture with graph structure
    Uses selective state space models (SSM) for graph representation learning
    """
    def __init__(self, args):
        super(GCN_ssm_block, self).__init__()
        self.args = args
        
        # Input projection to split into main path and residual
        self.in_proj = ssm_linear(args.d_model, args.d_inner * 2, with_bias=args.bias)
        
        # State space model parameters projection
        self.x_proj = ssm_linear(args.d_inner, args.dt_rank + args.d_state * 2, with_bias=args.bias)
        self.dropout = args.dropout
        self.dt_proj = ssm_linear(args.dt_rank, args.d_inner, with_bias=args.bias)
        
        # Batch normalization layers
        self.bns = torch.nn.ModuleList()
        for i in range(args.layer_num-1):
            self.bns.append(torch.nn.BatchNorm1d(args.d_inner))

        # State space model matrices
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(args.d_inner))
        
        # Output projection
        self.out_proj = ssm_linear(args.d_inner, args.d_model, with_bias=args.bias)

        # Action networks for adaptive edge weighting
        self.in_act_net = ActionNet(args)
        self.out_act_net = ActionNet(args)
        self.reset_parameters()

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        self.x_proj.reset_parameters()
        self.dt_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.in_act_net.reset_parameters()
        self.out_act_net.reset_parameters()

    def forward(self, x, edge_index, adj, layer_num):
        (b, d) = x.shape
        
        # Expand input across layers
        expanded_x = x.unsqueeze(1)
        x = expanded_x.expand(b, layer_num, d)
        (b, l, d) = x.shape
        
        # Split into main path and residual connection
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = F.relu(x)

        # Apply selective state space model
        y = self.ssm(x, self.args, edge_index, adj)
        
        # Gated residual connection
        y = y * F.relu(res)

        # Output projection
        output = self.out_proj(y)

        return output[:,-1,:]
    
    def ssm(self, x, args, edge_index, adj):
        """Selective State Space Model"""
        (d_in, n) = self.A_log.shape
        
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        # Selective scan with graph structure
        y = self.selective_scan(x, edge_index, delta, A, B, C, D, args, adj)
        
        return y
    
    def selective_scan(self, u, edge_index, delta, A, B, C, D, args, adj):
        """
        Selective scan operation with dynamic graph structure
        Integrates SSM's selective mechanism with GCN
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # Initialize state
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = [] 
        
        # Layer-wise processing with adaptive graph structure
        for i in range(args.layer_num):
            # State update
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            
            # Apply dropout and GCN
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x.reshape(b, d_in * n)
            x = adj @ x
            x = x.reshape(b, d_in, n)
            in_logits = self.in_act_net(x=y)
            out_logits = self.out_act_net(x=y)
            temp = 0.01
            in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)
            
            edge_weight = self.create_edge_weight(edge_index=edge_index,
                                                  keep_in_prob=in_probs[:, 0], 
                                                  keep_out_prob=out_probs[:, 0])
            adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                             value=edge_weight, sparse_sizes=(len(y), len(y)))

            adj = adj.set_diag()
            deg = adj.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        
        # Add skip connection with learnable D
        y = y + u * D

        return y
    
    def create_edge_weight(self, edge_index, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob


class IGN_Net(torch.nn.Module):
    """
     Instance-aware Graph Network (IGN)
    """
    def __init__(self, in_channels, hidden_channels, num_classes=None, args=None):
        super().__init__()
        self.args = args
        self.dropout = args.mamba_dropout if hasattr(args, 'mamba_dropout') else 0.1
        
        # Feature projection layer
        self.lin1 = ssm_linear(in_channels, hidden_channels, with_bias=args.bias)
        
        # SSM block
        self.layer_num = args.layer_num
        self.mamba = GCN_ssm_block(args)
        
        self.norm_1 = RMSNorm(hidden_channels)
        self.norm_2 = RMSNorm(hidden_channels)
        
        self.has_classifier = num_classes is not None
        if self.has_classifier:
            self.lin2 = ssm_linear(hidden_channels, num_classes, with_bias=args.bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin1.reset_parameters()
        if self.has_classifier:
            self.lin2.reset_parameters()
    
    def forward(self, x, edge_index, adj_t=None, return_features=True):
        """
        Forward pass
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            adj_t: Optional adjacency matrix (for GCN_ssm_block)
            return_features: Whether to return features (vs classification results)
        """
        # Feature extraction
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Build adjacency matrix if not provided
        if adj_t is None:
            adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], 
                                sparse_sizes=(x.size(0), x.size(0)))
        
        # SSM processing with residual connection
        output = self.mamba(self.norm_1(x), edge_index, adj_t, self.layer_num) + x
        features = self.norm_2(output)
        
        # Return features
        if return_features or not self.has_classifier:
            return features
        else:
            return F.log_softmax(self.lin2(features), dim=-1)
                
    def normalize_adj_tensor(self, adj):
        """Symmetric normalization of adjacency matrix"""
        mx = adj
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output