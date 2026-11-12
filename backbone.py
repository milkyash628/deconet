


import os
import torch
import warnings
import torch.nn as nn


from waffleiron import WI_SCATTER_REDUCE
if WI_SCATTER_REDUCE:
    from .helper_projection import projection_3d_to_2d_scatter_reduce as projection_3d_to_2d #inflate
    from .helper_projection import get_all_projections_scatter_reduce as get_all_projections
else:
    from .helper_projection import projection_3d_to_2d_sparse_matrix as projection_3d_to_2d #flatten
    from .helper_projection import get_all_projections_sparse_matrices as get_all_projections


class myLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)

NORM_OPTIONS = {
	"batchnorm": nn.BatchNorm1d,
	"layernorm": myLayerNorm,
}

class DropPath(nn.Module):
    """
    Stochastic Depth

    Original code of this module is at:
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def extra_repr(self):
        return f"prob={self.drop_prob}"

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output



class ElementScale(nn.Module):
    #A learnable element-wise scaler.

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale



class ChannelAggregationFFN(nn.Module):


    def __init__(self,
                 embed_dims,
                 kernel_size=3,
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = int(embed_dims*2)

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1,
            )
        self.drop = nn.Dropout(ffn_drop)
        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = nn.ReLU(inplace=True)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LCFA(nn.Module):
    def __init__(self, channels, grid_shape, drop_path_prob, which_norm="batchnorm"):
        super().__init__()
        self.compressed = False
        self.H, self.W = grid_shape
        self.norm = NORM_OPTIONS[which_norm](channels)
        self.ffn = ChannelAggregationFFN(
            embed_dims=channels,
            kernel_size=3,  # Assuming kernel size is 3, as in the original FFN
            ffn_drop=0.       # Assuming no dropout in FFN, as in the original FFN
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale
        self.grid_shape = grid_shape
        self.drop_path = DropPath(drop_path_prob)

    def extra_repr(self):
        return f"(grid): [{self.grid_shape[0]}, {self.grid_shape[1]}]"

    def compress(self):
        # Join scale and last conv
        self.ffn.fc2.weight.data = (
                self.ffn.fc2.weight.data * self.scale.weight.data[..., None]
        )
        self.ffn.fc2.bias.data = (
                self.ffn.fc2.bias.data * self.scale.weight.data[:, 0, 0]
        )
        # Flag
        self.compressed = True


    def forward_compressed(self, tokens, sp_mat):
       # tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )
        # Make sure we are not in training mode
        assert not self.training
        # Forward pass
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        # Flatten
        residual = projection_3d_to_2d(residual, sp_mat, B, C, self.H, self.W)
        residual = residual.reshape(B, C, self.H, self.W)
        # FFN
        residual = self.ffn(residual)
        # Inflate
        residual = residual.reshape(B, C, self.H * self.W)
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual)

    def forward(self, tokens, sp_mat):
        
        if self.compressed:
            return self.forward_compressed(tokens, sp_mat)
        #
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        # Flatten
        residual = projection_3d_to_2d(residual, sp_mat, B, C, self.H, self.W)
        residual = residual.reshape(B, C, self.H, self.W)
        # FFN
        residual = self.ffn(residual)
        # LayerScale
        residual = residual.reshape(B, C, self.H * self.W)
        residual = self.scale(residual)
        # Inflate
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=-1, keepdim=True)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

class AWSM(nn.Module):
    def __init__(self, channels, drop_path_prob, which_norm="batchnorm"):
        super().__init__()
        self.compressed = False
        self.which_norm = which_norm
        self.norm = NORM_OPTIONS[which_norm](channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(inplace=True),
            SEBlock(channels),
            nn.Conv1d(channels, channels, 1),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale
        self.drop_path = DropPath(drop_path_prob)

    def compress(self):
        if self.which_norm == "layernorm":
            warnings.warn(
                
            )
            return
        # Join Batch norm and first conv
        norm_weight = self.norm.weight.data / torch.sqrt(
            self.norm.running_var.data + 1e-05
        )
        norm_bias = self.norm.bias.data - norm_weight * self.norm.running_mean.data
        # Careful the order of the two lines below should not be changed
        self.mlp[0].bias.data = (
            self.mlp[0].weight.data[:, :, 0] @ norm_bias + self.mlp[0].bias.data
        )
        self.mlp[0].weight.data = self.mlp[0].weight.data * norm_weight[None, :, None]
        # Join scale and last conv
        self.mlp[-1].weight.data = self.mlp[-1].weight.data * self.scale.weight.data
        self.mlp[-1].bias.data = (
            self.mlp[-1].bias.data * self.scale.weight.data[:, 0, 0]
        )
        # Flag
        self.compressed = True

    def forward(self, tokens):
        
        if self.compressed:
            assert not self.training
            return tokens + self.drop_path(self.mlp(tokens))
        else:
            return tokens + self.drop_path(self.scale(self.mlp(self.norm(tokens))))





class WaffleIron(nn.Module):
    def __init__(self, channels, depth, grids_shape, drop_path_prob, which_norm="batchnorm"):
        super().__init__()
        self.depth = depth
        self.grids_shape = grids_shape
        self.awsm = nn.ModuleList(
            [AWSM(channels, drop_path_prob, which_norm) for _ in range(depth)]
        )
        self.lcfa = nn.ModuleList(
            [
                LCFA(channels, grids_shape[d % len(grids_shape)], drop_path_prob, which_norm)
                for d in range(depth)
            ]
        )

    def compress(self):
        for d in range(self.depth):
            self.awsm[d].compress()
            self.lcfa[d].compress()

    def forward(self, tokens, cell_ind, occupied_cell):
        # Build all 3D to 2D projection matrices
        batch_size, nb_feat, num_points = tokens.shape
        sp_mat = get_all_projections(
            cell_ind, nb_feat, batch_size, num_points,
            occupied_cell, tokens.device, self.grids_shape,tokens.dtype,
        )

        # Actual backbone
        for d, (smix, cmix) in enumerate(zip(self.lcfa, self.awsm)):
            tokens = smix(tokens, sp_mat[d % len(sp_mat)])
            tokens = cmix(tokens)


        return tokens