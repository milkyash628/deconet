

import torch
import numpy as np
from torch import autocast


def projection_3d_to_2d_scatter_reduce(feat, sp_mat, B, C, H, W):

    residual = torch.zeros(
        (B, C, H * W), 
        device=feat.device, 
        dtype=feat.dtype
    )
    residual.scatter_reduce_(
        2, 
        sp_mat["inflate"],
        feat * sp_mat["mask_zero_padding"],
        "sum",
        include_self=False,
    )
    
    return residual


def projection_3d_to_2d_sparse_matrix(feat, sp_mat, *args, **kwargs):

    with autocast("cuda", enabled=False):
        residual = torch.bmm(
            sp_mat["flatten"], feat.transpose(1, 2).float()
        ).transpose(1, 2)
    
    return residual


def get_all_projections_scatter_reduce(
    cell_ind, nb_feat, batch_size, num_points, occupied_cell, device, grids_shape, dtype,
):
    sp_mat = []
    B, C = batch_size, nb_feat
    occupied_cell = occupied_cell.unsqueeze(1).to(dtype)
    temp_mask = torch.ones_like(occupied_cell)
    for i in range(cell_ind.shape[1]):
        # Pre-compute number of points falling in each 2D cells
        temp_sp_mat = {"inflate": cell_ind[:, i:i + 1], "mask_zero_padding": temp_mask}
        num_points_per_cell = projection_3d_to_2d_scatter_reduce(
            occupied_cell, temp_sp_mat, B, 1, *grids_shape[i],
        )
        num_points_per_cell = torch.gather(
            num_points_per_cell, 2, temp_sp_mat["inflate"],
        )
        # Store projection information
        sp_mat += [dict()]
        sp_mat[i]["mask_zero_padding"] = occupied_cell / (num_points_per_cell + 1e-6)
        sp_mat[i]["inflate"] = cell_ind[:, i:i + 1].expand(-1, C, -1)
    return sp_mat


def get_all_projections_sparse_matrices(
    cell_ind, nb_feat, batch_size, num_points, occupied_cell, device, grids_shape, *args, **kwargs,
):
    point_ind = (
        torch.arange(num_points, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .reshape(1, -1)
    )
    batch_ind = (
        torch.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(-1, num_points)
        .reshape(1, -1)
    )
    non_zeros_ind = []
    for i in range(cell_ind.shape[1]):
        non_zeros_ind.append(
            torch.cat((batch_ind, cell_ind[:, i].reshape(1, -1), point_ind), axis=0)
        )
    sp_mat = [
        build_proj_matrix(
            id,
            occupied_cell,
            batch_size,
            np.prod(sh),
            cell_ind[:, i],
            nb_feat,
        )
        for i, (id, sh) in enumerate(zip(non_zeros_ind, grids_shape))
    ]

    return sp_mat


def build_proj_matrix(
    indices_non_zeros, occupied_cell, batch_size, num_2d_cells, inflate_ind, channels,
):
    """Build sparse matrix for 3D to 2D projection"""
    num_points = indices_non_zeros.shape[1] // batch_size
    matrix_shape = (batch_size, num_2d_cells, num_points)


    inflate = torch.sparse_coo_tensor(
        indices_non_zeros, occupied_cell.reshape(-1), matrix_shape
    ).transpose(1, 2)
    inflate_ind = inflate_ind.unsqueeze(1).expand(-1, channels, -1)


    with autocast("cuda", enabled=False):
        num_points_per_cells = torch.bmm(
            inflate, torch.bmm(inflate.transpose(1, 2), occupied_cell.unsqueeze(-1))
        )


    weight_per_point = 1.0 / (num_points_per_cells.reshape(-1) + 1e-6)
    weight_per_point *= occupied_cell.reshape(-1)
    flatten = torch.sparse_coo_tensor(indices_non_zeros, weight_per_point, matrix_shape)

    return {"flatten": flatten, "inflate": inflate_ind}
