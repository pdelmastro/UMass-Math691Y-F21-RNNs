"""
Functions for randomly sampling the spatiotemporal domain
"""
import torch

def sample_t0_points(n_pts, d, T=1):
    """
    Samples 'n_pts' points (t_i, x_i) from {0} x [0,1]^d
    where each x_ij ~ U([0,1])
    """
    points = torch.rand(size=(n_pts,d+1))
    points[:,0] = 0 # set time dimension to zero
    return points


def sample_boundary_points(n_pts, d, T=1):
    """
    Samples 'n_pts' points (t_i, x_i) from the spactial
    boundary of [0,T] x [0,1]^d. In particular,

        t_i  ~ U(0,T),
        x_ij ~ U(0,1), and
        x_ij = 0 or 1 for one randomly selected j
    """
    points = torch.rand(size=(n_pts,d+1))
    # Randomly select a spatial index to be set to
    # 0 or 1 to enforce that this point is on the boundary
    bndy_inds = torch.randint(low=1,high=d+1,size=(n_pts,))
    bndy_vals = torch.randint(low=0,high=2,size=(n_pts,))
    points[torch.arange(n_pts),bndy_inds] = bndy_vals.float()
    # Rescale time
    points[:,0] *= T
    return points


def sample_boundary_pairs(n_pairs, d, T=1):
    """
    Samples 'n_pairs' pairs of points (t_i, x_i)
    (t_i, x'_i) from the spatial boundary of the
    [0,T] x [0,1]^d. In particular,

        t_i  ~ U(0,T),
        x_ij, x'_ij ~ U(0,1), and
        x_ij = 0 and x'_ij 1 for one randomly selected j
    """
    bndry_pts = torch.rand(size=(n_pairs,d+1))
    # Randomly select a spatial index to be set to
    # 0 or 1 to enforce that this point is on the boundary
    bndry_inds = torch.randint(low=1,high=d+1,size=(n_pairs,))
    # Set the values at the selected boundary indexes to 0
    # for the 'bndry_pts' array
    bndry_pts[torch.arange(n_pairs),bndry_inds] = 0
    # Rescale time
    bndry_pts[:,0] *= T
    # Duplicate 'points' and set the values at the selected
    # boundary indexes to 0 for the duplicate array
    opposite_bndry_pts = bndry_pts.clone()
    opposite_bndry_pts[torch.arange(n_pairs),bndry_inds] = 1
    # Return the pairs
    return bndry_pts, opposite_bndry_pts


def sample_interior_points(n_pts, d, T=1):
    """
    Samples 'n_pts' points (t_i, x_i) from (0,T) x (0,1)^d
    with t_i  ~ U(0,T) and x_ij ~ U(0,1)
    """
    points = torch.rand(size=(n_pts,d+1))
    points[:,0] *= T  # Rescale time
    return points
