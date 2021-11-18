"""
Functions for loading data from file
"""

import os

from numpy.core.overrides import set_module
import torch
from torch import nn
import numpy as np

"""
--------------------------------------------------------------------------------
    European Regional Tracker, Germany
--------------------------------------------------------------------------------
"""
def ERT_germany(dtype='cases_daily', seq_len=None, 
        non_negative=True, country_wide_total=False,
        per_10k_population=True, normalize=True, squash_to=None,
        smooth=None, datadir='euro_regional_tracker/germany_data/data'
    ):
    """
    PARAMETERS

        dtype        : String. Controls which column the returned data comes from
                       Options: 'cases', 'deaths', 'cases_daily', 'deaths_daily'
                       Default: 'causes_daily'

        seq_len      : Int. Length of the sequences to return.
                       If 'None', returns the maximum sequence length
                       Default: None

        country_wide_total : Boolean. Set to true to collapse the data into a single
                       time series representing the sum over the entire country
                       Default: False

        non_negative : Boolean. Set to True to clip all data points to be positive.
                       Set of the daily cases numbers are negative due to 
                       inconsistencies in the dataset

        per_10k_population :


        normalize    : Boolean. Set to true to normalize the returned data to have
                       zero mean and unit variance

        squash_to    : Tuple or None. If tuple (u_min, u_max), each sequence u[i] 
                        will be rescaled such that u[i].min() = u_min and 
                        u[i].max() = u_max 

        smooth       : None or Int. If Int, each data point y[i,t] will be an
                        average over the past 'smooth' days
                       Default: None
                    
    RETURNS

        u            : (# regions, seq len)-PyTorch tensor storing the time-series data
                       for Germany. Here, # regions = 1 if country_wide_avg=True,
                       otherwise 401

    """
    # Verify the data type if valid
    if dtype not in ['cases', 'deaths', 'cases_daily', 'deaths_daily']:
        raise ValueError('dtype %s not recognized' % dtype)

    # Load the raw Torch data from file
    u = torch.load(os.path.join(datadir, '%s.pt' % dtype)).float()

    # Clip to non-negative values
    if non_negative:
        u = torch.clamp(u, min=0)

    # Adjust sequence length based on smoothing window
    if (smooth is not None) and (smooth > 0):
        seq_len += smooth-1

    # Set the sequence length
    if (seq_len is not None) and (seq_len < u.shape[1]):
        u = u[:,:seq_len]

    # Compute the average over all regions
    if country_wide_total:
        u = u.sum(axis=0)

    # Smooth
    if smooth is not None:
        # Object to perform 1d convolutions
        with torch.no_grad():
            mean_conv = nn.Conv1d(
                            in_channels=1,
                            out_channels=1,
                            kernel_size=smooth
            )
            # Set kernel to calculate mean
            kernel_weights = (np.ones(smooth) / smooth).astype('float32')
            mean_conv.weight.data = torch.Tensor(kernel_weights).view(1, 1, smooth)
            mean_conv.bias.data *= 0
            # Apply the convolution and reshape u back to the 
            # correct size
            u = mean_conv(u.view(-1,1,u.shape[-1]))
            u = u.reshape(-1,u.shape[-1])

    # Normalize by population
    if per_10k_population:
        # Load the population data from file
        population_data = np.loadtxt(
            os.path.join(datadir, 'region_data.csv'),
            dtype='object', delimiter=','
        )[:,1].astype('float32')

        # Normalize
        if country_wide_total:
            u *= 10000 / population_data.sum()
        else:
            u = (u.T * 10000 / population_data).T

    # Normalize to zero mean, unit variance
    if normalize:
        u -= torch.mean(u)
        u /= torch.std(u)
    
    # Squash to an interval
    elif isinstance(squash_to, tuple) and len(squash_to) == 2:
        u_min, u_max = squash_to
        # Compute the current min and max
        curr_min = u.min(axis=-1)[0]
        curr_max = u.max(axis=-1)[0]
        # Rescale to the interval (0,1)
        u = ((u.T - curr_min) / (curr_max - curr_min)).T
        # Rescale to (u_min, u_max) now
        u = ((u_max - u_min) * u + u_min)
        
    # Return the sequence(s)
    return u
