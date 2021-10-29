"""
Functions for loading data from file
"""

import os
import torch
import numpy as np

"""
--------------------------------------------------------------------------------
    European Regional Tracker, Germany
--------------------------------------------------------------------------------
"""
def ERT_germany(dtype='cases_daily', seq_len=None, country_wide_total=False,
        per_10k_population=True, normalize=True,
        datadir='euro_regional_tracker/germany_data/data'
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

        per_10k_population :


        normalize    : Boolean. Set to true to normalize the returned data to have
                       zero mean and unit variance

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

    # Set the sequence length
    if (seq_len is not None) and (seq_len < u.shape[1]):
        u = u[:,:seq_len]

    # Compute the average over all regions
    if country_wide_total:
        u = u.sum(axis=0)

    # Normalize by population
    elif per_10k_population:
        # Load the population data from file
        population_data = np.loadtxt(
            os.path.join(datadir, 'region_data.csv'),
            dtype='object', delimiter=','
        )[:,1].astype('int')

        # Normalize
        u = (u.T * 10000 / population_data).T

    # Normalize to zero mean, unit variance
    if normalize:
        u -= torch.mean(u)
        u /= torch.std(u)

    # Return the sequence(s)
    return u
