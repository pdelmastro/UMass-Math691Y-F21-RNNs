"""
Functions for loading data from file
"""

import os

from numpy.core.defchararray import count
import torch
from torch import nn
import numpy as np

"""
--------------------------------------------------------------------------------
    JHU Dataset, United States
--------------------------------------------------------------------------------
"""
from .util.US_state_abbr import abbrev_to_state

def JHU_US(
        dtype='cases_daily', 
        seq_len=None, 
        granularity='country',
        select_state=None,
        rescaling='population',
        squash_to=None,
        smooth=None, 
        datadir='datasets/jhu',
        lib='np'
    ):
    """
    PARAMETERS

        dtype        : String. Controls which dataset the returned data comes from
                       Options: 'cases' *, 'deaths' *, 'cases_daily', 'deaths_daily'
                         * cumulative
                       Default: 'causes_daily'

        seq_len      : Int. Length of the sequences to return.
                       If 'None', returns the maximum sequence length
                       Default: None

        granularity  : String. Sets the level of granularity of the returned data.
                       Options:
                        'country' : Country-wide total. Use this option to get  
                                    a *single* time series representing the total 
                                    across the entire country.
                        'state'   : State-level totals. Use this option to get 
                                    one time series for each state in the US
                        'county'  : County-level totals. Use this option to get 
                                    one time series for each *county* in the dataset
                       Default: 'county'

        select_state    : String or None. If using the 'state' or 'county' granularity
                          settings, you can set this arg to an abbreviation for a US 
                          state (e.g. MA) to get the time series only for that state. 
                          Two examples:
                          1. If granularity='state' and select-state='MA', this function
                             will return a single time series representing the total
                             across all counties in Massachusetts
                          2. If granularity='county' and select-state='MA', this function
                             will a time series for each county in Massachusetts
                          Default: 'MA'

        rescaling    : String. Used to select a data rescaling method
                       Options:
                        'none' or None  : No rescaling applied
                        'squash'        : All sequences squashed to the interval
                                            [squash_to[0], squash_to[1]]. See 
                                            'squash_to' below for more details.
                        'population'    : Rescaled by the population of the 
                                            {country, state, or county} depending on 
                                            granularity
                       Default: 'population'

        squash_to    :  Tuple or None. If rescaling='squash' and squash_to = (u_min, u_max), 
                        each time series u[i] will be rescaled such that min(u[i]) = u_min
                        and max(u[i]) = u_max 
                        Default: None

        smooth       : None or Int. If Int, each data point will be an average over 
                        the past 'smooth' days
                       Default: None

        lib          : String. Return time series will be an object from the package 
                        specificed by 'lib'. 
                       Options: 'np' for numpy array, 'torch' for PyTorch Tensor
                       Default: 'np'

    RETURNS

        states       :

        counties     :

        u            : (# sequences, seq len)-PyTorch tensor storing the time-series data
                       for Germany. Here, # regions = 1 if country_wide_avg=True,
                       otherwise 401

    """

    # Verify the data type is valid
    if dtype not in ['cases', 'deaths', 'cases_daily', 'deaths_daily']:
        raise ValueError('dtype %s not recognized' % dtype)
    
    # Verify the granularity is valid
    if granularity not in ['country', 'state', 'county']:
        raise ValueError('granularity %s not recognized' % granularity)


    # Load the raw data from file
    if dtype.startswith('cases'):
        fname_suffix = 'cases.csv'
    else:
        fname_suffix = 'deaths.csv'
    all_columns = np.loadtxt(
        os.path.join(
            datadir, 
            'formatted', 
            '%s_%s' % (granularity, fname_suffix)
        ),
        delimiter=',',
        dtype=object
    )


    # Extract individual columns
    # Case 1: Country-wide
    states = None
    counties = None
    if granularity == 'country':
        u = all_columns.reshape(1,-1).astype('float32')
    # Case 2: State-level
    elif granularity == 'state':
        states = all_columns[:,0].astype('str')
        u = all_columns[:,1:].astype('float32')
    # Case 3: County-level
    else:
        counties = all_columns[:,0].astype('str')
        states = all_columns[:,1].astype('str')
        u = all_columns[:,2:].astype('float32')


    # Convert from cumulative to differences
    if dtype.endswith('daily'):
        u = np.diff(u, axis=1)


    # Select a specific state
    if granularity != 'country' and select_state is not None:
        # Convert the state abbrev to the full state name
        if select_state not in abbrev_to_state:
            raise ValueError('State %s not recognized' % select_state)
        # Indices of 'states' that match 'select_state'
        state_inds = np.argwhere(states == abbrev_to_state[select_state])
        # Restrict counties and time series u down to these indices
        if granularity == 'county':
            counties = counties[state_inds].reshape(-1)
        u = u[state_inds,:].reshape(state_inds.shape[0], -1)


    # Adjust the sequence length
    if (seq_len is not None) and (seq_len < u.shape[1]):
        u = u[:,:seq_len]


    # Apply rescaling
    # Case 1: Population Rescaling
    if rescaling == 'population':
        # Load the population data from file
        population_fname = ('county' if granularity == 'county' else 'state')
        population_fname += '_populations.csv'
        population_data = np.loadtxt(
            os.path.join(datadir, 'formatted', population_fname),
            dtype='object', delimiter=','
        )[:,1 if granularity=='state' else 2].astype('float32')
        # Exclude the populations of other states
        # if the user selected a specific state
        if granularity != 'country' and select_state is not None:
            population_data = population_data[state_inds].reshape(-1)   
        # Perform the rescaling
        if granularity == 'country':
            u /= np.sum(population_data)
        else:
            u = (u.T / population_data).T

    # Case 2: Squashing Rescaling
    elif rescaling == 'squash':
        if isinstance(squash_to, tuple) and len(squash_to) == 2:
            u_min, u_max = squash_to
            # Compute the current min and max
            curr_min = u.min(axis=-1)[0]
            curr_max = u.max(axis=-1)[0]
            # Rescale to the interval (0,1)
            u = ((u.T - curr_min) / (curr_max - curr_min)).T
            # Rescale to (u_min, u_max) now
            u = ((u_max - u_min) * u + u_min)


    # Apply smoothing
    if smooth is not None:
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
            u_torch = torch.from_numpy(u)
            u_torch = mean_conv(u_torch.view(-1,1,u_torch.shape[-1]))
            u_torch = u_torch.reshape(-1,u_torch.shape[-1])
            # Convert back to numpy array if the user didnt want to get
            # a torch array back
            if lib == 'torch':
                u = u_torch.numpy()
            else:
                u = u_torch
                

    # Convert 'u' to torch tensor?
    if lib == 'torch' and smooth is None:
        u = torch.from_numpy(u)
        

    # Return the dataset
    # Case 1: 'country' granularity
    if (granularity == 'country') or (
            granularity == 'state' and select_state is not None
        ):
        return u
    # Cases 2 and 3: 'state' or 'county' granularity
    to_return = []
    if granularity != 'country' and select_state is None:
        to_return.append(states)
    if granularity == 'county':
        to_return.append(counties)
    to_return.append(u)
    return to_return
    



"""
--------------------------------------------------------------------------------
    European Regional Tracker, Germany
--------------------------------------------------------------------------------
"""
def ERT_germany(dtype='cases_daily', seq_len=None, 
        non_negative=True, country_wide_total=False,
        per_10k_population=True, normalize=True, squash_to=None,
        smooth=None, 
        datadir='datasets/euro_regional_tracker/germany_data/data'
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
