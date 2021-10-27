"""
Functions for clustering neurons based on activity
"""

import torch
import numpy as np
import scipy.cluster.hierarchy as spc
import matplotlib.pyplot as plt

def hierarchical_clustering(H):
    """
    Standard hierarchical clustering, using scipy

    ARGUMENTS

        H       :   (# hidden states, # neurons)-numpy array representing
                    hidden states of the RNN

    RETURNS

        order   :   (# neurons,)-numpy array. order[i] denotes the index of
                    neuron i, based on the order imposed by the hierarchical
                    clustering
    """
    # Covariance matrix
    corr = (H.T @ H) / H.shape[0]

    # Pairwise distanced, based on neuron-neuron correlation vectors
    pdist = spc.distance.pdist(corr)
    # Hierarchical Clustering
    linkage = spc.linkage(pdist, method='complete')
    # Convert to cluster indices
    idx = spc.fcluster(linkage, 0.7 * pdist.max(), 'distance')
    order = np.argsort(idx)

    # Return the new neuron ordering
    return order
