"""
Functions for clustering neurons based on activity
"""

import numpy as np
import scipy.cluster.hierarchy as spc

def hierarchical_clustering(H, alpha=0.7):
    """
    Standard hierarchical clustering, using scipy

    ARGUMENTS

        H       :   (# hidden states, # neurons)-numpy array representing
                    hidden states of the RNN

        alpha   :   Parameter that controls the    

    RETURNS

        order   :   (# neurons,)-numpy array. order[i] denotes the index of
                    neuron i, based on the order imposed by the hierarchical
                    clustering

        H_clustered     :
    """
    # Covariance matrix
    cov = (H.T @ H) / H.shape[0]

    # Pairwise distanced, based on neuron-neuron correlation vectors
    pdist = spc.distance.pdist(cov)
    # Hierarchical Clustering
    linkage = spc.linkage(pdist, method='complete')
    # Convert to cluster indices
    idx = spc.fcluster(linkage, alpha * pdist.max(), 'distance')
    order = np.argsort(idx)

    # Return the new neuron ordering
    return order
