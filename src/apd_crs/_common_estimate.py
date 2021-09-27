"""
Methods implemented for generating an estimate for the cured/non-cured labels
for censored data and for initializing hardEM algorithm
Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)
Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)
Company: Aimpoint Digital, LP
"""

import numpy as np
from sklearn.cluster import KMeans  # type: ignore
from apd_crs._parameters import _CENSOR_LABEL, _NON_CURE_LABEL, _CURE_LABEL, _LO_INT, _HI_INT


def _estimate_via_clustering(training_data, training_labels, rnd_gen, n_init):  # pylint: disable=too-many-locals
    '''
    Estimate labels for censored data using clustering. Two clusters are
    built from censored data. The cluster that is closer to the centroid of
    non-censored data is assigned the label non-cured.

    Parameters
    ----------
    training_data : {array-like} of shape
                   (n_samples, n_features)
          Training data

    training_labels : {array-like} of shape (n_samples, 1)
        Labels for the training data.
        Value of _censor_label_ implies training point was censored,
        and _non_censor_label_ implies non censored.

    rnd_gen : Random number generator object

    n_init : int,
        Number of random initializations to try for Kmeans clustering
    Returns
    -------
    training_labels : {array-like} of shape (n_samples, 1)
        Labels for training data where each of the censored data points are
        now assigned one of two labels, _cure_label_, _non_cured_label_.
        The values for _cure_label_ and non_cured_label_ can be obtained
        with get_cure_label and get_non_cure_label methods
    '''
    censored_inputs = training_data[training_labels == _CENSOR_LABEL]
    non_censored_inputs = training_data[training_labels != _CENSOR_LABEL]

    # Build one cluster out of non-censored data. Essentially, we are taking
    # the centroid of the non-censored data
    # By design, all non-censored data is also non-cured
    random_state = rnd_gen.integers(_LO_INT, _HI_INT)
    cluster_nc = KMeans(n_clusters=1, n_init=1, random_state=random_state).fit(non_censored_inputs)
    non_censored_center = cluster_nc.cluster_centers_[0]

    random_state = rnd_gen.integers(_LO_INT, _HI_INT)
    cluster_c = KMeans(n_clusters=2, n_init=n_init, random_state=random_state).fit(censored_inputs)
    # The cluster that is closer to non-censored data is assigned the label non-cured.
    # The other cluster is assigned the label cured
    cluster_c_cent = cluster_c.cluster_centers_

    # Identify which clsuter is closer to cluster_nc
    dist_center_0 = np.linalg.norm(cluster_c_cent[0] - non_censored_center)
    dist_center_1 = np.linalg.norm(cluster_c_cent[1] - non_censored_center)
    if dist_center_0 > dist_center_1:
        closer_cluster_label = 1
    else:
        closer_cluster_label = 0

    censored_labels = cluster_c.labels_
    closer_cluster = censored_labels == closer_cluster_label
    farther_cluster = censored_labels == (1 - closer_cluster_label)
    censored_labels[closer_cluster] = _NON_CURE_LABEL
    censored_labels[farther_cluster] = _CURE_LABEL
    training_labels[training_labels == _CENSOR_LABEL] = censored_labels

    return training_labels


def _generate_sample(n_sample, probability, rnd_gen):
    '''
    Generate n_sample samples from a binomial distribution assuming probability
    of a sample being cured as given

    Parameters
    ----------
    n_sample : int
        Number of samples to generate

    probability : float
        Probability parameter for binomial distribution, probability of a
        sample being non-cured

    rnd_gen : Random number generator object
    '''
    sample = rnd_gen.binomial(1, probability, n_sample)
    sample[sample == 0] = _CURE_LABEL
    sample[sample == 1] = _NON_CURE_LABEL

    return sample
