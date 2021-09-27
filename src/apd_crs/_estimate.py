"""
Methods implemented for generating an estimate for the cured/non-cured labels
for censored data.
Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)
Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)
Company: Aimpoint Digital, LP
"""
from apd_crs._parameters import _CENSOR_LABEL, _CURE_LABEL
from apd_crs._core import _estimate_via_hard_em
from apd_crs._common_estimate import _generate_sample, _estimate_via_clustering
from apd_crs._validate import _validate_kwargs

def _estimate_all_cured(_training_data, training_labels, _rnd_gen):
    '''
    Estimate all labels for censored data as cured. A quick heuristic for
    initialization or comparison

    Parameters
    ----------
    training_data : {array-like} of shape
                   (n_samples, n_features)
          Training data. Not used, maintained for consistency

    training_labels : {array-like} of shape (n_samples, 1)
        Labels for the training data.
        Value of _censor_label_ implies training point was censored,
        and _non_censor_label_ implies non censored.

    rnd_gen : Random number generator object

    Returns
    -------
    training_labels : {array-like} of shape (n_samples, 1)
        Labels for training data where each of the censored data points are
        now assigned one of two labels, _cure_label_, _non_cured_label_.
    '''
    training_labels[training_labels == _CENSOR_LABEL] = _CURE_LABEL
    return training_labels


def _estimate_randomly(training_data, training_labels, rnd_gen):
    '''
    Estimates the cured/non-cured values by assuming that every censored
    label has a 0.5 probability of being cured and samples from an
    appropriate distribution
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

    Returns
    -------
    training_labels : {array-like} of shape (n_samples, 1)
        Labels for training data where each of the censored data points are
        now assigned one of two labels, _cure_label_, _non_cured_label_.
        The values for _cure_label_ and non_cured_label_ can be obtained
        with get_cure_label and get_non_cure_label methods
    '''
    probability = 0.5
    n_cen = len(training_data[training_labels == _CENSOR_LABEL])
    guess_labels = _generate_sample(n_cen, probability, rnd_gen)
    training_labels[training_labels == _CENSOR_LABEL] = guess_labels
    return training_labels


def _estimate_labels(self, training_data, training_labels, **kwargs):
    '''
    Estimates the cured/non-cured values for censored data using one of the
    many implemented methods
    Parameters
    ----------
    training_data : {array-like} of shape
                   (n_samples, n_features)
          Training data

    training_labels : {array-like} of shape (n_samples, 1)
        Labels for the training data.
        Value of _censor_label_ implies training point was censored,
        and _non_censor_label_ implies non censored.

    Optional kwargs are used only when the estimator selected is hard_EM.
    They are otherwise ignored, They include:

    pu_reg_term : float, default=0.5
        The strength of the quadratic regularization term (C*w^2) on the
        non-intercept model covariate weights.


    pu_initialize : {'censoring_rate', 'use_clustering', 'use_random'},
        default='use_random'
        Method to determine how initial guesses for the SLSQP minimization
        are generated. The covariate weights are initialized at
        random from a uniform distribution.
        If the option 'censoring_rate' is selected, cure labels are
        initialized assuming the probability of being cured is the
        censoring rate. If 'use_clustering' is selected then a single
        cluster is created from the noncensored rows, and two clusters
        are created from the censored rows. The cluster closest to the
        noncensored rows is assigned the label non_cured as an initial
        guess.  If "use_random" is selected, multiple guesses for the
        unkown cure labels are generated randomly, multiple minimization
        problems are solved. The output chosen is the one corresponding to
        the lowest objective

    pu_max_guesses : int, default=50
        Maximum local searches to launch when initialize method is
        use_random. Otherwise ignored

    pu_max_processes : int, default=1
        Maximum parallel local searches to launch when initialize method is
        use_random. Otherwise ignored.
        if -1, all available processors are utilized

    pu_max_iter: int, default=1000
        maximum number of iterations for SLSQP method.

    pu_weight_lo : {scalar, array-like} of shape (n_features, 1), default=-0.5
        Lower bounds on weights for sampling from uniform distribution for
        initial value guesses

    pu_weight_hi : {scalar, array-like} of shape (n_features, 1), default=0.5
        Upper bounds on weights for sampling from uniform distribution for
        initial value guesses

    pu_kmeans_init : int
        Number of Kmeans initializations to try when estimator is clustering

    is_scar : {True, False}, default=False
        Is Selected completely at random (SCAR) assumption satisfied for data

    Returns
    -------
    training_labels : {array-like} of shape (n_samples, 1)
        Labels for training data where each of the censored data points are
        now assigned one of two labels, _cure_label_, _non_cured_label_.
        The values for _cure_label_ and non_cured_label_ can be obtained
        with get_cure_label and get_non_cure_label methods
    '''
    rnd_gen = self.rnd_gen
    kwargs = _validate_kwargs(kwargs)

    if self.estimator_ == "all_cens_cured":
        # All censored points are assumed to be cured
        training_labels = _estimate_all_cured(training_data, training_labels, rnd_gen)

    elif self.estimator_ == "clustering":
        # Use clustering to identify labels for censored data
        n_init = kwargs["pu_kmeans_init"]
        training_labels = _estimate_via_clustering(training_data, training_labels, rnd_gen, n_init)

    elif self.estimator_ == "hard_EM":
        training_labels = _estimate_via_hard_em(training_data, training_labels, rnd_gen,
                                                **kwargs)

    else:
        # Assume that each censored row has a 50% chance of being cured
        assert self.estimator_ == "fifty_fifty"
        training_labels = _estimate_randomly(training_data, training_labels, rnd_gen)
    return training_labels
