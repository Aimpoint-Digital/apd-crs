"""
Methods implemented for validation of input training and testing data passed to
a SurvivalAnalysis object. These will then be imported into the main class
DO NOT IMPORT DIRECTLY
Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)
Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)
Company: Aimpoint Digital, LP
"""
import warnings
import multiprocessing
import copy
import numpy as np
from apd_crs._parameters import _ZEROTOL, _DEFAULT_ARGS


def _val_test_data(self, test_data):
    '''
    Ensures that the test data passed is consistent with what the class expects

    Parameters
    ----------
    self : SurvivalAnalysis object

    test_data : {array_like, sparse matrix} of shape (n_samples, n_features)
        Test data

    Returns
    -------
    None
    '''
    test_data_ = np.asarray(test_data, np.float_)
    n_rows, n_features = test_data_.shape
    if n_rows == 0:
        raise ValueError("No test data passed")

    if n_features != self.n_features:
        raise ValueError("Training data included {0} features. Test data passed has {1}"
                         " features".format(self.n_features, n_features))

    return test_data_


def _val_train_data(self, training_data, training_labels, times=None):
    '''
    Validates the training data and labels and returns numpy arrays for them

    Parameters
    ----------
    self : SurvivalAnalysis object

    training_data : {array-like, sparse matrix} of shape
                    (n_samples, n_features)
          Training data

    training_labels : {array-like, sparse matrix} of shape (n_samples, 1)
        Labels for the training data. Value of _censor_label_ implies
        the training point was censored, value of _non_censor_label_
        implies non censored.
        The _censor_label_ and _non_censor_label_ values can be obtained
        with get_censor_label and get_non_censor_label methods

    times : {array-like} of shape (n_samples, 1) or None
            Times for all training points (censoring time or event time)


    Returns
    -------
    training_data_ : ndarray of shape (n_samples, n_features)
                     Training data with covariates
    training_labels_ : ndarray of shape (n_samples, ) or (n_samples, 1)
                      Labels for the training points
    times_ : ndarray of shape (n_samples, ) or (n_samples, 1) or None
            Censoring or event times for all training points if present in
            input
    '''

    training_data_ = np.asarray(training_data, np.float_)
    training_labels_ = np.asarray(training_labels, np.int_)

    if times is not None:
        times_ = np.asarray(times)
        n_times = times_.shape[0]
        if np.any(times_ < 0):
            raise ValueError("Event times cannot be negative")

    else:
        n_times = 0
        times_ = None

    n_rows = training_data_.shape[0]
    labels_shape = training_labels_.shape
    n_labels = labels_shape[0]

    if n_rows != n_labels:
        raise ValueError("Training data and training labels must be have "
                         "equal number of rows")

    if n_times > 0 and n_times != n_rows:
        raise ValueError("Training data and event times must have equal number of rows")

    if len(labels_shape) > 1 and labels_shape[1] > 1:
        n_cols = labels_shape[1]
        warnings.warn("Expected training_labels as an array of shape "
                      "({0}, ) or {0}, 1). Instead received array of shape"
                      " ({0}, {1}). Extra columns ignored".format(n_rows,
                                                                  n_cols))

        training_labels_ = np.array(training_labels_[:, 0], np.int_)

    non_cure_label = self.get_non_cure_label()
    censor_label = self.get_censor_label()

    if not np.isin(training_labels_, (censor_label, non_cure_label)).all():
        raise ValueError("Training labels include ambigous values. Use {0} as label for non cured"
                         " rows and {1} as label for censored rows".format(non_cure_label,
                                                                           censor_label))

    if times_ is not None:
        idx = np.isclose(times_, 0.0, _ZEROTOL)
        if np.any(idx):
            warnings.warn("Obtained censoring/cure times in training data that "
                          "are close to zero. These values will be dropped from"
                          " training set")
        training_data_ = training_data_[~idx]
        training_labels_ = training_labels_[~idx]
        times_ = times_[~idx]

    return training_data_, training_labels_, times_


def _is_int(input_val):
    '''
    Check if input is int like
    '''
    try:
        int(input_val)
        return True
    except TypeError:
        return False

def _is_float(input_val):
    '''
    Check if input is float like
    '''
    try:
        float(input_val)
        return True
    except TypeError:
        return False


def _get_int_val(input_val, keyword):
    '''
    Raise exception if value for a keyword is not an integer. Otherwise return
    casted integer
    '''
    if not _is_int(input_val):
        raise TypeError("Integer value expected for {0}. Received {1}".format(keyword, input_val))

    return int(input_val)


def _validate_val(input_val, keyword, lower_bound, upper_bound, is_int):
    '''
    Validate an input integer value and check if it is within acceptable bounds
    '''
    if is_int:
        opt_val = _get_int_val(input_val, keyword)
    else:
        opt_val = _get_float_val(input_val, keyword)

    if lower_bound is not None:
        if opt_val < lower_bound:
            raise ValueError(f"{keyword} must be > {lower_bound}. Received {opt_val}")
    if upper_bound is not None:
        if opt_val > upper_bound:
            raise ValueError(f"{keyword} must be < {upper_bound}. Received {opt_val}")
    return opt_val


def _get_float_val(input_val, keyword):
    '''
    Raise exception if value for a keyword is not a float. Otherwise return
    casted float
    '''
    if not _is_float(input_val):
        raise TypeError("Float value expected for {0}. Received {1}".format(keyword, input_val))

    return float(input_val)


def _get_float_array(input_val, keyword):
    '''
    Return input_val for keyword as a np float. Raise exception if value cannot
    be treated as float
    '''
    try:
        float_val = np.asarray(input_val, np.float_)
        return float_val
    except ValueError:
        raise ValueError("A single float or array of floats expected for {0}. "
                         "Received {1}".format(keyword, input_val))


def _validate_kwargs(kwargs):  # pylint: disable=too-many-branches
    '''
    Validates the keyword arguments received by the SurvivalAnalysis fit
    methods (both pu_fit, stochastic_fit and survival_fit)
    '''
    validated_kwargs = copy.copy(_DEFAULT_ARGS)

    for key, val in kwargs.items():
        if key in ("pu_reg_term", "surv_reg_term"):
            opt_val = _validate_val(val, key, 0, None, False)
            validated_kwargs[key] = opt_val

        elif key == "pu_initialize":
            if val not in ("censoring_rate", "use_clustering", "use_random"):
                raise ValueError("Unexpected value: '{0}' provided for 'initialize' keyword. "
                                 "Allowable values are: ('use_random', 'use_clustering', "
                                 "'censoring_rate')".format(val))

        elif key in ("pu_max_guesses", "pu_max_iter", "surv_max_iter",
                     "pu_kmeans_init", "surv_batch_size"):
            opt_val = _validate_val(val, key, 1, None, True)
            validated_kwargs[key] = opt_val

        elif key == "pu_max_processes":
            n_proc = multiprocessing.cpu_count()
            opt_val = _validate_val(val, key, -1, n_proc, True)
            if opt_val == 0:
                opt_val = 1
            validated_kwargs[key] = opt_val

        elif key == "pu_weight_lo":
            opt_val = _get_float_array(val, key)
            if opt_val.ndim == 0:
                # If scalar was provided
                opt_val = opt_val.item()

            validated_kwargs[key] = opt_val

        elif key == "pu_weight_hi":
            opt_val = _get_float_array(val, key)
            if opt_val.ndim == 0:
                # If scalar was provided
                opt_val = opt_val.item()

            validated_kwargs[key] = opt_val

        elif key in ("pu_suppress", "surv_suppress", "is_scar"):
            opt_val = bool(val)
            validated_kwargs[key] = opt_val

        else:
            if key in validated_kwargs:
                warnings.warn("{0} not implemented. Report error".format(key))
            else:
                warnings.warn("Unexpected keyword argument '{0}' provided to fit "
                              "method. Will be ignored".format(key))

    return validated_kwargs
