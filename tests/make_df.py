"""
The functions in this file help generate test datasets with specified covariates and specified
properties for censored and cured populations. These are useful in testing survival analysis
algorithms
Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)
Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)
Company: Aimpoint Digital, LP
"""
import numpy as np
import pandas as pd  # type: ignore
from scipy.special import expit  # type: ignore # pylint: disable=no-name-in-module
from _parameters import _CENSOR_LABEL, _NON_CURE_LABEL  # type: ignore # pylint: disable=import-error


def create_df(column_names, dist_parameters, n_rows, model_weights, censoring_prob, #pylint: disable=too-many-arguments
              seed):
    '''
    Create an artificial dataframe for testing survival analysis models. The inputs include
    column_names for covariates, list of normal distribution parameters (mean, stddev) for sampling
    each of the covariates independently, model_weights for a sigmoidal distribution to generate
    cure labels. If censoring_prob is a float, we assume that the dataset follows a SCAR (selected
    completely at random) assumption for censoring. If censoring_prob is a one-dimensional array
    of size (column_names), we assume that the dataset violates a non-scar assumption. In this
    scenario, we determine censoring labels based on a sigmoidal distribution with weights as
    censoring_prob
    '''
    censor_weights = np.asarray(censoring_prob)
    if censor_weights.ndim == 0:
        is_scar = True
    elif censor_weights.ndim == 1:
        is_scar = False
    else:
        raise ValueError("censoring prob must be a float (SCAR) or an array (non-SCAR)")

    rnd_gen = np.random.default_rng(seed=seed)
    data_frame = generate_covariates(n_rows, column_names, dist_parameters, rnd_gen)
    data_frame = generate_labels(data_frame, column_names, model_weights, "cure_label")
    if not is_scar:
        data_frame = generate_labels(data_frame, column_names, censor_weights, "nscar_label")
    data_frame["intercept"] = np.ones(n_rows)

    def to_censor(row):
        '''
        If you are cured, you must be censored. Otherwise you will be censored according to given
        probability in the scar situation, and nscar_label in the non-scar situation
        '''
        if row["cure_label"] == 0:
            censoring_indicator = 0
        elif is_scar:
            censoring_indicator = 1 - rnd_gen.binomial(1, censoring_prob)
        else:
            censoring_indicator = row["nscar_label"]

        if censoring_indicator == 0:
            return _CENSOR_LABEL
        return _NON_CURE_LABEL

    data_frame["label"] = data_frame.apply(to_censor, axis=1)
    if len(np.unique(data_frame["label"])) == 1:
        raise ValueError("Only one label in randomly generated data. Try running with "
                         "different parameters")

    if not is_scar:
        data_frame.drop("nscar_label", axis=1, inplace=True)

    return data_frame


def generate_covariates(n_rows, column_names, dist_parameters, rnd_gen):
    '''
    Generate a dataframe of covariates with n_rows as the number of rows and column_names as the
    names of the columns. dist_parameters are a list of tuples that contain the mean and standard
    deviation for a normal distribution from which the sample is drawn.
    All columns are drawn from independent distributions
    '''
    if len(column_names) != len(dist_parameters):
        raise ValueError("Incorrect inputs to generate_covariates")

    if n_rows <= 0:
        raise ValueError("Cannot generate a dataframe with number of rows = {0}".format(n_rows))

    # Separate list of tuples into two separate tuples
    mean, std = list(zip(*dist_parameters))
    data = rnd_gen.multivariate_normal(mean, np.diag(std), size=n_rows)
    data_frame = pd.DataFrame(data, columns=column_names)
    return data_frame


def generate_labels(data_frame, covariates, model_weights, cure_label):
    '''
    Generates labels for each of the rows in the dataframe using a sigmoidal distribution and
    using 'model_weights' for each of the covariates in 'covariates'. The label assigned to this
    column is 'cure_label'
    '''
    n_rows, _ = data_frame.shape
    intercept = np.ones(n_rows)

    # Probability of not being cured
    predictions_float = 1 - expit(model_weights[0]*intercept + np.dot(data_frame[covariates],
                                                                      model_weights[1:]))

    data_frame[cure_label] = np.rint(predictions_float)
    return data_frame
