"""
Estimation of a survival function from the results of the Hard-EM algorithm that return estimated
cured/non-cured labels for censored rows. The reference paper that contains the equations for many
of these methods is below.
    REFERENCE:    Kosovalić, N., Barui, S. A Hard EM algorithm for prediction
                  of the cured fraction in survival data.
                  Comput Stat (2021). https://doi.org/10.1007/s00180-021-01140-0

Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)
Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)
Company: Aimpoint Digital, LP
"""
import warnings
import autograd.numpy as np  # type: ignore
from autograd import grad, hessian
from scipy.optimize import minimize # type: ignore
from numpy.linalg import inv
from apd_crs._parameters import _CENSOR_LABEL, _ZERO_BOUND, _SCALETOL, _SHAPE_BOUND, _MAXEXP
from apd_crs._core import _slsqp_status
from apd_crs._validate import _validate_kwargs


def _check_scale_shape(scale, shape):
    '''
    Checks that the scale and shape parameter are positive for Weibull distribution
    '''
    if scale < 0:
        raise ValueError("Scaling parameter for Weibull distribution must be positive. Received"
                         f"{scale}")

    if shape < 0:
        raise ValueError("Shape parameter for Weibull distribution must be positive. Received"
                         f"{shape}")


def _susc_survival(time, covariate_vector, scale, shape, gamma):
    """
    This is the survival function of a susceptible individual.
    Eq 17 from REFERENCE

    Parameters:
    -----------
    time : float
        Time at which survival of an individual is to be determined

    covariate_vector: {array-like} of shape (n_features, )
        Covariates associated with the individual

    scale: float
        positive parameter of Weibull distribution

    shape: float
        positive parameter of Weibull distribution

    gamma: {array-like} of shape (n_features, )
        Survival function parameters.


    Returns:
    --------
    probability: float
        Probability that a susceptible individual with given covariates and given parameters
        assuming a Weibull distribution for the proportional _hazards model with the baseline
        _hazard function having given shape and scale will survive beyond given time
    """

    _check_scale_shape(scale, shape)
    exp = np.exp  # pylint: disable=no-member
    dot = np.dot # pylint: disable=no-member
    dot_product = dot(gamma, covariate_vector)
    arg = (-((time/scale)**shape))*(exp(dot_product))
    probability = exp(arg)

    if probability < 0:
        raise ValueError("Incorrect parameters for Weibull distribution. "
                         "Estimated survival probability is negative")

    return probability


def _hazard(time, covariate_vector, scale, shape, gamma):
    """
    This is hazard function of susceptible population.
    Return -d/dt ln(S(t,x,shape, scale, gamma)) giving the _hazard function.
    Derived from Eq 17 in REFERENCE

    Parameters:
    -----------
    time : float
        Time at which survival of an individual is to be determined

    covariate_vector: {array-like} of shape (n_features, )
        Covariates associated with the individual

    scale: float
        positive parameter of Weibull distribution

    shape: float
        positive parameter of Weibull distribution

    gamma: {array-like} of shape (n_features, )
        Survival function parameters.

    Returns:
    --------
    probability: float
        Hazard associated with a susceptible individual at time
    """

    _check_scale_shape(scale, shape)

    dot_product = np.dot(gamma, covariate_vector) # pylint: disable=no-member
    exp = np.exp(dot_product) # pylint: disable=no-member
    probability = shape*((time/scale)**(shape-1))*exp

    if probability < 0:
        raise RuntimeError(f"_hazard: {probability} is negatve")

    return probability


def _overall_survival(time, prob, covariate_vector, scale, shape, gamma): # pylint: disable=too-many-arguments
    """
    Predicts the overall survival function (Eq 18 from REFERENCE) based on given parameters.
    It is the combination of probability of being cured and the probability of surviving upto time
    t given probability an individual will not be cured

    Parameters:
    -----------
    time : float
        Time at which survival of an individual is to be determined

    prob: float
        Estimated probability of being cured returned by HardEM

    covariate_vector: {array-like} of shape (n_features, )
        Covariates associated with the individual

    scale: float
        positive parameter of Weibull distribution

    shape: float
        positive parameter of Weibull distribution

    gamma: {array-like} of shape (n_features, )
        Survival function parameters.

    Returns:
    --------
    out : float
        Overall survival function of any (susceptible or non-susceptible) individual
    """

    _check_scale_shape(scale, shape)

    out = prob + (1 - prob) * _susc_survival(time, covariate_vector, scale, shape, gamma)
    if out < 0 or out > 1:
        raise RuntimeError("Estimated probability for overall survival is not in [0, 1]")
    return out


def _prob_density(time, prob, covariate_vector, scale, shape, gamma): # pylint: disable=too-many-arguments
    '''
    Predicts the overall probability for time of event (Eq 19 from REFERENCE) based on given
    parameters

    Parameters:
    -----------
    time : float
        Time at which survival of an individual is to be determined

    prob: float
        Estimated probability returned by HardEM
    covariate_vector: {array-like} of shape (n_features, )
        Covariates associated with the individual

    scale: float
        positive parameter of Weibull distribution

    shape: float
        positive parameter of Weibull distribution

    gamma: {array-like} of shape (n_features, )
        Survival function parameters.

    Returns:
    --------
    out : float
        Overall survival function of any (susceptible or non-susceptible) individual
    '''

    _check_scale_shape(scale, shape)

    dot_product = np.dot(gamma, covariate_vector) # pylint: disable=no-member
    exp = np.exp(dot_product) # pylint: disable=no-member
    # Adding the zero bound is a hack, but small values are problematic...
    out = prob*(shape/scale)*((time/scale)**(shape-1))*exp + _ZERO_BOUND
    return out

def _generate_optimization_loss(training_data, training_labels, times, cure_probabilities,
                                regularization_term):
    '''
    Generate the optimization model needed to estimate covariate weights
    '''
    censored_inputs = training_data[training_labels == _CENSOR_LABEL]
    n_cens = len(censored_inputs)

    non_censored_inputs = training_data[training_labels != _CENSOR_LABEL]
    n_noncens = len(non_censored_inputs)


    def training_loss(param):
        '''
        Training loss for optimization
        '''
        dot = np.dot  # pylint: disable=no-member
        sum_arr = np.sum # pylint: disable=no-member
        array = np.array
        log = np.log # pylint: disable=no-member
        scale, shape, gamma = param[0], param[1], param[2:]

        # non-censored loss term
        known_loss = [_prob_density(times[i], cure_probabilities[i], non_censored_inputs[i, :],
                                    scale, shape, gamma) for i in range(n_noncens)]
        known_loss = sum_arr(log(array(known_loss)))

        # censored loss term
        unknown_loss = [_overall_survival(times[i], cure_probabilities[i], censored_inputs[i, :],
                                          scale, shape, gamma) for i in range(n_cens)]
        unknown_loss = sum_arr(log(array(unknown_loss)))

        # regularization term
        reg = dot(gamma, gamma)*regularization_term
        return reg - 1/(n_cens + n_noncens)*(known_loss + unknown_loss)

    gradient = grad(training_loss)  # pylint: disable=no-value-for-parameter
    hess = hessian(training_loss) # pylint: disable=no-value-for-parameter

    return training_loss, gradient, hess


def _survival_fit_weights(training_data, training_labels, times, cure_probabilities,  # pylint: disable=too-many-locals, too-many-arguments
                          **kwargs):
    '''
    Estimates the covariate weight vector, gamma using minimization of negative log of
    maximum-likelihood estimator (See Eq 19 in REFERENCE)

    Parameters:
    ----------

    training_data : {array-like} of shape
                   (n_samples, n_features)
          Training data

    training_labels : {array-like} of shape (n_samples, 1)
        Labels for the training data.
        Value of _censor_label_ implies training point was censored,
        and _non_censor_label_ implies non censored.

    times : {array-like} of shape (n_samples, 1), or None
            Indicate the time when a training point was censored or the event
            of interest was observed.

    cure_probabilities : {array-like} of shape (n_samples, 1), or None
            Indicate the probability that an training point is cured

    Optional keywords include:

    surv_reg_term : float, default=0.5
        The strength of the quadratic regularization term (C*w^2) on the
        non-intercept model covariate weights.

    surv_max_iter: int, default=100
        maximum number of iterations for SLSQP method.

    is_scar : {True, False}, default=False
        Is Selected completely at random (SCAR) assumption satisfied for data

    Returns:
    -------
    scale : float
        Scaling parameter of Weibull distribution

    shape : float
        Shape parameter of Weibull distribution

    gamma : np.ndarray of shape (n_samples, 1)
        Covariate weight vector
    '''
    kwargs = _validate_kwargs(kwargs)
    reg_term = kwargs["surv_reg_term"]
    max_iter = kwargs["surv_max_iter"]

    training_loss, gradient, hess = _generate_optimization_loss(training_data, training_labels,
                                                                times, cure_probabilities,
                                                                reg_term)

    n_rows, n_features = training_data.shape

    # Determine good bounds fo the weight to ensure training loss does not explode
    max_rowsum = np.abs(training_data).sum(axis=1).max() # pylint: disable=no-member
    if max_rowsum > _SCALETOL:
        warnings.warn("Training data should be scaled appropriately to ensure numerical "
                      "convergence", RuntimeWarning)

    bound = np.log(_MAXEXP)/max_rowsum  # pylint: disable=no-member

    # specify variable bounds, shape and gamma must be positive.
    variable_bounds = [[-bound, bound] for _ in range(n_features+2)]
    variable_bounds[0][0] = _ZERO_BOUND
    variable_bounds[1][0] = _ZERO_BOUND
    variable_bounds[0][1] = None
    variable_bounds[1][1] = _SHAPE_BOUND
    guess = bound/2.0*np.ones(n_features+2)  # pylint: disable=no-member
    suppress_ = kwargs.get("surv_suppress")
    if not suppress_:
        res = minimize(training_loss, guess, method="SLSQP", jac=gradient,
                       bounds=variable_bounds, options={"maxiter": max_iter})
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(training_loss, guess, method="SLSQP", jac=gradient,
                           bounds=variable_bounds, options={"maxiter": max_iter})

    _slsqp_status(res)

    model_weights = res.x

    log_likelihood = (-n_rows) * training_loss(model_weights)

    observed_information_matrix = n_rows * hess(model_weights)

    stand_errors = np.sqrt(inv(observed_information_matrix).diagonal())  # pylint: disable=no-member

    return model_weights, stand_errors, log_likelihood
