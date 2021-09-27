"""
The core of the Hard-EM algorithm. The methods in this file estimate the
cured/non-cured labels for every censored row which is further used for
survival analysis
Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)
Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)
Company: Aimpoint Digital, LP
"""
import warnings
from multiprocessing import cpu_count, Pool
from functools import partial
import autograd.numpy as np # type: ignore
from autograd import grad
from scipy.optimize import minimize  # type: ignore
from apd_crs._parameters import _CENSOR_LABEL, _LO_INT, _HI_INT, _SLSQP_ITER_LIM, \
        _SLSQP_INFEAS, _EPSILON
from apd_crs._common_estimate import _generate_sample, _estimate_via_clustering


def sigmoid(input_value):
    '''
    Calculate sigmoid function for input
    '''
    return 0.5*(np.tanh(input_value/2.0) + 1) # pylint: disable=no-member


def logistic_probability(covariate_weights, covariates):
    '''
    Calculates the probability of a sample being cured according to the
    logistic model
    '''
    # Outputs probability of a patient being cured according to logistic model.
    return sigmoid(np.dot(covariates, covariate_weights))   # pylint: disable=no-member


def _generate_optimization_model(training_data, training_labels, **kwargs):
    '''
    Generate the optimization model needed to solve the problem
    '''
    n_rows, n_features = training_data.shape
    censored_inputs = training_data[training_labels == _CENSOR_LABEL]
    non_censored_inputs = training_data[training_labels != _CENSOR_LABEL]
    model = {}

    def training_loss(param):
        '''
        Define the training loss objective for minimization
        '''
        covariate_weights, censored_labels = (param[0:n_features], param[n_features:])
        regularization_term = kwargs["pu_reg_term"]*np.dot(covariate_weights,  # pylint: disable=no-member
                                                           covariate_weights)

        known_loss = 1 - logistic_probability(covariate_weights, non_censored_inputs)
        logit_prob = logistic_probability(covariate_weights, censored_inputs)
        unknown_loss = (np.log(logit_prob) *  # pylint: disable=no-member
                        (1 - censored_labels) +
                        np.log(1-logit_prob)*censored_labels)  # pylint: disable=no-member

        return (regularization_term - 1.0/n_rows*(
            np.sum(np.log(known_loss)) +  # pylint: disable=no-member
            np.sum(unknown_loss)))  # pylint: disable=no-member

    model["obj"] = training_loss

    model["gradient"] = grad(training_loss)  # pylint: disable=no-member,no-value-for-parameter

    def lb_constraint(param):
        '''
        The lower bounding constraint for cure_labels
        '''
        return param[n_features:]*(param[n_features:] - 1) + _EPSILON

    # jacobian = autograd.jacobian(lb_constraint) # pylint: disable=no-value-for-parameter
    model["lb_cons"] = [{"type": "ineq", "fun": lb_constraint}]

    def ub_constraint(param):
        '''
        Upper bounding constraint for cure labels
        '''
        return _EPSILON - param[n_features:] * (param[n_features:]-1)

    # jacobian = autograd.jacobian(ub_constraint) # pylint: disable=no-value-for-parameter

    model["ub_cons"] = [{"type": "ineq", "fun": ub_constraint}]
    model["cons"] = model["lb_cons"] + model["ub_cons"]

    return model


def _generate_initial_values(training_data, training_labels,   # pylint: disable=too-many-arguments, too-many-locals
                             rnd_gen, initialize, weight_lo, weight_hi):
    '''
    Generate initial values for the optimization solver.
    Parameters:
    -----------
    training_data : {array-like} of shape
                   (n_samples, n_features)
          Training data

    training_labels : {array-like} of shape (n_samples, 1)
        Labels for the training data.
        Value of _censor_label_ implies training point was censored,
        and _non_censor_label_ implies non censored.

    rnd_gen : A random number generator object

    initialize : {'censoring_rate', 'use_clustering', 'use_random'},
        default='use_random'

        Method to determine how initial guesses for the SLSQP minimization
        are generated.

    weight_lo : {scalar, array-like} of shape (n_features, 1), default=-0.5
        Lower bounds on weights for sampling from uniform distribution for
        initial value guesses

    weight_hi : {scalar, array-like} of shape (n_features, 1), default=0.5
        Upper bounds on weights for sampling from uniform distribution for
        initial value guesses

    Returns
    -------
    initial_values : {array-like} of shape (n_features + 1 + n_cen_samples),
        Initial guesses for covariate weights as well as labels for all
        censored samples
    '''
    n_rows, n_features = training_data.shape

    n_cens = np.count_nonzero(training_labels == _CENSOR_LABEL)  # pylint: disable=no-member
    # Covariate weights are sampled from a uniform distribution
    guess_weights = rnd_gen.uniform(weight_lo, weight_hi, n_features) # pylint: disable=no-member

    if initialize == "censoring_rate":
        # Use censoring rate to initialize the minimization
        non_cure_rate = 1 - float(n_cens)/n_rows
        guess_labels = _generate_sample(n_cens, non_cure_rate, rnd_gen)

    elif initialize == "use_clustering":
        # Use clustering to generate initial guess for labels
        # Do not modify input data
        # Generate a random seed to initialize clustering
        random_state = rnd_gen.integers(_LO_INT, _HI_INT)
        # Use only one random initialization for initializing slsqp
        n_init = 1
        training_labels_ = _estimate_via_clustering(training_data, np.copy(training_labels), # pylint: disable=no-member
                                                    random_state, n_init)
        guess_labels = training_labels_[training_labels == _CENSOR_LABEL]

    elif initialize == "use_random":
        probability = rnd_gen.uniform(0.1, 0.9) # pylint: disable=no-member
        guess_labels = _generate_sample(n_cens, probability, rnd_gen)
    initial_values = np.concatenate((guess_weights, guess_labels), axis=None)

    return initial_values


def _solve_model(model, max_iter, suppress):
    '''
    Solve optimization model, wrapper to catch warnings
    '''
    if not suppress:
        return _solve_model_slsqp(model, max_iter)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _solve_model_slsqp(model, max_iter)
    return res


def _slsqp_status(res):
    '''
    Check slsqp model status and warn user in case of issues
    '''
    if res.status == _SLSQP_INFEAS:
        warnings.warn("Optimization model infeasible. Please report this issue",
                      RuntimeWarning)

    if res.status == _SLSQP_ITER_LIM:
        warnings.warn("Optimization terminated by iteration limit. Try"
                      " increasing iteration limit", RuntimeWarning)

    if not res.success:
        warnings.warn("Optimization did not terminate successfully. Error "
                      "reported is {0}".format(res.message), RuntimeWarning)


def _solve_model_slsqp(model, max_iter):
    '''
    Solve the hardEM optimization model with slsqp

    Parameters
    ----------
    model: object containing specification of optimization model

    max_iter: int
        maximum iterations for slsqp
    '''
    res = minimize(model["obj"], model["init"], method="SLSQP",
                   jac=model["gradient"], constraints=model["cons"],
                   options={"maxiter": max_iter})
    _slsqp_status(res)

    return res


def _solve_model_in_pool(model, init, max_iter, suppress):
    '''
    A wrapper around _solve_model to be used in a multiprocessing pool
    '''
    model["init"] = init
    return _solve_model(model, max_iter, suppress)


def _find_best_fit(results, training_labels, n_features):
    '''
    Find the best result from all optimization calls and return training_labels
    accordingly
    '''

    def get_solution_quality(res):
        '''
        Find the quality of an slsqp solution
        '''
        if res.success:
            return res.fun
        # If optimization returned in an error, ignore solution
        return np.inf # pylint: disable=no-member

    # Find best optimization call
    best_solution = min(results, key=get_solution_quality, default=np.inf) # pylint: disable=no-member
    if np.isposinf(best_solution.fun): # pylint: disable=no-member
        raise RuntimeError("Optimization model failed to converge. Please report this issue")
    unknown_cure_labels = np.rint(best_solution.x[n_features:]) # pylint: disable=no-member

    training_labels[training_labels == _CENSOR_LABEL] = unknown_cure_labels
    return training_labels


def _estimate_via_hard_em(training_data, training_labels, rnd_gen,
                          **kwargs):  # pylint: disable=too-many-locals
    '''
    This is the primary contribution of this package.
    Estimates the cured/non-cured values for censored data by formulating
    labeling as an optimization problem where training loss is minimized.
    The minimizer used currently is the constrained SLSQP method. For
    better success with SLSQP, suitable initialization or multiple random
    starts are desired, which are determined by values for the input
    parameters to this method

    Parameters:
    -----------
    training_data : {array-like} of shape
                   (n_samples, n_features)
          Training data

    training_labels : {array-like} of shape (n_samples, 1)
        Labels for the training data.
        Value of _censor_label_ implies training point was censored,
        and _non_censor_label_ implies non censored.

    rnd_gen : A random number generator object

    Optional kwargs can include:

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

    pu_weight_lo : {array-like} of shape (n_features, 1), default=-0.5
        Lower bounds on weights for sampling from uniform distribution for
        initial value guesses

    pu_weight_hi : {array-like} of shape (n_features, 1), default=0.5
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
    suppress_ = kwargs.get("pu_suppress")

    training_data_ = training_data
    _, n_features = training_data_.shape

    model = _generate_optimization_model(training_data_, training_labels, **kwargs)

    if kwargs["pu_initialize"] == "use_random":
        # Use multi-start local search
        n_calls = kwargs["pu_max_guesses"]
    else:
        # Use only one local search
        n_calls = 1

    # Generate initial values needed for SLSQP
    if kwargs["pu_initialize"] == "use_clustering":
        # clustering only needs to be run once
        # Generate a random seed to initialize clustering
        random_state = rnd_gen.integers(_LO_INT, _HI_INT)
        n_init = 1
        training_labels_ = _estimate_via_clustering(training_data_, np.copy(training_labels), # pylint: disable=no-member
                                                    random_state, n_init)
        guess_labels = training_labels_[training_labels == _CENSOR_LABEL]
        guess_weights = [rnd_gen.uniform(kwargs["pu_weight_lo"], kwargs["pu_weight_hi"], n_features) # pylint: disable=no-member
                         for _ in range(n_calls)]
        initial_values = [np.concatenate((guess_weight, guess_labels), axis=None)
                          for guess_weight in guess_weights]
    else:
        initial_values = [_generate_initial_values(training_data_, training_labels, rnd_gen,
                                                   kwargs["pu_initialize"],
                                                   kwargs["pu_weight_lo"],
                                                   kwargs["pu_weight_hi"])
                          for _ in range(n_calls)]

    n_processes = kwargs.get("pu_max_processes")
    if n_calls > 1 and n_processes != 1:
        if n_processes == -1:
            n_processes = cpu_count()

        with Pool(processes=n_processes) as pool:
            results = pool.map(partial(_solve_model_in_pool, model=model,
                                       max_iter=kwargs["pu_max_iter"], suppress=suppress_),
                               initial_values)
    else:
        results = [_solve_model_in_pool(model, initial_values[i], kwargs["pu_max_iter"],
                                        suppress=suppress_) for i in range(n_calls)]

    training_labels = _find_best_fit(results, training_labels, n_features)
    return training_labels
