"""
Implementation of the cure rate survival analysis, based on the work by 
Dr. Nemanja Kosovalic and Prof. Sandip Barui, both formerly at the 
University of South Alabama. Current implementation supports option
to use Hard EM algorithm and SCAR (selected completely at random), 
among others, for estimating cure probabilites.  

Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)

Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)

Company: Aimpoint Digital, LP
"""
import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
from tqdm import tqdm # type: ignore
from sksurv.metrics import concordance_index_censored as cindex  # type: ignore
from apd_crs._parameters import _CENSOR_LABEL, _NON_CURE_LABEL, _CURE_LABEL, _LO_INT, \
        _HI_INT
from apd_crs._validate import _val_train_data, _val_test_data, _validate_kwargs  # type: ignore
from apd_crs._estimate import _estimate_labels  # type: ignore
from apd_crs._survival_func import _survival_fit_weights, _overall_survival


class SurvivalAnalysis:  # pylint: disable=too-many-instance-attributes
    """
    SurvivalAnalysis class

    This class implements methods for survival analysis: i.e. time-to-event
    analysis using a dataset for which outcomes of a fraction of the
    dataset are unknown (censored), and for which some individuals never
    experience the event. For example, a medical trial where time since diagnosis is
    measured and some participants drop out during the trial period, some of which are
    cured. Or a manufacturing dataset where time-to-failure since maintenace is measured,
    and some equipment takes so long to fail that it "never sees" the event.

    For every training point (a patient in a medical trial, or machine in a
    manufacturing dataset), we have three potential labels:
        Cured: The event never occurs.
        Non-cured: The event occurs at some (possibly unobserved) time.
        Censored: The event does not occur during the time period, while it may
        or may not occur later, but is unobserved.

    The methods in this class help estimate the labels (Cured/Non-cured) for
    each of the points in the censored group. Using the estimated labels, a
    classifier is built to predict whether a training point will belong to the
    cured group or the non-cured group.

    In addition, the methods in the class help predict the overall survival
    probability of any training point upto time t, as well as the risk for a
    training point at time t

    Parameters
    ----------
    estimator : {'hard_EM', 'clustering', 'fifty_fifty', 'all_cens_cured'},
                default='hard_EM'
        The method used for estimating labels for censored training data.
        If 'hard_EM', the hard expectation maximization algorithm is utilized to
        estimate labels for censored population. If 'clustering', a clustering
        model is built with a single cluster for the non-censored rows, and two
        clusters for the censored rows. By comparing the distance of the two
        censored clusters to the noncensored cluster centor, cure labels are
        assigned to the censored rows. If 'fifty_fifty', each censored point is
        assigned to the cured/non-cured group with equal probability. Finally,
        'all-cens-cured' assumes that all censored population are cured.

    classifier : A classifier object, default=None
        A fully initialized classification model object that follows the
        scikit-learn classification API. The model is used to classify
        between cured and non-cured labels. If no classifier input is
        provided, a logistic regression model with default parameters will
        be utilized.

    random_state : int, array_like, SeedSquence, BitGenerator or Generator or None, default=None
        Random state for reproducible random number generation in the
        algorithms.

    Attributes
    ----------
    estimator_ : str
        Method used for estimating censored data

    classifier_ : A classification object
        A fully initialized classification object with scikit-learn
        classification API

    scale_ : float
        Scale parameter for the fitted Weibull distribution

    shape_ : float
        Shape parameter for the fitted Weibull distribution

    gamma_ : np.array
        Gamma parameter for survival function

    Examples
    --------
    TODO: Fill

    References
    ---------
    Kosovalić, N., Barui, S. A Hard EM algorithm for prediction
    of the cured fraction in survival data.
    Comput Stat (2021). https://doi.org/10.1007/s00180-021-01140-0
    """

    # Parameter attributes
    _censor_label = _CENSOR_LABEL
    _cure_label = _CURE_LABEL
    _non_cure_label = _NON_CURE_LABEL

    # Imported methods
    _validate_train_data = _val_train_data
    _validate_test_data = _val_test_data
    _estimate_censored_labels = _estimate_labels

    def __init__(self, estimator="hard_EM", classifier=None, random_state=None):
        if estimator not in {"hard_EM", "clustering", "fifty_fifty", "all_cens_cured"}:
            raise ValueError("estimator must be 'hard_EM', 'clustering', 'fifty_fifty' or "
                             f"'all_cens_cured', got {estimator}")

        self.estimator_ = estimator
        self.classifier_ = classifier
        self.random_state_ = random_state
        self.rnd_gen = np.random.default_rng(seed=random_state)
        self.n_features = -1
        self._is_fitted_ = False
        self._is_scar_ = False
        self.scale_ = None
        self.shape_ = None
        self.gamma_ = None

    def get_censor_label(self):
        """
        Return the value for the label to be utilized to pass 'censored' rows
        Parameters
        ----------
        None

        Returns
        -------
        censor_label : int
            Integer value to be used to denote censored training points
        """
        return self._censor_label

    def get_cure_label(self):
        """
        Return the value for the label to be utilized to indicate 'cured' rows
        Parameters
        ----------
        None

        Returns
        -------
        cure_label : int
            Integer value to be used to denote cured training points
        """
        return self._cure_label

    def get_non_cure_label(self):
        """
        Return the value for the label to be utilized to indicate 'non-cured'
        rows
        Parameters
        ----------
        None

        Returns
        -------
        non_cure_label : int
            Integer value to be used to denote non-cured training points
        """
        return self._non_cure_label

    def _reset_pu_fit(self):
        '''
        Resets the classifier built for estimating labels for censored
        individuals
        '''
        self.classifier_ = None
        self._is_fitted_ = False
        self._is_scar_ = False

    def pu_fit(self, training_data, training_labels, **kwargs):
        """
        Fits a model using censored and non censored inputs to estimate
        cured/non-cured labels

        Parameters
        ----------
        training_data : {array-like} of shape
                        (n_samples, n_features)
              Training data

        training_labels : {array-like} of shape (n_samples, 1)
            Labels for the training data.
            Value of _censor_label_ implies training point was censored,
            and _non_censor_label_ implies non censored.
            The _censor_label_ and _non_censor_label_ values can be obtained
            with get_censor_label and get_non_censor_label methods

        Optional kwargs include:

        pu_reg_term : float, default=0.5
            The strength of the quadratic regularization term (C*w^2) on the
            non-intercept model covariate weights for PU learning

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
            True if SCAR (selected completely at random) assumption holds for
            the dataset. In this situation, we find the probability of being NOT
            censored given covariates. The latter is then divided by an appropriate
            constant to get probability of being NOT cured.
            See the paper https://cseweb.ucsd.edu/~elkan/posonly.pdf for more
            details

        Returns
        --------
        self
            Fitted estimator.
        """

        training_data_, training_labels_, _ = self._validate_train_data(training_data,
                                                                        training_labels)

        _, self.n_features = training_data_.shape
        random_state = self.rnd_gen.integers(_LO_INT, _HI_INT)

        # Initialize classifier
        if self.classifier_ is None:
            self.classifier_ = LogisticRegression(random_state=random_state)

        self._is_scar_ = kwargs.get("is_scar", False)

        # Estimate values for censored individuals based on estimation method
        # chosen, if scar is false
        if not self._is_scar_:
            training_labels_estimated = self._estimate_censored_labels(training_data_,
                                                                       np.copy(training_labels_),
                                                                       **kwargs)
        else:
            training_labels_estimated = training_labels_

        self.classifier_.fit(training_data_, training_labels_estimated)
        self._is_fitted_ = True
        return self

    def predict_overall_survival(self, test_data, test_times, test_labels=None):
        '''
        Predict overall survival function for test data based on fitting
        survival function.
        Parameters
        ----------
        test_data : {array-like} of shape
                    (n_samples, n_features)
            Test data

        test_times : {array-like} of shape (n_samples, k)
            Times at which survival of an individual is to be determined. k can be greater than 1
            when the survival of an individual at multiple time points is to be determined

        test_labels: {array-like} of shape (n_samples, 1), default=None
            Test labels indicating censored/non censored status for test data.
            Method will provide cure predictions for censored individuals. This
            is only needed if model is fit with is_scar=True assumption

        Returns
        -------
        predictions : {array-like} of shape (n_samples, k)
            Overall survival function of any suspectible or non-susceptible
            individual
        '''
        test_data_ = self._validate_test_data(test_data)
        test_times_ = np.asarray(test_times)
        n_test, n_times = test_times_.shape
        if len(test_data_) != n_test:
            raise ValueError("Size of times and test_data do not match")

        probabilities = self.predict_cure_proba(test_data_, test_labels)
        if self.scale_ is None or self.shape_ is None or self.gamma_ is None:
            raise Exception("Fit survival function first by calling survival fit "
                            "or stochastic fit")

        survival = [[_overall_survival(test_times_[i, j], probabilities[i, 0], test_data_[i, :],
                                       self.scale_, self.shape_, self.gamma_)
                     for j in range(n_times)] for i in range(n_test)]
        return np.array(survival, np.float_)


    def predict_cure_labels(self, test_data, test_labels=None):
        """
        Predict cured/non-cured labels for data
        Parameters
        ----------
        test_data : {array-like} of shape
                    (n_samples, n_features)
            Test data

        test_labels: {array-like} of shape (n_samples, 1), default=None
            Test labels indicating censored/non censored status for test data.
            Method will provide cure predictions for censored individuals


        Returns
        -------
        predictions : {array-like} of shape (n_samples, 1)
            Predicted cured/non_cured labels

        """
        if not self._is_fitted_:
            raise Exception("This instance is not fitted yet. Call 'fit' first")

        test_data_ = self._validate_test_data(test_data)
        if self._is_scar_:
            if test_labels is None:
                raise ValueError("Model fit under SCAR Assumption. Must pass "
                                 "censor labels in test data")

            if len(test_labels) != len(test_data_):
                raise ValueError("Size of test_labels and test_data don't match")

            # Under scar assumption, labels are calculated a little differently
            probabilities = self._predict_proba(test_data_, test_labels, self._is_scar_)
            cure_labels = np.rint(probabilities[:, 1])
            return cure_labels

        return self.classifier_.predict(test_data_)

    def _predict_proba(self, test_data, test_labels, is_scar):
        '''
        Helper function that calculates probabilities of being cured/not cured
        under and not under SCAR assumption
        '''
        probabilities = self.classifier_.predict_proba(test_data)
        if self.classifier_.classes_[0] != self._cure_label:
            # Must flip probability columns for correct output
            cured_prob = probabilities[:, 1]
            probabilities[:, 1] = probabilities[:, 0]
            probabilities[:, 0] = cured_prob

        # Need to normalize under scar assumption
        if is_scar:
            noncens_probs = probabilities[:, 1][test_labels == 1]
            constant = noncens_probs.mean()
            probabilities[:, 1] /= constant
            non_cure_prob = probabilities[:, 1]
            non_cure_prob[non_cure_prob > 1.] = 1. # cut noncure probs off at 1
            probabilities[:, 1] = non_cure_prob
            probabilities[:, 0] = (1 - probabilities[:, 1])
        return probabilities

    def predict_cure_proba(self, test_data, test_labels=None):
        """
        Generate probability estimates for each training point as to whether it
        is cured/noncured
        Parameters
        ----------
        test_data : {array-like} of shape
                    (n_samples, n_features)
            Test data

        test_labels: {array-like} of shape (n_samples, 1), default=None
            Test labels indicating censored/non censored status for test data.
            Method will provide cure predictions for censored individuals

        Returns
        -------
        probabilities : {array-like} shape (n_samples, 2)
            Predicted cured, non_cured probabilities

        """
        if not self._is_fitted_:
            raise Exception("This instance is not fitted yet. Call 'fit' first")

        test_data_ = self._validate_test_data(test_data)

        if self._is_scar_:
            if test_labels is None:
                raise ValueError("Model fit under SCAR Assumption. Must pass "
                                 "censor labels in test data")

            test_labels_ = np.asarray(test_labels)
            if len(test_labels_) != len(test_data_):
                raise ValueError("Size of test_labels and test_data don't match")

        else:
            test_labels_ = None
        return self._predict_proba(test_data_, test_labels_, self._is_scar_)

    def get_params(self, deep=True):
        """
        Get parameters for the estimator

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for SurvivalAnalysis class and
            contained subojects

        Returns
        -------
        params : dict
            Parameter names for class mapped to values
        """
        param_dict = {"estimator_": self.estimator_, "random_state_": self.random_state_,
                      "n_features": self.n_features, "_is_fitted_": self._is_fitted_,
                      "scale_": self.scale_, "shape_": self.shape_, "gamma_": self.gamma_}

        if deep:
            if hasattr(self.classifier_, "get_params"):
                classifier_params = self.classifier_.get_params().items()
                for key, val in classifier_params:
                    param_dict["classifier__" + key] = val
        return param_dict

    def set_params(self, **params):
        """
        Set the parameters of this SurvivalAnalysis object.
        Parameters
        ----------
        **params : dict
            Estimator parameters

        Returns
        -------
        self : SurvivalAnalysis object
        """
        valid_params = self.get_params(deep=True)
        classifier_params = {}
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError("Invalid parameter {0} for SurvivalAnalysis class".format(key))
            if delim:
                classifier_params[sub_key] = value
            else:
                setattr(self, key, value)
                params[key] = value

        self.classifier_.set_params(**classifier_params)

        return self

    def score_labels(self, test_data, test_labels, sample_weight=None):
        """
        Returns the accuracy score on a given test_data provided labels
        Parameters
        ----------
        test_data : {array-like} of shape
                    (n_samples, n_features)
            test data

        test_labels : {array-like} of shape (n_samples, 1)
            Labels for the test data.

        sample_weight : array_like of shape (n_samples,), default = None
            Sample weights

        Returns
        -------
        score : float
            Mean accuracy
        """

        test_data_ = np.assarray(test_data)
        n_rows, _ = test_data.shape
        if n_rows != len(test_labels):
            raise ValueError(
                "test_data and test_labels do not have the same number of rows"
            )
        predicted_labels = self.predict_cure_labels(test_data_)

        return accuracy_score(
            test_labels, predicted_labels, sample_weight=sample_weight
        )

    def cindex(self, test_times, test_labels, danger):
        '''
        Returns the concordance index on test/score data. I.e. the ratio
        of concordant pairs to comparable ones.

        Parameters
        ----------
        test_times : {array-like} of shape (n_samples, 1)
            Column of times of event/times of censoring.

        test_labels : {array-like} of shape (n_samples, 1)
            Column indicating whether censored or event was observed

        danger : (float)
            Measures the risk of an individual. The higher the riskier. Sometimes
            taken as ln of proportionality factor (exponential term) from proportional
            hazard function. In this setting, a risk also incorporating the (non) cure
            probability is reasonable. E.g. ln(1-pi)+gamma*x, or something similar.

        Returns
        -------
        c-index: (float)
            Concordance index
        '''
        non_cure_label = self.get_non_cure_label()

        # Convert to boolean for use with sksurv
        test_labels_ = np.asarray(test_labels) == non_cure_label
        return cindex(test_labels_, test_times, danger)[0]

    def _fit_weights(self, training_data, training_labels, training_times, **kwargs):
        '''
        Fit the lifetime parameters and survival function and returns the fit
        weights
        '''
        # We first need to fit a model to estimate labels
        if not self._is_fitted_:
            self.pu_fit(training_data, training_labels, **kwargs)


        if self._is_scar_:
            probabilities = self.predict_cure_proba(training_data, training_labels)[:, 0]


        else:
            probabilities = self.predict_cure_proba(training_data)[:, 0]

        weights, errors, log_likelihood = \
                _survival_fit_weights(training_data, training_labels, training_times,
                                      probabilities, **kwargs)

        return weights, errors, log_likelihood

    def survival_fit(self, training_data, training_labels, training_times, **kwargs):
        '''
        Fits the lifetime parameters and survival function and returns a fitted
        object
        Parameters
        ----------
        training_data : {array-like} of shape (n_samples, n_features)
            Training data

        training_labels : {array-like} of shape (n_samples, n_features)
            Labels for training_data

        training_times : {array-like} of shape (n_samples, 1)
            Times for all training points (censoring time or event time)

        Optional kwargs include:

        surv_reg_term : float, default=0.5
            The strength of the quadratic regularization term (C*w^2) on the
            non-intercept model covariate weights for survival fit.

        surv_max_iter: int, default=1000
            maximum number of iterations for SLSQP method.

        is_scar : {True, False}, default=False
            True if SCAR (selected completely at random) assumption holds for
            the dataset. In this situation, we find the probability of being NOT
            censored given covariates. The latter is then divided by an appropriate
            constant to get probability of being NOT cured.
            See the paper https://cseweb.ucsd.edu/~elkan/posonly.pdf for more
            details

        Returns
        -------
        self
            Fitted estimator.

        '''
        training_data_, training_labels_, training_times_ = self._validate_train_data(
            training_data, training_labels, training_times)

        weights, _, _ = self._fit_weights(training_data_, training_labels_,
                                          training_times_, **kwargs)
        _, self.n_features = training_data_.shape

        self.shape_ = weights[0]
        self.scale_ = weights[1]
        self.gamma_ = weights[2:]

        return self

    def stochastic_fit(self, training_data, training_labels, training_times, # pylint: disable=too-many-arguments, too-many-locals
                       **kwargs):
        '''
        Fits the lifetime parameters and survival function and returns a fitted
        object. This is a meta algorithm heuristic for large datasets where
        survival_fit does not scale well, due to the computational bottleneck
        of censored individuals.

        Step 1: split into smaller datasets
        Step 2: for each dataset run survival fit
        Step 3: as final output take average of parameters.

        Parameters
        ----------
        training_data : {array-like} of shape (n_samples, n_features)
            Training data

        training_labels : {array-like} of shape (n_samples, n_features)
            Labels for training_data

        training_times : {array-like} of shape (n_samples, 1)
            Times for all training points (censoringtime or event time)

        Optional kwargs include:

        surv_reg_term : float, default=0.5
            strength of regularization parameter

        surv_max_iter: int, default=1000
            maximum number of iterations for SLSQP method.

        surv_batch_size : int, default=200
            The maximum size for a batch for stochastic fit

        is_scar : {True, False}, default=False
            True if SCAR (selected completely at random) assumption holds for
            the dataset. In this situation, we find the probability of being NOT
            censored given covariates. The latter is then divided by an appropriate
            constant to get probability of being NOT cured.
            See the paper https://cseweb.ucsd.edu/~elkan/posonly.pdf for more
            details

        Returns
        -------
        self
            Fitted estimator.
        '''

        training_data_, training_labels_, training_times_ = self._validate_train_data(
            training_data, training_labels, training_times)
        kwargs = _validate_kwargs(kwargs)
        batch_size = kwargs.get("surv_batch_size")
        _, self.n_features = training_data_.shape
        is_scar = kwargs.get("is_scar", False)

        if is_scar:
            # Things are fast under SCAR assumption. Can stick to one split
            batch_size = len(training_data_)
            n_splits = 1
        else:
            n_cens = training_labels_[training_labels_ == _CENSOR_LABEL].shape[0]
            n_rows = training_labels_.shape[0]
            n_splits = int(np.ceil(n_cens/batch_size))

        shuffled = self.rnd_gen.permutation(n_rows)
        print(f"Splitting up data into {n_splits} pieces")
        outputs = []
        for i in tqdm(range(n_splits)):
            # Must reset cure label estimator each time
            self._reset_pu_fit()
            indices = shuffled[batch_size*i:batch_size*(i+1)]
            data_chunk = training_data_[indices]
            label_chunk = training_labels_[indices]
            time_chunk = training_times_[indices]

            weights, _, _ = self._fit_weights(data_chunk, label_chunk, time_chunk,
                                              **kwargs)
            outputs.append(weights)

        mean_weights = np.mean(outputs, axis=0)
        self.shape_ = mean_weights[0]
        self.scale_ = mean_weights[1]
        self.gamma_ = mean_weights[2:]
        return self
