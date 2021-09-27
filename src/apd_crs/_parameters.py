"""
Identifies all parameters used within the package
Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)
Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)
Company: Aimpoint Digital, LP
"""

# Label for censored data
_CENSOR_LABEL = 2

# No separate label exists for non censored data, as non censored data is also
# non cured by way of study design

# Label for uncured data
_NON_CURE_LABEL = 1

# Label for cured data
_CURE_LABEL = 0

# Integers to sample between for random_state initialization in clustering
_LO_INT = 1
_HI_INT = 10000000


# Epsilon threshold for enforcing a label is 0 or 1
_EPSILON = 0.001

# SLSQP return status codes. defined here:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html
_SLSQP_INFEAS = 4
_SLSQP_ITER_LIM = 9

# Use bound to enforce nonzero values for scale and shape parameters
_ZERO_BOUND = 0.001

# Use bound to detect poorly scaled models for survival fitting
_SCALETOL = 5

# Bound for shape parameter
_SHAPE_BOUND = 5

# Maximum allowed value for exp before it explodes
_MAXEXP = 20

# Default arguments for fit functions
_DEFAULT_ARGS = {"pu_reg_term": 0.5,             # Regularization term for PU Fit
                 "pu_initialize": "use_random",  # Initialization method for slsqp for pu fit
                 "pu_max_guesses": 50,           # Max slsqp guesses to run for PU fit
                 "pu_max_processes": 1,          # Max parallel processes to exploit for pu fit
                 "pu_max_iter": 1000,            # Max slsqp iterations for pu fit
                 "pu_weight_lo": -0.5,           # Lower bound for sampling for pu initialization
                 "pu_weight_hi": 0.5,            # Lower bound for sampling for pu initialization
                 "pu_suppress": True,            # Whether to suppress slsqp messages for PU
                 "pu_kmeans_init": 20,           # Number of kmeans initializations for pu fit
                 "surv_reg_term": 0.5,           # Regularization term for survival fit
                 "surv_max_iter": 1000,          # Max slsqp iterations for survival fit
                 "surv_batch_size": 200,         # Max batch size in stochastic fit
                 "surv_supress": True,           # Whether to suppress slsqp for survival fit
                 "is_scar": False                # Is scar assumption satisfied
                }

# Zero tolerance for censoring/cure times to be nonzero
_ZEROTOL = 1e-3
