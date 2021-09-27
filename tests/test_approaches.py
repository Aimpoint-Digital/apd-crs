"""
Provides methods to generate random datasets and run tests comparing Hard-EM algorithm against other
baseline approaches
Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)
Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)
Company: Aimpoint Digital, LP
"""
import os
import sys
import copy

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
try:
    from survival_analysis import SurvivalAnalysis # type: ignore # pylint: disable=import-error
except ImportError:
    # This is bit of a hack :-(
    __CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    __PARENT_DIR = os.path.dirname(__CURRENT_DIR)
    __SRC_DIR = os.path.join(__PARENT_DIR, "src")
    sys.path.append(__SRC_DIR)
    from survival_analysis import SurvivalAnalysis # type: ignore
from make_df import create_df


def score_model(model, test_data, test_labels):
    '''
    Provides metrics for the model on test_data
    '''
    pred_labels = model.predict_cure_labels(test_data)
    pred_scores = model.predict_cure_proba(test_data)[:, 0]
    accuracy = accuracy_score(test_labels, pred_labels)
    auc_score = roc_auc_score(test_labels, pred_scores)
    loss = log_loss(test_labels, pred_scores)
    return accuracy, auc_score, loss


def score_approach(training_data, training_labels, true_labels, test_data, # pylint: disable=too-many-arguments
                   true_test_labels, **kwargs):
    '''
    Applies hard-em approach along with three other baseline approaches for PU learning
    and tabulates the accuracy scores for each of the approaches, both in sample and out of sample
    **kwargs helps build models as needed
    '''
    kwargs_copy = copy.copy(kwargs)
    init_args = {}
    for key in ("estimator", "classifier", "random_state"):
        val = kwargs_copy.pop(key, None)
        if val is not None:
            init_args[key] = val

    model = SurvivalAnalysis(**init_args)
    model.pu_fit(training_data, training_labels, **kwargs_copy)
    train_scores = score_model(model, training_data, true_labels)
    test_scores = score_model(model, test_data, true_test_labels)
    return {"train_scores": train_scores, "test_scores": test_scores}


def score_all_approaches(training_data, training_labels, true_labels, test_data,
                         true_test_labels):
    '''
    Applies four approaches on data and compares on both training and test data
    '''
    hard_em_scores = score_approach(training_data, training_labels, true_labels, test_data,
                                    true_test_labels, pu_suppress=False)
    cluster_scores = score_approach(training_data, training_labels, true_labels, test_data,
                                    true_test_labels, estimator="clustering", pu_suppress=False)
    random_scores = score_approach(training_data, training_labels, true_labels, test_data,
                                   true_test_labels, estimator="fifty_fifty")
    all_cured_scores = score_approach(training_data, training_labels, true_labels, test_data,
                                      true_test_labels, estimator="all_cens_cured")
    return (hard_em_scores, cluster_scores, random_scores, all_cured_scores)


if __name__ == "__main__":
    COLUMN_NAMES = ["x1", "x2", "x3"]
    DIST_PARAMETERS = [[1, 2], [1, 2], [1, 2]]
    N_ROWS = 200
    SEED = 42
    MODEL_WEIGHTS = [0.3, 0.1, 0.8, 0.6]
    CENSORING_PROB = 0.5
    DATA_FRAME = create_df(COLUMN_NAMES, DIST_PARAMETERS, N_ROWS, MODEL_WEIGHTS, CENSORING_PROB,
                           SEED)
    TRAIN_FRAME, TEST_FRAME = train_test_split(DATA_FRAME, test_size=0.2, random_state=SEED,
                                               shuffle=True)
    TRAINING_DATA, TRAINING_LABELS, TRUE_LABELS = TRAIN_FRAME[["x1", "x2", "x3"]], \
            TRAIN_FRAME["label"], TRAIN_FRAME["cure_label"]
    TEST_DATA, TRUE_TEST_LABELS = TEST_FRAME[["x1", "x2", "x3"]], TEST_FRAME["cure_label"]
    SCORES = score_all_approaches(TRAINING_DATA, TRAINING_LABELS, TRUE_LABELS, TEST_DATA,
                                  TRUE_TEST_LABELS)
