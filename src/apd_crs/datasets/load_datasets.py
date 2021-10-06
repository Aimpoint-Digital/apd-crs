"""
Makes available datasets for testing survival analysis methods
Primary Author: Nem Kosovalic (nem.kosovalic@aimpointdigital.com)
Secondary Author: Yash Puranik (yash.puranik@aimpointdigital.com)
Company: Aimpoint Digital, LP
"""
from os.path import abspath, dirname, join
import pandas as pd  # type: ignore
from sklearn.utils import Bunch  # type: ignore
from apd_crs._parameters import _CENSOR_LABEL, _NON_CURE_LABEL  # type: ignore

PATH = dirname(abspath(__file__))
TELCO_FILE = join(PATH, "Telco-Customer-Churn.csv")
ADVERTISING_FILE = join(PATH, "advertising.csv")
MANUFACTURING_FILE = join(PATH, "manufacturing.csv")
MELANOMA_FILE = join(PATH, "melanoma.csv")


def return_data(data_frame, feature_names, target_label, time_label, return_X): # pylint: disable=invalid-name
    '''
    Returns the dataframe to user
    '''

    data = data_frame[feature_names]
    training_labels = data_frame[target_label]
    if time_label is not None:
        training_times = data_frame[time_label]
    else:
        training_times = None

    if return_X:
        if time_label is not None:
            return (data, training_labels, training_times)
        return (data, training_labels)

    return Bunch(data=data, target=training_labels, target_times=training_times,
                 feature_names=feature_names, target_name=target_label, target_time=time_label)

def load_melanoma(return_X_y=False):  # pylint: disable=invalid-name
    '''
    Returns a melanoma dataset from:
        https://stat.ethz.ch/R-manual/R-patched/library/boot/html/melanoma.html


    Parameters
    -----------
    return_X_y : {bool}
        If True, output is returned as (training_data, training_label)

    Returns
    -------
    data : Bunch
        The object has the following attributes:

        data : {dataframe} of shape (205, 5)
            The data matrix.

        target : {Series) of shape (205, )
            The classification target.

        target_times: (205, )
            The time at which data was cured/censored.

        features_names : list
            The names of columns

        target_name : str
            The columns containing target class

        target_time : str
            The column containing target time

        (data, target) : tuple if return_X_y is True
    '''
    data_frame = pd.read_csv(MELANOMA_FILE)
    time_label = "time"
    target_label = "status"
    feature_names = ["sex", "age", "year", "thickness", "ulcer"]
    data_frame[target_label].replace({2: _CENSOR_LABEL, 3: _CENSOR_LABEL, 1: _NON_CURE_LABEL},
                                     inplace=True)

    return return_data(data_frame, feature_names, target_label, time_label, return_X_y)

def load_advertising(return_X_y=False):  # pylint: disable=invalid-name
    '''
    Returns an advertising dataset available on kaggle at
    https://www.kaggle.com/farhanmd29/predicting-customer-ad-clicks/data

    Parameters
    -----------
    return_X_y : {bool}
        If True, output is returned as (training_data, training_label)

    Returns
    -------
    data : Bunch
        The object has the following attributes:

        data : {dataframe} of shape (1000, 9)
            The data matrix.

        target : {Series) of shape (1000, )
            The classification target.

        target_times: None
            This dataset is for a PU learning problem. No target times available

        features_names : list
            The names of columns

        target_name : str
            The columns containing target class

        target_time : None
            This dataset is for a PU learning problem. No target times available

        (data, target) : tuple if return_X_y is True
    '''
    data_frame = pd.read_csv(ADVERTISING_FILE)
    time_label = None
    target_label = "Clicked on Ad"
    data_frame[target_label].replace({0: _CENSOR_LABEL, 1: _NON_CURE_LABEL}, inplace=True)

    feature_names = ["Daily Time Spent on Site", "Age", "Area Income", "Daily Internet Usage",
                     "Ad Topic Line", "City", "Male", "Country", "Timestamp"]

    return return_data(data_frame, feature_names, target_label, time_label, return_X_y)

def load_manufacturing(return_X_y_t=False):  # pylint: disable=invalid-name
    '''
    Returns the manufacturing dataset available in the pysurvival library. For more details, see:
        https://square.github.io/pysurvival/tutorials/maintenance.html

    Parameters
    -----------
    return_X_y_t : {bool}
        If True, output is returned as (training_data, training_label, training_times)

    Returns
    -------
    data : Bunch
        The object has the following attributes:

        data : {dataframe} of shape (1000, 5)
            The data matrix.

        target : {Series) of shape (1000, )
            The classification target.

        target_times : {Series) of shape (1000, )
            The time at which data was cured/censored.

        features_names : list
            The names of columns

        target_name : str
            The columns containing target class

        target_time : str
            The column containing target time

        (data, target) : tuple if return_X_y is True
    '''
    data_frame = pd.read_csv(MANUFACTURING_FILE)
    time_label = "lifetime"
    target_label = "broken"

    data_frame[target_label].replace({0: _CENSOR_LABEL, 1: _NON_CURE_LABEL}, inplace=True)

    feature_names = ["pressureInd", "moistureInd", "temperatureInd", "team", "provider"]

    return return_data(data_frame, feature_names, target_label, time_label, return_X_y_t)

def load_telco(return_X_y_t=False):  # pylint: disable=invalid-name
    '''
    Returns the telco dataset available on kaggle at
    https://www.kaggle.com/blastchar/telco-customer-churn

    Parameters
    -----------
    return_X_y_t : {bool}
        If True, output is returned as (training_data, training_label, training_times)

    Returns
    -------
    data : Bunch
        The object has the following attributes:

        data : {dataframe} of shape (7043, 19)
            The data matrix.

        target : {Series) of shape (7043, )
            The classification target.

        target_times : {Series) of shape (7043, )
            The time at which data was cured/censored.

        features_names : list
            The names of columns

        target_name : str
            The columns containing target class

        target_time : str
            The column containing target time

        (data, target) : tuple if return_X_y is True
    '''
    data_frame = pd.read_csv(TELCO_FILE)
    time_label = "tenure"
    target_label = "Churn"
    data_frame[target_label].replace({"No": _CENSOR_LABEL, "Yes": _NON_CURE_LABEL}, inplace=True)
    feature_names = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
                     "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                     "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                     "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
                     "MonthlyCharges", "TotalCharges"]
    data_frame["TotalCharges"] = pd.to_numeric(data_frame["TotalCharges"], errors="coerce")
    return return_data(data_frame, feature_names, target_label, time_label, return_X_y_t)
