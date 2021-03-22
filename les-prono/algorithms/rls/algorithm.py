import datetime as dt
import logging
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import algorithms.useful_functions as uf


def predict(loc, dcfg):
    """Every time this is called a whole prediction is computed."""
    locname = loc["name"]
    rlscfg = dcfg["algorithm"]["rls"]  # algorithm parameters
    logging.debug("Load configuration")

    # get last image's datetime and needed lead times
    imgs_list, last_img, last_dtime, c_dtime = uf.analyze_psat(
        dcfg, True
    )
    logging.debug("Check last img dtime")

    period = uf.get_consistent_period(
        imgs_list
    )  # assume constant period
    needed_lts = uf.get_needed_lead_times(
        dcfg["forecasts"]["limit"], period
    )
    logging.debug("Get needed lead times")

    aux_variables, dict_of_filters = load_variables(
        locname, rlscfg, needed_lts
    )
    logging.debug("Start predict")

    if (
        aux_variables[locname]["last_run_dtime"] < c_dtime
    ) and uf.is_name_of_day_img(  # if new image
        last_img
    ):
        input_vector = get_input_vector(
            loc, last_img, imgs_list, rlscfg
        )
        logging.debug("Get input vector from images")

        aux_variables[locname]["last_input_vector"] = input_vector
        old_inputs = update_old_inputs(
            input_vector, aux_variables[locname]
        )
        old_target = uf.get_pixel(loc, last_img)
        for lt, rls_filter in dict_of_filters[locname].items():
            rls_filter.update(old_target, old_inputs[:, -int(lt)])
        logging.debug("Update every filters' coefficients")

    else:
        logging.debug("Reuse old forecast")

    new_predictions = {"predictions": {}, "inf": {}}
    for lt, rls_filter in dict_of_filters[locname].items():
        prediction = rls_filter.predict(
            aux_variables[locname]["last_input_vector"]
        )
        new_predictions["predictions"][lt] = prediction
        new_predictions["inf"][lt] = 215
    logging.debug("Create new predictions and inf")

    df2, last_run_dtime = update_forecast(
        loc,
        dcfg["forecasts"]["time_step"],
        new_predictions,
        last_dtime,
        needed_lts,
        period,
        limit=dcfg["forecasts"]["limit"],
    )
    logging.info("Make new forecast")

    history_file_path = get_history_file_path(
        dcfg, last_run_dtime, loc
    )
    df2.to_csv(history_file_path)

    save_variables(
        locname, aux_variables, dict_of_filters, last_run_dtime
    )
    logging.debug("Finish predict")


### psat stats functions
def analyze_psat(dcfg):
    """Returns list of images in psat (only the name), the Path of the last
    image, and the datetime for the last image"""
    imgs_list = sorted(
        [
            img
            for img in os.listdir(dcfg["paths"]["psat"])
            if img.endswith("rp.npy")
        ]
    )
    last_img = Path(dcfg["paths"]["psat"]) / imgs_list[-1]
    last_dtime = uf.get_dtime(last_img.name)
    return imgs_list, last_img, last_dtime


def get_imgs_mask(imgs_list):
    """Returns a mask with True if there is an image when it should be
    and False if there is no image near that datetime.
    Assumes constant period."""
    period, dtimes = uf.get_consistent_period(
        imgs_list, return_dtimes=True
    )
    dtimes_to_evaluate = [
        dtimes[0] + period * i
        for i in range(round((dtimes[-1] - dtimes[0]) / period))
    ]
    mask = []
    for adtime in dtimes_to_evaluate:
        isthere = False
        for imgdtime in dtimes:
            if (adtime - imgdtime) < (period / 8):
                isthere = True
        mask.append(isthere)
    return mask


### variables handler functions
def load_variables(locname, rlscfg, needed_lts):
    """Loads pickles in this folder or creates them if nonexistent"""
    # load variables
    aux_variables_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "aux_variables.pkl",
    )
    dict_of_filters_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "dict_of_filters.pkl",
    )

    try:
        with open(aux_variables_path, "rb") as f:
            aux_variables = pickle.load(f)
    except FileNotFoundError:
        logging.warning("There is no aux_variables.pkl")
        aux_variables = {}

    if not locname in aux_variables:
        dim = sum(rlscfg["n_values"]) + 1  # input dimension
        aux_variables[locname] = {
            "last_run_dtime": (
                dt.datetime.now(dt.timezone.utc)  # long ago
                - dt.timedelta(days=1000)
            ),
            "last_input_vector": 0.5 + 0.5 * np.random.randn(dim),
            "old_inputs": np.zeros((dim, needed_lts)),
        }

    try:
        with open(dict_of_filters_path, "rb") as f:
            dict_of_filters = pickle.load(f)
            # reset if number of filters is not ok
        assert needed_lts == len(
            list(dict_of_filters.values())[0].keys()
        )
    except FileNotFoundError:
        logging.warning("There is no dict_of_filters.pkl")
        dict_of_filters = {}
    except AssertionError:
        logging.warning(
            "dict_of_filters.pkl has bad number of filters"
        )
        dict_of_filters = {}

    if not locname in dict_of_filters:
        dim = sum(rlscfg["n_values"]) + 1  # input dimension
        dict_of_filters[locname] = {
            lt: FilterRLS(
                dim, w="random"
            )  # aux_variables[locname]['last_input_vector'])
            for lt in range(1, needed_lts + 1)
        }

    return aux_variables, dict_of_filters


def update_old_inputs(input_vector, aux_variables_lname):
    """ updates aux_variables by modifying aux_variables[locname]"""
    old_inputs = aux_variables_lname["old_inputs"]
    shifted_old_inputs = np.roll(
        old_inputs, -1, axis=1
    )  # put first col last
    shifted_old_inputs[:, -1] = np.array(
        input_vector
    )  # insert new value
    aux_variables_lname["old_inputs"] = shifted_old_inputs  # update

    aux_variables_lname["last_input_vector"] = input_vector
    return aux_variables_lname["old_inputs"]


def save_variables(
    locname, aux_variables, dict_of_filters, last_run_dtime
):
    # update last run datetime and account for 'else' clause
    aux_variables[locname]["last_run_dtime"] = (
        last_run_dtime  # to account for 'else'
        - dt.timedelta(minutes=1)
    )

    # save variables
    aux_variables_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "aux_variables.pkl",
    )
    pickle.dump(aux_variables, open(aux_variables_path, "wb"))

    dict_of_filters_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "dict_of_filters.pkl",
    )
    pickle.dump(dict_of_filters, open(dict_of_filters_path, "wb"))


### img operations
def get_input_vector(loc, last_img, imgs_list, rlscfg):
    """Returns input vector if img exists or interpolates the value
    otherwise"""
    max_lag = max(rlscfg["n_values"])
    sizes, lags = rlscfg["winsizes"], rlscfg["n_values"]
    imgs_exist = get_imgs_mask(imgs_list)

    vector, mask = [], []
    for i, size in enumerate(sizes):
        for lag in range(1, lags[i] + 1):
            if imgs_exist[-lag]:
                value = uf.get_value(
                    loc, size, last_img.parent / imgs_list[-lag]
                )
                vector.append(value)
                mask.append(True)
            else:
                vector.append(-1)
                mask.append(False)

    vector, mask = np.array(vector), np.array(mask)
    vector[~mask] = np.interp(
        np.arange(len(vector))[~mask],
        np.arange(len(vector))[mask],
        vector[mask],
    )
    vector, mask = np.append(vector, 1), np.append(mask, True)
    return vector  # , flag


### forecast update functions
def update_forecast(
    loc,
    time_step,
    predictions,
    last_dtime,
    needed_lts,
    period,
    limit=4,
):
    """Update forecasts file up to limit hours"""
    last_run_datetime = dt.datetime.now(dt.timezone.utc)

    df1 = uf.create_df1_from_dict(
        predictions, last_dtime, period, needed_lts
    )
    df2 = uf.df1_to_df2(df1, last_run_datetime, time_step, limit)

    pred_dir_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "predictions",
        loc["name"],
    )
    uf.write_to_daydf(df2, pred_dir_path, loc)

    return df2, last_run_datetime


def get_history_file_path(dcfg, last_dtime, loc):
    history_file_path = Path(
        os.path.join(
            dcfg["paths"]["history"],
            "rls",
            loc['name'],
            str(last_dtime.date()),
            str(last_dtime).replace(" ", "_") + ".csv",
        )
    )
    history_file_path.parent.mkdir(parents=True, exist_ok=True)
    return history_file_path


def load_pandas_series(filename):
    df1 = pd.read_csv(
        filename,
        index_col=0,
        names=["", "values"],
        header=None,
        parse_dates=True,
    )["values"]
    df1.name = None
    df1 = df1[~df1.index.isnull()]
    return df1


def next_step_aux(dtime, time_step):
    maxi = 60 // time_step
    if dtime.minute // time_step + 1 == maxi:
        next_hour = 1
    else:
        next_hour = 0
    next_minute = (time_step * (dtime.minute // time_step + 1)) % 60
    return next_hour, next_minute


def next_step_starts(dtime, time_step):
    next_hour, next_minute = next_step_aux(dtime, time_step)
    start = dt.datetime(
        dtime.year,
        dtime.month,
        dtime.day,
        dtime.hour + next_hour,
        next_minute,
    )
    return start


### RLS filter class and functions  (modified from padasip)
class FilterRLS:
    """
    Adaptive RLS filter.
    
    **Args:**

    * `n` : length of filter (integer) - how many input is input array
      (row of input matrix)

    **Kwargs:**

    * `mu` : forgetting factor (float). It is introduced to give exponentially
      less weight to older error samples. It is usually chosen
      between 0.98 and 1.

    * `eps` : initialisation value (float). It is usually chosen
      between 0.1 and 1.

    * `w` : initial weights of filter. Possible values are:
        
        * array with initial weights (1 dimensional array) of filter size
    
        * "random" : create random weights
        
        * "zeros" : create zero value weights
    """

    def __init__(self, n, mu=0.999, eps=0.001, w="random"):
        self.kind = "RLS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError("The size of filter must be an integer")
        self.mu = self.check_float_param(mu, 0, 1, "mu")
        self.eps = self.check_float_param(eps, 0, 1, "eps")
        self.init_weights(w, self.n)
        self.P = 1 / self.eps * np.identity(n)

    def predict(self, x):
        """
        This function calculates the new output value `y` from input array `x`.
        **Args:**
        * `x` : input vector (1 dimension array) in length of filter.
        **Returns:**
        * `y` : output value (float) calculated from input array.
        """
        y = np.dot(self.w.flatten(), x.flatten())
        return y

    def update(self, d, x_lagged_h):
        """
        Update weights according to one desired value and its input.
        Remember that the input must be lagged h observations.

        **Args:**

        * `d` : desired value (float)

        * `x` : input array (1-dimensional array)
        """
        z = np.reshape(x_lagged_h, (x_lagged_h.size, 1))
        K = (self.P @ z) / (self.mu + z.T @ self.P @ z)
        self.P = (self.P - K @ z.T @ self.P) / self.mu
        self.w += K @ (d - z.T @ self.w)

    def check_float_param(self, param, low, high, name):
        """
        Check if the value of the given parameter is in the given range
        and a float.
        Designed for testing parameters like `mu` and `eps`.
        To pass this function the variable `param` must be able to be converted
        into a float with a value between `low` and `high`.
        **Args:**
        * `param` : parameter to check (float or similar)
        * `low` : lowest allowed value (float), or None
        * `high` : highest allowed value (float), or None
        * `name` : name of the parameter (string), it is used for an error message
            
        **Returns:**
        * `param` : checked parameter converted to float
        """
        try:
            param = float(param)
        except:
            raise ValueError(
                "Parameter {} is not float or similar".format(name)
            )
        if low != None or high != None:
            if not low <= param <= high:
                raise ValueError(
                    "Parameter {} is not in range <{}, {}>".format(
                        name, low, high
                    )
                )
        return param

    def init_weights(self, w, n=-1):
        """
        This function initialises the adaptive weights of the filter.
        **Args:**
        * `w` : initial weights of filter. Possible values are:
        
            * array with initial weights (1 dimensional array) of filter size
        
            * "random" : create random weights
            
            * "zeros" : create zero value weights
       
        **Kwargs:**
        * `n` : size of filter (int) - number of filter coefficients.
        **Returns:**
        * `y` : output value (float) calculated from input array.
        """
        if n == -1:
            n = self.n
        if type(w) == str:
            if w == "random":
                w = np.random.normal(0, 0.5, n)
            elif w == "zeros":
                w = np.zeros(n)
            else:
                raise ValueError("Impossible to understand the w")
        elif len(w) == n:
            try:
                w = np.array(w, dtype="float64")
            except:
                raise ValueError("Impossible to understand the w")
        else:
            raise ValueError("Impossible to understand the w")
        self.w = w

 
def MSE(x1, x2=-1):
    """ Mean squared error - this function accepts two series of data or 
    directly one series with error.

    Args:
        x1 ([type]): first data series or error (1d array)
        x2 (int, optional): second series (1d array) if first series was not error directly,\\
            then this should be the second series. Defaults to -1.

    Returns:
        float: mean squared error
    """
    e = get_valid_error(x1, x2)
    return np.dot(e, e) / float(len(e))


    """
    
    
    **Args:**
    * `x1` - first data series or error (1d array)
    **Kwargs:**
    * `x2` - second series (1d array) if first series was not error directly,\\
        then this should be the second series
    **Returns:**
    * `e` - mean error value (float) obtained directly from `x1`, \\
        or as a difference of `x1` and `x2`
    """
 
def get_mean_error(x1, x2=-1, function="MSE"):
    """ This function returns desired mean error. Options are: MSE, MAE, RMSE

    Args:
        x1 (1d array): first data series or error
        x2 (1d array or int, optional): second series if first series was not error directly,\\
        then this should be the second series. Defaults to -1.
        function (str, optional): Type of error [MAE, RMSE, MSE]. Defaults to "MSE".

    Raises:
        ValueError: when function is not 'MSE'

    Returns:
        float: mean error value obtained directly from `x1`, \\
        or as a difference of `x1` and `x2`
    """
    if function == "MSE":
        return MSE(x1, x2)
    else:
        raise ValueError("The provided error function is not known")


def get_valid_error(x1, x2=-1):
    """
    Function that validates:
    * x1 is possible to convert to numpy array
    * x2 is possible to convert to numpy array (if exists)
    * x1 and x2 have the same length (if both exist)
    """
    # just error
    if type(x2) == int and x2 == -1:
        try:
            e = np.array(x1)
        except:
            raise ValueError(
                "Impossible to convert series to a numpy array"
            )
    # two series
    else:
        try:
            x1 = np.array(x1)
            x2 = np.array(x2)
        except:
            raise ValueError(
                "Impossible to convert one of series to a numpy array"
            )
        if not len(x1) == len(x2):
            raise ValueError("The length of both series must agree.")
        e = x1 - x2
    return e

