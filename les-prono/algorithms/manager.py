import copy
import datetime as dt
import glob
import importlib
from itertools import chain
import os
import logging
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder
import pytz
import yaml

import algorithms.useful_functions as uf
from algorithms.new_CIM_module.fGenerar_CIM_GHI import (
    fGenerar_CIM_de_FR as alf,
)  # agustin laguarda function
from algorithms.CIM_module.funcion_CIMGHI_dummy import (
    generar_CIM_GHI_dumy as alf_old,
)  # agustin laguarda function


"""
This script executes the selected algorithms and combines them.
"""


def main():
    """Runs all algorithms and postprocess the results.
    """
    dcfg = uf.initialize("algorithms")
    for name, method in dcfg["algorithm"].items():
        if method["enabled"]:
            start = dt.datetime.now()
            module = importlib.import_module(
                "algorithms." + name + ".main_program"
            )
            module.main_program()
            end = dt.datetime.now()
            print(name, end - start)
    postprocessing(dcfg)
    average_over_locations(dcfg)


def average_over_locations(dcfg):
    """Computes the Rp and GHI average over all locations in order to give the
    aggregated forecast

    Args:
        dcfg (dict): configuration dict
    """
    # first find out the last date
    location_dirs = [
        dirname
        for dirname in os.listdir("predictions")
        if not dirname == "PXX"
    ]
    filenames = sorted(
        [
            Path(
                sorted(
                    glob.glob(
                        os.path.join(
                            "predictions",
                            locdir,
                            "ghi",
                            "predictions",
                            "*.csv",
                        )
                    )
                )[-1]
            )
            for locdir in location_dirs
        ]
    )
    last_filename = filenames[-1].name
    # load dataframes
    pred_loc_dataframes, gt_loc_dataframes = [], []
    for locdir in location_dirs:
        locfilename = os.path.join(
            "predictions", locdir, "ghi", "predictions", last_filename,
        )
        if os.path.isfile(locfilename):
            pred_loc_dataframes.append(uf.load_pandas_dataframe(locfilename))
            gt_loc_dataframes.append(
                uf.load_pandas_dataframe(
                    rreplace(locfilename, "predictions", "groundtruth", 1)
                )
            )
    pred_concat = pd.concat(pred_loc_dataframes)
    gt_concat = pd.concat(gt_loc_dataframes)

    pred_by_row_index = pred_concat.groupby(pred_concat.index)
    pred_means = pred_by_row_index.mean()

    target_filename = Path(f"predictions/PXX/ghi/predictions/{last_filename}")
    target_filename.parent.mkdir(parents=True, exist_ok=True)
    pred_means.to_csv(target_filename)

    gt_by_row_index = gt_concat.groupby(gt_concat.index)
    gt_means = gt_by_row_index.mean()

    gt_target_filename = Path(
        rreplace(str(target_filename), "predictions", "groundtruth", 1)
    )
    gt_target_filename.parent.mkdir(parents=True, exist_ok=True)
    gt_means.to_csv(gt_target_filename)


def postprocessing(dcfg):
    """ Iterates over locations and calls <postprocess_location> for enabled locations.

    Args:
        dcfg (dict): configuration dicts
    """
    # load forecasts in dataframe
    for loc in dcfg["locations"].values():
        logging.debug("--- Start location postprocessing")
        if loc["enable"]:  # for every location
            postprocess_location(loc, dcfg)
        logging.debug("--- End location postprocessing")


def postprocess_location(loc, dcfg):
    """ Loads the forecasts, combines them, zeroes the night values, converts to GHI and saves everything.

    Args:
        loc (dict): location dictionary
        dcfg (dict): configuration dictionary
    """
    forecasts_df, today = get_forecasts_df(loc, dcfg)
    combiner = Combiner(dcfg)
    ndf = combiner(forecasts_df)
    ts_out = zero_night_values(ndf, loc)
    output_filename, rpgtname, ghiname, ghigtname = get_dest_paths(loc, today)
    ts_out = save_update_only(
        dcfg, loc, ts_out, output_filename, rpgtname, ghiname, ghigtname
    )


def save_update_only(
    dcfg, loc, ts_out, output_filename, rpgtname, ghiname, ghigtname
):
    """ Saves results only where they are different from past results (in the future!).
    Conversion to GHI is called here.

    Args:
        dcfg ([type]): [description]
        loc ([type]): [description]
        ts_out ([type]): [description]
        output_filename ([type]): [description]
        rpgtname ([type]): [description]
        ghiname ([type]): [description]
        ghigtname ([type]): [description]
    """
    now = dt.datetime.now(dt.timezone.utc)
    if os.path.isfile(output_filename):
        last_df = uf.load_pandas_dataframe(output_filename)
        mask = pd.Series(ts_out.index < now, index=ts_out.index)
        ts_out = ts_out.mask(mask)
        last_df.update(ts_out)
        ts_out = last_df
    # save to paths
    ts_out.to_csv(output_filename)
    logging.debug("Save to csv")
    groundtruth_rp = postprocess_ground_truth(ts_out, output_filename)
    # convert to ghi
    ghi_out = alf_wrapper(ts_out, loc)
    ghi_out.to_csv(ghiname)
    ghi_groundtruth = alf_wrapper(uf.load_pandas_dataframe(rpgtname), loc)
    ghi_groundtruth.to_csv(ghigtname)
    save_to_history(
        dcfg, loc, now, ts_out, ghi_out, groundtruth_rp, ghi_groundtruth
    )


def get_dest_paths(loc, today):
    """Gets paths where to save the predictions

    Args:
        loc (dict): location dictionary
        today (datetime.date): local today

    Returns:
        pathlib.Path: four Path objects (2 ground truth x [GHI, Rp])
    """
    # create dest paths
    output_filename = Path(
        os.path.join(
            "predictions", loc["name"], "rp", "predictions", f"{today}.csv",
        )
    )
    rpgtname = Path(
        rreplace(str(output_filename), "predictions", "groundtruth", 1)
    )
    ghiname = Path(str(output_filename).replace("rp/", "ghi/"))
    ghigtname = Path(rreplace(str(ghiname), "predictions", "groundtruth", 1))
    for filename in [output_filename, rpgtname, ghiname, ghigtname]:
        filename.parent.mkdir(parents=True, exist_ok=True)
    return output_filename, rpgtname, ghiname, ghigtname


def save_to_history(
    dcfg, loc, now, ts_out, ghi_out, groundtruth_rp, ghi_groundtruth
):
    """ Saves the output dataframes (Rp and GHI) to their corresponding path.

    Args:
        dcfg (dict): [description]
        loc (dict): [description]
        now (datetime.datetime): [description]
        ts_out (pandas.DataFrame): [description]
        ghi_out (pandas.DataFrame): [description]
        groundtruth_rp (pandas.DataFrame): [description]
        ghi_groundtruth (pandas.DataFrame): [description]
    """
    # save to history
    for subfolder, dataframe in (
        ["forecast_rp", ts_out],
        ["forecast_ghi", ghi_out],
        ["groundtruth_rp", groundtruth_rp],
        ["groundtruth_ghi", ghi_groundtruth],
    ):
        history_file_path = get_history_file_path(dcfg, now, loc, subfolder)
        dataframe.to_csv(history_file_path)


def get_history_file_path(dcfg, last_dtime, loc, subfolder):
    """ Returns the path to the history file to be written. Uses <last_dtime> in the name."""
    history_file_path = Path(
        os.path.join(
            dcfg["paths"]["history"],
            "combination",
            subfolder,
            loc["name"],
            str(last_dtime.date()),
            str(last_dtime).replace(" ", "_") + ".csv",
        )
    )
    history_file_path.parent.mkdir(parents=True, exist_ok=True)
    return history_file_path


def rreplace(s, old, new, occurrence):
    """ takes string s and replaces old by new from end to beggining ocurrence times
    from https://stackoverflow.com/a/2556252/8462678"""
    li = s.rsplit(old, occurrence)
    return new.join(li)


def alf_wrapper(rp_dataframe, loc):
    """Wrapper for Agustin Laguarda Function. Uses new module when possible (some locations are not supported).

    Args:
        rp_dataframe (pandas.DataFrame): Rp values to convert to GHI
        loc (dict): location dictionary

    Returns:
        pandas.DataFrame: GHI values corresponding to input Rp values
    """
    ts_out = rp_dataframe
    paso = round(pd.to_timedelta(ts_out.index.freq).total_seconds() / 60)
    try:
        ghi, *_ = alf(  # agustin laguarda function
            ts_out.index.tz_convert("UTC"),
            ts_out.iloc[:, 0].values,
            paso, est=loc['name'], tipo='Rp',
        )
    except KeyError:
        logging.error('New CIM module can not handle location, using old one')
        ghi = alf_old(  # agustin laguarda function
            ts_out.index.tz_convert("UTC"),
            ts_out.iloc[:, 0].values,
            loc["lat"],
            loc["lon"],
            paso,
            est=loc["name"],
        )
    ghi_out = pd.DataFrame(
        index=ts_out.index, data=ghi.values, columns=["ghi"]
    )
    return ghi_out


def get_forecasts_df(loc, dcfg):
    """ Load forecasts from each algorithm folder and concatenates them in a big dataframe.
    Column names have the folder name in it."""
    forecasts_df = pd.DataFrame()  # create empty df
    today = get_local_today(loc)
    a_dirs = get_algorithms_dirs(dcfg)
    for a_folder in a_dirs:  # for each algorithm folder
        filename = os.path.join(
            "algorithms", a_folder, "predictions", loc["name"], f"{today}.csv",
        )
        logging.debug("Get filename for each algorithm")
        if os.path.isfile(filename):
            pred = uf.load_pandas_dataframe(filename)
            pred.rename(
                columns={
                    cname: cname + f"_{a_folder}" for cname in pred.columns
                },
                inplace=True,
            )
            forecasts_df = pd.concat([forecasts_df, pred], axis=1)
        else:  # tried to add predictions to df
            logging.warning(
                f"No prediction found at {a_folder} " + f"for {loc['name']}"
            )
        logging.debug(f"Create big dataframe with predictions for {loc['name']}")
    return forecasts_df, today


def get_local_today(loc):
    """Get current day at some location (could be different from utc current day)

    Args:
        loc (dict): location dict

    Returns:
        datetime.Date: current day in location  (maybe type is incorrect)
    """
    tzf = TimezoneFinder()
    tzone = pytz.timezone(tzf.timezone_at(lat=loc["lat"], lng=loc["lon"]))
    utcnow = dt.datetime.now(dt.timezone.utc)
    today = utcnow.astimezone(tzone).date()  # date at loc
    logging.debug("Set local timezone")
    return today


def get_algorithms_dirs(dcfg):
    """Gets list of algorithms folder names with order defined
    by priority in the configuration dict.

    Args:
        dcfg (dict): configuration dict

    Returns:
        list: list of directories. E.g. ['climatology', 'cmv', ...]
    """
    priorities = sorted(
        [
            algo["priority"]  # list of ordered priorities
            for algo in dcfg["algorithm"].values()
        ]
    )
    a_dirs = list(
        chain.from_iterable(
            [
                [
                    a
                    for a in sorted(os.listdir("algorithms"))
                    if os.path.isdir(os.path.join("algorithms", a))
                    and a in dcfg["algorithm"].keys()
                    and dcfg["algorithm"][a]["enabled"]
                    and dcfg["algorithm"][a]["priority"] == priority
                ]
                for priority in priorities
            ]
        )
    )
    return a_dirs


class Combiner:
    """ Class that implements different combination methods.
    The methods are selected in <dcfg> or `config.yaml`.
    """
    def __init__(self, dcfg):
        """ Initialization: defines <self.method> as the name of the first method in dcfg that carries a <True> value.

        Args:
            dcfg (dict): configuration dict as loaded from `config.yaml`
        """
        self.method = [key for key, value in dcfg["combine"].items() if value][
            0
        ]

    def __call__(self, forecasts_df):
        """ Calls the method defined in <self.method>

        Args:
            forecasts_df (pandas.DataFrame): DataFrame with all the predictions and confidence intervals (one for each method). Its index is a DatetimeIndex with all the desired datetimes in the day.

        Returns:
            pandas.DataFrame: Resulting DataFrame with columns defined as ['predictions', 'inf', 'sup']
        """
        if self.method == "most_complex":
            ndf = self.combine_most_complex(forecasts_df)
        elif self.method == "average":
            ndf = self.combine_average(forecasts_df)
        elif self.method in ["trees", "mlp", "earth"]:
            ndf = self.combine_supervised(forecasts_df, self.method)
        return ndf

    def combine_supervised(
        self, forecasts_df, method, mins_since_last_img=5, limit_in_hours=4
    ):
        """ Combines the input data using some trained method ('trees', 'earth', 'mlp')
        Assumes a number of <mins_since_last_img> instead of calculating it. 
        Defaults to 'cmv' in many cases.

        Args:
            forecasts_df (pandas.DataFrame): DataFrame with all the predictions and confidence intervals (one for each method). Its index is a DatetimeIndex with all the desired datetimes in the day.
            method (str): method name (one of ['trees', 'earth', 'mlp'])
            mins_since_last_img (int, optional): Assumed number of minutes since last image, used to compute the lead time of the predictions and choose the corresponding regressor. Defaults to 5.
            limit_in_hours (int, optional): Max number of hours after current datetime in which the combination is computed. After that moment no combination is made. Defaults to 5.

        Returns:
            pandas.DataFrame: Resulting DataFrame with columns defined as ['predictions', 'inf', 'sup']
        """
        method_ind = {"trees": 0, "mlp": 2, "earth": 1}[method]
        if not all(
            [
                algorithmname in forecasts_df.columns
                for algorithmname in ["predictions_persistence", "predictions_cmv", "predictions_climatology"]
            ]
        ):  # check needed columns
            logging.error(
                f"Combine {method} could not work, taking most complex"
            )
            return self.combine_most_complex(forecasts_df)
        with open("algorithms/regressors.pkl", "rb") as f:  # load model
            regressors = pickle.load(f)
        pred_dtime = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(
            minutes=mins_since_last_img  # approximate datetime of last prediction
        )
        aux_forecasts_df = forecasts_df[  # select only relevant columns
            ['predictions_persistence', 'predictions_climatology', 'predictions_cmv']]
        intervals = forecasts_df[['inf_cmv', 'sup_cmv']].rename(columns={'inf_cmv':'inf', 'sup_cmv':'sup'})  # optimistic (will be over 3)
        forecasts_df = aux_forecasts_df.rename(columns={colname:colname.split('_')[-1] for colname in aux_forecasts_df.columns})
        forecasts_df = forecasts_df.reindex(
            sorted(forecasts_df.columns), axis=1
        )  # sort (input should be like this)
        aux_forecasts_df = copy.deepcopy(forecasts_df)
        aux_forecasts_df["comb"] = forecasts_df["cmv"]  # initialize
        for i in range(len(forecasts_df.index)):
            delta_t = (forecasts_df.index[i] - pred_dtime.astimezone(forecasts_df.index[i].tzinfo)).total_seconds()
            if 0 < delta_t < 60 * 60 * limit_in_hours:
                lt = min(
                    list(regressors.keys()), key=lambda x: abs(x * 60 - delta_t)
                )
                input_row = np.array(forecasts_df.iloc[i, :]).reshape(1, 3)
                if np.isnan(input_row).any():
                    logging.warning(f'Invalid values when combining: {input_row}')
                    continue
                aux_forecasts_df["comb"].iloc[i] = regressors[lt][
                    method_ind
                ].predict(input_row)
        ndf = pd.concat([aux_forecasts_df[["comb"]].rename(
            columns={"comb": "predictions"}
        ), intervals / 3], axis=1)  # reduce confidence interval to a third
        return ndf

    def combine_most_complex(self, forecasts_df):
        """ Assumes higher priority at the end with
        [pred, inf, sup] order

        Args:
            forecasts_df (pandas.DataFrame): forecasts at desired datetimes concatenated according to priority and containing [predictions_algorithmname, inf_algorithmname, sup_algorithmname]

        Returns:
            pandas.DataFrame: only the relevant columns without the name of the method
        """
        ndf = forecasts_df.iloc[:, -3:]
        ndf.rename(
            columns={cname: cname.split("_")[0] for cname in ndf.columns},
            inplace=True,
        )
        logging.debug("Select most complex method")
        return ndf  # new forecasts df

    def combine_average(self, forecasts_df):
        """Replaces the repeated columns with an only column containing the mean of the values. This does not work in the new formulation and will raise an error.

        Args:
            forecasts_df (pandas.DataFrame): forecasts at desired datetimes concatenated according to priority and containing [predictions_algorithmname, inf_algorithmname, sup_algorithmname]

        Returns:
            pandas.DataFrame: only the relevant columns without the name of the method
        """
        raise NotImplementedError('This combination is deprecated. You have to rewrite the code again.')
        ndf = forecasts_df.groupby(level=0, axis=1).mean()
        # the variance becomes var = mean(var_i) / 3
        # logging.debug("Select averaging method")
        return ndf  # new forecasts df


def zero_night_values(ndf, loc):
    """Calls <useful_functions.get_cosangs_nonvector> to obtain a mask and zero the values at night.

    Args:
        ndf (pandas.DataFrame): only the relevant columns without the name of the method
        loc (dict): Location dict (needed to extract lat and loc)

    Returns:
        pandas.Series: Predictions with 0 value at night (maybe type is incorrect)
    """
    cos_mask = []
    for dtime in ndf.index:
        cos, mask = uf.get_cosangs_nonvector(
            dtime.tz_convert(dt.timezone.utc), loc["lat"], loc["lon"]
        )
        cos_mask.append(mask)
    ts_out = ndf.multiply(cos_mask, axis=0)  # zero night values
    logging.debug("Zero night values")
    return ts_out


def postprocess_ground_truth(ts_out, output_filename):
    """ Takes ground truth extracted from new images in preprocessing and
    makes an interpolation in order to have the ground truth in the desired
    datetimes. Interpolation is done 'inside'.

    Args:
        ts_out (pandas.Series): Output Rp time series.
        output_filename (pathlib.Path): Path to the output Rp filename.

    Returns:
        [type]: [description]
    """
    filename = Path(
        os.path.join(
            output_filename.parent.parent, "groundtruth", output_filename.name,
        )
    )
    hidden_filename = Path(os.path.join(filename.parent, "." + filename.name))
    if not os.path.isfile(hidden_filename):
        return
    df = uf.load_pandas_dataframe(hidden_filename)
    placeholder = ts_out.copy()
    placeholder_idx = placeholder.index  # save index to filter later
    for col in placeholder.columns:  # clean cells
        placeholder[col].loc[:] = np.nan
    placeholder = pd.DataFrame(placeholder["predictions"]).rename(
        columns={"predictions": "rp_value"}  # rename column
    )
    placeholder = pd.concat([placeholder, df], axis=0)  # add grount truth data
    placeholder = placeholder.loc[  # delete duplicated rows
        ~placeholder.index.duplicated(keep="last")
    ]
    placeholder = placeholder.sort_index()  # sort by datetime
    placeholder = placeholder.interpolate(  # interpolate to desired datetimes
        "time", limit_area="inside"
    )
    placeholder = placeholder.loc[
        placeholder_idx
    ]  # filter and keep desired datetimes
    placeholder.to_csv(filename)  # write to csv
    return placeholder


if __name__ == "__main__":
    main()

