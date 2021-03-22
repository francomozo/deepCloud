import datetime as dt
import logging
import os
import json
from pathlib import Path
import pickle

from dateutil.parser import isoparse
import numpy as np
import pandas as pd

import algorithms.useful_functions as uf


def predict(loc, dcfg):
    """Makes predictions for loc using persistence
    with the passed configuration.

    Args:
        loc (dict): dict with individual location as loaded from config.yaml
        dcfg (dict): configuration dict
    """
    logging.debug('Start persistence predict')
    locname = loc['name']
    aux_variables = load_variables(locname)
    logging.debug('Load variables')
    imgs_list, last_img, last_dtime, created_dtime = uf.analyze_psat(dcfg,
                                                                     True)  # get useful names and datetimes    
    if aux_variables[locname]['last_run_dtime'] < created_dtime:  # if last run was before creation of the last image predict
        last_value = uf.get_value_gaussian(loc,  # get prediction for that location
                                  dcfg['algorithm']['persistence']['kernel'], 
                                  last_img)
        aux_variables[locname]['last_value'] = last_value  # save last value for next run
        logging.debug('Compute new forecast')
    else:  # if last run was after creation of last img
        last_value = float(aux_variables[locname]['last_value'])  # load last value
        logging.debug('Reuse old forecast')
    df2, last_run_dtime = update_forecast(loc, dcfg['forecasts']['time_step'],
                                          last_value,
                                          limit=dcfg['forecasts']['limit'])  # updates forecast. Last run datetime is now.
    history_file_path = get_history_file_path(dcfg, last_run_dtime, loc)
    df2.to_csv(history_file_path)  # save to history
    logging.info('Make new forecast')
    save_variables(aux_variables, locname, last_run_dtime)  # save aux variables
    logging.debug('Finish predict')    
    
   
def update_forecast(loc, time_step, value, limit=4):
    '''Update forecasts file up to limit hours'''
    last_run_datetime = dt.datetime.now(dt.timezone.utc)  # set last run to now
    idx = uf.create_df2_datetimeindex(  # df2 is aligned with the desired output datetimes
      last_run_datetime, last_run_datetime, 
      time_step, limit
    )
    df2 = pd.DataFrame(index=idx, columns=['predictions', 'inf', 'sup'])  # create output dataframe
    df2['predictions'] = np.ones(len(idx)) * value
    df2['inf'] = 932 * np.ones(len(idx))  # this is only a flag. A placeholder for confidence intervals.
    dir_path = Path(os.path.join(  # where to write the resulting dataframe
        os.path.dirname(os.path.realpath(__file__)),
        'predictions', loc['name']))
    uf.write_to_daydf(df2, dir_path, loc)
    logging.debug('Write to forecasts file')
    return df2, last_run_datetime


def get_history_file_path(dcfg, last_dtime, loc):
    """Gets path to file to include in the historical
    record. It carries as name the datetime of the forecast.

    Args:
        dcfg (dict): configuration dictionary
        last_dtime (datetime.datetime): time of run
        loc (dict): location dictionary 

    Returns:
        str: path to file where to save the forecasts done at <last_dtime>
    """
    history_file_path = Path(os.path.join(
        dcfg['paths']['history'],
        'persistence',
        loc['name'],
        str(last_dtime.date()),
        str(last_dtime).replace(' ', '_') + '.csv'
        ))
    history_file_path.parent.mkdir(parents=True, exist_ok=True)
    return history_file_path



def load_variables(locname):
    """Loads auxiliary variables saved in the json
    file. In the persistence case only last run
    datetime and last predicted value are saved
    for each location.

    Args:
        locname (str): location name (e.g. "LE")

    Returns:
        dict: {locname1:{"last_run_dtime": v1, "last_value":v2}, locname2:...}
    """
    aux_variables_path = os.path.join(os.path.dirname(  # get path
        os.path.realpath(__file__)), 'aux_variables.json')
    try:
        aux_variables = json.load(open(aux_variables_path, 'r'))  # try to open
        for loc in aux_variables:  # create dictionary
          aux_variables[loc]['last_run_dtime'] = isoparse(
              aux_variables[loc]['last_run_dtime']
            )
    except FileNotFoundError:  # there isn't such a file
        logging.warning("There is no 'aux_variables.json'")
        aux_variables = {}  # initialize as empty dict
    if not locname in aux_variables:  # if current location is not in the saved data
      aux_variables[locname] = {'last_run_dtime': 
          (dt.datetime.now(dt.timezone.utc) 
           - dt.timedelta(days=1000)),  # reset with a very old datetime
                                'last_value': 0}
    return aux_variables

def save_variables(aux_variables, locname, last_run_dtime):
    """Saves auxiliary variables for selected location.
    Note that <aux_variables> 'value' should be updated by now.
    However, 'last_run_dtime' is updated.

    Args:
        aux_variables (dict): auxiliary variables dict as loaded from json
        locname (str): name of location (a key in the aux_variables dict)
        last_run_dtime (datetime.datetime): when last run was executed (this is used to update the aux_variables dict)
    """
    aux_variables_path = os.path.join(os.path.dirname(
      os.path.realpath(__file__)), 'aux_variables.json')
    aux_variables[locname]['last_run_dtime'] = last_run_dtime
    json.dump(aux_variables, open(aux_variables_path, 'w'),
              default=str)
 
