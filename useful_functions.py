import calendar
import datetime as dt
import logging
import os
from pathlib import Path
import sys
import yaml

import cv2 as cv
import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder


def analyze_psat(dcfg, return_creation=False):
    '''Returns list of images in psat (only the name), the Path of the last
    image, and the datetime for the last image. Can also return creation time.'''
    imgs_list = sorted([img for img in os.listdir(dcfg['paths']['psat']) 
                       if img.endswith('rp.npy')])
    last_img = Path(dcfg['paths']['psat']) / imgs_list[-1] 
    last_dtime = get_dtime(last_img.name)
    if not return_creation:
      return imgs_list, last_img, last_dtime
    elif return_creation:
      modification = os.path.getmtime(last_img)
      creation_dtime = dt.datetime.fromtimestamp(modification,
                                                 tz=dt.timezone.utc)
      return imgs_list, last_img, last_dtime, creation_dtime


def get_dtime(img_name):
    '''Return image datetime assuming img_name is 'ART_yyyydoy_hhmmss_rp.npy' '''
    yyyydoy, hhmmss = img_name.split('_')[1], img_name.split('_')[2][:6]
    year, doy = int(yyyydoy[:4]), int(yyyydoy[4:])
    hs, mins, sec = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:])
    dtime = (dt.datetime(year, 1, 1, hour=hs,
                minute=mins, second=sec,
                tzinfo=dt.timezone.utc)
             + dt.timedelta(doy - 1))
    return dtime


def get_doy(dtime):
    '''Return day of year from datetime.
    From "https://stackoverflow.com/questions/
    620305/convert-year-month-day-to-day-
    of-year-in-python".'''
    return dtime.timetuple().tm_yday


def get_consistent_period(imgs_list, return_dtimes=False):
    dtimes = [get_dtime(img) for img in imgs_list]

    timedeltas = []
    for i in range(len(dtimes) - 1):
        timedeltas.append(dtimes[i+1] - dtimes[i])
    period = np.median(timedeltas)

    if return_dtimes:
        return period, dtimes
    else:
        return period
     

def get_last_period(imgs_list, return_dtimes=False):
    """Gets period as difference between the datetimes
    of the images. Assumes sorted <imgs_list>.

    Args:
        imgs_list (list): list with images names
        return_dtimes (bool, optional): [description]. Defaults to False.

    Returns:
        datetime.timedelta: difference in time between the last two images
        list: list with datetimes for each image in <imgs_list>
    """
    dtimes = [get_dtime(img) for img in imgs_list]
    period = dtimes[-1] - dtimes[-2]
    if return_dtimes:
        return period, dtimes
    else:
        return period
       
            
def get_needed_lead_times(limit, period):
    '''Returns number of time steps needed to surpass the limit
    assuming period'''
    lt = 1
    while period * lt < dt.timedelta(hours=limit):
        lt += 1
    return lt


def get_pixel(loc, img_path, lats=None, lons=None):
    '''Gets target pixel value'''
    if (lats is None) or (lons is None):
        lats = np.load(os.path.join(img_path.parent, 'meta', 'lats.npy'))
        lons = np.load(os.path.join(img_path.parent, 'meta', 'lons.npy'))

    i = find_nearest_index(lats, loc['lat']) 
    j = find_nearest_index(lons, loc['lon'])

    img = np.load(img_path)

    # print(img)
    # import matplotlib.pyplot as plt
    # plt.savefig('debug.png')

    target = img[i, j]
    logging.debug('Get pixel[i, j] value')
    return target

def get_value_gaussian(loc, kernelsize, img_path, lats=None, lons=None):
    '''Get average of RP using a gaussian blur. kernelsize must be odd'''
    assert kernelsize % 2 == 1  # check kernelsize is odd
    if (lats is None) or (lons is None):  # load lats and lons if not in arguments
        lats = np.load(os.path.join(img_path.parent, 'meta', 'lats.npy'))
        lons = np.load(os.path.join(img_path.parent, 'meta', 'lons.npy'))
    # Closest pixel. Interpolation could be used in the future.
    i = find_nearest_index(lats, loc['lat'])
    j = find_nearest_index(lons, loc['lon'])
    img = np.load(img_path) 
    blurred_img = cv.GaussianBlur(img, (kernelsize, kernelsize), 0)
    value = blurred_img[i, j]
    return value

def get_value(loc, winsize, img_path, lats=None, lons=None):
    '''Get average of RP over winsize'''
    if (lats is None) or (lons is None):
        lats = np.load(os.path.join(img_path.parent, 'meta', 'lats.npy'))
        lons = np.load(os.path.join(img_path.parent, 'meta', 'lons.npy'))

    i = find_nearest_index(lats, loc['lat']) 
    j = find_nearest_index(lons, loc['lon'])

    hw = winsize // 2  # half winsize
    img = np.load(img_path)
    value = np.nanmean(img[np.amax(i-hw, 0):i+1+hw,
                        np.amax(j-hw, 0):j+1+hw]) 
    # logging.debug('Average over window')
    return value


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def load_pandas_dataframe(filename):
    '''Correcty loads dataframe but freq is not maintained'''
    df = pd.read_csv(filename, index_col=0,
                     parse_dates=True)
    df.index.freq = df.index.inferred_freq
    return df


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
    start = (dt.datetime(dtime.year,
                         dtime.month,
                         dtime.day,
                         dtime.hour,
                         next_minute,
                         tzinfo=dtime.tzinfo)
             + dt.timedelta(hours=next_hour))
    return start


def previous_step_aux(dtime, time_step):
    previous_minute = (time_step * (dtime.minute // time_step)) % 60
    return previous_minute    


def previous_step_starts(dtime, time_step):
    previous_minute = previous_step_aux(dtime, time_step)
    start = dt.datetime(dtime.year,
                        dtime.month,
                        dtime.day,
                        dtime.hour,
                        previous_minute,
                        tzinfo=dtime.tzinfo)
    return start


def is_a_day_img(img_dtime, img_lats, img_lons, thresh=0.15):
    '''Checks if the four corners of image are under daylight'''
    cosang00, mask00 = get_cosangs_nonvector(
        img_dtime, img_lats[0], img_lons[0]
    )
    cosang01, mask01 = get_cosangs_nonvector(
        img_dtime, img_lats[0], img_lons[-1]
    )
    cosang10, mask10 = get_cosangs_nonvector(
        img_dtime, img_lats[-1], img_lons[0]
    )
    cosang11, mask11 = get_cosangs_nonvector(
        img_dtime, img_lats[-1], img_lons[-1]
    )
    mask00, mask01 = cosang00 > thresh, cosang01 > thresh
    mask10, mask11 = cosang10 > thresh, cosang11 > thresh
    return mask00 and mask01 and mask10 and mask11
   

def is_name_of_day_img(img_path):
  lats = np.load(os.path.join(img_path.parent, 'meta', 'lats.npy'))
  lons = np.load(os.path.join(img_path.parent, 'meta', 'lons.npy'))
  img_dtime = get_dtime(img_path.name)
  return is_a_day_img(img_dtime, lats, lons)



def get_day_imgs_list(imgs_list):
    '''imgs_list should be full Path object'''
    lats = np.load(os.path.join(imgs_list[0].parent, 'meta', 'lats.npy'))
    lons = np.load(os.path.join(imgs_list[0].parent, 'meta', 'lons.npy'))
    day_imgs_list = [img for img in imgs_list 
                     if is_a_day_img(
                         get_dtime(img.name), 
                         lats, lons)]
    return day_imgs_list



def daily_solar_angles(year, doy):
    days_year = calendar.isleap(year)*1 + 365
    gamma_rad =  2 * np.pi * (doy-1) / days_year
    delta_rad =  (0.006918
                  - 0.399912*np.cos(gamma_rad)
                  + 0.070257*np.sin(gamma_rad)
                  - 0.006758*np.cos(2*gamma_rad)
                  + 0.000907*np.sin(2*gamma_rad)
                  - 0.002697*np.cos(3*gamma_rad)
                  + 0.00148*np.sin(3*gamma_rad)
                  )
    eot_min = (60 * 3.8196667
               * (0.000075
                  + 0.001868*np.cos(gamma_rad)
                  - 0.032077*np.sin(gamma_rad)
                  - 0.014615*np.cos(2*gamma_rad)
                  - 0.040849*np.sin(2*gamma_rad)
                  )
               )
    return delta_rad, eot_min
 

def solar_angles(lat, lon, delta_rad, eot_min,
                 hour, minute, second, ret_mask=True):
    lat_rad = lat * np.pi / 180
    h_sol = (hour + minute / 60 + lon / 15
            + eot_min / 60 + second / 3600)
    w_rad = np.pi * (h_sol / 12 - 1)
    cos_zenith = (np.sin(lat_rad)
                  * np.sin(delta_rad)
                  + np.cos(lat_rad) 
                  * np.cos(delta_rad) 
                  * np.cos(w_rad)
                 )
    mask = cos_zenith > 0
    if cos_zenith.size > 1:
       cos_zenith[~mask] = 0
    elif cos_zenith.size == 1:
        cos_zenith = cos_zenith * mask
    if ret_mask:
        return cos_zenith, mask
    else:
        return cos_zenith


def get_cosangs_nonvector(dtime, lat, lon):
    '''Combine the two solar angle functions'''
    delta_rad, eot_min = daily_solar_angles(dtime.year,
                                            get_doy(dtime))
    cosangs, cos_mask = solar_angles(
        lat, lon,
        delta_rad, eot_min, dtime.hour,
        dtime.minute, dtime.second)
    return cosangs, cos_mask


# df1 refers to the forecasts dataframe with datetimeindex given by the period
# df2 refers to the forecasts dataframe with the required datetimeindex
# daydf refers to the dataframe with the required datetimeindex for one day

def write_to_daydf(df2, pred_dir_path, loc):
    """df2 is converted to local (loc) time and
    the current daydf is updated with df2's values

    Args:
        df2 (pandas.DataFrame): instantaneous forecasts dataframe with required datetimeindex
        pred_dir_path (pathlib.Path or str): path to predictions folder
        loc (dict): location dictionary as loaded from config.yaml
    """
    df2, tzone = timezone_convert(df2, loc)  # convert to local
    for day, daygroup in df2.groupby(df2.index.day):  # for each day the timezone conversion expanded the dataset into
        filename = Path(os.path.join(pred_dir_path,  # target filename for this day
                                     str(daygroup.index[0].date()) + '.csv'))
        if not os.path.isfile(filename):  # create if nonexistent
            daydf = create_daydf(daygroup.index[0].date(), 
                                 daygroup.index.freq.delta,
                                 columns=daygroup.columns,
                                 tzone=tzone)
            try:  # create daydf file
                daydf.to_csv(filename)
            except FileNotFoundError:  # try fix if FileNotFoundError. If not solved an error will be raised again.
                logging.warning(f"Directory of {filename} not found")
                (filename.parent).mkdir(parents=True, exist_ok=True)
                daydf.to_csv(filename)
        daydf = load_pandas_dataframe(filename)  # load daydf (freq is not maintained)
        daydf = df2_to_daydf(daygroup, daydf)  # updates daydf with data from df2
        daydf.to_csv(filename)  # writes to file


def create_df2_datetimeindex(dtime_last, dtime_now, time_step, limit):
    """Creates datetimeindex object to be used as
    index of df2. It has as indices the desired
    output datetimes (e.g. multiples of 10 min).
    Starts at the last index before <dtime_last>
    until <limit> hours after <dtime_now> with 
    timestep = <time_step>. It is utc localized.

    Args:
        dtime_last (datetime.datetime): index starts before this datetime
        dtime_now (datetime.datetime): index goes <limit> hours after this datetime
        time_step (int): minutes between each datetime in the index
        limit (int): hours after <dtime_now> that are included in the index

    Returns:
        pandas.DatetimeIndex: Regular index to be used in df2.
    """
    idx2 = pd.date_range(start=previous_step_starts(dtime_last,
                                                    time_step),
                          end=(dtime_now 
                              + dt.timedelta(hours=limit,
                                                    minutes=time_step)),
                          freq=f'{time_step}min',
                          tz=dt.timezone.utc
                          )
    return idx2



 
def df1_to_df2(df1, dtime_now, time_step, limit):
    '''Adapts values with arbitrary datetimeindex to the datetimeinxex defined
    by time_step mins and limit hours.'''
    idx2 = create_df2_datetimeindex(
      dtime_now, dtime_now, time_step, limit
    )
    columns = df1.columns
    df2 = pd.DataFrame(index=idx2, columns=columns)  # only has NaNs
    df2 = pd.concat([df1, df2]).sort_index().interpolate('time').loc[df2.index]
    logging.debug('Interpolate and get df2')

    df2['predictions'][df2.index < df1.index[0]] = df1.loc[df1.index[0]]
    df2['predictions'][df1.index[-1] < df2.index] = df1.loc[df1.index[-1]]
    logging.debug('Persist to the past/future (rls)')

    return df2


def df2_to_daydf(df2, daydf):
    '''Updates daydf with df2 values.
    Both should be located'''
    idx = daydf.index
    daydf.update(df2)
    idx = daydf.index
    daydf = daydf.loc[idx]
    return daydf


def create_df1_from_dict(predictions, last_dtime, period, needed_lts):
    """ Creates df1 from predictions.

    Args:
        predictions (dict): E.g. predictions = {'predictions':{1:pred1, 2:pred2, ..., n:predn}, 'flags':{1:flag1, ..., n:flagn}, 'inf':{...}, 'sup':{...}}
        last_dtime ([type]): [description]
        period ([type]): [description]
        needed_lts ([type]): [description]

    Returns:
        [type]: [description]
    """
    idx1 = pd.date_range(start=last_dtime + period,
                         end=(last_dtime + needed_lts * period),
                         freq=period, tz=dt.timezone.utc)
    df1 = pd.DataFrame(index=idx1, columns=predictions.keys())
    for column_name, dictionary in predictions.items():
        df1[column_name] = np.array([value for key, value
                                     in dictionary.items()])
    return df1.astype(float)
   

def create_daydf(datetimedate, period, 
                 columns=['predictions', 'inf', 'sup', 'flags'],
                 tzone=None):
    """Creates empty day dataframe to be filled with predictions"""
    idx = pd.date_range(
      start=datetimedate,
      end=datetimedate + dt.timedelta(days=1),
      freq=period, closed='left', tz=tzone
      )
    return pd.DataFrame(index=idx, columns=columns)


def date_to_datetime(date):
  return dt.datetime(date.year, date.month, date.day)


def timezone_convert(df, loc):
  tzf = TimezoneFinder()
  tzone = tzf.timezone_at(lat=loc['lat'], lng=loc['lon'])
  df_utc = df
  df = df_utc.tz_convert(tzone)  # df in localtime
  return df, tzone

 
def initialize(name):
    # Load confiuration
    stream = open("admin_scripts/config.yaml", "r")
    dcfg = yaml.load(stream, yaml.FullLoader)  # dict
    loglevel = dcfg["logging"]["loglevel"]

    # logging
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % loglevel)
    file_handler = logging.FileHandler(
        f"admin_scripts/logs/{name}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.WARNING)
    handlers = [
        file_handler,
        stream_handler,
    ]
    logging.basicConfig(
        format="%(levelname)s:%(asctime)s: %(message)s",
        level=numeric_level,
        handlers=handlers,
    )
    sys.excepthook = handle_exception
    logging.info("Initialize logging")
    return dcfg


def handle_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        logging.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    return

