import calendar
import datetime as dt
import logging
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from timezonefinder import TimezoneFinder
import pandas as pd
import pytz
import yaml


def preprocess(dcfg):
    img_folder_path, mk_folder_path, meta_path = get_paths(
        Path(dcfg["paths"]["some_fr_img"])
    )
    logging.debug("Load paths")
    for i in range(1, dcfg["buffers"]["psat"] + 1):
        last_img_name = Path(sorted(os.listdir(img_folder_path))[-i])
        rp_img_name = Path(
            os.path.join(
                dcfg["paths"]["psat"], last_img_name.stem + "_rp.npy"
            )
        )
        last_img_name = str(last_img_name)
        logging.debug(f"Find current img name (img # -{i})")
        if rp_img_name.exists():
            logging.debug("Target file exists, nothing to do")
            continue
        rp_image = process_img(dcfg, last_img_name)
        np.save(
            os.path.join(dcfg["paths"]["psat"], rp_img_name.name),
            rp_image,
        )
        logging.debug("Save RP image")
        logging.info("Normalize image")
        save_ground_truth(dcfg, rp_image, last_img_name)


def process_img(dcfg, last_img_name):
    img_folder_path, mk_folder_path, meta_path = get_paths(
        Path(dcfg["paths"]["some_fr_img"])
    )
    logging.debug("Load paths")

    lats, lons = read_meta(meta_path)
    save_to_psat_meta(lats, lons, dcfg)
    logging.debug("Load and save meta")

    dtime = get_dtime(last_img_name)
    logging.debug("Get datetime")

    cosangs, cos_mask = get_cosangs(dtime, lats, lons)
    logging.debug("Get cosine over all image")

    img_mask = load_mask(
        last_img_name, mk_folder_path, lats.size, lons.size
    )
    logging.debug("Load img mask")

    img = load_img(
        last_img_name, img_folder_path, lats.size, lons.size
    )
    logging.debug("Load img")

    rimg = cv.inpaint(img, img_mask, 3, cv.INPAINT_NS)
    logging.debug("Inpaint img")

    rp_image = normalize(rimg, cosangs, 0.15)
    logging.debug("Calculate RP image")

    return rp_image


def daily_solar_angles(year, doy):
    days_year = calendar.isleap(year) * 1 + 365
    gamma_rad = 2 * np.pi * (doy - 1) / days_year
    delta_rad = (
        0.006918
        - 0.399912 * np.cos(gamma_rad)
        + 0.070257 * np.sin(gamma_rad)
        - 0.006758 * np.cos(2 * gamma_rad)
        + 0.000907 * np.sin(2 * gamma_rad)
        - 0.002697 * np.cos(3 * gamma_rad)
        + 0.00148 * np.sin(3 * gamma_rad)
    )
    eot_min = (
        60
        * 3.8196667
        * (
            0.000075
            + 0.001868 * np.cos(gamma_rad)
            - 0.032077 * np.sin(gamma_rad)
            - 0.014615 * np.cos(2 * gamma_rad)
            - 0.040849 * np.sin(2 * gamma_rad)
        )
    )
    return delta_rad, eot_min


def solar_angles(
    lat, lon, delta_rad, eot_min, hour, minute, second, ret_mask=True
):
    lat_rad = lat * np.pi / 180
    h_sol = (
        hour + minute / 60 + lon / 15 + eot_min / 60 + second / 3600
    )
    w_rad = np.pi * (h_sol / 12 - 1)
    cos_zenith = np.sin(lat_rad) * np.sin(delta_rad) + np.cos(
        lat_rad
    ) * np.cos(delta_rad) * np.cos(w_rad)
    mask = cos_zenith > 0
    if cos_zenith.size > 1:
        cos_zenith[~mask] = 0
    elif cos_zenith.size == 1:
        cos_zenith = cos_zenith * mask
    if ret_mask:
        return cos_zenith, mask
    else:
        return cos_zenith


def get_cosangs(dtime, lats, lons):
    """Combine the two solar angle functions"""
    delta_rad, eot_min = daily_solar_angles(
        dtime.year, get_doy(dtime)
    )
    cosangs, cos_mask = solar_angles(
        lats[:, np.newaxis],
        lons[np.newaxis, :],
        delta_rad,
        eot_min,
        dtime.hour,
        dtime.minute,
        dtime.second,
    )
    return cosangs, cos_mask


def load_mask(img_name, mk_folder_path, ly, lx):
    MK2 = np.fromfile(
        os.path.join(mk_folder_path, img_name.split(".")[0] + ".MK"),
        dtype=np.int16,
    )
    MK2 = MK2.reshape(ly, lx)
    img_mask = (~MK2 + 2).astype(
        np.uint8
    )  # adapt to cv2.inpaint format
    return img_mask


def load_img(img_name, img_folder_path, ly, lx, datatype=np.float32):
    img = np.fromfile(
        os.path.join(img_folder_path, img_name), dtype=datatype
    )
    img = img.reshape(ly, lx)
    return img


def get_doy(dtime):
    """Return day of year from datetime.
    From "https://stackoverflow.com/questions/
    620305/convert-year-month-day-to-day-
    of-year-in-python"."""
    return dtime.timetuple().tm_yday


def get_dtime(img_name):
    """Assume img_name is 'ART_yyyydoy_hhmmss.FR' """
    yyyydoy, hhmmss = (
        img_name.split("_")[1],
        img_name.split("_")[2][:6],
    )
    year, doy = int(yyyydoy[:4]), int(yyyydoy[4:])
    hs, mins, sec = (
        int(hhmmss[:2]),
        int(hhmmss[2:4]),
        int(hhmmss[4:]),
    )
    dtime = dt.datetime(
        year,
        1,
        1,
        hour=hs,
        minute=mins,
        second=sec,
        tzinfo=dt.timezone.utc,
    ) + dt.timedelta(doy - 1)
    return dtime


def get_paths(some_fr_img_path):
    """Get paths assuming the directory structure found at the beggining.
    Take as input any FR image's path."""
    img_folder_path = some_fr_img_path.parent
    mk_folder_path = Path(str(img_folder_path).replace("FR", "MK"))
    meta_path = mk_folder_path.parent / "meta"
    return img_folder_path, mk_folder_path, meta_path


def read_meta(meta_path):
    """Load latitude and longitude vectors in degrees."""
    # meta = np.fromfile(os.path.join(meta_path, 'T000gri.META'),
    #    dtype=np.float32)
    lons = np.fromfile(
        os.path.join(meta_path, "T000gri.LONvec"), dtype=np.float32
    )
    lats = np.fromfile(
        os.path.join(meta_path, "T000gri.LATvec"), dtype=np.float32
    )
    lats = np.flipud(lats)  # reverse. now from -18 to -43
    # ly, lx = LATdeg_vec.size, LONdeg_vec.size
    return lats, lons


def save_to_psat_meta(lats, lons, dcfg):
    psat_meta_path = os.path.join(dcfg["paths"]["psat"], "meta")
    if os.listdir(psat_meta_path) == []:
        # check that arrays are sorted
        assert np.all(lats == (np.array(sorted(lats))[::-1]))
        assert np.all(lons == np.array(sorted(lons)))
        # this will raise an error if the directory is not there
        np.save(os.path.join(psat_meta_path, "lats.npy"), lats)
        np.save(os.path.join(psat_meta_path, "lons.npy"), lons)


def normalize(img, cosangs, thresh):
    """Normalization involves dividing by cosangs if greater
  than 0, and clipping if cosangs is below threshold"""
    n1 = np.divide(
        img, cosangs, out=np.zeros_like(img), where=thresh < cosangs
    )
    n2 = np.divide(
        img,
        cosangs,
        out=np.zeros_like(img),
        where=(0 < cosangs) * (cosangs <= thresh),
    )
    return n1 + np.clip(n2, 0, np.nanmean(n1)) #np.amax(n1))


def save_ground_truth(dcfg, rp_image, last_img_name):
    """ Writes pixels to csv at time of last_img_name

    Args:
        dcfg (dict): configuration dict
        rp_image (numpy.ndarray): normalized images
        last_img_name (str): 'ART_yyyy_hhmmss_*'
    """
    img_folder_path, mk_folder_path, meta_path = get_paths(
        Path(dcfg["paths"]["some_fr_img"])
    )
    lats, lons = read_meta(meta_path)
    for loc in dcfg["locations"].values():
        logging.debug("--- Start ground truth save")
        if loc["enable"]:  # for every location
            value = get_pixel2(loc, rp_image, lats, lons)
            local_dtime = get_local_dtime(
                loc, get_dtime(last_img_name)
            )
            hidden_filename = Path(
                os.path.join(
                    "predictions",
                    loc["name"],
                    "rp",
                    "groundtruth",
                    f".{local_dtime.date()}.csv",
                )
            )
            hidden_filename.parent.mkdir(parents=True, exist_ok=True)
            new_row = pd.DataFrame(
                [[value]], columns=["rp_value"], index=[local_dtime]
            )
            if os.path.isfile(hidden_filename):
                pred = load_pandas_dataframe(hidden_filename)
                df = pd.concat([pred, new_row], ignore_index=False)
                df = df.sort_index()
                df.to_csv(
                    hidden_filename
                )  # this will create a single not sorted file with the real ground truths (without taking into account the satellite capture time)
            else:
                new_row.to_csv(hidden_filename)


def load_pandas_dataframe(filename):
    """Correcty loads dataframe but freq is not maintained"""
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    df.index.freq = df.index.inferred_freq
    return df


def get_local_dtime(loc, utcdtime):
    tzf = TimezoneFinder()
    tzone = pytz.timezone(
        tzf.timezone_at(lat=loc["lat"], lng=loc["lon"])
    )
    local_dtime = utcdtime.astimezone(
        tzone
    )  # .date()  # date at loc
    logging.debug("Set local timezone")
    return local_dtime


def get_pixel2(loc, img, lats, lons):
    """Gets target pixel value"""
    i = find_nearest_index(lats, loc["lat"])
    j = find_nearest_index(lons, loc["lon"])
    target = img[i, j]
    return target


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


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
