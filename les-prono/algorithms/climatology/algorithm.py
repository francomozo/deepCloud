import datetime as dt
import logging
import os
from pathlib import Path
import pickle

import cv2 as cv
import numpy as np
import pandas as pd

import algorithms.useful_functions as uf


def predict(loc, dcfg):
    """We now know climatology best Rp forecast is
    only one value: 41.25 .

    Args:
        loc (dict): location dict
        dcfg (dict): configuration dict
    """
    logging.debug("Start predict")
    # climatology_img_path = os.path.join(
    #     os.path.dirname(os.path.realpath(__file__)), "cimg.npy"
    # )
    # if os.path.isfile(climatology_img_path):
    #     cimg = np.load(climatology_img_path)
    #     logging.debug("Load climatology image")
    # else:
    #     logging.warning("Couldn't find 'cimg.npy'")
    #     dcfg["algorithm"]["climatology"]["recompute"] = True

    # if dcfg["algorithm"]["climatology"]["recompute"]:
    #     cimg = compute_new_cimg(dcfg, climatology_img_path)

    # # assume the same meta than cimg.npy
    # lats = np.load(
    #     os.path.join(dcfg["paths"]["psat"], "meta", "lats.npy")
    # )
    # lons = np.load(
    #     os.path.join(dcfg["paths"]["psat"], "meta", "lons.npy")
    # )
    # climatology_value = uf.get_pixel(
    #     loc, climatology_img_path, lats=lats, lons=lons
    # )
    climatology_value = 41.25
    logging.debug("Get value from climatology image")
    update_forecast(
        loc,
        dcfg["forecasts"]["time_step"],
        climatology_value,
        limit=dcfg["forecasts"]["limit"],
    )
    logging.info("Make new forecast")
    logging.debug("Finish predict")


# def compute_new_cimg(dcfg, climatology_img_path):
#     imgs_list = sorted(
#         [
#             Path(dcfg["paths"]["history"]) / img
#             for img in os.listdir(dcfg["paths"]["history"])
#             if img.endswith("rp.npy")
#         ]
#     )
#     logging.debug("Get historical list of images")

#     cimg = get_climatology_img(
#         imgs_list, dcfg["algorithm"]["climatology"]["winsize"]
#     )
#     logging.debug("Start new climatology image computation")

#     np.save(climatology_img_path, cimg)
#     logging.debug("Save locally new climatology image")
#     # history_file_path = get_history_file_path(dcfg)
#     # np.save(history_file_path, cimg)
#     # logging.debug('Save to history new climatology image')


# def get_climatology_img(imgs_list, winsize):
#     """Get average of RP over winsize over all day images"""
#     day_imgs_list = uf.get_day_imgs_list(imgs_list)
#     N = len(day_imgs_list)

#     # blur and average
#     cimg = 0
#     for n, img_path in enumerate(day_imgs_list):
#         bimg = cv.blur(
#             np.load(img_path), ksize=(winsize, winsize)
#         )  # reflect
#         cimg = cimg * (n / (n + 1)) + bimg / (n + 1)
#     logging.debug("Get climatology image")
#     return cimg  # , flag


def update_forecast(loc, time_step, value, limit=4):
    """Update forecasts file up to limit hours"""
    last_run_datetime = dt.datetime.now(dt.timezone.utc)
    logging.debug("Get current UTC time")

    idx = uf.create_df2_datetimeindex(
        last_run_datetime, last_run_datetime, time_step, limit
    )
    df2 = pd.DataFrame(
        index=idx, columns=["predictions", "inf", "sup"]
    )
    df2["predictions"] = np.ones(len(idx)) * value
    logging.debug("Create df2 dataframe (only predictions)")

    dir_path = Path(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "predictions",
            loc["name"],
        )
    )
    uf.write_to_daydf(df2, dir_path, loc)
    logging.debug("Call write_to_daydf(df2, dir_path, loc)")


# def get_history_file_path(dcfg):
#     history_file_path = Path(
#         os.path.join(
#             dcfg["paths"]["history"],  # save to history
#             "climatology",
#             f"cimg_{dt.datetime.now(dt.timezone.utc)}.npy",
#         )
#     )
#     history_file_path.parent.mkdir(parents=True, exist_ok=True)
#     return history_file_path


# def get_climatology_img_test(imgs_list, winsize):
#     """Get average of RP over winsize over all day images"""
#     day_imgs_list = uf.get_day_imgs_list(imgs_list)
#     N = len(day_imgs_list)
#     print(day_imgs_list)

#     # blur and average
#     # cimg = 0
#     for n, img_path in enumerate(day_imgs_list):
#         img = np.load(img_path)
#         print(img)
#         print(img_path)
#         if not np.all(np.isfinite(img)):
#             rem = input("is not finite, remove?[y|n]")
#             if rem == "y":
#                 os.remove(img_path)
#             elif rem == "n":
#                 pass
#             else:
#                 print("Invalid value")
#                 raise ValueError
        # input('print another?')
    #     bimg = cv.blur(np.load(img_path), ksize=(winsize, winsize))  # reflect
    #     cimg = cimg * (n / (n + 1)) + bimg / (n + 1)
    # logging.debug('Get climatology image')
    # return cimg#, flag

    # this was in the main function of this script
    # # debug
    # imgs_list = sorted([Path(dcfg['paths']['dataset']) / img for img in
    #                     os.listdir(dcfg['paths']['dataset'])
    #                     if img.endswith('rp.npy')])
    # # debug
    # get_climatology_img_test(imgs_list, dcfg['algorithm']['climatology']['winsize'])

