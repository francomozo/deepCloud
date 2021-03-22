import datetime
import os
import logging
from pathlib import Path
import sys

import cv2 as cv
import numpy as np
import yaml

import preprocessing.preprocessing_functions as pf
import preprocessing.sync_scripts.psat2dataset as p2d

"""
This script creates normalized images from the last (psatbuffersize) images
found in sat. These normalized images are saved to the psat directory.
It also syncs psat with dataset in order to save the processed images while
cleaning the psat directory. 'Cleaning' is removing the not so recent images.
"""

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
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def main():
    dcfg = pf.initialize('preprocessing')
    logging.info(f"Start {os.path.basename(__file__)}")
    pf.preprocess(dcfg)
    # p2d.sync_psat2dataset(dcfg)
    p2d.clean_buffer(dcfg)
    logging.info("Sync psat to dataset")
    logging.info("Finish run_preprocessing.py")


if __name__ == '__main__':
    main()
# if False:  # only for testing purposes
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(30, 30))
#     plt.imshow(rp_image)
#     plt.colorbar()
#     plt.show()

