import calendar
import datetime as dt
from itertools import chain
import os
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import yaml
from timezonefinder import TimezoneFinder
import pytz

import algorithms.useful_functions as uf
'''
This script executes the selected algorithms and combines them.
'''

def sender_program():
    # Load confiuration
    stream = open("admin_scripts/config.yaml", 'r')
    dcfg = yaml.load(stream, yaml.FullLoader)  # dict
    loglevel = dcfg['logging']['loglevel']

    # logging
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(filename='admin_scripts/logs/admin.log',
                        format='%(levelname)s:%(asctime)s: %(message)s',
                        level=numeric_level)
    logging.info('Start data sender.py')  # exclusive of this script
    # prepare_forecasts()
    # send_forecasts()
    logging.info('End data sender.py')


def send_forecasts():
    pass


def prepare_forecasts():
    pass


if __name__ == '__main__':
    sender_program()