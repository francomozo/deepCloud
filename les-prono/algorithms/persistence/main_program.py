import os
import logging
from pathlib import Path
import sys

import numpy as np
import yaml

import algorithms.persistence.algorithm as algorithm


def main_program():
    '''
    This script computes persistence forecasts for the selected locations
    '''
    # Load confiuration
    stream = open("admin_scripts/config.yaml", 'r')
    dcfg = yaml.load(stream, yaml.FullLoader)  # dict
    loglevel = dcfg['logging']['loglevel']

    # logging
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(filename='admin_scripts/logs/algorithms.log',
                        format='%(levelname)s:%(asctime)s: %(message)s',
                        level=numeric_level)
    logging.info('Start persistence/main_program.py')  # exclusive of this script

    locs = dcfg['locations']
    for loc in locs.values():
      if loc['enable']:
        algorithm.predict(loc, dcfg)


if __name__ == '__main__':
    main_program()
