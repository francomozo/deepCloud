import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pickle
import argparse
import re

import torch
from torch.utils.data import DataLoader

from src import data, evaluate, model, preprocessing, visualization
from src.lib import utils
from src.data import MontevideoFoldersDataset, MontevideoFoldersDataset_w_name
from src.dl_models.unet import UNet, UNet2

#guardar con el tiempo de prediccion
ap = argparse.ArgumentParser(description='Generate cmv prediction dataset')

ap.add_argument("--csv-path-base", default=None,
                help="Csv path for baseline models (CMV, Persistence, BCMV). Defaults to None.")
ap.add_argument("--data-path", default='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/',
                help="Defaults to /clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/")
ap.add_argument("--dest-path", default='/clusteruy/home03/DeepCloud/deepCloud/data/cmv_mvd/train/',
                help="Defaults to /clusteruy/home03/DeepCloud/deepCloud/data/cmv_mvd/train/")

params = vars(ap.parse_args())
PATH_DATA = params['data_path']
csv_path_base = params['csv_path_base']
destination_path = params['dest-path']

cmv = model.Cmv2()
val_mvd = MontevideoFoldersDataset_w_name(path = PATH_DATA, in_channel = 2, out_channel=1, output_last=True)
val_loader = DataLoader(val_mvd)

for idx, (inputs, targets, out_name) in enumerate(val_loader):
  prediction = cmv.predict(
                          imgi=inputs[0], 
                          imgf=inputs[1],
                          period=10*60, delta_t=10*60, 
                          predict_horizon=1) 
  
  day = re.sub("[^0-9]", "", out_name[0])[:7]
  try:
      os.makedirs(os.path.join(
          os.getcwd(), destination_path, day))
  except:
      pass
  path = os.path.join(destination_path,
                      day, os.path.splitext(out_name[0])[0] + ".npy")
  
  np.save(path, prediction)

