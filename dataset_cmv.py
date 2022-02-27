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

ap.add_argument("--data-path", default='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/')
ap.add_argument("--dest-path", default='/clusteruy/home03/DeepCloud/deepCloud/data/cmv/cmv_mvd_10min/train/')
ap.add_argument("--csv-path", default='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train_cosangs_mvd.csv')
                
ap.add_argument("--predict-horizon", default=1, type=int,
                help="Defaults to 1")

params = vars(ap.parse_args())
PATH_DATA = params['data_path']
destination_path = params['dest_path']

cmv = model.Cmv2()
val_mvd = MontevideoFoldersDataset_w_name(path = PATH_DATA, 
                                          in_channel = 2, out_channel=params['predict_horizon'], 
                                          output_last=True,
                                          csv_path=params['csv_path'])
val_loader = DataLoader(val_mvd)

for idx, (inputs, targets, out_name) in enumerate(val_loader):
  if idx % 1000 == 0:
      print(idx)
  prediction = cmv.predict(
                          imgi=inputs[0][0], 
                          imgf=inputs[0][1],
                          period=10*60, delta_t=10*60, 
                          predict_horizon=params['predict_horizon']) 
  
  day = re.sub("[^0-9]", "", out_name[0])[:7]
  try:
      os.makedirs(os.path.join(
          os.getcwd(), destination_path, day))
  except:
      pass
  path = os.path.join(destination_path,
                      day, os.path.splitext(out_name[0])[0] + ".npy")
  
  np.save(path, prediction[-1])

