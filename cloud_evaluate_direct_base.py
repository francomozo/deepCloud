import os
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import pickle
import argparse

import torch
from torch.utils.data import DataLoader

from src import data, evaluate, model, preprocessing, visualization
from src.lib import utils
from src.data import MontevideoFoldersDataset

ap = argparse.ArgumentParser(description='Evaluate multiple models with multiple metrics')

ap.add_argument("--horizon-step", default=6, type=int,
                help="Defaults to 6. Distance between predictions")                
ap.add_argument("--start-horizon", default=6, type=int,
                help="Defaults to 6. Starting point of prediction")
ap.add_argument("--predict-length", default=5, type=int,
                help="Defaults to 5. Amount of predicted images")

ap.add_argument("--models", nargs="+", default=["CMV"], 
                help="Options: CMV, Persistence, BCMV. Defaults to CMV")
ap.add_argument("--metrics", nargs="+", default=["RMSE"],
                help="Defaults to RMSE. Add %% for percentage metric")
ap.add_argument("--csv-path-base", default=None,
                help="Csv path for baseline models (CMV, Persistence, BCMV). Defaults to None.")
ap.add_argument("--data-path", default='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/',
                help="Defaults to /clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/")
ap.add_argument("--save-errors", default=False, type=bool,
                help="Save results in file. Defaults to False.")
ap.add_argument("--save-name", default=None,
                help="Add name to results file. Defaults to None.")

ap.add_argument("--window-pad", default=0, type=int,
                help="Size of padding for evaluation, eval window is [w_p//2 : M-w_p//2, w_p//2 : N-w_p//2]. Defaults to 0.")
ap.add_argument("--window-pad-height", default=0, type=int,
                help="Size of height padding for evaluation, eval window is [w_p_h//2 : M-w_p_h//2]. Defaults to 0.")
ap.add_argument("--window-pad-width", default=0, type=int,
                help="Size of width padding for evaluation, eval window is [w_p_w//2 : N-w_p_w//2]. Defaults to 0.")

params = vars(ap.parse_args())
csv_path_base = params['csv_path_base']
PATH_DATA = params['data_path']
models_names = params['models']
metrics = params['metrics']
#Calculate out_channels
start_horizon = params['start_horizon']
horizon_step = params['horizon_step']
predict_length = params['predict_length']
out_channels = list(range(start_horizon, start_horizon+horizon_step*predict_length, horizon_step))
print("out_channels:", out_channels)

models_names = [each_string.lower() for each_string in models_names]
metrics = [each_string.upper() for each_string in metrics]

#Definition of models
models = []
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
for a_model_name in models_names:
  if "cmv" == a_model_name:
    models.append(model.Cmv2())
  if "bcmv" == a_model_name:
    models.append(model.Cmv2(kernel_size_list=[(5,5),(17,17),(41,41),(65,65),(89,89),(113,113),(137,137),(157,157),(181,181),(205,205),(225,225),(249,249)]))
  if "p" == a_model_name or "persistence" == a_model_name:
    models.append(model.Persistence())
  if "bp" == a_model_name or "blurredpersistence" == a_model_name:
    models.append(model.BlurredPersistence(kernel_size_list=[(33,33),(73,73),(109,109),(145,145),(181,181),(213,213),(249,249),(285,285),(321,321),(361,361),(405,405),(445,445)])) 
  if "gt_blur" == a_model_name:
    models.append("gt_blur")

errors_metrics = {}
for metric in metrics:
    print("\n", metric)
    fix = 1
    #variables for percentage evaluation:
    percentage_pos = metric.find("%") 
    if percentage_pos != -1:
        end_metric = percentage_pos
        error_percentage=True
    else:
        end_metric = len(metric)
        error_percentage=False
        fix = fix*100
    #fix to normalize errors:
    if metric[:end_metric] == 'MSE':
        fix = fix*100
    if metric[:end_metric] == 'FS':
        fix = fix/100

    errors_metric = {}
    for idx, a_model in enumerate(models):
      print('\nPredicting', models_names[idx])
      time.sleep(1)
      error_mean_all_horizons = []
      for out_channel in out_channels:
        #DataLoaders
        val_mvd = MontevideoFoldersDataset(path = PATH_DATA, 
                                          in_channel=2, 
                                          out_channel=out_channel,
                                          min_time_diff=5, max_time_diff=15, 
                                          csv_path=csv_path_base)
        val_loader = DataLoader(val_mvd)

        error_array = evaluate.evaluate_model(a_model, val_loader, 
                                              predict_horizon=out_channel,
                                              metric=metric[:end_metric],
                                              error_percentage=error_percentage,
                                              window_pad=params['window_pad'],
                                              window_pad_height=params['window_pad_height'],
                                              window_pad_width=params['window_pad_width'])
        error_mean = np.mean(error_array, axis=0)
        error_mean = error_mean/fix
        error_mean_all_horizons.append(error_mean[-1])
      print(f'Error_mean_{models_names[idx]}: {error_mean_all_horizons}')
      print(f'Error_mean_mean_{models_names[idx]}: {np.mean(error_mean_all_horizons)}')
      errors_metric[models_names[idx]] = error_mean_all_horizons

    errors_metrics[metric] = errors_metric

if params['save_errors']:
  PATH = "reports/errors_evaluate_model/"
  ts = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
  NAME = "base_"
  if params['save_name']:
    NAME = NAME + params['save_name'] + "_"
  NAME = NAME + str(ts) + '.pkl'
  a_file = open(os.path.join(PATH, NAME), "wb")
  pickle.dump(errors_metrics, a_file)
  a_file.close()
