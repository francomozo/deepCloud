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
from src.dl_models.unet import UNet, UNet2


ap = argparse.ArgumentParser(description='Evaluate direct models with multiple metrics')

# ap.add_argument("--out-channel", default=None, type=int,
#                 help="Defaults to None")
ap.add_argument("--horizon-step", default=6, type=int,
                help="Defaults to 6. Distance between predictions")                
ap.add_argument("--start-horizon", default=6, type=int,
                help="Defaults to 6. Starting point of prediction")
ap.add_argument("--predict-horizon", default=5, type=int,
                help="Defaults to 5. Amount of predicted images")

ap.add_argument("--metrics", nargs="+", default=["RMSE"],
                help="Defaults to RMSE. Add %% for percentage metric")
ap.add_argument("--csv-path-unet", default=None,
                help="Csv path for unets. Defaults to None.")
ap.add_argument("--data-path", default='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/',
                help="Defaults to /clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/")
ap.add_argument("--model-path", default="checkpoints",
                help="Model path without names for NNs.")
ap.add_argument("--model-names", nargs="*", default=None,
                help="Model names for NNs.")
ap.add_argument("--save-errors", default=False, type=bool,
                help="Save results in file. Defaults to False.")
ap.add_argument("--save-name", default=None,
                help="Add name to results file. Defaults to None.")
ap.add_argument("--output-activation", default='sigmoid',
                help="Output activation for unets. Defaults to sigmoid.")
ap.add_argument("--unet-type", default=1, type=int,
                help="Type of unet. Defaults to None.")
ap.add_argument("--bias", default=False, type=bool,
                help="bias of unet. Defaults to False.")
ap.add_argument("--filters", default=64, type=int,
                help="Amount of filters of unet. Defaults to 64.")

ap.add_argument("--window-pad", default=0, type=int,
                help="Size of padding for evaluation, eval window is [w_p//2 : M-w_p//2, w_p//2 : N-w_p//2]. Defaults to 0.")
ap.add_argument("--window-pad-height", default=0, type=int,
                help="Size of height padding for evaluation, eval window is [w_p_h//2 : M-w_p_h//2]. Defaults to 0.")
ap.add_argument("--window-pad-width", default=0, type=int,
                help="Size of width padding for evaluation, eval window is [w_p_w//2 : N-w_p_w//2]. Defaults to 0.")

params = vars(ap.parse_args())
csv_path_unet = params['csv_path_unet']
PATH_DATA = params['data_path']
metrics = params['metrics']
metrics = [each_string.upper() for each_string in metrics]
#start_horizon = params['start_horizon'] if params['start_horizon'] is not None else 0
#out_channel = params['out_channel'] if params['out_channel'] is not None else params['predict_horizon']
start_horizon = params['start_horizon']
horizon_step = params['horizon_step']
predict_horizon = params['predict_horizon']
out_channels = list(range(start_horizon, start_horizon+horizon_step*predict_horizon, horizon_step))
print("out_channels:", out_channels)

#Definition of models
models = []
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
for i in range(params['predict_horizon']):
  model_path = os.path.join(params["model_path"], params["model_names"][i])
  if params['unet_type'] == 1:
    model_Unet = UNet(n_channels=3, n_classes=1, bilinear=True, output_activation=params["output_activation"], bias=params["bias"], filters=params["filters"]).to(device)
  elif params['unet_type'] == 2:
    model_Unet = UNet2(n_channels=3, n_classes=1, bilinear=True, output_activation=params["output_activation"], bias=params["bias"], filters=params["filters"]).to(device)
  model_Unet.load_state_dict(torch.load(model_path)["model_state_dict"])
  model_Unet.eval()
  models.append(model_Unet)

errors_metrics = {}
for metric in metrics:
  print("\n", metric)
  #variables for percentage evaluation:
  percentage_pos = metric.find("%") 
  if percentage_pos != -1:
    end_metric = percentage_pos
    error_percentage=True
  else:
    end_metric = len(metric)
    error_percentage=False
    
  errors_metric = {}
  time.sleep(1)

  normalize = preprocessing.normalize_pixels(mean0=False) 
  error_mean_all_models = []
  for i in range(len(models)):
    #DataLoaders
    val_dataset_Unet = MontevideoFoldersDataset(path = PATH_DATA, 
                                        in_channel=3, 
                                        out_channel=out_channels[i],
                                        min_time_diff=5, max_time_diff=15,
                                        transform=normalize, 
                                        csv_path=csv_path_unet,
                                        output_last=True)
    val_loader_Unet = DataLoader(val_dataset_Unet)

    error_array = evaluate.evaluate_model(models[i], val_loader_Unet, 
                                        predict_horizon=1, 
                                        start_horizon=None,
                                        device=device, 
                                        metric=metric[:end_metric], 
                                        error_percentage=error_percentage,
                                        window_pad=params['window_pad'],
                                        window_pad_height=params['window_pad_height'],
                                        window_pad_width=params['window_pad_width'])
    error_mean = np.mean(error_array, axis=0)
    error_mean_all_models.append(error_mean)
  print(f'Error_mean: {error_mean_all_models}')
  print(f'Error_mean_mean: {np.mean(error_mean_all_models)}')
  errors_metric["unet_direct"] = error_mean_all_models

  errors_metrics[metric] = errors_metric

if params['save_errors']:
  PATH = "reports/errors_evaluate_model/"
  ts = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
  NAME = 'errors_models_direct_' + str(ts)
  if params['save_name']:
    NAME = NAME + "_" + params['save_name']
  NAME = NAME + '.pkl'
  a_file = open(os.path.join(PATH, NAME), "wb")
  pickle.dump(errors_metrics, a_file)
  a_file.close()
