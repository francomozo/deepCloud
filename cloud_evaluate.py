import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import optuna
import datetime
import pickle

import torch
from torch.utils.data import DataLoader

from src import data, evaluate, model, preprocessing, visualization
from src.lib import utils
from src.data import MontevideoDataset, MontevideoFoldersDataset
from src.dl_models.unet import UNet

PATH_PROJECT = 'C:/Users/Ignacio/Desktop/Facultad/2021/proyecto/'
PATH_DATA = os.path.join(PATH_PROJECT, 'data/mvd/validation')
print(PATH_DATA)

#Evaluate Unet
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)

normalize = preprocessing.normalize_pixels(mean0=False)

val_mvd = MontevideoFoldersDataset(
                                    path=PATH_DATA,
                                    in_channel=3,
                                    out_channel=6,
                                    min_time_diff=5,
                                    max_time_diff=15,
                                    transform=normalize
                                    )
val_loader = DataLoader(val_mvd)

load_path = 'checkpoints/model_epoch20_18-06-2021_07_35.pt'

model_Unet = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
model_Unet.load_state_dict(torch.load(load_path)["model_state_dict"])
model_Unet.eval()

print('Predicting Unet')
time.sleep(1)
error_array_Unet = evaluate.evaluate_model(model_Unet, val_loader, 6, device=device, metric='MBD')

error_mean_Unet = np.mean(error_array_Unet, axis=0)
print(f'error_array_Unet: {error_mean_Unet}')
visualization.plot_graph(error_mean_Unet, model='Unet')