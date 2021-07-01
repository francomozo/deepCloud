# from src.data import save_imgs_list_2npy
# import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import time
import argparse

import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import data, evaluate, model, preprocessing, train, visualization
from src.data import MontevideoFoldersDataset
from src.dl_models.unet import UNet
from src.train import train_model

# from src.lib import utils

ap = argparse.ArgumentParser()

ap.add_argument("--seed", default=50, type=int,
                help="Seed for PyTorch. Defaults to 50.")
ap.add_argument("--epochs", default=50, type=int,
                help="Defaults to 50.")
ap.add_argument("--batch-size", default=20, type=int,
                help="Defaults to 20.")
ap.add_argument("--num-val-samples", default=10, type=int,
                help="Defaults to 10.")
ap.add_argument("--eval-every", default=50, type=int,
                help="Defaults to 50.")
ap.add_argument("--csv-path", default=None,
                help="String. Defaults str 'None'.")
ap.add_argument("-lr", "--learning-rates", default=[1e-3], nargs="+", type=float,
                help="List. Floats for learning rates (ie: 1e-3). Defaults to [1e-3].")
ap.add_argument("-wd", "--weight-decay", default=[0], nargs="+", type=float,
                help="List. Floats for the weight decay. Defaults to [0].")
ap.add_argument("--checkpoint-every", default=10, type=int,
                help="Checkpoint every x epochs. Defaults to 10.")

params = vars(ap.parse_args())
csv_path = params['csv_path'] if params['csv_path'] != 'None' else None

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)

# TRAINNING WITH TRAIN.PY
torch.manual_seed(params['seed'])
criterion = nn.L1Loss()

epochs = params['epochs']
batch_size = params['batch_size']
num_val_samples = params['num_val_samples']
eval_every = params['eval_every']

normalize = preprocessing.normalize_pixels()

train_mvd = MontevideoFoldersDataset(path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/',    
                                     in_channel=3,
                                     out_channel=1,
                                     min_time_diff=5, max_time_diff=15,
                                     csv_path=csv_path,
                                     transform=normalize)

                                     
val_mvd = MontevideoFoldersDataset(path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/',    
                                   in_channel=3,
                                   out_channel=1,
                                   min_time_diff=5, max_time_diff=15,
                                   transform=normalize)

train_loader = DataLoader(train_mvd, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_mvd, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

learning_rates = params['learning_rates']
weight_decay = params['weight_decay']
grid_search = [ (lr, wd) for lr in learning_rates for wd in weight_decay ]

for lr, wd in grid_search:
  model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=False)
  print('lr =', lr, 'weight_decay =', wd)

  comment = f' batch_size = {batch_size} lr = {lr} weight_decay = {wd}'
  writer = SummaryWriter(comment=comment)

  TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL = train_model(model=model,
                                                   criterion=criterion,
                                                   optimizer=optimizer,
                                                   device=device,
                                                   train_loader=train_loader,
                                                   epochs=epochs,
                                                   val_loader=val_loader,
                                                   num_val_samples=num_val_samples,
                                                   checkpoint_every=params['checkpoint_every'],
                                                   verbose=True,
                                                   eval_every=eval_every,
                                                   writer=writer)
  
  writer.add_hparams(
                    {"lr": lr, "bsize": batch_size, "weight_decay":wd},
                    {
                        "loss train": TRAIN_LOSS_GLOBAL[-1],
                        "loss validation": VAL_LOSS_GLOBAL[-1] ,
                    },)
                            

writer.close()
