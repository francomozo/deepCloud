from src.data import save_imgs_list_2npy
import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import data, evaluate, model, preprocessing, visualization, train
from src.lib import utils

from src.data import MontevideoFoldersDataset
from src.dl_models.unet import UNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)
#TRAINNING WITH TRAIN.PY

torch.manual_seed(50)
criterion = nn.L1Loss()

epochs = 50
batch_size = 20
num_val_samples = 10
eval_every = 50

normalize = preprocessing.normalize_pixels()

train_mvd = MontevideoFoldersDataset(path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/',    
                                    in_channel=3,
                                    out_channel=1,
                                     min_time_diff=5,max_time_diff=15,
                                     transform = normalize)
                                     
val_mvd = MontevideoFoldersDataset(path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/',    
                                  in_channel=3,
                                  out_channel=1,
                                   min_time_diff=5,max_time_diff=15,
                                     transform = normalize)

train_loader = DataLoader(train_mvd, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory = True)
val_loader = DataLoader(val_mvd, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True)

learning_rates = [1e-3]
weight_decay = [0]
grid_search = [ (lr, wd) for lr in learning_rates for wd in weight_decay ]

for lr, wd in grid_search:
  model = UNet(n_channels=3,n_classes=1,bilinear=True).to(device)
  optimizer = optim.Adam(model.parameters(), lr=lr ,betas = (0.9,0.999),eps =1e-08, weight_decay=wd ,amsgrad=False)
  print('lr =', lr, 'weight_decay =', wd)

  comment = f' batch_size = {batch_size} lr = {lr} weight_decay = {wd}'
  writer = SummaryWriter(comment=comment)

  TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL = train.train_model(model = model,
                                                  criterion= criterion,
                                                  optimizer=optimizer,
                                                  device=device,
                                                  train_loader= train_loader,
                                                  epochs=epochs,
                                                  val_loader = val_loader,
                                                  num_val_samples=num_val_samples,
                                                  checkpoint_every=10,
                                                  verbose = True,
                                                  eval_every = eval_every,
                                                  writer=writer)
  
  writer.add_hparams(
                    {"lr": lr, "bsize": batch_size, "weight_decay":wd},
                    {
                        "loss train": TRAIN_LOSS_GLOBAL[-1],
                        "loss validation": VAL_LOSS_GLOBAL[-1] ,
                    },)
                            
writer.close()
