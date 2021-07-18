import os
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
print('import basic')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
print('import torch')
from src import data, evaluate, model, preprocessing, visualization, train
from src.lib import utils

from src.data import MontevideoFoldersDataset
from src.dl_models.unet import UNet
from src.dl_models.unet_meteor_france import UNet_France
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
print('finis import')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)
#TRAINNING WITH TRAIN.PY

torch.manual_seed(50)
criterion = nn.L1Loss()

epochs = 120
batch_size = 20

select_frame = preprocessing.select_output_frame(1)
normalize = preprocessing.normalize_pixels(mean0 = False) #values between [0,1]

transforms = [select_frame,normalize]

train_mvd = MontevideoFoldersDataset(	path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/train/',    
                                    	in_channel=3,
                                    	out_channel=2,
                                     	min_time_diff=5,
					max_time_diff=15,
                                     	transform = transforms)
                                     
val_mvd = MontevideoFoldersDataset( 	path='/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/',    
                                   	in_channel=3,
                                  	out_channel=2,
                                   	min_time_diff=5,
					max_time_diff=15,
                                     	transform = transforms)

train_loader = DataLoader(train_mvd, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_mvd, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

learning_rates = [1e-3]
weight_decay = [0]

grid_search = [ (lr, wd) for lr in learning_rates for wd in weight_decay]

for lr, wd in grid_search:
  
  model = UNet(n_channels=3, n_classes=1, bilinear=True, p=0, output_sigmoid=True).to(device)
  model.apply(train.weights_init)  

  optimizer = optim.Adam(model.parameters(), lr=lr ,betas = (0.9,0.999),eps =1e-08, weight_decay=wd ,amsgrad=False)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-9) 

  comment = f' batch_size:{batch_size} lr:{lr} weight_decay:{wd} '
  writer = SummaryWriter(log_dir='runs/predict_20min' ,comment=comment)
  #writer=None

  TRAIN_LOSS_GLOBAL, VAL_LOSS_GLOBAL = train.train_model_2(model=model,
                                                           criterion= criterion,
                                                 	   optimizer=optimizer,
                                                 	   device=device,
                                                 	   train_loader=train_loader,
                                                  	   epochs=epochs,
                                                  	   val_loader=val_loader,
                                                 	   checkpoint_every=10,
                                                 	   verbose = True,
                                                	   writer=writer,
							   scheduler=scheduler,
							   model_name='UNET_20min_prediction',
							   save_images=True)
  if writer:
    writer.add_hparams(
                      {"lr": lr, "bsize": batch_size, "weight_decay":wd},
                      {
                          "loss train": TRAIN_LOSS_GLOBAL[-1],
                          "loss validation": VAL_LOSS_GLOBAL[-1] ,
                      },)
                           
if writer:
  writer.close()
