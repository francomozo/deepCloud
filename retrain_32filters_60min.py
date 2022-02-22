import os
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print('import basic')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

print('import torch')
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src import data, evaluate, model, preprocessing, train, visualization
from src.data import MontevideoFoldersDataset
from src.dl_models.unet import UNet, UNet2
from src.dl_models.unet_advanced import (AttU_Net, NestedUNet, R2AttU_Net,
                                         R2U_Net)
from src.lib import utils

print('finis import')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)
#TRAINNING WITH TRAIN.PY

torch.manual_seed(50)

batch_size = 3
init_filters = 32
MODEL_PATH = 'checkpoints/R3/60min/60min_UNET__region3_mae_filters32_sigmoid_diffFalse_retrainFalse_40_01-02-2022_20:43.pt' # AGREGAR NOMBRE COMPLETO DEL MODELO



dataset = 'region3'  # 'mvd', 'uru', 'region3'
epochs = 100

normalize = preprocessing.normalize_pixels(mean0 = False) #values between [0,1]

train_mvd = MontevideoFoldersDataset(path='/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/train/',    
                                     in_channel=3,
                                     out_channel=6,
                                     min_time_diff=5,
				     max_time_diff=15,
                                     transform = normalize,
				     output_last=True)
                                     
val_mvd = MontevideoFoldersDataset(path='/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/validation/',    
                                   in_channel=3,
                                   out_channel=6,
                                   min_time_diff=5,
				   max_time_diff=15,
                                   transform = normalize,
				   output_last=True)

train_loader = DataLoader(train_mvd, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_mvd, batch_size=batch_size, shuffle=True, num_workers=2)

retrain = True


learning_rates = [1e-3]
arquitecture = ['']  # ['R2', 'Att', 'R2Att', 'Nested']


grid_search = [ (lr, mdl) for lr in learning_rates for mdl in arquitecture]

for lr, mdl in grid_search:
  if mdl == '':
    model = UNet(n_channels=3, n_classes=1, bilinear=True, output_activation='sigmoid', filters=init_filters).to(device)
  if torch.cuda.device_count() > 1:
    print('Model Paralleling')
    model = nn.DataParallel(model)

  if retrain:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if torch.cuda.device_count() == 1:
        for _ in range(len(checkpoint['model_state_dict'])):
            key, value = checkpoint['model_state_dict'].popitem(False)
            checkpoint['model_state_dict'][key[7:] if key[:7] == 'module.' else key] = value
    model.load_state_dict(checkpoint['model_state_dict']) 

 
  model.to(device)
  
  if not retrain:
      model.apply(train.weights_init)  
  
  optimizer = optim.Adam(model.parameters(), lr=lr ,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
 
  if retrain:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('actual learning rate', checkpoint['optimizer_state_dict']['param_groups'][0]['lr'])
 
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-7) 

  save_dict = True
  
  train_loss = 'mae'  # ['mae', 'mse', 'ssim']
  loss_for_scheduler = 'mae'
  predict_diff = False
  checkpoint_folder = 'R3/60min/'
  model_name = f'60min_UNET_{mdl}_{dataset}_{train_loss}_filters{init_filters}_sigmoid_diff{predict_diff}_retrain{retrain}'
  
  comment = f' batch_size:{batch_size} lr:{lr} model:{mdl} train_loss:{train_loss} predict_diff{predict_diff}'
  
  writer = SummaryWriter(log_dir='runs/predict_60min/'+model_name, comment=comment)
  #writer = None
  print(model_name)
  TRAIN_LOSS, VAL_MAE_LOSS, VAL_MSE_LOSS, VAL_SSIM_LOSS = train.train_model_full(	
                                                               	model=model,
                                                                train_loss=train_loss,
                                                 	 	optimizer=optimizer,
                                                 	 	device=device,
                                                 	        train_loader=train_loader,
                                                  	 	val_loader=val_loader,
								epochs=epochs,		
                                                 	 	checkpoint_every=1,
                                                 	 	verbose=True,
                                                	 	writer=writer,
                                                                scheduler=scheduler,
								loss_for_scheduler=loss_for_scheduler,
								model_name=checkpoint_folder + model_name,
								predict_diff=predict_diff,
                                                                retrain=retrain,
                                                                trained_model_dict=checkpoint,
                                                                testing_loop=False)
  
  if writer and False:
    writer.add_hparams(
                      {"lr": lr, "bsize": batch_size, "model":mdl},
                      {
                          "loss train": TRAIN_LOSS[-1],
                          "loss validation": VAL_MAE_LOSS[-1] ,
                      },)
                           
  if writer:
    writer.close()

  if save_dict:
    learning_values = {
      'model_name': model_name,
      'train_loss': train_loss,
      'predict diff': predict_diff,
      'validation_loss': loss_for_scheduler,
      'train_loss_epoch_mean': TRAIN_LOSS,
      'val_mae_loss': VAL_MAE_LOSS,
      'val_mse_loss': VAL_MSE_LOSS,
      'val_ssim_loss': VAL_SSIM_LOSS
     }
    utils.save_pickle_dict(name=model_name, dict_=learning_values)