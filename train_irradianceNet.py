import numpy as np
import pandas as pd
import json
import argparse
import os
import cv2 as cv
import matplotlib.pyplot as plt
import time
from piqa import SSIM

print('import basic')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
print('import torch')

# IrradianceNET
from src.dl_models.ConvLSTM_large import ConvLSTM_patch
from src.dl_models.ConvLSTM_small import ConvLSTM
from src.lib.utils_irradianceNet import convert_to_full_res, interpolate_borders

# DeepCloud
from src import data, evaluate, model, preprocessing, visualization, train
from src.lib import utils
from src.data import MontevideoFoldersDataset, PatchesFoldersDataset

print('finis import')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)

torch.manual_seed(50)

normalize = preprocessing.normalize_pixels(mean0 = False) #values between [0,1]

in_channel = 1 # 1 if only image, higher if more metadata in training
n_future_frames = 6
input_seq_len = 3
img_size = 512
patch_size = 128

train_mvd = PatchesFoldersDataset(
        path='/clusteruy/home03/DeepCloud/deepCloud/data/uru/train/',    
        in_channel=input_seq_len,
        out_channel=n_future_frames,
        min_time_diff=5,
        max_time_diff=15,
        transform=normalize,
        output_last=False,
        img_size=img_size,
        patch_size=patch_size,
        train=True
        )
                             
val_mvd = PatchesFoldersDataset(
        path='/clusteruy/home03/DeepCloud/deepCloud/data/uru/validation/',                          	
        in_channel=input_seq_len,
        out_channel=n_future_frames,
        min_time_diff=5,
        max_time_diff=15,
        transform=normalize,
        output_last=False,
        img_size=img_size,
        patch_size=patch_size,
        train=False
        )


batch_size = 10
epochs = 1


train_loader = DataLoader(train_mvd, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_mvd, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


patch_model = True

if patch_model:
    # patch model
    model = ConvLSTM_patch(
        input_seq_len=input_seq_len,
        seq_len=n_future_frames,
        in_chan=in_channel,
        image_size=patch_size).cuda()

else:
    # full image model
    model = ConvLSTM(
        seq_len=n_future_frames,
        in_chan=in_channel,
        input_seq_len=input_seq_len).cuda()

model.apply(train.weights_init)  


  
# OPTIMIZER
# Paper:
# We use the Adam optimizer with an initial learning
# rate of 0.002 and gradually decrease it by half once the validation loss
# stops improving beyond a small threshold for at least five consecutive
# epochs, also called reduce learning rate on plateau

lr = 2e-3

optimizer = optim.Adam(model.parameters(), lr=lr , betas=(0.9,0.999), eps=1e-08, weight_decay=0 ,amsgrad=False)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-9)

train_loss = 'mae'  # ['mae', 'mse', 'ssim']
loss_for_scheduler = 'mae'
model_name = f'60min_IrradianceNet_uru_{train_loss}'
comment = f' batch_size:{batch_size} lr:{lr} model:irradianceNet train_loss:{train_loss}'
writer = SummaryWriter(log_dir='runs/predict_60min/irradianceNet', comment=comment)

# TRAIN LOOP

TRAIN_LOSS, VAL_MAE_LOSS, VAL_MSE_LOSS, VAL_SSIM_LOSS = train.train_irradianceNet(
    model=model,
    train_loss=train_loss,
    optimizer=optimizer,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=epochs,
    img_size=512,
    patch_size=128,
    checkpoint_every=10,
    verbose=True,
    writer=writer,
    scheduler=scheduler,
    loss_for_scheduler=loss_for_scheduler,
    model_name=model_name,
    save_images=True
    )

if writer:
  writer.close()
  
if save_dict:
    learning_values = {
        'model_name': model_name,
        'train_loss': train_loss,
        'validation_loss': loss_for_scheduler,
        'train_loss_epoch_mean': TRAIN_LOSS,
        'val_mae_loss': VAL_MAE_LOSS,
        'val_mse_loss': VAL_MSE_LOSS,
        'val_ssim_loss': VAL_SSIM_LOSS
        }                                         
    utils.save_pickle_dict(name=model_name, dict_=learning_values)   

