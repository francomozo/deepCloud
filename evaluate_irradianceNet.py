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
from src.data import PatchesFoldersDataset_w_geodata, PatchesFoldersDataset

print('finis import')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)

torch.manual_seed(50)

patch_model = True
MODEL_PATH = 'checkpoints/R3/240min/240min_UNET2_mae_sigmoid_f32_R3_48_31-08-2021_11:34.pt'

n_future_frames = 4
input_seq_len = 3
patch_size = 128
dataset = 'uru'

geo_data = False
direct = False
train_w_last = False

if train_w_last and direct:
    raise ValueError('To train with only last predict horizon the model shouldnt be direct')
if direct and n_future_frames > 1:
    raise ValueError('When direct the output is only on of 1 image')

if geo_data:
    in_channel = 4  # 1 if only image, higher if more metadata in training
else:
    in_channel = 1  # 1 if only image, higher if more metadata in training

if dataset == 'mvd':
    img_size = 256
elif dataset == 'uru':
    img_size = 512
elif dataset == 'region3':
    img_size = 1024

normalize = preprocessing.normalize_pixels(mean0 = False) #values between [0,1]
if geo_data:
    val_mvd = PatchesFoldersDataset_w_geodata(
        path='/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/validation/',
        in_channel=input_seq_len,
        out_channel=n_future_frames,
        min_time_diff=5,
        max_time_diff=15,
        output_last=True,
        img_size=img_size,
        patch_size=patch_size,
        geo_data_path='reports/',
        train=False)

else:
    val_mvd = PatchesFoldersDataset(
            path='/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/validation/',                          	
            in_channel=input_seq_len,
            out_channel=n_future_frames,
            min_time_diff=5,
            max_time_diff=15,
            transform=normalize,
            output_last=direct,
            img_size=img_size,
            patch_size=patch_size,
            train=False
            )

val_loader = DataLoader(val_mvd, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

if patch_model:
    model = ConvLSTM_patch(
        input_seq_len=input_seq_len,
        seq_len=n_future_frames,
        in_chan=in_channel,
        image_size=patch_size).cuda()

checkpoint = torch.load(MODEL_PATH, map_location=device)
if torch.cuda.device_count() == 1:
    for _ in range(len(checkpoint['model_state_dict'])):
        key, value = checkpoint['model_state_dict'].popitem(False)
        checkpoint['model_state_dict'][key[7:] if key[:7] == 'module.' else key] = value
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

with torch.no_grad():
    
    if direct or train_w_last:
        mse_val_loss = 0
        mae_val_loss = 0
        ssim_val_loss = 0
    else:
        mse_val_loss = np.zeros((n_future_frames))
        mae_val_loss = np.zeros((n_future_frames))
        ssim_val_loss = np.zeros((n_future_frames))
    
    for val_batch_idx, (in_frames, out_frames) in enumerate(val_loader):

        if not geo_data:
            in_frames = torch.unsqueeze(in_frames, dim=2)
            
        in_frames = in_frames.to(device=device)

        if not train_w_last:
            out_frames = torch.unsqueeze(out_frames, dim=2)
            
        out_frames = out_frames.to(device=device)
        
        if direct or train_w_last:
            mae_val_loss_Q = 0
            mse_val_loss_Q = 0
            ssim_val_loss_Q = 0
        else:
            mae_val_loss_Q = np.zeros((n_future_frames))
            mse_val_loss_Q = np.zeros((n_future_frames))
            ssim_val_loss_Q = np.zeros((n_future_frames))
        
        for i in range(dim):
            for j in range(dim):
                n = i * patch_size
                m = j * patch_size
                
                frames_pred_Q = model(in_frames[:,:,:, n:n+patch_size, m:m+patch_size])
                
                if direct:
                    mae_val_loss_Q += mae_loss(frames_pred_Q,
                                        out_frames[:,:,:, n:n+patch_size, m:m+patch_size]).detach().item()
                    mse_val_loss_Q += mse_loss(frames_pred_Q,
                                            out_frames[:,:,:, n:n+patch_size, m:m+patch_size]).detach().item()
                    
                    frames_pred_Q = torch.clamp(torch.squeeze(frames_pred_Q, dim=1), min=0, max=1)   
                    ssim_val_loss_Q += ssim_loss(frames_pred_Q,
                                                torch.squeeze(out_frames[:,:,:, n:n+patch_size, m:m+patch_size],
                                                                dim=1)
                                                ).detach().item()
                else:
                    if train_w_last:
                        mae_val_loss_Q += mae_loss(frames_pred_Q[:,-1,:,:,:],
                                                    out_frames[:,:, n:n+patch_size, m:m+patch_size]).detach().item()
                        mse_val_loss_Q += mse_loss(frames_pred_Q[:,-1,:,:,:],
                                                    out_frames[:,:, n:n+patch_size, m:m+patch_size]).detach().item()

                        frames_pred_Q = torch.clamp(frames_pred_Q[:,-1,:,:,:], min=0, max=1)
                        
                        ssim_val_loss_Q += ssim_loss(frames_pred_Q,
                                                    out_frames[:,:, n:n+patch_size, m:m+patch_size]).detach().item()
                        
                    else:
                        for x in range(n_future_frames):
                            mae_val_loss_Q[x] += mae_loss(frames_pred_Q[:, :, x, :, :],
                                                          out_frames[:,:,x, n:n+patch_size, m:m+patch_size]).detach().item()
                            mse_val_loss_Q[x] += mse_loss(frames_pred_Q[:, :, x, :, :],
                                                          out_frames[:,:,x, n:n+patch_size, m:m+patch_size]).detach().item()
                            ssim_val_loss_Q[x] += ssim_loss(torch.clamp(frames_pred_Q[:, x, :, : , :], min=0, max=1),
                                                            out_frames[:,:, n:n+patch_size, m:m+patch_size]).detach().item()
                
        if direct or train_w_last:
            mae_val_loss += (mae_val_loss_Q / (dim*dim))
            mse_val_loss += (mse_val_loss_Q / (dim**2))
            ssim_val_loss += (ssim_val_loss_Q / (dim**2))
        else:
            for x in range(n_future_frames):
                mae_val_loss[x] += (mae_val_loss_Q[x] / (dim**2))
                mse_val_loss[x] += (mse_val_loss_Q[x] / (dim**2))
                ssim_val_loss[x] += (ssim_val_loss_Q[x] / (dim**2))

print('MAE:', mse_val_loss)
print('MSE:', mse_val_loss)
print('SSIM:', mse_val_loss)
