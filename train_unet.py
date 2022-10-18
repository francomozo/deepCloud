import os
import numpy as np
import pandas as pd
import time
import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import multiprocessing
import re

from src import data, model, preprocessing, train
from src.lib import utils
from src.data import MontevideoFoldersDataset
from src.dl_models.unet import UNet, UNet2
from src.dl_models.unet_advanced import R2U_Net, AttU_Net, NestedUNet


# parse arguments
parser = argparse.ArgumentParser(description='Script for training U-Net model.') 

parser.add_argument("--dataset", type=str, default="uru", help="Options: mvd, uru, region3")
parser.add_argument("--epochs", type=int, default=100, help="length of training")
parser.add_argument("--batch_size", type=int, default=7, help="batch size")
parser.add_argument("--architecture", type=str, default="UNET", help="UNET, R2, Att, Nested")
parser.add_argument("--init_filters", type=int, default=16, help="number of filters in initial layer of U-Net")
parser.add_argument("--output_last", type=bool, default=True, help="Dataloader only outputs objective time horizon")
parser.add_argument("--time_horizon", type=str, default="60min",
			help="Time horizon for the predictions, Options: 10min, 60min, 120min, 180min, 240min, 300min")
parser.add_argument("--output_activation", type=str, default="sigmoid",
			help="Output activation in the U-Net, Options: sigmoid, tanh, relu")
parser.add_argument("--load_checkpoint_path", type=str, default="",
			help="Path to model checkpoint for retraining")
parser.add_argument("--retrain", type=bool, default=False, help="Retrain checkpoint")
parser.add_argument("--save_dict", type=bool, default=True, help="Save dict with training information")
parser.add_argument("--train_loss", type=str, default="mae", help="Options: mae, mse, ssim")
parser.add_argument("--loss_for_scheduler", type=str, default="mae",
			help="Metric used to control scheduler, Options: mae, mse, ssim")
parser.add_argument("--predict_diff", type=bool, default=False, help="Train U-Net to predict difference")
parser.add_argument("--testing_loop", type=bool, default=False, help="Run fast pass through training loop")
parser.add_argument("--writer", type=bool, default=True, help="Save results on a TensorBoard folder")
parser.add_argument("--day_pct_filter", type=float, default=1, help="filter images with lower percentage of day pixels")

args = parser.parse_args()

def out_channel_calculator(time_horizon: str):
	number = int(re.findall(r'\d+', time_horizon)[0])
	return number//10

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)

torch.manual_seed(50)

train_mvd = MontevideoFoldersDataset(
	path='/clusteruy/home03/DeepCloud/deepCloud/data/' + args.dataset + '/train/',
	in_channel=3,
	out_channel=out_channel_calculator(args.time_horizon),
	min_time_diff=5,
	max_time_diff=15,
        transform=preprocessing.normalize_pixels(mean0=False),
	output_last=args.output_last,
	day_pct=args.day_pct_filter
)
                                     
val_mvd = MontevideoFoldersDataset(
	path='/clusteruy/home03/DeepCloud/deepCloud/data/' + args.dataset + '/validation/',    
        in_channel=3,
        out_channel=out_channel_calculator(args.time_horizon),
        min_time_diff=5,
	max_time_diff=15,
        transform=preprocessing.normalize_pixels(mean0=False),
	output_last=args.output_last,
	day_pct=args.day_pct_filter
)

train_loader = DataLoader(train_mvd, batch_size=args.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())
val_loader = DataLoader(val_mvd, batch_size=args.batch_size, shuffle=True, num_workers=multiprocessing.cpu_count())

learning_rates = [1e-3]

grid_search = [(lr) for lr in learning_rates]

for lr in grid_search:
  if args.architecture == 'UNET':
    model = UNet2(n_channels=3, n_classes=1, bilinear=True, output_activation=args.output_activation, filters=args.init_filters).to(device)
  if args.architecture == 'R2':
    model = R2U_Net(img_ch=3, output_ch=1, t=2)
  if args.architecture == 'Att':
    model = AttU_Net(img_ch=3, output_ch=1, output_activation=args.output_activation, init_filter=args.init_filters)
  if args.architecture == 'Nested':
    model = NestedUNet(in_ch=3, out_ch=1, output_activation=args.output_activation, init_filter=args.init_filters)
       
  if torch.cuda.device_count() > 1:
    print('Model Paralleling')
    model = nn.DataParallel(model)

  if args.retrain:
      checkpoint = torch.load(args.load_checkpoint_path, map_location=device)
      if torch.cuda.device_count() == 1:
          for _ in range(len(checkpoint['model_state_dict'])):
              key, value = checkpoint['model_state_dict'].popitem(False)
              checkpoint['model_state_dict'][key[7:] if key[:7] == 'module.' else key] = value
      model.load_state_dict(checkpoint['model_state_dict']) 

  model.to(device)
  
  if not args.retrain:
    model.apply(train.weights_init)  
    checkpoint = False

  optimizer = optim.Adam(model.parameters(), lr=lr ,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
 
  if args.retrain:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('actual learning rate', checkpoint['optimizer_state_dict']['param_groups'][0]['lr'])
 
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-7) 

  checkpoint_folder = f'paper/{args.dataset}/{args.time_horizon}/'
  model_name = f'{args.time_horizon}_{args.architecture}_{args.dataset}_{args.train_loss}_filters{args.init_filters}_{args.output_activation}_diff{args.predict_diff}_retrain{args.retrain}'
  
  comment = f'batch_size:{args.batch_size} lr:{lr} model:{args.architecture} train_loss:{args.train_loss} predict_diff{args.predict_diff}'
  if args.writer:  
  	writer = SummaryWriter(log_dir=f'runs/paper/{args.time_horizon}/' + model_name, comment=comment)
  else:
  	writer = None
  print(model_name)
  TRAIN_LOSS, VAL_MAE_LOSS, VAL_MSE_LOSS, VAL_SSIM_LOSS = train.train_model_full(	
  	model=model,
        train_loss=args.train_loss,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
	epochs=args.epochs,		
        checkpoint_every=1,
        verbose=True,
        writer=writer,
        scheduler=scheduler,
	loss_for_scheduler=args.loss_for_scheduler,
	model_name=checkpoint_folder + model_name,
	predict_diff=args.predict_diff,
        retrain=args.retrain,
        trained_model_dict=checkpoint,
        testing_loop=args.testing_loop
  )
                         
  if writer:
    writer.close()

  if args.save_dict:
    learning_values = {
      'model_name': model_name,
      'args': args,
      'train_loss': args.train_loss,
      'predict diff': args.predict_diff,
      'validation_loss': args.loss_for_scheduler,
      'train_loss_epoch_mean': TRAIN_LOSS,
      'val_mae_loss': VAL_MAE_LOSS,
      'val_mse_loss': VAL_MSE_LOSS,
      'val_ssim_loss': VAL_SSIM_LOSS
     }
    utils.save_pickle_dict(name=model_name, dict_=learning_values)

