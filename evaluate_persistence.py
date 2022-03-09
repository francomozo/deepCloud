import numpy as np
import pandas as pd
import json
import os
import cv2 as cv
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
print('import basic')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
print('import torch')

# DeepCloud
from src import data, evaluate, model, preprocessing, visualization, train
from src.lib import utils
from src.lib.latex_options import Colors, Linestyles
from src.data import MontevideoFoldersDataset, MontevideoFoldersDataset_w_time
print('Finish imports')

### SETUP #############################################################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)
MSE = nn.MSELoss()
MAE = nn.L1Loss()
normalize = preprocessing.normalize_pixels(mean0 = False) #values between [0,1]
borders = np.linspace(1, 450, 100)
#######################################################################################

REGION = 'R3'  # [URU, R3]

if REGION == 'URU':
    dataset = 'uru'
    img_size = 512
elif REGION == 'R3':
    dataset = 'region3'
    img_size = 1024
    
PREDICT_T_LIST = [6, 12, 18, 24, 30]  # 1->10min, 2->20min, 3->30min... [1,6] U [12] U [18] U [24] U [30]

evaluate_test = True

RMSE_pct_maps_list = []
RMSE_maps_list = []
MAE_maps_list = []

for PREDICT_T in PREDICT_T_LIST:
    
    if PREDICT_T == 6:
        PREDICT_HORIZON = '60min'
    if PREDICT_T == 12:
        PREDICT_HORIZON = '120min'
    if PREDICT_T == 18:
        PREDICT_HORIZON = '180min'
    if PREDICT_T == 24:
        PREDICT_HORIZON = '240min'
    if PREDICT_T == 30:
        PREDICT_HORIZON = '300min'

    print('Predict Horizon:', PREDICT_HORIZON)

    if evaluate_test:
        CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/region3/test_cosangs_region3.csv'
        PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/test/'
        SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/test/persistence/'   
        SAVE_BORDERS_ERROR = 'reports/borders_cut/' + REGION + '/' + PREDICT_HORIZON + '/test/persistence/'

    else:
        CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/region3/val_cosangs_region3.csv'
        PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/validation/'
        SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/persistence/'
        SAVE_BORDERS_ERROR = 'reports/borders_cut/' + REGION + '/' + PREDICT_HORIZON

    #########################################################################################

    try:
        os.mkdir(SAVE_IMAGES_PATH)
    except:
        pass

    try:
        os.mkdir(SAVE_BORDERS_ERROR)
    except:
        pass
    
    try:
        os.mkdir(SAVE_BORDERS_ERROR)
    except:
        pass

    val_dataset = MontevideoFoldersDataset(
        path=PATH_DATA,
        in_channel=3,
        out_channel=PREDICT_T,
        min_time_diff=5,
        max_time_diff=15,
        csv_path=CSV_PATH,
        transform=normalize,
        output_last=True
    )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    mean_image = np.zeros((img_size, img_size))

    MAE_error_image = np.zeros((img_size, img_size))
    MAE_pct_error_image = np.zeros((img_size, img_size))
    RMSE_pct_error_image = np.zeros((img_size, img_size))
    RMSE_error_image = np.zeros((img_size, img_size))

    for val_batch_idx, (in_frames, out_frames) in enumerate(tqdm(val_loader)):

        in_frames = in_frames.to(device=device)
        out_frames = out_frames.to(device=device)

        mean_image += out_frames[0,0].cpu().numpy()

        MAE_error_image += torch.abs(torch.subtract(out_frames[0,0], in_frames[0,0])).cpu().numpy()

        RMSE_error_image += torch.square(torch.multiply(torch.subtract(out_frames[0,0], in_frames[0, 0]), 100)).cpu().numpy()

    mean_image = (mean_image / len(val_dataset)) * 100  # contains the mean value of each pixel independently 
    MAE_error_image = (MAE_error_image / len(val_dataset))
    MAE_pct_error_image = (MAE_error_image / mean_image) * 100
    RMSE_pct_error_image = (np.sqrt((RMSE_error_image) / len(val_dataset)) / mean_image) * 100
    RMSE_error_image = (np.sqrt((RMSE_error_image) / len(val_dataset))) / 100

    RMSE_pct_maps_list.append(RMSE_pct_error_image)
    RMSE_maps_list.append(RMSE_error_image)
    MAE_maps_list.append(MAE_error_image)
    
    np.save(os.path.join(SAVE_IMAGES_PATH, 'mean_image.npy'), mean_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'mean_image.pdf')
    visualization.show_image_w_colorbar(
        image=mean_image,
        title=None,
        fig_name=fig_name,
        save_fig=True,
        bar_max=1,
        colormap='viridis'
    )
    plt.close()
    
    np.save(os.path.join(SAVE_IMAGES_PATH, 'MAE_error_image.npy'), MAE_error_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_error_image.pdf')
    visualization.show_image_w_colorbar(
        image=MAE_error_image,
        title=None,
        fig_name=fig_name,
        save_fig=True,
        bar_max=1,
        colormap='coolwarm'
    )
    plt.close()
    
    np.save(os.path.join(SAVE_IMAGES_PATH, 'MAE_pct_error_image.npy'), MAE_pct_error_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_pct_error_image.pdf')
    visualization.show_image_w_colorbar(
        image=MAE_pct_error_image,
        title=None,
        fig_name=fig_name,
        save_fig=True,
        bar_max=100,
        colormap='coolwarm'
    )
    plt.close()

    np.save(os.path.join(SAVE_IMAGES_PATH, 'RMSE_error_image.npy'), RMSE_error_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_error_image.pdf')
    visualization.show_image_w_colorbar(
        image=RMSE_error_image,
        title=None,
        fig_name=fig_name,
        save_fig=True,
        bar_max=1,
        colormap='coolwarm'
    )
    plt.close()

    np.save(os.path.join(SAVE_IMAGES_PATH, 'RMSE_pct_error_image.npy'), RMSE_pct_error_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_pct_error_image.pdf')
    visualization.show_image_w_colorbar(
        image=RMSE_pct_error_image,
        title=None,
        fig_name=fig_name,
        save_fig=True,
        bar_max=100,
        colormap='coolwarm'
    )
    plt.close()

    mae_errors_borders = []
    r_RMSE_errors_borders = []

    for i in borders:
        p = int(i)
        mae_errors_borders.append(np.mean(MAE_error_image[p:-p, p:-p]))
        r_RMSE_errors_borders.append(np.mean(RMSE_pct_error_image[p:-p, p:-p]))
        
    if SAVE_BORDERS_ERROR:
        dict_values = {
            'model_name': 'persistence',
            'test_dataset': evaluate_test,
            'csv_path': CSV_PATH,
            'predict_t': PREDICT_T,
            'borders': borders,
            'mae_errors_borders': mae_errors_borders,
            'r_RMSE_errors_borders': r_RMSE_errors_borders
        }                                                                                                                      

        utils.save_pickle_dict(path=SAVE_BORDERS_ERROR, name='persistence', dict_=dict_values)

fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_pct_maps_together.pdf')
visualization.error_maps_for_5_horizons(
    error_maps_list=RMSE_pct_maps_list,
    vmax=100,
    fig_name=fig_name,
    save_fig=True,
    colormap='coolwarm'
)

fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_maps_together.pdf')
visualization.error_maps_for_5_horizons(
    error_maps_list=MAE_maps_list,
    vmax=1,
    fig_name=fig_name,
    save_fig=True,
    colormap='coolwarm'
)

fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_maps_together.pdf')
visualization.error_maps_for_5_horizons(
    error_maps_list=RMSE_maps_list,
    vmax=1,
    fig_name=fig_name,
    save_fig=True,
    colormap='coolwarm'
)
