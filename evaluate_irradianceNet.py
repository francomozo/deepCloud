import numpy as np
import pandas as pd
import json
import os
import cv2 as cv
import matplotlib.pyplot as plt
import time
from piqa import SSIM
from tqdm import tqdm
print('import basic')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
print('import torch')

# IrradianceNET
from src.dl_models.ConvLSTM_large import ConvLSTM_patch
from src.lib.utils_irradianceNet import convert_to_full_res, interpolate_borders

# DeepCloud
from src import data, evaluate, model, preprocessing, visualization, train
from src.lib import utils
from src.lib.latex_options import Colors, Linestyles
from src.data import PatchesFoldersDataset_w_geodata, PatchesFoldersDataset
from src.lib.utils import get_model_name
print('Finish imports')

### SETUP #############################################################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)
MSE = nn.MSELoss()
MAE = nn.L1Loss()
SSIM = SSIM(n_channels=1).cuda()
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

input_seq_len = 4
patch_size = 128

dim = img_size // patch_size

GEO_DATA = False
TRAIN_W_LAST = True
    
PREDICT_T_LIST = [6, 12, 18, 24, 30]  # 1->10min, 2->20min, 3->30min... [1,6] U [12] U [18] U [24] U [30]

evaluate_test = True
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
    
    MODEL_NAME = get_model_name(PREDICT_HORIZON, architecture='irradianceNet', geo=GEO_DATA)
    MODEL_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/' + REGION + '/' + PREDICT_HORIZON +  '/' + MODEL_NAME

    if evaluate_test:
        CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/region3/test_cosangs_region3.csv'
        PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/test/'
        SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/test/' + MODEL_PATH.split('/')[-1][:-9]  
        SAVE_BORDERS_ERROR = 'reports/eval_per_hour/' + REGION + '/test'

    else:
        CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/region3/val_cosangs_region3.csv'
        PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/validation/'
        SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/' + MODEL_PATH.split('/')[-1][:-9]  
        SAVE_BORDERS_ERROR = 'reports/eval_per_hour/' + REGION + '/' + PREDICT_HORIZON

    #########################################################################################

    try:
        os.mkdir(SAVE_IMAGES_PATH)
    except:
        pass

    try:
        os.mkdir(SAVE_BORDERS_ERROR)
    except:
        pass

    if GEO_DATA:
        in_channel = 4  # 1 if only image, higher if more metadata in training
    else:
        in_channel = 1  # 1 if only image, higher if more metadata in training

    if GEO_DATA:
        val_dataset = PatchesFoldersDataset_w_geodata(
            path=PATH_DATA,
            csv_path=CSV_PATH,
            in_channel=input_seq_len,
            out_channel=PREDICT_T,
            min_time_diff=5,
            max_time_diff=15,
            output_last=TRAIN_W_LAST,
            img_size=img_size,
            patch_size=patch_size,
            geo_data_path='reports/',
            train=False
        )

    else:
        val_dataset = PatchesFoldersDataset(
                path=PATH_DATA,
                csv_path=CSV_PATH,
                in_channel=input_seq_len,
                out_channel=PREDICT_T,
                min_time_diff=5,
                max_time_diff=15,
                transform=normalize,
                output_last=TRAIN_W_LAST,
                output_30min=False,
                img_size=img_size,
                patch_size=patch_size,
                train=False
        )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = ConvLSTM_patch(
        input_seq_len=input_seq_len,
        seq_len=PREDICT_T//3,
        in_chan=in_channel,
        image_size=patch_size
    ).cuda()

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if torch.cuda.device_count() == 1:
        for _ in range(len(checkpoint['model_state_dict'])):
            key, value = checkpoint['model_state_dict'].popitem(False)
            checkpoint['model_state_dict'][key[7:] if key[:7] == 'module.' else key] = value

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print('Model Loaded')

    mae_list = []
    mae_pct_list = []
    rmse_list = []
    rmse_pct_list = []
    mbd_list = []
    mbd_pct_list = []
    fs_list = []
    ssim_list = []

    mean_image = np.zeros((img_size, img_size))

    MAE_error_image = np.zeros((img_size, img_size))
    MAE_pct_error_image = np.zeros((img_size, img_size))
    RMSE_pct_error_image = np.zeros((img_size, img_size))
    RMSE_error_image = np.zeros((img_size, img_size))

    with torch.no_grad():
        for val_batch_idx, (in_frames, out_frames) in enumerate(tqdm(val_loader)):

            if not GEO_DATA:
                in_frames = torch.unsqueeze(in_frames, dim=2)

            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)
            reconstructed_pred = torch.zeros((1, 1, img_size, img_size)).to(device=device)

            mean_image += out_frames[0,0].cpu().numpy()

            for i in range(dim):
                for j in range(dim):
                    n = i * patch_size
                    m = j * patch_size
                    frames_pred_Q = model(in_frames[:, :, :, n:n + patch_size, m:m + patch_size])
                    reconstructed_pred[0, 0, n:n + patch_size, m:m + patch_size] = torch.clamp(frames_pred_Q[0, 1, 0, :, :], min=0, max=1)

            MAE_loss = (MAE(reconstructed_pred[0, 0], out_frames[0, 0]).detach().item() * 100)
            MAE_pct_loss = (MAE_loss / (torch.mean(out_frames[0,0]).cpu().numpy() * 100)) * 100

            MAE_error_image += torch.abs(torch.subtract(out_frames[0,0], reconstructed_pred[0,0])).cpu().numpy()

            RMSE_loss = torch.sqrt(MSE(reconstructed_pred[0, 0], out_frames[0, 0])).detach().item() * 100
            RMSE_pct_loss = (RMSE_loss / (torch.mean(out_frames[0, 0]).cpu().numpy() * 100)) * 100

            RMSE_error_image += torch.square(torch.multiply(torch.subtract(out_frames[0,0], reconstructed_pred[0, 0]), 100)).cpu().numpy()

            SSIM_loss = SSIM(reconstructed_pred, out_frames).detach().item()

            MBD_loss = (torch.mean(torch.subtract(reconstructed_pred[0, 0], out_frames[0, 0])).detach().item() * 100)
            MBD_pct_loss = (MBD_loss / (torch.mean(out_frames[0,0]).cpu().numpy() * 100)) * 100

            persistence_rmse = torch.sqrt(MSE(in_frames[0, -1, 0], out_frames[0, 0])).detach().item() * 100
            forecast_skill = 1 - (RMSE_loss / persistence_rmse)

            mbd_list.append(MBD_loss)
            mbd_pct_list.append(MBD_pct_loss)
            fs_list.append(forecast_skill)
            mae_list.append(MAE_loss)
            mae_pct_list.append(MAE_pct_loss)
            rmse_list.append(RMSE_loss)
            rmse_pct_list.append(RMSE_pct_loss)
            ssim_list.append(SSIM_loss)

    print('MAE', np.mean(mae_list))
    print('MAE%', np.mean(mae_pct_list))
    print('RMSE', np.mean(rmse_list))
    print('RMSE%', np.mean(rmse_pct_list))
    print('SSIM', np.mean(ssim_list))
    print('MBD', np.mean(MBD_loss))
    print('MBD%', np.mean(MBD_pct_loss))
    print('FS', np.mean(forecast_skill))
    
    mean_image = (mean_image / len(val_dataset)) * 100  # contains the mean value of each pixel independently 
    MAE_error_image = (MAE_error_image / len(val_dataset))
    MAE_pct_error_image = (MAE_error_image / mean_image) * 100
    RMSE_pct_error_image = (np.sqrt((RMSE_error_image) / len(val_dataset)) / mean_image) * 100
    RMSE_error_image = (np.sqrt((RMSE_error_image) / len(val_dataset))) * 100

    np.save(os.path.join(SAVE_IMAGES_PATH, 'mean_image.npy'), mean_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'mean_image.pdf')
    visualization.show_image_w_colorbar(
        image=MAE_error_image,
        title=None,
        fig_name=fig_name,
        save_fig=True
    )
    plt.close()

    np.save(os.path.join(SAVE_IMAGES_PATH, 'MAE_error_image.npy'), MAE_error_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_error_image.pdf')
    visualization.show_image_w_colorbar(
        image=MAE_error_image,
        title=None,
        fig_name=fig_name,
        save_fig=True
    )
    plt.close()

    np.save(os.path.join(SAVE_IMAGES_PATH, 'MAE_pct_error_image.npy'), MAE_pct_error_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_pct_error_image.pdf')
    visualization.show_image_w_colorbar(
        image=MAE_pct_error_image,
        title=None,
        fig_name=fig_name,
        save_fig=True
    )
    plt.close()

    np.save(os.path.join(SAVE_IMAGES_PATH, 'RMSE_error_image.npy'), RMSE_error_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_error_image.pdf')
    visualization.show_image_w_colorbar(
        image=RMSE_error_image,
        title=None,
        fig_name=fig_name,
        save_fig=True
    )
    plt.close()

    np.save(os.path.join(SAVE_IMAGES_PATH, 'RMSE_pct_error_image.npy'), RMSE_pct_error_image)
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_pct_error_image.pdf')
    visualization.show_image_w_colorbar(
        image=RMSE_pct_error_image,
        title=None,
        fig_name=fig_name,
        save_fig=True
    )
    plt.close()

    mae_errors_borders = []
    mae_std_borders = []
    r_RMSE_errors_borders = []
    r_RMSE_std_borders = []

    for i in borders:
        p = int(i)
        mae_errors_borders.append(np.mean(MAE_error_image[p:-p, p:-p]))
        r_RMSE_errors_borders.append(np.mean(RMSE_pct_error_image[p:-p, p:-p]))
        
    if SAVE_BORDERS_ERROR:
        dict_values = {
            'model_name': MODEL_PATH.split('/')[-1],
            'test_dataset': evaluate_test,
            'csv_path': CSV_PATH,
            'predict_t': PREDICT_T,
            'geo_data': GEO_DATA,
            'MAE': np.mean(mae_list),
            'MAE%': np.mean(mae_pct_list),
            'RMSE': np.mean(rmse_list),
            'RMSE%': np.mean(rmse_pct_list),
            'SSIM': np.mean(ssim_list),
            'MBD': np.mean(MBD_loss),
            'MBD%': np.mean(MBD_pct_loss),
            'FS': np.mean(forecast_skill),
            'borders': borders,
            'mae_errors_borders': mae_errors_borders,
            'r_RMSE_errors_borders': r_RMSE_errors_borders
        }                                                                                                                      

        utils.save_pickle_dict(path=SAVE_BORDERS_ERROR, name=MODEL_PATH.split('/')[-1][:-14], dict_=dict_values)
    
    del model