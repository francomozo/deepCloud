import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from src import data, evaluate, model, preprocessing, visualization
from src.lib import utils
from src.data import MontevideoFoldersDataset, MontevideoFoldersDataset_w_time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from piqa import SSIM
from src.dl_models.unet import UNet, UNet2
from src.dl_models.unet_advanced import R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
import scipy.stats as st
from src.lib.latex_options import Colors, Linestyles
from src.lib.utils import get_model_name

### SETUP #############################################################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)
MSE = nn.MSELoss()
MAE = nn.L1Loss()
SSIM = SSIM(n_channels=1).cuda()
normalize = preprocessing.normalize_pixels(mean0 = False) #values between [0,1]
fontsize = 22 # 22 generates the font more like the latex text
borders = np.linspace(1, 450, 100)
#######################################################################################

REGION = 'R3' # [MVD, URU, R3]
if REGION == 'MVD':
    dataset = 'mvd'
elif REGION == 'URU':
    dataset = 'uru'
elif REGION == 'R3':
    dataset = 'region3'

PREDICT_T_LIST = [6, 12, 18, 24, 30]
OUTPUT_ACTIVATION = 'sigmoid'
FILTERS = 16
PREDICT_DIFF = False

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
    
    MODEL_NAME = get_model_name(predict_horizon=PREDICT_HORIZON, architecture='UNET2', predict_diff=PREDICT_DIFF)
    MODEL_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/' + REGION + '/' + PREDICT_HORIZON +  '/' + MODEL_NAME 
    
    model = UNet2(
        n_channels=3,
        n_classes=1,
        output_activation=OUTPUT_ACTIVATION,
        filters=FILTERS
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))["model_state_dict"])
    
    if evaluate_test:
        CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/region3/test_cosangs_region3.csv'
        PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/test/'
        SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/test/' + MODEL_PATH.split('/')[-1][:-9]  
        SAVE_PER_HOUR_ERROR = 'reports/eval_per_hour/' + REGION + '/' + PREDICT_HORIZON + '/test'
        SAVE_BORDERS_ERROR = 'reports/borders_cut/' + REGION + '/' + PREDICT_HORIZON + '/test'

    else:
        CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/region3/val_cosangs_region3.csv'
        PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/validation/'
        SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/' + MODEL_PATH.split('/')[-1][:-9]  
        SAVE_PER_HOUR_ERROR = 'reports/eval_per_hour/' + REGION + '/' + PREDICT_HORIZON
        SAVE_BORDERS_ERROR = 'reports/borders_cut/' + REGION + '/validation'
        
    
    try:
        os.mkdir(SAVE_IMAGES_PATH)
    except:
        pass

    try:
        os.mkdir(SAVE_PER_HOUR_ERROR)
    except:
        pass
    
    try:
        os.mkdir(SAVE_BORDERS_ERROR)
    except:
        pass
    
    val_dataset = MontevideoFoldersDataset_w_time(
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
    in_frames, out_frames, _, _ = next(iter(val_loader))
    M, N = out_frames[0,0].shape[0], out_frames[0,0].shape[1]

    MAE_per_hour = {}  # MAE
    MAE_pct_per_hour = {}
    RMSE_per_hour = {}  # RMSE
    RMSE_pct_per_hour = {}
    MBD_per_hour = {}  # MBD
    MBD_pct_per_hour = {}
    FS_per_hour = {}  # FS
    SSIM_per_hour = {}  # SSIM

    mean_image = np.zeros((M,N))
    MAE_error_image = np.zeros((M,N))
    MAE_pct_error_image = np.zeros((M,N))
    RMSE_pct_error_image = np.zeros((M,N))
    RMSE_error_image = np.zeros((M,N))

    model.eval()
    with torch.no_grad():
        for val_batch_idx, (in_frames, out_frames, in_time, out_time) in enumerate(tqdm(val_loader)):
            in_frames = in_frames.to(device=device)
            out_frames = out_frames.to(device=device)
            day, hour, minute  = int(out_time[0, 0, 0]), int(out_time[0, 0, 1]), int(out_time[0, 0, 2])

            mean_image += out_frames[0,0].cpu().numpy()

            if not PREDICT_DIFF:
                frames_pred = model(in_frames)

            if PREDICT_DIFF:
                diff_pred = model(in_frames)        
                frames_pred = torch.add(diff_pred[:,0], in_frames[:,2]).unsqueeze(1)  
                frames_pred = torch.clamp(frames_pred, min=0, max=1)  

            # MAE
            MAE_loss = (MAE(frames_pred, out_frames).detach().item() * 100)
            MAE_pct_loss = (MAE_loss / (torch.mean(out_frames[0,0]).cpu().numpy() * 100)) * 100

            MAE_error_image += torch.abs(torch.multiply(torch.subtract(out_frames[0,0], frames_pred[0,0]), 100)).cpu().numpy()

            if minute < 30:
                if (hour, 0) in MAE_per_hour.keys():
                    MAE_per_hour[(hour, 0)].append(MAE_loss)
                    MAE_pct_per_hour[(hour, 0)].append(MAE_pct_loss)
                else:
                    MAE_per_hour[(hour, 0)] = [MAE_loss]
                    MAE_pct_per_hour[(hour, 0)] = [MAE_pct_loss]
            else:
                if (hour, 30) in MAE_per_hour.keys():
                    MAE_per_hour[(hour, 30)].append(MAE_loss)
                    MAE_pct_per_hour[(hour, 30)].append(MAE_pct_loss)
                else:
                    MAE_per_hour[(hour, 30)] = [MAE_loss]
                    MAE_pct_per_hour[(hour, 30)] = [MAE_pct_loss]
            
            # RMSE
            RMSE_loss = torch.sqrt(MSE(frames_pred, out_frames)).detach().item() * 100
            RMSE_pct_loss = (RMSE_loss / (torch.mean(out_frames[0, 0]).cpu().numpy() * 100)) * 100
            
            RMSE_error_image += torch.square(torch.multiply(torch.subtract(out_frames[0,0], frames_pred[0,0]), 100)).cpu().numpy()
        
            if minute<30:
                if (hour, 0) in RMSE_per_hour.keys():
                    RMSE_per_hour[(hour, 0)].append(RMSE_loss)
                    RMSE_pct_per_hour[(hour, 0)].append(RMSE_pct_loss)
                else:
                    RMSE_per_hour[(hour, 0)] = [RMSE_loss]
                    RMSE_pct_per_hour[(hour, 0)] = [RMSE_pct_loss]
            else:
                if (hour, 30) in RMSE_per_hour.keys():
                    RMSE_per_hour[(hour, 30)].append(RMSE_loss)
                    RMSE_pct_per_hour[(hour, 30)].append(RMSE_pct_loss)
                else:
                    RMSE_per_hour[(hour, 30)] = [RMSE_loss]
                    RMSE_pct_per_hour[(hour, 30)] = [RMSE_pct_loss]

            # SSIM
            SSIM_loss = SSIM(frames_pred, out_frames)
                        
            if minute<30:
                if (hour,0) in SSIM_per_hour.keys():
                    SSIM_per_hour[(hour,0)].append(SSIM_loss.detach().item())
                else:
                    SSIM_per_hour[(hour,0)] = [SSIM_loss.detach().item()]
            else:
                if (hour,30) in SSIM_per_hour.keys():
                    SSIM_per_hour[(hour,30)].append(SSIM_loss.detach().item())
                else:
                    SSIM_per_hour[(hour,30)] = [SSIM_loss.detach().item()]

            # MBD and FS
            MBD_loss = (torch.mean(torch.subtract(frames_pred, out_frames)).detach().item() * 100)
            MBD_pct_loss = (MBD_loss / (torch.mean(out_frames[0,0]).cpu().numpy() * 100)) * 100
            
            persistence_rmse = torch.sqrt(MSE(in_frames[0, 2], out_frames[0, 0])).detach().item() * 100
            forecast_skill = 1 - (RMSE_loss / persistence_rmse)

            if minute < 30:
                if (hour, 0) in MBD_per_hour.keys():
                    MBD_per_hour[(hour, 0)].append(MBD_loss)
                    MBD_pct_per_hour[(hour, 0)].append(MBD_pct_loss)
                    FS_per_hour[(hour, 0)].append(forecast_skill)
                else:
                    MBD_per_hour[(hour, 0)] = [MBD_loss]
                    MBD_pct_per_hour[(hour, 0)] = [MBD_pct_loss]
                    FS_per_hour[(hour, 0)] = [forecast_skill]
            else:
                if (hour, 30) in MBD_per_hour.keys():
                    MBD_per_hour[(hour, 30)].append(MBD_loss)
                    MBD_pct_per_hour[(hour, 30)].append(MBD_pct_loss)
                    FS_per_hour[(hour, 30)].append(forecast_skill)
                else:
                    MBD_per_hour[(hour, 30)] = [MBD_loss]
                    MBD_pct_per_hour[(hour, 30)] = [MBD_pct_loss]
                    FS_per_hour[(hour, 30)] = [forecast_skill]

    # GENERATE ERROR IMAGES
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

    mean_MAE = []
    mean_MAE_pct = []
    mean_RMSE = []
    mean_RMSE_pct = []
    mean_SSIM = []
    std_MAE = []
    std_MAE_pct = []
    std_RMSE = []
    std_RMSE_pct = []
    std_SSIM = []
    mean_MBD = []
    mean_MBD_pct = []
    std_MBD = []
    std_MBD_pct = []
    mean_FS = []
    std_FS = []

    sorted_keys = sorted(MAE_per_hour.keys(), key=lambda element: (element[0], element[1]))
    hour_list = []

    for key in sorted_keys:
        hour_list.append(str(key[0]).zfill(2) + ':' + str(key[1]).zfill(2))
        mean_MAE.append(np.mean(MAE_per_hour[key]))
        std_MAE.append(np.std(MAE_per_hour[key]))
        mean_MAE_pct.append(np.mean(MAE_pct_per_hour[key]))
        std_MAE_pct.append(np.std(MAE_pct_per_hour[key]))
        mean_RMSE.append(np.mean(RMSE_per_hour[key]))
        std_RMSE.append(np.std(RMSE_per_hour[key]))
        mean_RMSE_pct.append(np.mean(RMSE_pct_per_hour[key]))
        std_RMSE_pct.append(np.std(RMSE_pct_per_hour[key]))
        mean_SSIM.append(np.mean(SSIM_per_hour[key]))
        std_SSIM.append(np.std(SSIM_per_hour[key]))
        mean_MBD.append(np.mean(MBD_per_hour[key]))
        std_MBD.append(np.std(MBD_per_hour[key]))
        mean_MBD_pct.append(np.mean(MBD_pct_per_hour[key]))
        std_MBD_pct.append(np.std(MBD_pct_per_hour[key]))
        mean_FS.append(np.mean(FS_per_hour[key]))
        std_FS.append(np.std(FS_per_hour[key]))
        
    if SAVE_PER_HOUR_ERROR:
        dict_values = {
            'model_name': MODEL_PATH.split('/')[-1],
            'csv_path': CSV_PATH,
            'PREDICT_T': PREDICT_T,
            'predict diff': PREDICT_DIFF,
            'hour_list': hour_list,
            'mean_MAE': mean_MAE,
            'std_MAE': std_MAE,
            'mean_MAE_pct': mean_MAE_pct,
            'std_MAE_pct': std_MAE_pct,
            'mean_SSIM': mean_SSIM,
            'std_SSIM': std_SSIM,
            'mean_RMSE': mean_RMSE,
            'std_RMSE': std_RMSE,
            'mean_RMSE_pct': mean_RMSE_pct,
            'std_RMSE_pct': std_RMSE_pct,
            'mean_MBD': mean_MBD,
            'std_MBD': std_MBD,
            'mean_MBD_pct': mean_MBD_pct,
            'std_MBD_pct': std_MBD_pct,
            'mean_FS': mean_FS,
            'std_FS': std_FS
        }                                                                                                                      

        utils.save_pickle_dict(path=SAVE_PER_HOUR_ERROR, name=MODEL_PATH.split('/')[-1][:-12], dict_=dict_values) 

        mae_errors_borders = []
        r_RMSE_errors_borders = []

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
                'borders': borders,
                'mae_errors_borders': mae_errors_borders,
                'r_RMSE_errors_borders': r_RMSE_errors_borders
            }                                                                                                                      

            utils.save_pickle_dict(path=SAVE_BORDERS_ERROR, name=MODEL_PATH.split('/')[-1][:-12], dict_=dict_values)
        
    print('Dict with error values saved.')
    del model

