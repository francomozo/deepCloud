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

## CONFIGURATION #########

REGION = 'R3' # [MVD, URU, R3]
if REGION == 'MVD':
    dataset = 'mvd'
elif REGION == 'URU':
    dataset = 'uru'
elif REGION == 'R3':
    dataset = 'region3'

PREDICT_HORIZON = '180min'
FRAME_OUT = 17  # 0->10min, 1->20min, 2->30min... [0,5] U [11] U [17] U [23] 
OUTPUT_ACTIVATION = 'sigmoid'
PREDICT_DIFF = False

evaluate_test = True

MODEL_NAME = get_model_name(predict_horizon=PREDICT_HORIZON, predict_diff=PREDICT_DIFF)
MODEL_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/' + REGION + '/' + PREDICT_HORIZON +  '/' + MODEL_NAME 

if evaluate_test:
    CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/region3/test_cosangs_region3.csv'
    PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/test/'
    SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/test/' + MODEL_PATH.split('/')[-1][:-9]  
    SAVE_VALUES_PATH = 'reports/eval_per_hour/' + REGION + '/test'

else:
    CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/region3/val_cosangs_region3.csv'
    PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/validation/'
    SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/' + MODEL_PATH.split('/')[-1][:-9]  
    SAVE_VALUES_PATH = 'reports/eval_per_hour/' + REGION + '/' + PREDICT_HORIZON 

CROP_SIZE = 50

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)

#model = UNet(n_channels=3, n_classes=1, bilinear=True, p=0, output_activation=OUTPUT_ACTIVATION, bias=False, filters=16).to(device)
model = UNet2(n_channels=3, n_classes=1, bilinear=True, p=0, output_activation=OUTPUT_ACTIVATION, bias=False, filters=16).to(device)

###########################

# LATEX CONFIG
fontsize = 22 # 22 generates the font more like the latex text

save_fig = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

try:
    os.mkdir(SAVE_IMAGES_PATH)
except:
    pass

try:
    os.mkdir(SAVE_VALUES_PATH)
except:
    pass

#Evaluate Unet

normalize = preprocessing.normalize_pixels(mean0=False) #values between [0,1]

val_dataset = MontevideoFoldersDataset_w_time(
    path=PATH_DATA,
    in_channel=3,
    out_channel=FRAME_OUT+1,
    min_time_diff=5,
    max_time_diff=15,
    csv_path=CSV_PATH,
    transform=normalize,
    output_last=True
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
in_frames, out_frames, _, _ = next(iter(val_loader))
M, N = out_frames[0,0].shape[0], out_frames[0,0].shape[1] 

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))["model_state_dict"])

gt_mean = []
gt_std = []
pred_mean = []
pred_std = []

# MAE
MAE = nn.L1Loss()
worst_MAE_error = 0
worst_MAE_time = ''
best_MAE_error = 1
best_MAE_time = ''
best_MAE_images = np.zeros((5, M, N))
worst_MAE_images = np.zeros((5, M, N))
MAE_per_hour = {}
MAE_per_hour_crop = {}
# MAE PORCENTUAL
MAE_pct_per_hour = {}

# RMSE
MSE = nn.MSELoss()
worst_RMSE_error = 0
worst_RMSE_time = ''
best_RMSE_error = 1
best_RMSE_time = ''    
best_RMSE_images = np.zeros((5, M, N))
worst_RMSE_images = np.zeros((5, M, N))
RMSE_per_hour = {}
# RMSE PORCENTUAL
RMSE_pct_per_hour = {}


# MBD
MBD_per_hour = {}
MBD_pct_per_hour = {}

# FS
FS_per_hour = {}

try:
    SSIM = SSIM(n_channels=1).to(device)
except:
    pass

worst_SSIM_error = 1
worst_SSIM_time = ''
best_SSIM_error = 0
best_SSIM_time = ''
best_SSIM_images = np.zeros((5, M, N))
worst_SSIM_images = np.zeros((5, M, N))
SSIM_per_hour = {}
SSIM_per_hour_crop = {}

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
        
        MAE_loss_crop = MAE(frames_pred[:, :, CROP_SIZE:M-CROP_SIZE, CROP_SIZE:N-CROP_SIZE], out_frames[:, :, CROP_SIZE:M-CROP_SIZE, CROP_SIZE:N-CROP_SIZE])
        if minute < 30:
            if (hour, 0) in MAE_per_hour.keys():
                MAE_per_hour[(hour, 0)].append(MAE_loss)
                MAE_per_hour_crop[(hour, 0)].append(MAE_loss_crop)
                MAE_pct_per_hour[(hour, 0)].append(MAE_pct_loss)
            else:
                MAE_per_hour[(hour, 0)] = [MAE_loss]
                MAE_pct_per_hour[(hour, 0)] = [MAE_pct_loss]
                MAE_per_hour_crop[(hour, 0)] = [MAE_loss_crop]
        else:
            if (hour, 30) in MAE_per_hour.keys():
                MAE_per_hour[(hour, 30)].append(MAE_loss)
                MAE_pct_per_hour[(hour, 30)].append(MAE_pct_loss)
                MAE_per_hour_crop[(hour, 30)].append(MAE_loss_crop)
            else:
                MAE_per_hour[(hour, 30)] = [MAE_loss]
                MAE_pct_per_hour[(hour, 30)] = [MAE_pct_loss]
                MAE_per_hour_crop[(hour, 30)] = [MAE_loss_crop]
        if MAE_loss > worst_MAE_error:
            worst_MAE_error = MAE_loss
            
            worst_MAE_time = out_time[0, 0]
            worst_MAE_input_time = in_time[0].numpy()
            
            worst_MAE_time_list = [
                str(int((worst_MAE_input_time[0][1]))).zfill(2) + ':' + str(int((worst_MAE_input_time[0][2]))).zfill(2),
                str(int((worst_MAE_input_time[1][1]))).zfill(2) + ':' + str(int((worst_MAE_input_time[1][2]))).zfill(2),
                str(int((worst_MAE_input_time[2][1]))).zfill(2) + ':' + str(int((worst_MAE_input_time[2][2]))).zfill(2),
                str(int(worst_MAE_time[1].numpy())).zfill(2) + ':' + str(int(worst_MAE_time[2].numpy())).zfill(2)
            ]
            
            worst_MAE_images[0:3] = in_frames[0].cpu().numpy()
            worst_MAE_images[3] = out_frames[0, 0].cpu().numpy()
            worst_MAE_images[4] = frames_pred[0, 0].cpu().numpy()
            
        if MAE_loss < best_MAE_error:
            best_MAE_error = MAE_loss
            best_MAE_input_time = in_time[0].numpy()
            best_MAE_time = out_time[0, 0]
            
            best_MAE_time_list = [
                str(int((best_MAE_input_time[0][1]))).zfill(2) + ':' + str(int((best_MAE_input_time[0][2]))).zfill(2),
                str(int((best_MAE_input_time[1][1]))).zfill(2) + ':' + str(int((best_MAE_input_time[1][2]))).zfill(2),
                str(int((best_MAE_input_time[2][1]))).zfill(2) + ':' + str(int((best_MAE_input_time[2][2]))).zfill(2),
                str(int(best_MAE_time[1].numpy())).zfill(2) + ':' + str(int(best_MAE_time[2].numpy())).zfill(2)
            ]
            
            best_MAE_images[0:3] = in_frames[0].cpu().numpy()
            best_MAE_images[3] = out_frames[0, 0].cpu().numpy()
            best_MAE_images[4] = frames_pred[0, 0].cpu().numpy()
        
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
                
        if RMSE_loss > worst_RMSE_error:
            worst_RMSE_error = RMSE_loss
            worst_RMSE_input_time = in_time[0].numpy()
            worst_RMSE_time = out_time[0, 0]
            
            worst_RMSE_time_list = [
                str(int((worst_RMSE_input_time[0][1]))).zfill(2) + ':' + str(int((worst_RMSE_input_time[0][2]))).zfill(2),
                str(int((worst_RMSE_input_time[1][1]))).zfill(2) + ':' + str(int((worst_RMSE_input_time[1][2]))).zfill(2),
                str(int((worst_RMSE_input_time[2][1]))).zfill(2) + ':' + str(int((worst_RMSE_input_time[2][2]))).zfill(2),
                str(int(worst_RMSE_time[1].numpy())).zfill(2) + ':' + str(int(worst_RMSE_time[2].numpy())).zfill(2)
            ] 
            
            worst_RMSE_images[0:3] = in_frames[0].cpu().numpy()
            worst_RMSE_images[3] = out_frames[0, 0].cpu().numpy()
            worst_RMSE_images[4] = frames_pred[0, 0].cpu().numpy()
        if RMSE_loss < best_RMSE_error:
            best_RMSE_error = RMSE_loss
            best_RMSE_input_time = in_time[0].numpy()
            best_RMSE_time = out_time[0, 0]
            
            best_RMSE_time_list = [
                str(int((best_RMSE_input_time[0][1]))).zfill(2) + ':' + str(int((best_RMSE_input_time[0][2]))).zfill(2),
                str(int((best_RMSE_input_time[1][1]))).zfill(2) + ':' + str(int((best_RMSE_input_time[1][2]))).zfill(2),
                str(int((best_RMSE_input_time[2][1]))).zfill(2) + ':' + str(int((best_RMSE_input_time[2][2]))).zfill(2),
                str(int(best_RMSE_time[1].numpy())).zfill(2) + ':' + str(int(best_RMSE_time[2].numpy())).zfill(2)
            ]
            
            best_RMSE_images[0:3] = in_frames[0].cpu().numpy()
            best_RMSE_images[3] = out_frames[0,0].cpu().numpy()
            best_RMSE_images[4] = frames_pred[0, 0].cpu().numpy()

        # SSIM
        SSIM_loss = SSIM(frames_pred, out_frames)
        SSIM_loss_crop = SSIM(frames_pred[:, :, CROP_SIZE:M-CROP_SIZE, CROP_SIZE:N-CROP_SIZE], out_frames[:, :, CROP_SIZE:M-CROP_SIZE, CROP_SIZE:N-CROP_SIZE])
                    
        if minute<30:
            if (hour,0) in SSIM_per_hour.keys():
                SSIM_per_hour[(hour,0)].append(SSIM_loss.detach().item())
                SSIM_per_hour_crop[(hour,0)].append(SSIM_loss_crop.detach().item())
            else:
                SSIM_per_hour[(hour,0)] = []
                SSIM_per_hour[(hour,0)].append(SSIM_loss.detach().item())
                SSIM_per_hour_crop[(hour,0)] = []
                SSIM_per_hour_crop[(hour,0)].append(SSIM_loss_crop.detach().item())
        else:
            if (hour,30) in SSIM_per_hour.keys():
                SSIM_per_hour[(hour,30)].append(SSIM_loss.detach().item())
                SSIM_per_hour_crop[(hour,30)].append(SSIM_loss_crop.detach().item())
            else:
                SSIM_per_hour[(hour,30)] = []
                SSIM_per_hour[(hour,30)].append(SSIM_loss.detach().item())
                SSIM_per_hour_crop[(hour,30)] = []
                SSIM_per_hour_crop[(hour,30)].append(SSIM_loss_crop.detach().item())
        
        if SSIM_loss.detach().item() < worst_SSIM_error:
            worst_SSIM_error = SSIM_loss.detach().item()
            worst_SSIM_time = out_time[0, 0]
            worst_SSIM_input_time = in_time[0].numpy()

            worst_SSIM_time_list = [
                str(int((worst_SSIM_input_time[0][1]))).zfill(2) + ':' + str(int((worst_SSIM_input_time[0][2]))).zfill(2),
                str(int((worst_SSIM_input_time[1][1]))).zfill(2) + ':' + str(int((worst_SSIM_input_time[1][2]))).zfill(2),
                str(int((worst_SSIM_input_time[2][1]))).zfill(2) + ':' + str(int((worst_SSIM_input_time[2][2]))).zfill(2),
                str(int(worst_SSIM_time[1].numpy())).zfill(2) + ':' + str(int(worst_SSIM_time[2].numpy())).zfill(2)
            ] 
            
            worst_SSIM_images[0:3] = in_frames[0].cpu().numpy()
            worst_SSIM_images[3] = out_frames[0,0].cpu().numpy()
            worst_SSIM_images[4] = frames_pred[0,0].cpu().numpy()
        if SSIM_loss.detach().item() > best_SSIM_error:
            best_SSIM_error = SSIM_loss.detach().item()
            best_SSIM_input_time = in_time[0].numpy() 
            best_SSIM_time = out_time[0, 0]
            
            best_SSIM_time_list = [
                str(int((best_SSIM_input_time[0][1]))).zfill(2) + ':' + str(int((best_SSIM_input_time[0][2]))).zfill(2),
                str(int((best_SSIM_input_time[1][1]))).zfill(2) + ':' + str(int((best_SSIM_input_time[1][2]))).zfill(2),
                str(int((best_SSIM_input_time[2][1]))).zfill(2) + ':' + str(int((best_SSIM_input_time[2][2]))).zfill(2),
                str(int(best_SSIM_time[1].numpy())).zfill(2) + ':' + str(int(best_SSIM_time[2].numpy())).zfill(2)
            ]
            
            best_SSIM_images[0:3] = in_frames[0].cpu().numpy()
            best_SSIM_images[3] = out_frames[0,0].cpu().numpy()
            best_SSIM_images[4] = frames_pred[0,0].cpu().numpy()

        # MBD and FS
        
        MBD_loss = (torch.subtract(frames_pred, out_frames).detach().item() * 100)
        MBD_pct_loss = (MBD_loss / (torch.mean(out_frames[0,0]).cpu().numpy() * 100)) * 100
        
        persistence_rmse = torch.sqrt(MSE(in_frames[0, 2], out_frames[0, 0])).detach().item() * 100
        forecast_skill = RMSE_loss / persistence_rmse

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
        
        gt_mean.append(torch.mean(out_frames[0,0]).cpu().numpy())
        gt_std.append(torch.std(out_frames[0,0]).cpu().numpy())
        pred_mean.append(torch.mean(frames_pred[0,0]).cpu().numpy())
        pred_std.append(torch.std(frames_pred[0,0]).cpu().numpy())


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

np.save(os.path.join(SAVE_IMAGES_PATH, 'MAE_error_image.npy'), MAE_error_image)
fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_error_image.pdf')
visualization.show_image_w_colorbar(
    image=MAE_error_image,
    title=None,
    fig_name=fig_name,
    save_fig=True
)

np.save(os.path.join(SAVE_IMAGES_PATH, 'MAE_pct_error_image.npy'), MAE_pct_error_image)
fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_pct_error_image.pdf')
visualization.show_image_w_colorbar(
    image=MAE_pct_error_image,
    title=None,
    fig_name=fig_name,
    save_fig=True
)

np.save(os.path.join(SAVE_IMAGES_PATH, 'RMSE_error_image.npy'), RMSE_error_image)
fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_error_image.pdf')
visualization.show_image_w_colorbar(
    image=RMSE_error_image,
    title=None,
    fig_name=fig_name,
    save_fig=True
)

np.save(os.path.join(SAVE_IMAGES_PATH, 'RMSE_pct_error_image.npy'), RMSE_pct_error_image)
fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_pct_error_image.pdf')
visualization.show_image_w_colorbar(
    image=RMSE_pct_error_image,
    title=None,
    fig_name=fig_name,
    save_fig=True
)

mean_MAE = []
mean_MAE_pct = []
mean_MAE_crop = []
mean_RMSE = []
mean_RMSE_pct = []
mean_SSIM = []
mean_SSIM_crop = []
std_MAE = []
std_MAE_pct = []
std_MAE_crop = []
std_RMSE = []
std_RMSE_pct = []
std_SSIM = []
std_SSIM_crop = []
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
    mean_MAE_crop.append(np.mean(MAE_per_hour_crop[key]))
    std_MAE_crop.append(np.std(MAE_per_hour_crop[key]))
    mean_RMSE.append(np.mean(RMSE_per_hour[key]))
    std_RMSE.append(np.std(RMSE_per_hour[key]))
    mean_RMSE_pct.append(np.mean(RMSE_pct_per_hour[key]))
    std_RMSE_pct.append(np.std(RMSE_pct_per_hour[key]))
    mean_SSIM.append(np.mean(SSIM_per_hour[key]))
    std_SSIM.append(np.std(SSIM_per_hour[key]))
    mean_SSIM_crop.append(np.mean(SSIM_per_hour_crop[key]))
    std_SSIM_crop.append(np.std(SSIM_per_hour_crop[key]))
    mean_MBD.append(np.mean(MBD_per_hour[key]))
    std_MBD.append(np.std(MBD_per_hour[key]))
    mean_MBD_pct.append(np.mean(MBD_pct_per_hour[key]))
    std_MBD_pct.append(np.std(MBD_pct_per_hour[key]))
    mean_FS.append(np.mean(FS_per_hour[key]))
    std_FS.append(np.std(FS_per_hour[key]))
    
if SAVE_VALUES_PATH:
    dict_values = {
        'model_name': MODEL_PATH.split('/')[-1],
        'csv_path': CSV_PATH,
        'frame_out': FRAME_OUT,
        'predict diff': PREDICT_DIFF,
        'crop_size': CROP_SIZE,
        'hour_list': hour_list,
        'mean_MAE': mean_MAE,
        'std_MAE': std_MAE,
        'mean_MAE_pct': mean_MAE_pct,
        'std_MAE_pct': std_MAE_pct,
        'mean_MAE_crop': mean_MAE_crop,
        'std_MAE_crop': std_MAE_crop,
        'mean_SSIM': mean_SSIM,
        'std_SSIM': std_SSIM,
        'mean_SSIM_crop': mean_SSIM_crop,
        'std_SSIM_crop': std_SSIM_crop,
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

    utils.save_pickle_dict(path=SAVE_VALUES_PATH, name=MODEL_PATH.split('/')[-1][:-12], dict_=dict_values) 

print('Dict with error values saved.')

#SCATTER PLOT
print('Scatter Plot')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
m, b = np.polyfit(gt_mean, pred_mean, 1)
fig = plt.figure()
fig.set_size_inches(6, 6)
ax = fig.add_subplot(1, 1, 1)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.0)
ax.tick_params(direction='out', length=6, width=1, colors='k')
ax.scatter(x=gt_mean, y=pred_mean, color=Colors.peterRiver)
ax.plot(gt_mean, m*np.array(gt_mean) + b, color=Colors.pomegranate)
textstr = '\n'.join((
    r'$m=%.2f$' % (m, ),
    r'$n=%.2f$' % (b, )))

        va='top', ha='left')

if SAVE_IMAGES_PATH:
    fig.tight_layout() 
    fig.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'scatterplot_mean.pdf'))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
m, b = np.polyfit(gt_std, pred_std, 1)
fig = plt.figure()
fig.set_size_inches(6, 6)
ax = fig.add_subplot(1, 1, 1)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.0)
ax.tick_params(direction='out', length=6, width=1, colors='k')
ax.scatter(x=gt_std, y=pred_std, color=Colors.peterRiver)
ax.plot(gt_std, m*np.array(gt_std) + b, color=Colors.pomegranate)
textstr = '\n'.join((
    r'$m=%.2f$' % (m, ),
    r'$n=%.2f$' % (b, )))

        va='top', ha='left')

if SAVE_IMAGES_PATH:
    fig.tight_layout() 
    fig.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'scatterplot_std.pdf'))
plt.close()

#MEANS DENSITY DISTRIBUTION
xmin, xmax = np.min(gt_mean)-0.01, np.max(gt_mean)+0.01
ymin, ymax = np.min(pred_mean)-0.01, np.max(pred_mean)+0.01

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([gt_mean, pred_mean])
kernel = st.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure()
fig.set_size_inches(6, 6)
ax = fig.add_subplot(1, 1, 1)
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(gt_mean, pred_mean, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_xlabel('GT mean')
ax.set_ylabel('Pred mean')
if SAVE_IMAGES_PATH:
    fig.tight_layout() 
    fig.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'mean_distribution.pdf'))
plt.show()
plt.close()

#STD DENSITY DISTRIBUTION
xmin, xmax = np.min(gt_std)-0.01, np.max(gt_std)+0.01
ymin, ymax = np.min(pred_std)-0.01, np.max(pred_std)+0.01

# Peform the kernel density estimate
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([gt_std, pred_std])
kernel = st.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure()
fig.set_size_inches(6, 6)
ax = fig.add_subplot(1, 1, 1)
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(gt_std, pred_std, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_xlabel('GT std')
ax.set_ylabel('Pred std')
if SAVE_IMAGES_PATH:
    fig.tight_layout() 
    fig.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'std_distribution.pdf'))
plt.show()
plt.close()

# IMG MEANS HISTOGRAM
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure()
fig.set_size_inches(12, 6)
ax = fig.add_subplot(1, 1, 1)


ax.grid(alpha=.2)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.0)
ax.tick_params(direction='out', length=6, width=1, colors='k')

ax.hist(gt_mean, bins=100, alpha=0.5, label="Ground Truth")
ax.hist(pred_mean, bins=100, alpha=0.5, label="Prediction")
l1 = plt.axvline(np.mean(gt_mean), color=Colors.peterRiver)
l2 = plt.axvline(np.mean(pred_mean), color=Colors.pomegranate)
plt.legend(loc='upper right')
if SAVE_IMAGES_PATH:
    fig.tight_layout() 
    fig.savefig(os.path.join(SAVE_IMAGES_PATH, 'means_histogram.pdf'))

plt.close()

# IMG STD DEV HISTOGRAM
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure()
fig.set_size_inches(12, 6)
ax = fig.add_subplot(1, 1, 1)


ax.grid(alpha=.2)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.0)
ax.tick_params(direction='out', length=6, width=1, colors='k')

ax.hist(gt_std, bins=100, alpha=0.5, label="Ground Truth")
ax.hist(pred_std, bins=100, alpha=0.5, label="Prediction")
l1 = plt.axvline(np.mean(gt_std), color=Colors.peterRiver)
l2 = plt.axvline(np.mean(pred_std), color=Colors.pomegranate)
plt.legend(loc='upper right')

if SAVE_IMAGES_PATH:
    fig.tight_layout() 
    fig.savefig(os.path.join(SAVE_IMAGES_PATH, 'stds_histogram.pdf'))
plt.close()

#BEST AND WORST PREDICTIONS

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'MAE best prediction, day:' + str(int(best_MAE_time[0].numpy())) + \
                        'hour:' + str(int(best_MAE_time[1].numpy())).zfill(2)+str(int(best_MAE_time[2].numpy())).zfill(2) + '.pdf')
visualization.show_seq_and_pred(best_MAE_images,
                                time_list=best_MAE_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'MAE worst prediction, day:'+str(int(worst_MAE_time[0].numpy())) + \
                        'hour:' + str(int(worst_MAE_time[1].numpy())).zfill(2) + str(int(worst_MAE_time[2].numpy())).zfill(2) + '.pdf')
visualization.show_seq_and_pred(worst_MAE_images,
                                time_list=worst_MAE_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'MSE best prediction, day:' + str(int(best_RMSE_time[0].numpy())) + \
                        'hour:'+ str(int(best_RMSE_time[1].numpy())).zfill(2) + str(int(best_RMSE_time[2].numpy())).zfill(2) + '.pdf')
visualization.show_seq_and_pred(best_RMSE_images,
                                time_list=best_RMSE_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'MSE worst prediction, day:' + str(int(worst_RMSE_time[0].numpy())) + \
                        'hour:'+ str(int(worst_RMSE_time[1].numpy())).zfill(2) + str(int(worst_RMSE_time[2].numpy())).zfill(2) + '.pdf')
visualization.show_seq_and_pred(worst_RMSE_images,
                                time_list=worst_RMSE_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'SSIM best prediction, day:'+str(int(best_SSIM_time[0].numpy())) + \
                        'hour:' + str(int(best_SSIM_time[1].numpy())).zfill(2) + str(int(best_SSIM_time[2].numpy())).zfill(2) + '.pdf')
visualization.show_seq_and_pred(best_SSIM_images,
                                time_list=best_SSIM_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'SSIM worst prediction, day:' + str(int(worst_SSIM_time[0].numpy())) + \
                        'hour:' + str(int(worst_SSIM_time[1].numpy())).zfill(2) + str(int(worst_SSIM_time[2].numpy())).zfill(2) + '.pdf')
visualization.show_seq_and_pred(worst_SSIM_images,
                                time_list=worst_SSIM_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

if not evaluate_test:
    # OUTPUT WITH MOST NANS SEQUENCE
    print('OUTPUT WITH MOST NANS SEQUENCE')
    img0 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_115017.npy'))
    img1 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_120017.npy'))
    img2 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_121017.npy'))
    time_list = ['11:50', '12:00', '12:10']
    if FRAME_OUT == 0:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_122017.npy'))
        time_list.append('12:20')
    elif FRAME_OUT == 1:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_123017.npy'))
        time_list.append('12:30')
    elif FRAME_OUT == 2:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_124017.npy'))
        time_list.append('12:40')
    elif FRAME_OUT == 3:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_125017.npy'))
        time_list.append('12:50')   
    elif FRAME_OUT == 4:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_130017.npy'))
        time_list.append('13:00')
    elif FRAME_OUT == 5:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_131017.npy'))
        time_list.append('13:10')
    elif FRAME_OUT == 11:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_141017.npy'))
        time_list.append('14:10')
    elif FRAME_OUT == 17:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_151017.npy'))
        time_list.append('15:10')
    elif FRAME_OUT == 23:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_161017.npy'))
        time_list.append('16:10')
    elif FRAME_OUT == 29:
        output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_171017.npy'))
        time_list.append('17:10')
    else:
        raise ValueError('Prediction time must be 10,20,30,40,50,60,120,180,240 or 300 minutes.')
            
    in_frames= torch.tensor(np.ones((1, 3, M, N))).to(device)
    out_frames= torch.tensor(np.ones((1, 1, M, N))).to(device)
    in_frames[0,0] = torch.from_numpy(img0/100).float().to(device)
    in_frames[0,1] = torch.from_numpy(img1/100).float().to(device)
    in_frames[0,2] = torch.from_numpy(img2/100).float().to(device)
    out_frames[0,0] = torch.from_numpy(output/100).float().to(device)

    model.eval()
    with torch.no_grad():
        if not PREDICT_DIFF:
            frames_pred = model(in_frames.type(torch.cuda.FloatTensor))

        if PREDICT_DIFF:
            diff_pred = model(in_frames.type(torch.cuda.FloatTensor))
            img_diff_pred = diff_pred[0, 0, :, :].cpu().numpy()
            frames_pred = torch.add(diff_pred[:,0], in_frames[:,2]).unsqueeze(1) 
            
    frames_array = np.ones((5, M, N))
    frames_array[0:3] = in_frames[0].cpu().numpy()
    frames_array[3]= out_frames[0,0].cpu().numpy()
    frames_array[4] = frames_pred[0,0].cpu().numpy()
        
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'most_nan_sequence.pdf')
    visualization.show_seq_and_pred(frames_array,
                                    time_list=time_list,
                                    prediction_t=FRAME_OUT+1,
                                    fig_name=fig_name, save_fig=True)

    if PREDICT_DIFF:
        fig_name = os.path.join(SAVE_IMAGES_PATH, 'most_nan_sequence_diff_pred.pdf')
        visualization.show_image_w_colorbar(img_diff_pred, fig_name=fig_name, save_fig=True)
        
    # LARGEST MOVEMENT left to right --->

    img0 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_135018.npy'))
    img1 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_140018.npy'))
    img2 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_141018.npy'))
    time_list = ['13:50', '14:00', '14:10']

    if FRAME_OUT == 0:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_142018.npy'))
        time_list.append('14:20')
    elif FRAME_OUT == 1:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_143018.npy'))
        time_list.append('14:30')
    elif FRAME_OUT == 2:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_144018.npy'))      
        time_list.append('14:40')
    elif FRAME_OUT == 3:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_145018.npy'))
        time_list.append('14:50')
    elif FRAME_OUT == 4:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_150018.npy'))
        time_list.append('15:00')
    elif FRAME_OUT == 5:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_151018.npy'))
        time_list.append('15:10')
    elif FRAME_OUT ==11:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_161018.npy'))
        time_list.append('16:10')
    elif FRAME_OUT == 17:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_171017.npy'))
        time_list.append('17:10')
    elif FRAME_OUT == 23:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_181017.npy'))
        time_list.append('18:10')
    elif FRAME_OUT == 29:
        output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_191017.npy'))
        time_list.append('19:10')
    else:
        raise ValueError('Prediction time must be 10,20,30,40,50,60,120,180,240 or 300 minutes.')
            
    in_frames= torch.tensor(np.ones((1, 3, M, N))).to(device)
    out_frames= torch.tensor(np.ones((1, 1, M, N))).to(device)
    in_frames[0,0] = torch.from_numpy(img0/100).float().to(device)
    in_frames[0,1] = torch.from_numpy(img1/100).float().to(device)
    in_frames[0,2] = torch.from_numpy(img2/100).float().to(device)
    out_frames[0,0] = torch.from_numpy(output/100).float().to(device)

    model.eval()
    with torch.no_grad():
        if not PREDICT_DIFF:
            frames_pred = model(in_frames.type(torch.cuda.FloatTensor))
            
        if PREDICT_DIFF:
            diff_pred = model(in_frames.type(torch.cuda.FloatTensor))
            img_diff_pred = diff_pred[0, 0, :, :].cpu().numpy()
            frames_pred = torch.add(diff_pred[:,0], in_frames[:,2]).unsqueeze(1)
        
    frames_array = np.ones((5, M, N))
    frames_array[0:3] = in_frames[0].cpu().numpy()
    frames_array[3]= out_frames[0,0].cpu().numpy()
    frames_array[4] = frames_pred[0,0].cpu().numpy()
        
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'most_moved_sequence.pdf')
    visualization.show_seq_and_pred(frames_array,
                                    time_list=time_list,
                                    prediction_t=FRAME_OUT+1,
                                    fig_name=fig_name, save_fig=True)

    if PREDICT_DIFF:
        fig_name = os.path.join(SAVE_IMAGES_PATH, 'most_moved_sequence_diff_pred.pdf')
        visualization.show_image_w_colorbar(img_diff_pred, fig_name=fig_name, save_fig=True)
