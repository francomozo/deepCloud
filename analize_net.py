import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from src import data, evaluate, model, preprocessing, visualization
from src.lib import utils
from src.data import MontevideoFoldersDataset, MontevideoFoldersDataset_w_time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from piqa import SSIM , MS_SSIM
from src.dl_models.unet import UNet, UNet2
from src.dl_models.unet_advanced import R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
import scipy.stats as st
from src.lib.latex_options import Colors, Linestyles


## CONFIGURATION #########

REGION = 'R3' # [MVD, URU, R3]
PREDICT_HORIZON = '60min'
FRAME_OUT = 5  # 0->10min, 1->20min, 2->30min... [0,5] U [11] U [17] U [23] 
CSV_PATH = None
# CSV_PATH = 'data/mvd/val_seq_in3_out1_cosangs.csv'
MODEL_PATH = 'checkpoints/'+REGION+'/'+PREDICT_HORIZON+'/60min_UNET__region3_mae_filters16_sigmoid_diffFalse_retrainTrue_80_01-02-2022_15:18.pt'
OUTPUT_ACTIVATION = 'sigmoid'
CROP_SIZE = 50


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)

model = UNet(n_channels=3, n_classes=1, bilinear=True, p=0, output_activation='sigmoid', bias=False, filters=16).to(device)
#model = UNet2(n_channels=3, n_classes=1, bilinear=True, p=0, output_activation='sigmoid', bias=False, filters=32).to(device)


SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/' + MODEL_PATH.split('/')[-1][:-9]  
SAVE_VALUES_PATH = 'reports/eval_per_hour/' + REGION + '/' + PREDICT_HORIZON 


###########################

# LATEX CONFIG
fontSize = 22 # 22 generates the font more like the latex text
save_fig = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


if REGION == 'MVD':
    dataset = 'mvd'
elif REGION == 'URU':
    dataset = 'uru'
elif REGION == 'R3':
    dataset = 'region3'
PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/validation/'

if OUTPUT_ACTIVATION == 'tanh':
    PREDICT_DIFF = True
else:
    PREDICT_DIFF = False

try:
    os.mkdir(SAVE_IMAGES_PATH)
except:
    pass

try:
    os.mkdir(SAVE_VALUES_PATH)
except:
    pass

#Evaluate Unet

normalize = preprocessing.normalize_pixels(mean0 = False) #values between [0,1]

val_mvd = MontevideoFoldersDataset_w_time(
                                            path=PATH_DATA,
                                            in_channel=3,
                                            out_channel=FRAME_OUT+1,
                                            min_time_diff=5,
                                            max_time_diff=15,
                                            csv_path=CSV_PATH,
                                            transform=normalize,
                                            output_last=True
                                            )

val_loader = DataLoader(val_mvd, batch_size=1, shuffle=False)
in_frames, out_frames, out_time = next(iter(val_loader))
M, N = out_frames[0,0].shape[0], out_frames[0,0].shape[1] 

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))["model_state_dict"])

gt_mean = []
gt_std = []
pred_mean = []
pred_std = []

worst_MAE_error = 0
worst_MAE_time = ''
best_MAE_error = 1
best_MAE_time = ''
best_MAE_images = np.zeros((5, M, N))
worst_MAE_images = np.zeros((5, M, N))
MAE = nn.L1Loss()
MAE_per_hour = {}
MAE_per_hour_crop = {}

worst_MSE_error = 0
worst_MSE_time = ''
best_MSE_error = 1
best_MSE_time = ''    
MSE = nn.MSELoss()
best_MSE_images = np.zeros((5, M, N))
worst_MSE_images = np.zeros((5, M, N))
MSE_per_hour = {}

worst_PSNR_error = 100
worst_PSNR_time = ''
best_PSNR_error = 0
best_PSNR_time = ''
best_PSNR_images = np.zeros((5, M, N))
worst_PSNR_images = np.zeros((5, M, N))
PSNR_per_hour = {}

try:
    SSIM = SSIM(n_channels=1).cuda()
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

MAE_error_image = np.zeros((M,N))

model.eval()
with torch.no_grad():
    for val_batch_idx, (in_frames, out_frames, in_time, out_time) in enumerate(val_loader):
        
        in_frames = in_frames.to(device=device)
        out_frames = out_frames.to(device=device)
               
        day, hour, minute  = int(out_time[0, 0, 0]), int(out_time[0, 0, 1]), int(out_time[0, 0, 2]) 

        if not PREDICT_DIFF:
            frames_pred = model(in_frames)
        
        if PREDICT_DIFF:
            diff_pred = model(in_frames)        
            frames_pred = torch.add(diff_pred[:,0], in_frames[:,2]).unsqueeze(1)  
            frames_pred = torch.clamp(frames_pred, min=0, max=1)     

        # MAE
        MAE_loss = MAE(frames_pred, out_frames)
        MAE_error_image += torch.abs(torch.subtract(out_frames[0,0], frames_pred[0,0])).cpu().numpy()
        MAE_loss_crop = MAE(frames_pred[:, :, CROP_SIZE:M-CROP_SIZE, CROP_SIZE:N-CROP_SIZE], out_frames[:, :, CROP_SIZE:M-CROP_SIZE, CROP_SIZE:N-CROP_SIZE])
        if minute < 30:
            if (hour, 0) in MAE_per_hour.keys():
                MAE_per_hour[(hour, 0)].append(MAE_loss.detach().item())
                MAE_per_hour_crop[(hour, 0)].append(MAE_loss_crop.detach().item())
            else:
                MAE_per_hour[(hour,0)] = []
                MAE_per_hour[(hour,0)].append(MAE_loss.detach().item())
                MAE_per_hour_crop[(hour,0)] = []
                MAE_per_hour_crop[(hour,0)].append(MAE_loss_crop.detach().item())
        else:
            if (hour,30) in MAE_per_hour.keys():
                MAE_per_hour[(hour, 30)].append(MAE_loss.detach().item())
                MAE_per_hour_crop[(hour, 30)].append(MAE_loss_crop.detach().item())
            else:
                MAE_per_hour[(hour, 30)] = []
                MAE_per_hour[(hour, 30)].append(MAE_loss.detach().item())
                MAE_per_hour_crop[(hour, 30)] = []
                MAE_per_hour_crop[(hour, 30)].append(MAE_loss_crop.detach().item())
        if MAE_loss.detach().item() > worst_MAE_error:
            worst_MAE_error = MAE_loss.detach().item()
            
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
            
        if MAE_loss.detach().item() < best_MAE_error:
            best_MAE_error = MAE_loss.item()
            best_MAE_input_time = in_time[0].numpy()
            best_MAE_time = out_time[0, 0]
            
            best_MAE_time_list = [
                str(int((best_MAE_input_time[0][1]))).zfill(2) + ':' + str(int((best_MAE_input_time[0][2]))).zfill(2),
                str(int((best_MAE_input_time[1][1]))).zfill(2) + ':' + str(int((best_MAE_input_time[1][2]))).zfill(2),
                str(int((best_MAE_input_time[2][1]))).zfill(2) + ':' + str(int((best_MAE_input_time[2][2]))).zfill(2),
                str(int(best_MAE_time[1].numpy())).zfill(2) + ':' + str(int(best_MAE_time[2].numpy())).zfill(2)
            ]
            
            best_MAE_images[0:3] = in_frames[0].cpu().numpy()
            best_MAE_images[3] = out_frames[0,0].cpu().numpy()
            best_MAE_images[4] = frames_pred[0,0].cpu().numpy()
        
        # MSE
        MSE_loss = MSE(frames_pred, out_frames)
        if minute<30:
            if (hour,0) in MSE_per_hour.keys():
                MSE_per_hour[(hour,0)].append(MSE_loss.detach().item())
            else:
                MSE_per_hour[(hour,0)] = []
                MSE_per_hour[(hour,0)].append(MSE_loss.detach().item())
        else:
            if (hour,30) in MSE_per_hour.keys():
                MSE_per_hour[(hour,30)].append(MSE_loss.detach().item())
            else:
                MSE_per_hour[(hour,30)] = []
                MSE_per_hour[(hour,30)].append(MSE_loss.detach().item())
                
        if MSE_loss.detach().item() > worst_MSE_error:
            worst_MSE_error = MSE_loss.detach().item()
            worst_MSE_input_time = in_time[0].numpy()
            worst_MSE_time = out_time[0, 0]
            
            worst_MSE_time_list = [
                str(int((worst_MSE_input_time[0][1]))).zfill(2) + ':' + str(int((worst_MSE_input_time[0][2]))).zfill(2),
                str(int((worst_MSE_input_time[1][1]))).zfill(2) + ':' + str(int((worst_MSE_input_time[1][2]))).zfill(2),
                str(int((worst_MSE_input_time[2][1]))).zfill(2) + ':' + str(int((worst_MSE_input_time[2][2]))).zfill(2),
                str(int(worst_MSE_time[1].numpy())).zfill(2) + ':' + str(int(worst_MSE_time[2].numpy())).zfill(2)
            ] 
            
            worst_MSE_images[0:3] = in_frames[0].cpu().numpy()
            worst_MSE_images[3] = out_frames[0, 0].cpu().numpy()
            worst_MSE_images[4] = frames_pred[0, 0].cpu().numpy()
        if MSE_loss.detach().item() < best_MSE_error:
            best_MSE_error = MSE_loss.detach().item()
            best_MSE_input_time = in_time[0].numpy()
            best_MSE_time = out_time[0, 0]
            
            best_MSE_time_list = [
                str(int((best_MSE_input_time[0][1]))).zfill(2) + ':' + str(int((best_MSE_input_time[0][2]))).zfill(2),
                str(int((best_MSE_input_time[1][1]))).zfill(2) + ':' + str(int((best_MSE_input_time[1][2]))).zfill(2),
                str(int((best_MSE_input_time[2][1]))).zfill(2) + ':' + str(int((best_MSE_input_time[2][2]))).zfill(2),
                str(int(best_MSE_time[1].numpy())).zfill(2) + ':' + str(int(best_MSE_time[2].numpy())).zfill(2)
            ]
            
            best_MSE_images[0:3] = in_frames[0].cpu().numpy()
            best_MSE_images[3] = out_frames[0,0].cpu().numpy()
            best_MSE_images[4] = frames_pred[0, 0].cpu().numpy()
            
        # PSNR
        if minute < 30:
            minute_key = 0
            if (MSE_per_hour[(hour,0)][-1] != 0):
                if (hour,0) in PSNR_per_hour.keys():
                    PSNR_per_hour[(hour,0)].append(10* np.log10(1**2/MSE_per_hour[(hour,0)][-1]))
                else:
                    PSNR_per_hour[(hour,0)] = [10* np.log10(1**2/MSE_per_hour[(hour,0)][-1])]
            else:
                if (hour,0) in PSNR_per_hour.keys():
                    PSNR_per_hour[(hour,0)].append(20*np.log10(1))
                else:
                    PSNR_per_hour[(hour,0)] = [20*np.log10(1)]
        else:
            minute_key = 30
            if (MSE_per_hour[(hour,30)][-1] != 0):
                if (hour,30) in PSNR_per_hour.keys():
                    PSNR_per_hour[(hour,30)].append(10* np.log10(1**2/MSE_per_hour[(hour,30)][-1]))
                else:
                    PSNR_per_hour[(hour,30)] = [10* np.log10(1**2/MSE_per_hour[(hour,30)][-1])]
            else:
                if (hour,30) in PSNR_per_hour.keys():
                    PSNR_per_hour[(hour,30)].append(20*np.log10(1))
                else:
                    PSNR_per_hour[(hour,30)] = [20*np.log10(1)]
                    
        if PSNR_per_hour[(hour, minute_key)][-1] < worst_PSNR_error:
            worst_PSNR_error = PSNR_per_hour[(hour, minute_key)][-1]
            worst_PSNR_input_time = in_time[0].numpy()
            worst_PSNR_time = out_time[0, 0]
            
            worst_PSNR_time_list = [
                str(int((worst_PSNR_input_time[0][1]))).zfill(2) + ':' + str(int((worst_PSNR_input_time[0][2]))).zfill(2),
                str(int((worst_PSNR_input_time[1][1]))).zfill(2) + ':' + str(int((worst_PSNR_input_time[1][2]))).zfill(2),
                str(int((worst_PSNR_input_time[2][1]))).zfill(2) + ':' + str(int((worst_PSNR_input_time[2][2]))).zfill(2),
                str(int(worst_PSNR_time[1].numpy())).zfill(2) + ':' + str(int(worst_PSNR_time[2].numpy())).zfill(2)
            ] 
            
            worst_PSNR_images[0:3] = in_frames[0].cpu().numpy()
            worst_PSNR_images[3] = out_frames[0,0].cpu().numpy()
            worst_PSNR_images[4] = frames_pred[0,0].cpu().numpy()
            
        if PSNR_per_hour[(hour, minute_key)][-1] > best_PSNR_error:
            best_PSNR_error = PSNR_per_hour[(hour, minute_key)][-1]
            best_PSNR_input_time = in_time[0].numpy()
            best_PSNR_time = out_time[0, 0]
            
            best_PSNR_time_list = [
                str(int((best_PSNR_input_time[0][1]))).zfill(2) + ':' + str(int((best_PSNR_input_time[0][2]))).zfill(2),
                str(int((best_PSNR_input_time[1][1]))).zfill(2) + ':' + str(int((best_PSNR_input_time[1][2]))).zfill(2),
                str(int((best_PSNR_input_time[2][1]))).zfill(2) + ':' + str(int((best_PSNR_input_time[2][2]))).zfill(2),
                str(int(best_PSNR_time[1].numpy())).zfill(2) + ':' + str(int(best_PSNR_time[2].numpy())).zfill(2)
            ] 
            
            best_PSNR_images[0:3] = in_frames[0].cpu().numpy()
            best_PSNR_images[3] = out_frames[0,0].cpu().numpy()
            best_PSNR_images[4] = frames_pred[0,0].cpu().numpy()
        
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

        gt_mean.append(torch.mean(out_frames[0,0]).cpu().numpy())
        gt_std.append(torch.std(out_frames[0,0]).cpu().numpy())
        pred_mean.append(torch.mean(frames_pred[0,0]).cpu().numpy())
        pred_std.append(torch.std(frames_pred[0,0]).cpu().numpy())

MAE_error_image = MAE_error_image/len(val_mvd)
fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'MAE_error_image.png')
visualization.show_image_w_colorbar(image=MAE_error_image, title=None,
                                    fig_name=fig_name, save_fig=True)

mean_MAE = []
mean_MAE_crop = []
mean_MSE = []
mean_PSNR = []
mean_SSIM = []
mean_SSIM_crop = []
std_MAE = []
std_MAE_crop = []
std_MSE = []
std_PSNR = []
std_SSIM = []
std_SSIM_crop = []

sorted_keys = sorted(MAE_per_hour.keys(), key=lambda element: (element[0], element[1]))
hour_list = []

for key in sorted_keys:
    hour_list.append(str(key[0]).zfill(2) + ':' + str(key[1]).zfill(2))
    mean_MAE.append(np.mean(MAE_per_hour[key]))
    std_MAE.append(np.std(MAE_per_hour[key]))
    mean_MAE_crop.append(np.mean(MAE_per_hour_crop[key]))
    std_MAE_crop.append(np.std(MAE_per_hour_crop[key]))
    mean_MSE.append(np.mean(MSE_per_hour[key]))
    std_MSE.append(np.std(MSE_per_hour[key]))
    mean_PSNR.append(np.mean(PSNR_per_hour[key]))
    std_PSNR.append(np.std(PSNR_per_hour[key]))
    mean_SSIM.append(np.mean(SSIM_per_hour[key]))
    std_SSIM.append(np.std(SSIM_per_hour[key]))
    mean_SSIM_crop.append(np.mean(SSIM_per_hour_crop[key]))
    std_SSIM_crop.append(np.std(SSIM_per_hour_crop[key]))
    
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
        'mean_MAE_crop': mean_MAE_crop,
        'std_MAE_crop': std_MAE_crop,
        'mean_SSIM': mean_SSIM,
        'std_SSIM': std_SSIM,
        'mean_SSIM_crop': mean_SSIM_crop,
        'std_SSIM_crop': std_SSIM_crop,
        'mean_MSE': mean_MSE,
        'std_MSE': std_MSE,
        'mean_PSNR': mean_PSNR,
        'std_PSNR': std_PSNR
        }                                                                                                                      

    utils.save_pickle_dict(path=SAVE_VALUES_PATH, name=MODEL_PATH.split('/')[-1][:-9], dict_=dict_values) 

print('Dict with error values saved.')

plt.figure(figsize=(12, 6))
plt.plot(mean_MAE, '-o', label='Full window')
plt.plot(mean_MAE_crop, '-o', label='Crop')
plt.legend(loc='upper right')
plt.xticks(range(len(hour_list)), hour_list)
plt.gcf().autofmt_xdate()
plt.title('Mean MAE error per hour')
plt.xlabel('Time of day')
plt.ylabel('MAE')
plt.grid()
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'MAE_p_hour.png')
                )
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(mean_MSE, '-o')
plt.xticks(range(len(hour_list)), hour_list)
plt.gcf().autofmt_xdate()
plt.title('Mean MSE error per hour')
plt.xlabel('Time of day')
plt.ylabel('MSE')
plt.grid()
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'MSE_p_hour.png')
                )
plt.show()
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(mean_SSIM, '-o', label='Full Window')
plt.plot(mean_SSIM_crop, '-o', label='Crop')
plt.legend(loc='upper right')
plt.xticks(range(len(hour_list)), hour_list)
plt.gcf().autofmt_xdate()
plt.title('Mean SSIM error per hour')
plt.xlabel('Time of day')
plt.ylabel('SSIM')
plt.grid()
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'SSIM_p_hour.png')
                )
plt.show()
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(mean_PSNR, '-o')
plt.xticks(range(len(hour_list)), hour_list)
plt.gcf().autofmt_xdate()
plt.title('Mean PSNR error per hour')
plt.xlabel('Time of day')
plt.ylabel('PSNR')
plt.grid()
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'PSNR_p_hour.png')
                )
plt.show()


#SCATTER PLOT
print('Scatter Plot')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
m, b = np.polyfit(gt_mean, pred_mean, 1)
fig = plt.figure()
fig.set_size_inches(6, 6)
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_tick_params(labelsize=fontSize)
ax.yaxis.set_tick_params(labelsize=fontSize)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.0)
ax.tick_params(direction='out', length=6, width=1, colors='k')
ax.scatter(x=gt_mean, y=pred_mean, color=Colors.peterRiver)
ax.plot(gt_mean, m*np.array(gt_mean) + b, color=Colors.pomegranate)
textstr = '\n'.join((
    r'$m=%.2f$' % (m, ),
    r'$n=%.2f$' % (b, )))
ax.set_xlabel('GT Mean', fontsize=fontSize)
ax.set_ylabel('Prediction Mean', fontsize=fontSize)
ax.text(np.min(gt_mean), np.max(pred_mean),textstr, fontsize=fontSize,
        va='top', ha='left')

if SAVE_IMAGES_PATH:
    fig.tight_layout() 
    fig.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'scatterplot_mean.png'))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
m, b = np.polyfit(gt_std, pred_std, 1)
fig = plt.figure()
fig.set_size_inches(6, 6)
ax = fig.add_subplot(1, 1, 1)
ax.xaxis.set_tick_params(labelsize=fontSize)
ax.yaxis.set_tick_params(labelsize=fontSize)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.0)
ax.tick_params(direction='out', length=6, width=1, colors='k')
ax.scatter(x=gt_std, y=pred_std, color=Colors.peterRiver)
ax.plot(gt_std, m*np.array(gt_std) + b, color=Colors.pomegranate)
textstr = '\n'.join((
    r'$m=%.2f$' % (m, ),
    r'$n=%.2f$' % (b, )))
ax.set_xlabel('GT Standard Deviation', fontsize=fontSize)
ax.set_ylabel('Prediction Standard Deviation', fontsize=fontSize)
ax.text(np.min(gt_std), np.max(pred_std), textstr, fontsize=fontSize,
        va='top', ha='left')

if SAVE_IMAGES_PATH:
    fig.tight_layout() 
    fig.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'scatterplot_std.png'))
plt.close()

#MEANS DENSITY DISTRIBUTION
xmin, xmax = np.min(gt_mean)-0.01, np.max(gt_mean)+0.01
ymin, ymax = np.min(pred_mean)-0.01, np.max(pred_mean)+0.01

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([gt_mean, pred_mean])
kernel = st.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(gt_mean, pred_mean, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_title('Scatter plot of Means with prob distribution')
ax.set_xlabel('GT mean')
ax.set_ylabel('Pred mean')
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'mean_distribution.png')
                )
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

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(gt_std, pred_std, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_title('Scatter plot of STD with prob distribution')
ax.set_xlabel('GT std')
ax.set_ylabel('Pred std')
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'std_distribution.png')
                )
plt.show()
plt.close()

# IMG MEANS HISTOGRAM
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure()
fig.set_size_inches(12, 6)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r"Mean Value of Image", fontsize=fontSize)
ax.set_ylabel(r"Quantity", fontsize=fontSize)
ax.xaxis.set_tick_params(labelsize=fontSize)
ax.yaxis.set_tick_params(labelsize=fontSize)

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
    fig.savefig(os.path.join(SAVE_IMAGES_PATH, 'means_histogram.png'))

plt.close()

# IMG STD DEV HISTOGRAM
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure()
fig.set_size_inches(12, 6)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r"STD of Image", fontsize=fontSize)
ax.set_ylabel(r"Quantity", fontsize=fontSize)
ax.xaxis.set_tick_params(labelsize=fontSize)
ax.yaxis.set_tick_params(labelsize=fontSize)

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
    fig.savefig(os.path.join(SAVE_IMAGES_PATH, 'stds_histogram.png'))
plt.close()

#BEST AND WORST PREDICTIONS

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'MAE best prediction, day:' + str(int(best_MAE_time[0].numpy())) + \
                        'hour:' + str(int(best_MAE_time[1].numpy())).zfill(2)+str(int(best_MAE_time[2].numpy())).zfill(2))
visualization.show_seq_and_pred(best_MAE_images,
                                time_list=best_MAE_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'MAE worst prediction, day:'+str(int(worst_MAE_time[0].numpy())) + \
                        'hour:' + str(int(worst_MAE_time[1].numpy())).zfill(2) + str(int(worst_MAE_time[2].numpy())).zfill(2))
visualization.show_seq_and_pred(worst_MAE_images,
                                time_list=worst_MAE_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'MSE best prediction, day:' + str(int(best_MSE_time[0].numpy())) + \
                        'hour:'+ str(int(best_MSE_time[1].numpy())).zfill(2) + str(int(best_MSE_time[2].numpy())).zfill(2))
visualization.show_seq_and_pred(best_MSE_images,
                                time_list=best_MSE_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'MSE worst prediction, day:' + str(int(worst_MSE_time[0].numpy())) + \
                        'hour:'+ str(int(worst_MSE_time[1].numpy())).zfill(2) + str(int(worst_MSE_time[2].numpy())).zfill(2))
visualization.show_seq_and_pred(worst_MSE_images,
                                time_list=worst_MSE_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'PSNR best prediction, day:' + str(int(best_PSNR_time[0].numpy())) + \
                        'hour:'+ str(int(best_PSNR_time[1].numpy())).zfill(2) + str(int(best_PSNR_time[2].numpy())).zfill(2))
visualization.show_seq_and_pred(best_PSNR_images,
                                time_list=best_PSNR_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'PSNR worst prediction, day:'+str(int(worst_PSNR_time[0].numpy())) + \
                        'hour:' + str(int(worst_PSNR_time[1].numpy())).zfill(2) + str(int(worst_PSNR_time[2].numpy())).zfill(2))
visualization.show_seq_and_pred(worst_PSNR_images,
                                time_list=worst_PSNR_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'SSIM best prediction, day:'+str(int(best_SSIM_time[0].numpy())) + \
                        'hour:' + str(int(best_SSIM_time[1].numpy())).zfill(2) + str(int(best_SSIM_time[2].numpy())).zfill(2))
visualization.show_seq_and_pred(best_SSIM_images,
                                time_list=best_SSIM_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

fig_name = os.path.join(SAVE_IMAGES_PATH,
                        'SSIM worst prediction, day:' + str(int(worst_SSIM_time[0].numpy())) + \
                        'hour:' + str(int(worst_SSIM_time[1].numpy())).zfill(2) + str(int(worst_SSIM_time[2].numpy())).zfill(2))
visualization.show_seq_and_pred(worst_SSIM_images,
                                time_list=worst_SSIM_time_list,
                                prediction_t=FRAME_OUT+1,
                                fig_name=fig_name,
                                save_fig=True)
plt.close()

# OUTPUT WITH MOST NANS SEQUENCE
print('OUTPUT WITH MOST NANS SEQUENCE')
img0 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_115017.npy'))
img1 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_120017.npy'))
img2 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_121017.npy'))
if FRAME_OUT == 0:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_122017.npy'))
elif FRAME_OUT == 1:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_123017.npy'))
elif FRAME_OUT == 2:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_124017.npy'))      
elif FRAME_OUT == 3:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_125017.npy'))
elif FRAME_OUT == 4:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_130017.npy'))
elif FRAME_OUT == 5:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_131017.npy'))
elif FRAME_OUT ==11:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_141017.npy'))
elif FRAME_OUT == 17:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_151017.npy'))
elif FRAME_OUT == 23:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_161017.npy'))
else:
    raise ValueError('Prediction time must be 10,20,30,40,50,60,120,180 or 240 minutes.')
        
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
        frames_pred = torch.add(diff_pred[:,0], in_frames[:,2]).unsqueeze(1) 
    
frames_array = np.ones((5, M, N))
frames_array[0:3] = in_frames[0].cpu().numpy()
frames_array[3]= out_frames[0,0].cpu().numpy()
frames_array[4] = frames_pred[0,0].cpu().numpy()
    
fig_name = os.path.join(SAVE_IMAGES_PATH, 'most_nan_sequence.png')
visualization.show_seq_and_pred(frames_array, fig_name=fig_name, save_fig=True)

# LARGEST MOVEMENT left to right --->

img0 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_135018.npy'))
img1 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_140018.npy'))
img2 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_141018.npy'))
if FRAME_OUT == 0:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_142018.npy'))
elif FRAME_OUT == 1:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_143018.npy'))
elif FRAME_OUT == 2:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_144018.npy'))      
elif FRAME_OUT == 3:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_145018.npy'))
elif FRAME_OUT == 4:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_150018.npy'))
elif FRAME_OUT == 5:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_151018.npy'))
elif FRAME_OUT ==11:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_161018.npy'))
elif FRAME_OUT == 17:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_171017.npy'))
elif FRAME_OUT == 23:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_181017.npy'))
else:
    raise ValueError('Prediction time must be 10,20,30,40,50,60,120,180 or 240 minutes.')
        
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
    
fig_name = os.path.join(SAVE_IMAGES_PATH, 'most_moved_sequence.png')
visualization.show_seq_and_pred(frames_array, fig_name=fig_name, save_fig=True)

if PREDICT_DIFF:
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'most_moved_sequence_diff_pred.png')
    visualization.show_image_w_colorbar(img_diff_pred, fig_name=fig_name, save_fig=True)

# FIRST LAYER OF FILTERS OUTPUT
if M < 1000:
    output_list = []

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                for filter_ in m.weight:
                    output = visualization.use_filter(in_frames.cpu().numpy(), filter_.cpu().numpy()) 
                    output_list.append(output)
                break
    fig_name = os.path.join(SAVE_IMAGES_PATH, 'filter_layer_output.png')
    visualization.show_image_list(output_list, rows=8, fig_name=fig_name, save_fig=True)
    plt.close()
print('Done.')
