import numpy as np
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

## CONFIGURATION #########

PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/'
FRAME_OUT = 0  # 0->10min, 1->20min, 2->30min... 
CSV_PATH = None
# CSV_PATH = 'data/mvd/val_seq_in3_out1_cosangs.csv'
MODEL_PATH = 'checkpoints/10min_predict_ssim_60_05-07-2021_08:43.pt'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)

model = UNet(n_channels=3, n_classes=1, bilinear=True, p=0, output_activation='sigmoid', bias=False).to(device)
# model = UNet2(n_channels=3, n_classes=1, bilinear=True, p=0, output_activation='sigmoid', bias=False).to(device)
SAVE_IMAGES_PATH = 'prueba' 

###########################

#Evaluate Unet

select_frame = preprocessing.select_output_frame(FRAME_OUT)
normalize = preprocessing.normalize_pixels(mean0 = False) #values between [0,1]
transforms = [select_frame,normalize]

val_mvd = MontevideoFoldersDataset_w_time(
                                            path=PATH_DATA,
                                            in_channel=3,
                                            out_channel=FRAME_OUT+1,
                                            min_time_diff=5,
                                            max_time_diff=15,
                                            csv_path=CSV_PATH,
                                            transform=transforms
                                            )

val_loader = DataLoader(val_mvd, batch_size=1, shuffle=False)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))["model_state_dict"])

gt_mean = []
gt_std = []
pred_mean = []
pred_std = []

worst_MAE_error = 0
worst_MAE_time = ''
best_MAE_error = 1
best_MAE_time = ''
best_MAE_images = np.zeros((5, 256, 256))
worst_MAE_images = np.zeros((5, 256, 256))
MAE = nn.L1Loss()
MAE_per_hour = {}
MAE_per_hour_crop = {}
for i in range(0, 25):
    MAE_per_hour[i] = []
    MAE_per_hour_crop[i] = []

worst_MSE_error = 0
worst_MSE_time = ''
best_MSE_error = 1
best_MSE_time = ''    
MSE = nn.MSELoss()
best_MSE_images = np.zeros((5, 256, 256))
worst_MSE_images = np.zeros((5, 256, 256))
MSE_per_hour = {}
for i in range(0, 25):
    MSE_per_hour[i] = []    

worst_PSNR_error = 100
worst_PSNR_time = ''
best_PSNR_error = 0
best_PSNR_time = ''
best_PSNR_images = np.zeros((5, 256, 256))
worst_PSNR_images = np.zeros((5, 256, 256))
PSNR_per_hour = {}
for i in range(0, 25):
    PSNR_per_hour[i] = []   

try:
    SSIM = SSIM(n_channels=1).cuda()
except:
    pass

worst_SSIM_error = 1
worst_SSIM_time = ''
best_SSIM_error = 0
best_SSIM_time = ''
best_SSIM_images = np.zeros((5, 256, 256))
worst_SSIM_images = np.zeros((5, 256, 256))
SSIM_per_hour = {}
SSIM_per_hour_crop = {}
for i in range(0, 25):
    SSIM_per_hour[i] = [] 
    SSIM_per_hour_crop[i] = []

model.eval()
with torch.no_grad():
    for val_batch_idx, (in_frames, out_frames, out_time) in enumerate(val_loader):
        
        in_frames = in_frames.to(device=device)
        out_frames = out_frames.to(device=device)
               
        day, hour = int(out_time[0, FRAME_OUT, 0]), int(out_time[0 ,FRAME_OUT, 1]/1e4)

        frames_pred = model(in_frames)

        # MAE
        MAE_loss = MAE(frames_pred, out_frames)
        MAE_loss_crop = MAE(frames_pred[:, :, 28:-28, 28:-28], out_frames[:, :, 28:-28, 28:-28])
        MAE_per_hour[hour].append(MAE_loss.detach().item())
        MAE_per_hour_crop[hour].append(MAE_loss_crop.detach().item())
        if MAE_loss.detach().item() > worst_MAE_error:
            worst_MAE_error = MAE_loss.detach().item()
            worst_MAE_time = out_time[0, FRAME_OUT]
            worst_MAE_images[0:3] = in_frames[0].cpu().numpy()
            worst_MAE_images[3] = out_frames[0, 0].cpu().numpy()
            worst_MAE_images[4] = frames_pred[0, 0].cpu().numpy()
            
        if MAE_loss.item() < best_MAE_error:
            best_MAE_error = MAE_loss.item()
            best_MAE_time = out_time[0, FRAME_OUT]
            best_MAE_images[0:3] = in_frames[0].cpu().numpy()
            best_MAE_images[3] = out_frames[0,0].cpu().numpy()
            best_MAE_images[4] = frames_pred[0,0].cpu().numpy()
        
        # MSE
        MSE_loss = MSE(frames_pred, out_frames)
        MSE_per_hour[hour].append(MSE_loss.detach().item())
        if MSE_loss.detach().item() > worst_MSE_error:
            worst_MSE_error = MSE_loss.detach().item()
            worst_MSE_time = out_time[0, FRAME_OUT]
            worst_MSE_images[0:3] = in_frames[0].cpu().numpy()
            worst_MSE_images[3] = out_frames[0, 0].cpu().numpy()
            worst_MSE_images[4] = frames_pred[0, 0].cpu().numpy()
        if MSE_loss.item() < best_MSE_error:
            best_MSE_error = MSE_loss.detach().item()
            best_MSE_time = out_time[0, FRAME_OUT]
            best_MSE_images[0:3] = in_frames[0].cpu().numpy()
            best_MSE_images[3] = out_frames[0,0].cpu().numpy()
            best_MSE_images[4] = frames_pred[0, 0].cpu().numpy()
            
        # PSNR        
        if (MSE_per_hour[hour][-1] != 0 ):
            PSNR_per_hour[hour].append(10* np.log10(1**2/MSE_per_hour[hour][-1])) 
        else:
            PSNR_per_hour[hour].append(20*np.log10(1))
        if PSNR_per_hour[hour][-1] < worst_PSNR_error:
            worst_PSNR_error = PSNR_per_hour[hour][-1]
            worst_PSNR_time = out_time[0, FRAME_OUT]
            worst_PSNR_images[0:3] = in_frames[0].cpu().numpy()
            worst_PSNR_images[3] = out_frames[0,0].cpu().numpy()
            worst_PSNR_images[4] = frames_pred[0,0].cpu().numpy()
        if PSNR_per_hour[hour][-1] > best_PSNR_error:
            best_PSNR_error = PSNR_per_hour[hour][-1]
            best_PSNR_time = out_time[0, FRAME_OUT]
            best_PSNR_images[0:3] = in_frames[0].cpu().numpy()
            best_PSNR_images[3] = out_frames[0,0].cpu().numpy()
            best_PSNR_images[4] = frames_pred[0,0].cpu().numpy()
        
        # SSIM
        SSIM_loss = SSIM(frames_pred, out_frames)
        SSIM_per_hour[hour].append(SSIM_loss.detach().item())
        SSIM_loss_crop = SSIM(frames_pred[:, :, 28:-28, 28:-28], out_frames[:, :, 28:-28, 28:-28])
        SSIM_per_hour_crop[hour].append(SSIM_loss_crop.detach().item())
        if SSIM_loss.detach().item() < worst_SSIM_error:
            worst_SSIM_error = SSIM_loss.detach().item()
            worst_SSIM_time = out_time[0, FRAME_OUT]
            worst_SSIM_images[0:3] = in_frames[0].cpu().numpy()
            worst_SSIM_images[3] = out_frames[0,0].cpu().numpy()
            worst_SSIM_images[4] = frames_pred[0,0].cpu().numpy()
        if SSIM_loss.item() > best_SSIM_error:
            best_SSIM_error = SSIM_loss.detach().item()
            best_SSIM_time = out_time[0, FRAME_OUT]
            best_SSIM_images[0:3] = in_frames[0].cpu().numpy()
            best_SSIM_images[3] = out_frames[0,0].cpu().numpy()
            best_SSIM_images[4] = frames_pred[0,0].cpu().numpy()

        gt_mean.append(np.mean(out_frames[0,0].numpy()))
        gt_std.append(np.std(out_frames[0,0].numpy()))
        pred_mean.append(np.mean(frames_pred[0,0].numpy()))
        pred_std.append(np.std(frames_pred[0,0].numpy()))

mean_MAE = []
mean_MAE_crop = []
mean_MSE = []
mean_PSNR = []
mean_SSIM = []
mean_SSIM_crop = []
hour_list = []
for i in range(25): 
    if len(MAE_per_hour[i])>0:
        hour_list.append(i)
        mean_MAE.append(np.mean(MAE_per_hour[i]))
        mean_MAE_crop.append(np.mean(MAE_per_hour_crop[i]))
        mean_MSE.append(np.mean(MSE_per_hour[i]))
        mean_PSNR.append(np.mean(PSNR_per_hour[i]))
        mean_SSIM.append(np.mean(SSIM_per_hour[i]))
        mean_SSIM_crop.append(np.mean(SSIM_per_hour_crop[i]))
        
# ERROR GRAPHS
fig, axs = plt.subplots(1, 4, figsize=(20,5))
axs[0].plot(hour_list, mean_MAE, 'r-o', label='Full window')
axs[0].plot(hour_list, mean_MAE_crop, 'g-o', label='Crop')
axs[0].legend(loc='upper right')
axs[0].set_title('MAE')
#axs[0].set(xlabel='hour', ylabel='Error')
axs[1].plot(hour_list, mean_MSE,'r-o')
axs[1].set_title('MSE')
#axs[1].set(xlabel='hour', ylabel='Error')
axs[2].plot(hour_list, mean_PSNR,'r-o')
axs[2].set_title('PSNR')
#axs[2].set(xlabel='hour', ylabel='Error')
axs[3].plot(hour_list, mean_SSIM, 'r-o', label='Full Window')
axs[3].plot(hour_list, mean_SSIM_crop, 'g-o', label='Crop')
axs[3].legend(loc='upper left')
axs[3].set_title('SSIM')
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'error_graphs.png')
                )
plt.show()

# IMG MEANS HISTOGRAM
plt.figure(figsize=(8,6))
gt_mean_hist, gt_mean_bins, _ = plt.hist(gt_mean, bins=100, density=True, alpha=0.5, label="GT")
pred_mean_hist, pred_mean_bins, _ = plt.hist(pred_mean, bins=100, density=True, alpha=0.5, label="Pred")
l1 = plt.axvline(np.mean(gt_mean), c='b')
l2 = plt.axvline(np.mean(pred_mean), c='r')
plt.xlabel("Data", size=14)
plt.ylabel("Count", size=14)
plt.title("Image Mean value")
plt.legend(loc='upper right')
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'means_histogram.png')
                )
plt.show()

# IMG STD DEV HISTOGRAM
plt.figure(figsize=(8,6))
gt_std_hist, gt_std_bins, _ = plt.hist(gt_std, bins=100, density=True, alpha=0.5, label="GT")
pred_std_hist, pred_std_bins, _ =plt.hist(pred_std, bins=100, density=True, alpha=0.5, label="Pred")
l1 = plt.axvline(np.mean(gt_std), c='b')
l2 = plt.axvline(np.mean(pred_std), c='r')
plt.xlabel("Data", size=14)
plt.ylabel("Count", size=14)
plt.title("Image Std value")
plt.legend(loc='upper right')
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'stds_histogram.png')
                )
plt.show()

#BEST AND WORST PREDICTIONS

# print('MAE best prediction, day:', int(best_MAE_time[0].numpy()),'hour:', int(best_MAE_time[1].numpy()))
fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE best prediction, day:'+str(int(best_MAE_time[0].numpy()))+'hour:'+ str(int(best_MAE_time[1].numpy())))
visualization.show_seq_and_pred(best_MAE_images, fig_name=fig_name, save_fig=True)
# print('MAE worst prediction, day:', int(worst_MAE_time[0].numpy()),'hour:', int(worst_MAE_time[1].numpy())) 
fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE worst prediction, day:'+str(int(worst_MAE_time[0].numpy()))+'hour:'+ str(int(worst_MAE_time[1].numpy())))
visualization.show_seq_and_pred(worst_MAE_images, fig_name=fig_name, save_fig=True)
# print('MSE best prediction, day:', int(best_MSE_time[0].numpy()),'hour:', int(best_MSE_time[1].numpy()))
fig_name = os.path.join(SAVE_IMAGES_PATH, 'MSE best prediction, day:'+str(int(best_MSE_time[0].numpy()))+'hour:'+ str(int(best_MSE_time[1].numpy())))
visualization.show_seq_and_pred(best_MSE_images, fig_name=fig_name, save_fig=True)
# print('MSE worst prediction, day:', int(worst_MSE_time[0].numpy()),'hour:', int(worst_MSE_time[1].numpy()))
fig_name = os.path.join(SAVE_IMAGES_PATH, 'MSE worst prediction, day:'+str(int(worst_MSE_time[0].numpy()))+'hour:'+ str(int(worst_MSE_time[1].numpy())))
visualization.show_seq_and_pred(worst_MSE_images, fig_name=fig_name, save_fig=True)
# print('PSNR best prediction, day:', int(best_PSNR_time[0].numpy()),'hour:', int(best_PSNR_time[1].numpy()))
fig_name = os.path.join(SAVE_IMAGES_PATH, 'PSNR best prediction, day:'+str(int(best_PSNR_time[0].numpy()))+'hour:'+ str(int(best_PSNR_time[1].numpy())))
visualization.show_seq_and_pred(best_PSNR_images, fig_name=fig_name, save_fig=True)
# print('PSNR worst prediction, day:', int(worst_PSNR_time[0].numpy()),'hour:', int(worst_PSNR_time[1].numpy()))
fig_name = os.path.join(SAVE_IMAGES_PATH, 'PSNR worst prediction, day:'+str(int(worst_PSNR_time[0].numpy()))+'hour:'+ str(int(worst_PSNR_time[1].numpy())))
visualization.show_seq_and_pred(worst_PSNR_images, fig_name=fig_name, save_fig=True)
# print('SSIM best prediction, day:', int(best_SSIM_time[0].numpy()),'hour:', int(best_SSIM_time[1].numpy()))
fig_name = os.path.join(SAVE_IMAGES_PATH, 'SSIM best prediction, day:'+str(int(best_SSIM_time[0].numpy()))+'hour:'+ str(int(best_SSIM_time[1].numpy())))
visualization.show_seq_and_pred(best_SSIM_images, fig_name=fig_name, save_fig=True)
# print('SSIM worst prediction, day:', int(worst_SSIM_time[0].numpy()),'hour:', int(worst_SSIM_time[1].numpy()))
fig_name = os.path.join(SAVE_IMAGES_PATH, 'SSIM worst prediction, day:'+str(int(worst_SSIM_time[0].numpy()))+'hour:'+ str(int(worst_SSIM_time[1].numpy())))
visualization.show_seq_and_pred(worst_SSIM_images, fig_name=fig_name, save_fig=True)


# PRECITIONS WITH INPUT ALL ONES OR ZEROS
ones_frames = torch.tensor(np.ones((1, 3, 256, 256))).to(device)
zeros_frames = torch.tensor(np.zeros((1, 3, 256, 256))).to(device)

with torch.no_grad():
    ones_pred = model_Unet(ones_frames.float())
    zeros_pred = model_Unet(zeros_frames.float())

fig_name = os.path.join(SAVE_IMAGES_PATH, 'prediction_from_ones.png')
visualization.show_image_w_colorbar(ones_pred[0,0], fig_name=fig_name, save_fig=True)
fig_name = os.path.join(SAVE_IMAGES_PATH, 'prediction_from_zeros.png')
visualization.show_image_w_colorbar(zeros_pred[0,0], fig_name=fig_name, save_fig=True)

               
# OUTPUT WITH MOST MOVED SEQUENCE
               
img0 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_115017.npy'))
img1 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_120017.npy'))
img2 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_121017.npy'))
output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_122017.npy'))

in_frames= torch.tensor(np.ones((1, 3, 256, 256))).to(device)
out_frames= torch.tensor(np.ones((1, 1, 256, 256))).to(device)
in_frames[0,0] *= img0/100
in_frames[0,1] *= img1/100
in_frames[0,2] *= img2/100
out_frames[0,0] *= output/100 

model.eval()
with torch.no_grad():
    frames_pred = model(in_frames)
    
frames_array = np.ones((5, 256, 256))
frames_array[0:3] = in_frames[0]
frames_array[3]= out_frames[0,0]
frames_array[4] = frames_pred[0,0]
    
fig_name = os.path.join(SAVE_IMAGES_PATH, 'example_sequence.png')
show_seq_and_pred(frames_array, fig_name=fig_name, save_fig=True)

output_list = []

with torch.no_grad():
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            for filter_ in m.weight:
                output = visualization.use_filter(in_frames, filter_.numpy()) 
                output_list.append(output)
            break
fig_name = os.path.join(SAVE_IMAGES_PATH, 'filter_layer_output.png')
visualization.show_image_list(output_list, rows=8, fig_name=fig_name, save_fig=True)
