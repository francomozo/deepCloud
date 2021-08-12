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

## CONFIGURATION #########

PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/validation/'
FRAME_OUT = 2  # 0->10min, 1->20min, 2->30min... [0,5]
CSV_PATH = None
# CSV_PATH = 'data/mvd/val_seq_in3_out1_cosangs.csv'
MODEL_PATH = 'checkpoints/30min_UNet2_SSIM_relu_f64_40_01-08-2021_23:43.pt'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)

#model = UNet(n_channels=3, n_classes=1, bilinear=True, p=0, output_activation='sigmoid', bias=False).to(device)
model = UNet2(n_channels=3, n_classes=1, bilinear=True, p=0, output_activation='relu', bias=False, filters=64).to(device)
#model = AttU_Net(img_ch=3, output_ch=1, init_filter=32).to(device)
#model = NestedUNet(in_ch=3, out_ch=1, init_filter=32).to(device)

SAVE_IMAGES_PATH = 'graphs/30min/30min_UNet2_SSIM_relu_f64_40' 

CROP_SIZE = 28
PREDICT_DIFF = False

###########################

try:
    os.mkdir(SAVE_IMAGES_PATH)
except:
    pass

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
for i in range(0, 25):
    MAE_per_hour[i] = []
    MAE_per_hour_crop[i] = []

worst_MSE_error = 0
worst_MSE_time = ''
best_MSE_error = 1
best_MSE_time = ''    
MSE = nn.MSELoss()
best_MSE_images = np.zeros((5, M, N))
worst_MSE_images = np.zeros((5, M, N))
MSE_per_hour = {}
for i in range(0, 25):
    MSE_per_hour[i] = []    

worst_PSNR_error = 100
worst_PSNR_time = ''
best_PSNR_error = 0
best_PSNR_time = ''
best_PSNR_images = np.zeros((5, M, N))
worst_PSNR_images = np.zeros((5, M, N))
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
best_SSIM_images = np.zeros((5, M, N))
worst_SSIM_images = np.zeros((5, M, N))
SSIM_per_hour = {}
SSIM_per_hour_crop = {}
for i in range(0, 25):
    SSIM_per_hour[i] = [] 
    SSIM_per_hour_crop[i] = []

MAE_error_image = np.zeros((M,N))

model.eval()
with torch.no_grad():
    for val_batch_idx, (in_frames, out_frames, out_time) in enumerate(val_loader):
        
        in_frames = in_frames.to(device=device)
        out_frames = out_frames.to(device=device)
               
        day, hour = int(out_time[0, FRAME_OUT, 0]), int(out_time[0 ,FRAME_OUT, 1]/1e4)

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
        SSIM_loss_crop = SSIM(frames_pred[:, :, CROP_SIZE:M-CROP_SIZE, CROP_SIZE:N-CROP_SIZE], out_frames[:, :, CROP_SIZE:M-CROP_SIZE, CROP_SIZE:N-CROP_SIZE])
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

        gt_mean.append(torch.mean(out_frames[0,0]).cpu().numpy())
        gt_std.append(torch.std(out_frames[0,0]).cpu().numpy())
        pred_mean.append(torch.mean(frames_pred[0,0]).cpu().numpy())
        pred_std.append(torch.std(frames_pred[0,0]).cpu().numpy())

MAE_error_image = MAE_error_image/len(val_mvd)
fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_error_image.png')
visualization.show_image_w_colorbar(MAE_error_image, fig_name=fig_name, save_fig=True)

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


#SCATTER PLOT
plt.figure(figsize=(5,5))
plt.scatter(x=gt_mean, y=pred_mean)
plt.title('Image means scatter plot')
plt.xlabel('GT mean')
plt.ylabel('Prediction mean')
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'scatterplot_mean.png')
                )
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(x=gt_std, y=pred_std)
plt.title('Image std scatter plot')
plt.xlabel('GT std')
plt.ylabel('Prediction std')
if SAVE_IMAGES_PATH:
    plt.savefig(os.path.join(
                            SAVE_IMAGES_PATH, 'scatterplot_std.png')
                )
plt.show()

#MEANS DENSITY DISTRIBUTION
xmin, xmax = -0.01, 1.1
ymin, ymax = -0.01, 1.1

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

#STD DENSITY DISTRIBUTION
xmin, xmax = -0.01, 0.4
ymin, ymax = -0.01, 0.4

# Peform the kernel density estimate
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([gt_std, pred_std])
kernel = st.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(gt_mean, pred_mean, 'k.', markersize=2)
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
ones_frames = torch.tensor(np.ones((1, 3, M, N))).to(device)
zeros_frames = torch.tensor(np.zeros((1, 3, M, N))).to(device)

with torch.no_grad():
    ones_pred = model(ones_frames.float())
    zeros_pred = model(zeros_frames.float())

fig_name = os.path.join(SAVE_IMAGES_PATH, 'prediction_from_ones.png')
visualization.show_image_w_colorbar(ones_pred[0,0].cpu().numpy(), fig_name=fig_name, save_fig=True)
fig_name = os.path.join(SAVE_IMAGES_PATH, 'prediction_from_zeros.png')
visualization.show_image_w_colorbar(zeros_pred[0,0].cpu().numpy(), fig_name=fig_name, save_fig=True)

               
# OUTPUT WITH MOST NANS SEQUENCE
               
img0 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_115017.npy'))
img1 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_120017.npy'))
img2 = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_121017.npy'))
if FRAME_OUT == 0:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_122017.npy'))
if FRAME_OUT == 1:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_123017.npy'))
if FRAME_OUT == 2:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_124017.npy'))      
if FRAME_OUT == 3:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_125017.npy'))
if FRAME_OUT == 4:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_130017.npy'))
if FRAME_OUT == 5:
    output = np.load(os.path.join(PATH_DATA, '2020160/ART_2020160_131017.npy'))
        
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

img0 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_135018.npy'))
img1 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_140018.npy'))
img2 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_141018.npy'))
if FRAME_OUT == 0:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_142018.npy'))
if FRAME_OUT == 1:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_143018.npy'))
if FRAME_OUT == 2:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_144018.npy'))      
if FRAME_OUT == 3:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_145018.npy'))
if FRAME_OUT == 4:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_150018.npy'))
if FRAME_OUT == 5:
    output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_151018.npy'))
        
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
    
fig_name = os.path.join(SAVE_IMAGES_PATH, 'most_moved_sequence.png')
visualization.show_seq_and_pred(frames_array, fig_name=fig_name, save_fig=True)

# FIRST LAYER OF FILTERS OUTPUT

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
