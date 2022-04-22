import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
from src import data, evaluate, model, preprocessing, visualization
from src.lib import utils
from src.data import MontevideoFoldersDataset_w_time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from src.dl_models.unet import UNet, UNet2
from src.lib.latex_options import Colors, Linestyles
from src.lib.utils import get_model_name

# DISCLAIMER: do not look at this code!! 

# Modified script from evaluate_unet.py to generte error maps over MVD with gans and U-Net diff

### SETUP #####################################################################
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)
MSE = nn.MSELoss()
MAE = nn.L1Loss()
normalize = preprocessing.normalize_pixels(mean0 = False) #values between [0,1]
fontsize = 22 # 22 generates the font more like the latex text
borders = np.linspace(1, 450, 100)

###############################################################################

REGION = 'MVD'
dataset = 'mvd'
MODEL_TO_EVALUATE = 'GAN' # 'UNET'

REGION = 'MVD'
dataset = 'mvd'
PREDICT_T_LIST = [3, 6, 9]

if MODEL_TO_EVALUATE == 'UNET':

    
    OUTPUT_ACTIVATION = 'sigmoid'    
    FILTERS = 16

    # paths and names
    UNET_BASE_PATH = '/clusteruy/home03/DeepCloud/deepCloud/checkpoints/'
    UNET_NAMES = {
        3 : '30min_UNET2_mvd_mae_filters16_sigmoid_diffFalse_retrainFalse_34_16-02-2022_11:26_BEST_FINAL.pt',
        6 : '60min_UNET2_mvd_mae_filters16_sigmoid_diffFalse_retrainFalse_15_17-02-2022_11:13_BEST_FINAL.pt',
        9 : '90min_UNET2_mvd_mae_filters16_sigmoid_diffFalse_retrainFalse_16_17-02-2022_06:27_BEST_FINAL.pt'
    }

    evaluate_test = True
    GENERATE_ERROR_MAP = True

    if GENERATE_ERROR_MAP:
        RMSE_pct_maps_list = []
        RMSE_maps_list = []
        MAE_maps_list = []

    for PREDICT_T in PREDICT_T_LIST:

        if PREDICT_T == 3:
            PREDICT_HORIZON = '30min'    
        if PREDICT_T == 6:
            PREDICT_HORIZON = '60min'
        if PREDICT_T == 9:
            PREDICT_HORIZON = '90min'

        print('Predict Horizon:', PREDICT_HORIZON)


        model_name = UNET_NAMES[PREDICT_T]
        MODEL_PATH = os.path.join(UNET_BASE_PATH, REGION, PREDICT_HORIZON, model_name)

        model = UNet2(
            n_channels=3,
            n_classes=1,
            output_activation=OUTPUT_ACTIVATION,
            filters=FILTERS
        ).to(device)

        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))["model_state_dict"])

        CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/test_cosangs_mvd.csv'
        PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/test/'
        SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/test/' + MODEL_PATH.split('/')[-1][:-17]

        try:
            os.mkdir(SAVE_IMAGES_PATH)
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

        RMSE_pct_list = []
        RMSE_list = []
        MAE_list = []

        MAE_per_hour = {}  # MAE
        MAE_pct_per_hour = {}
        RMSE_per_hour = {}  # RMSE
        RMSE_pct_per_hour = {}
        MBD_per_hour = {}  # MBD
        MBD_pct_per_hour = {}
        FS_per_hour = {}  # FS

        if GENERATE_ERROR_MAP:
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


                frames_pred = model(in_frames)

                # MAE
                MAE_loss = (MAE(frames_pred, out_frames).detach().item() * 100)
                MAE_pct_loss = (MAE_loss / (torch.mean(out_frames[0,0]).cpu().numpy() * 100)) * 100

                MAE_list.append(MAE_loss)

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

                RMSE_list.append(RMSE_loss)
                RMSE_pct_list.append(RMSE_pct_loss)

                if GENERATE_ERROR_MAP:
                    mean_image += out_frames[0,0].cpu().numpy()
                    MAE_error_image += torch.abs(torch.subtract(out_frames[0,0], frames_pred[0,0])).cpu().numpy()
                    RMSE_error_image += torch.square(torch.subtract(out_frames[0,0], frames_pred[0,0])).cpu().numpy()

                if minute < 30:
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
        if GENERATE_ERROR_MAP:

            mean_image = (mean_image / len(val_dataset)) * 1  # contains the mean value of each pixel independently
            MAE_error_image = (MAE_error_image / len(val_dataset))
            MAE_pct_error_image = (MAE_error_image / mean_image) * 100
            RMSE_pct_error_image = (np.sqrt((RMSE_error_image) / len(val_dataset)) / mean_image) * 100
            RMSE_error_image = (np.sqrt((RMSE_error_image) / len(val_dataset))) / 1

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
                bar_max=0.3,
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
                bar_max=0.3,
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

        sorted_keys = sorted(MAE_per_hour.keys(), key=lambda element: (element[0], element[1]))
        hour_list = []

        del model

    if GENERATE_ERROR_MAP:
        fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_pct_maps_together.pdf')
        visualization.error_maps_for_3_horizons(
            error_maps_list=RMSE_pct_maps_list,
            vmax=100,
            fig_name=fig_name,
            save_fig=True,
            colormap='coolwarm'
        )

        fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_maps_together.pdf')
        visualization.error_maps_for_3_horizons(
            error_maps_list=MAE_maps_list,
            vmax=0.3,
            fig_name=fig_name,
            save_fig=True,
            colormap='coolwarm'
        )

        fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_maps_together.pdf')
        visualization.error_maps_for_3_horizons(
            error_maps_list=RMSE_maps_list,
            vmax=0.3,
            fig_name=fig_name,
            save_fig=True,
            colormap='coolwarm'
        )
elif MODEL_TO_EVALUATE == 'GAN':
    
    print('Evaluating GAN')
    OUTPUT_ACTIVATION = 'sigmoid'    
    FILTERS = 64

    evaluate_test = True
    GENERATE_ERROR_MAP = True

    if GENERATE_ERROR_MAP:
        RMSE_pct_maps_list = []
        RMSE_maps_list = []
        MAE_maps_list = []

    for PREDICT_T in PREDICT_T_LIST:
        

        if PREDICT_T == 3:
            PREDICT_HORIZON = '30min'    
        if PREDICT_T == 6:
            PREDICT_HORIZON = '60min'
        if PREDICT_T == 9:
            PREDICT_HORIZON = '90min'
        
        MODEL_PATH = f'/clusteruy/home/franco.mozo/deepCloud/checkpoints/GAN_FINAL_MODELS/wgan_64fil_20eps_ph{PREDICT_HORIZON}.pt'

        model = UNet2(
            n_channels=3, 
            n_classes=1, 
            bilinear=True, 
            bias=False, 
            filters=FILTERS
        ).to(device)
        
        print('Predict Horizon:', PREDICT_HORIZON)
        
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu'))["model_state_dict"])

        CSV_PATH = '/clusteruy/home03/DeepCloud/deepCloud/data/mvd/test_cosangs_mvd.csv'
        PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/test/'
        SAVE_IMAGES_PATH = 'graphs/' + REGION + '/' + PREDICT_HORIZON + '/test/' + 'gan'

        try:
            os.makedirs(SAVE_IMAGES_PATH)
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

        RMSE_pct_list = []
        RMSE_list = []
        MAE_list = []

        MAE_per_hour = {}  # MAE
        MAE_pct_per_hour = {}
        RMSE_per_hour = {}  # RMSE
        RMSE_pct_per_hour = {}
        MBD_per_hour = {}  # MBD
        MBD_pct_per_hour = {}
        FS_per_hour = {}  # FS

        if GENERATE_ERROR_MAP:
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


                frames_pred = model(in_frames)

                # MAE
                MAE_loss = (MAE(frames_pred, out_frames).detach().item() * 100)
                MAE_pct_loss = (MAE_loss / (torch.mean(out_frames[0,0]).cpu().numpy() * 100)) * 100

                MAE_list.append(MAE_loss)

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

                RMSE_list.append(RMSE_loss)
                RMSE_pct_list.append(RMSE_pct_loss)

                if GENERATE_ERROR_MAP:
                    mean_image += out_frames[0,0].cpu().numpy()
                    MAE_error_image += torch.abs(torch.subtract(out_frames[0,0], frames_pred[0,0])).cpu().numpy()
                    RMSE_error_image += torch.square(torch.subtract(out_frames[0,0], frames_pred[0,0])).cpu().numpy()

                if minute < 30:
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
        if GENERATE_ERROR_MAP:

            mean_image = (mean_image / len(val_dataset)) * 1  # contains the mean value of each pixel independently
            MAE_error_image = (MAE_error_image / len(val_dataset))
            MAE_pct_error_image = (MAE_error_image / mean_image) * 100
            RMSE_pct_error_image = (np.sqrt((RMSE_error_image) / len(val_dataset)) / mean_image) * 100
            RMSE_error_image = (np.sqrt((RMSE_error_image) / len(val_dataset))) / 1

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
                bar_max=0.3,
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
                bar_max=0.3,
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

        sorted_keys = sorted(MAE_per_hour.keys(), key=lambda element: (element[0], element[1]))
        hour_list = []

        del model

    if GENERATE_ERROR_MAP:
        fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_pct_maps_together.pdf')
        visualization.error_maps_for_3_horizons(
            error_maps_list=RMSE_pct_maps_list,
            vmax=100,
            fig_name=fig_name,
            save_fig=True,
            colormap='coolwarm'
        )

        fig_name = os.path.join(SAVE_IMAGES_PATH, 'MAE_maps_together.pdf')
        visualization.error_maps_for_3_horizons(
            error_maps_list=MAE_maps_list,
            vmax=0.3,
            fig_name=fig_name,
            save_fig=True,
            colormap='coolwarm'
        )

        fig_name = os.path.join(SAVE_IMAGES_PATH, 'RMSE_maps_together.pdf')
        visualization.error_maps_for_3_horizons(
            error_maps_list=RMSE_maps_list,
            vmax=0.3,
            fig_name=fig_name,
            save_fig=True,
            colormap='coolwarm'
        )
