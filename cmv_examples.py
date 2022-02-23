import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

from src.lib.latex_options import Colors, Linestyles
from src import data, evaluate, model, preprocessing, visualization
########################################################################################
dataset = 'region3'
REGION = 'R3'

PATH_DATA = '/clusteruy/home03/DeepCloud/deepCloud/data/' + dataset + '/validation/'
SAVE_IMAGES_PATH = 'graphs/' + REGION + '/CMV'

CMV = model.Cmv2()

#########################################################################################
# EXAMPLE DAY 77, 60 minutes
img1 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_140018.npy'))
img2 = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_141018.npy'))
output = np.load(os.path.join(PATH_DATA, '2020077/ART_2020077_151018.npy'))
time_list = ['14:00', '14:10', '15:10']

frames_pred = CMV.predict(imgi=img1,
                            imgf=img2,
                            imgf_ts = None, 
                            period=10*60, delta_t=10*60, 
                            predict_horizon=6)

frames_array = np.array([img1, img2, output, frames_pred[-1]])
fig_name = os.path.join(SAVE_IMAGES_PATH, 'most_moved_sequence_day77_60min.pdf')
visualization.show_seq_and_pred(frames_array,
                                time_list=time_list,
                                prediction_t=6,
                                fig_name=fig_name, save_fig=True)

# EXAMPLE DAY 61, 180 minutes
img1 = np.load(os.path.join(PATH_DATA, '2020061/ART_2020061_111017.npy'))
img2 = np.load(os.path.join(PATH_DATA, '2020061/ART_2020061_112017.npy'))
output = np.load(os.path.join(PATH_DATA, '2020061/ART_2020061_142017.npy'))
time_list = ['11:10', '11:20', '14:20']

frames_pred = CMV.predict(imgi=img1,
                            imgf=img2,
                            imgf_ts = None, 
                            period=10*60, delta_t=10*60, 
                            predict_horizon=18)

frames_array = np.array([img1, img2, output, frames_pred[-1]])
fig_name = os.path.join(SAVE_IMAGES_PATH, 'day61_180min.pdf')
visualization.show_seq_and_pred(frames_array,
                                time_list=time_list,
                                prediction_t=18,
                                fig_name=fig_name, save_fig=True)


# EXAMPLE DAY 365, 300 minutes
img1 = np.load(os.path.join(PATH_DATA, '2020365/ART_2020365_134021.npy'))
img2 = np.load(os.path.join(PATH_DATA, '2020365/ART_2020365_135021.npy'))
output = np.load(os.path.join(PATH_DATA, '2020365/ART_2020365_185021.npy'))
time_list = ['13:40', '13:50', '18:50']

frames_pred = CMV.predict(imgi=img1,
                            imgf=img2,
                            imgf_ts = None, 
                            period=10*60, delta_t=10*60, 
                            predict_horizon=30)

frames_array = np.array([img1, img2, output, frames_pred[-1]])
fig_name = os.path.join(SAVE_IMAGES_PATH, 'day356_300min.pdf')
visualization.show_seq_and_pred(frames_array,
                                time_list=time_list,
                                prediction_t=30,
                                fig_name=fig_name, save_fig=True)


# EXAMPLE DAY 61, 120 minutes
img1 = np.load(os.path.join(PATH_DATA, '2020061/ART_2020061_122017.npy'))
img2 = np.load(os.path.join(PATH_DATA, '2020061/ART_2020061_123017.npy'))
output = np.load(os.path.join(PATH_DATA, '2020061/ART_2020061_143017.npy'))
time_list = ['12:20', '12:30', '14:30']

frames_pred = CMV.predict(imgi=img1,
                            imgf=img2,
                            imgf_ts = None, 
                            period=10*60, delta_t=10*60, 
                            predict_horizon=12)

frames_array = np.array([img1, img2, output, frames_pred[-1]])
fig_name = os.path.join(SAVE_IMAGES_PATH, 'day61_120min.pdf')
visualization.show_seq_and_pred(frames_array,
                                time_list=time_list,
                                prediction_t=12,
                                fig_name=fig_name, save_fig=True)
