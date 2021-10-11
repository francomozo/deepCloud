# USAGE:
#   Utility functions and functions that perform part of the computation
#   for functions in other modules. Should be used as little as possible.
#

import csv
import datetime
import os
import sys
import shutil
from datetime import datetime, timedelta
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
import pandas as pd
import src.lib.preprocessing_functions as pf
import torch

def save_checkpoint(gen_dict, disc_dict, expId, sub_expId=None):

    PATH = f"checkpoints/{expId}/"
    curr_epoch = gen_dict['epoch']

    ts = datetime.now().strftime("%d-%m-%Y_%H:%M")
    if sub_expId is None:
        GEN_NAME = f'{expId}_gen_{ts}.pt'
        DISC_NAME = f'{expId}_disc_{ts}.pt'
    else:
        GEN_NAME = f'{expId}{sub_expId}_gen_{ts}.pt'
        DISC_NAME = f'{expId}{sub_expId}_disc_{ts}.pt'

    torch.save(gen_dict, PATH + GEN_NAME)
    torch.save(disc_dict, PATH + DISC_NAME)
    print("Checkpoints saved.")
    return


def save_gan_checkpoint(gen, disc, opt_gen, opt_disc, curr_epoch, gen_loss, disc_loss, desc):
    # gen_loss and disc_loss are lists with losses for each epoch

    PATH = 'checkpoints/'

    gen_dict = {
        'epoch': curr_epoch,
        'model_state_dict': gen.state_dict(),
        'opt_state_dict': opt_gen.state_dict(),
        'gen_epoch_loss': gen_loss,
    }

    disc_dict = {
            'epoch': curr_epoch,
            'model_state_dict': disc.state_dict(),
            'opt_state_dict': opt_disc.state_dict(),
            'disc_epoch_loss': disc_loss,
    }

    ts = datetime.now().strftime("%d-%m-%Y_%H:%M")
    GEN_NAME = f'gen_{curr_epoch}epochs_{desc}_{ts}.pt'
    DISC_NAME = f'disc_{curr_epoch}epochs_{desc}_{ts}.pt'

    torch.save(gen_dict, PATH + GEN_NAME)
    torch.save(disc_dict, PATH + DISC_NAME)
    print("Checkpoints saved.")
    return


def gradient_penalty(disc, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate disc scores
    mixed_scores = disc(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def datetime2str(datetime_obj):
    """ 
        Receives a datetime object and returns a string
        in format 'day/month/year hr:mins:secs' 
    """

    return datetime_obj.strftime('%d/%m/%Y %H:%M:%S')


def str2datetime(date_str):
    """
        Receives a string with format 'day/month/year hr:mins:secs'
        and returns a datetime object
    """
    date, time = date_str.split()
    day, month, year = date.split('/')
    hr, mins, secs = time.split(':')
    return datetime.datetime(int(year), int(month), int(day),
                             int(hr), int(mins), int(secs)
                             )


def find_inner_image(image):
    """
        Receives and image with some values equal to np.nan
        and returns a window with no np.nans
    """
    step = 5
    found_xmin = found_ymin = found_xmax = found_ymax = False
    xmin = ymin = xmax = ymax = 0
    while(not (found_xmin and found_ymin and found_xmax and found_ymax)):
        range_x = len(image[0]) - (xmin+xmax)
        range_y = len(image[0]) - (ymin+ymax)
        found_xmin_aux = found_ymin_aux = found_xmax_aux = found_ymax_aux = True
        ymin_aux, ymax_aux, xmin_aux, xmax_aux = ymin, ymax, xmin, xmax
        for i in range(range_x):
            if (found_ymin == False and found_ymin_aux == True and np.isnan(image[xmin_aux+i][ymin_aux])):
                ymin += step
                found_ymin_aux = False
            if (found_ymax == False and found_ymax_aux == True and np.isnan(image[-(xmax_aux+i+1)][-(ymax_aux+1)])):
                ymax += step
                found_ymax_aux = False
            if (i == range_x-1):
                found_ymin = found_ymin_aux
                found_ymax = found_ymax_aux

        for i in range(range_y):
            if (found_xmin == False and found_xmin_aux == True and np.isnan(image[xmin_aux][ymin_aux+i])):
                xmin += step
                found_xmin_aux = False
            if (found_xmax == False and found_xmax_aux == True and np.isnan(image[-(xmax_aux+1)][-(ymax_aux+i+1)])):
                xmax += step
                found_xmax_aux = False
            if (i == range_y-1):
                found_xmin = found_xmin_aux
                found_xmax = found_xmax_aux

    return xmin, xmax, ymin, ymax


def save_errorarray_as_csv(error_array, time_stamp, filename):
    """ Generates a CSV file with the error of the predictions at the different times of the day

    Args:
        error_array (array): Array containing the values of the error of a prediction
        time_stamp (list): Contains the diferent timestamps of the day
        filename (string): path and name of the generated file
    """

    M, N = error_array.shape
    fieldnames = []
    fieldnames.append('timestamp')
    for i in range(N):
        #fieldnames.append(str(10*(i+1)) + 'min')
        fieldnames.append(str(10*(i)) + 'min')

    with open(filename + '.csv', 'w', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(M):
            row_dict = {}
            row_dict['timestamp'] = time_stamp[i]
            for j in range(N):
                #row_dict[str(10*(j+1)) + 'min']  = error_array[i,j]
                row_dict[str(10*(j)) + 'min'] = error_array[i, j]

            writer.writerow(row_dict)


def get_cosangs_mask(meta_path='data/meta',
                     img_name='ART_2020020_111017.FR'
                     ):
    """ Returns zenithal cos from img_name, with and whitout threshold

    Args:
        meta_path (str, optional): Defaults to 'data/meta'.
        img_name (str, optional): 'ART....FR' or 'dd/mm/yyyy hh:mm:ss'
    """

    lats, lons = pf.read_meta(meta_path)

    if (img_name[0:3] == 'ART'):
        dtime = pf.get_dtime(img_name)
    else:
        dtime = datetime.datetime(year=int(img_name[6:10]), month=int(img_name[3:5]), day=int(img_name[0:2]),
                                  hour=int(img_name[11:13]), minute=int(img_name[14:16]), second=int(img_name[17:]))

    cosangs, _ = pf.get_cosangs(dtime, lats, lons)

    cosangs_thresh = cosangs.copy()

    cosangs_thresh[(0 < cosangs) & (cosangs <= 0.15)] = 0.5
    cosangs_thresh[0.15 < cosangs] = 1

    return cosangs, cosangs_thresh


def print_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    # f = r - a
    
    print('Memory Usage:')
    print(f'\t Total:                  {(t/1024**2):.3f} MB.')
    print(f'\t Reserved (cached):      {(r/1024**2):.3f} MB.')
    print(f'\t Allocated:              {(a/1024**2):.3f} MB.')
    # print(f'\t Free (inside reserved): {(f/1024**2):.3f} MB.')
    return

  # free inside reserved


def get_last_checkpoint(path):
    checkpoints = os.listdir(path)
    epochs = [int(epoch)
              for cp in checkpoints for epoch in re.findall(r'\d+', cp)]
    last_epoch = max(epochs)
    return last_epoch

def image_sequence_generator(path, in_channel,out_channel, min_time_diff, max_time_diff, csv_path):
    """Recieves a folder with images named as ART_2020XXX_hhmmss.npy and it generates a csv file with the 
    available sequences of a specified length.

    Args:
        path (str): path to folder containing images
        in_channel (int): Quantity of input images in sequence
        out_channel (int): Quantity of output images in sequence
        min_time_diff (int): Images separated by less than this time cannot be a sequence
        max_time_diff (int): Images separated by more than this time cannot be a sequence
        csv_path (int): Path and name og generated csv. 
    """    
     #file names
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    
    days_list = []
    dict_day = {}

    for i in range(len(onlyfiles)):
        day = int(onlyfiles[i][8:11]) #dia el anio
        if day not in(days_list):
            days_list.append(day)
        if day in dict_day.keys():
            dict_day[day].append(onlyfiles[i])
        else:
            dict_day[day] = []
            dict_day[day].append(onlyfiles[i])
    
    dt_min =timedelta(minutes = min_time_diff)
    dt_max =timedelta(minutes = max_time_diff)

    with open(csv_path, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        days_in_folder = sorted(dict_day.keys())
        for day in days_in_folder: #recorro cada dia por separado
            
            len_day = len(dict_day[day]) 
            day_images_sorted = sorted(dict_day[day])
            for i in range(len_day - (in_channel+out_channel)): #me fijo si puedo completar un conjunto de datos
                complete_seq = True
                image_sequence = []
                for j in range(in_channel+out_channel-1): #veo si puede rellenar un dato
                    if complete_seq:
                        dt_i = datetime(1997,5,28,hour = int(day_images_sorted[i+j][12:14]), 
                                        minute = int(day_images_sorted[i+j][14:16]), 
                                        second = int(day_images_sorted[i+j][16:18]) )
                        dt_f = datetime(1997,5,28,hour = int(day_images_sorted[i+j+1][12:14]), 
                                        minute = int(day_images_sorted[i+j+1][14:16]), 
                                        second = int(day_images_sorted[i+j+1][16:18]) )
                        
                        time_diff = dt_f - dt_i
                        
                        if  dt_min < time_diff < dt_max: #las imagenes estan bien espaciadas en el tiempo
                            if j == 0:
                                image_sequence.append(day_images_sorted[i+j])
                                image_sequence.append(day_images_sorted[i+j+1])
                            if j>0:
                                image_sequence.append(day_images_sorted[i+j+1])          
                        else:
                            complete_seq = False
                        
                if complete_seq: 
                    writer.writerow(image_sequence)

def image_sequence_generator_folders(path, in_channel,out_channel, min_time_diff, max_time_diff, csv_path):
    """Recieves a folder with images named as ART_2020XXX_hhmmss.npy and it generates a csv file with the 
    available sequences of a specified length.

    Args:
        path (str): path to folder containing images '/train'
        in_channel (int): Quantity of input images in sequence
        out_channel (int): Quantity of output images in sequence
        min_time_diff (int): Images separated by less than this time cannot be a sequence
        max_time_diff (int): Images separated by more than this time cannot be a sequence
        csv_path (int): Path and name og generated csv. 
    """    
    #folder names
    folders = os.listdir(path)
    
    dt_min =timedelta(minutes = min_time_diff)
    dt_max =timedelta(minutes = max_time_diff)
    
    with open(csv_path, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        for folder in folders:
            folderfiles = sorted([f for f in listdir(join(path,folder)) if isfile(join(path,folder, f))])
            len_day = len(folderfiles)
            for i in range(len_day - (in_channel+out_channel)): #me fijo si puedo completar un conjunto de datos
                complete_seq = True
                image_sequence = []
                for j in range(in_channel+out_channel-1): #veo si puede rellenar un dato
                    if complete_seq:
                        dt_i = datetime(1997,5,28,hour = int(folderfiles[i+j][12:14]), 
                                        minute = int(folderfiles[i+j][14:16]), 
                                        second = int(folderfiles[i+j][16:18]) )
                        dt_f = datetime(1997,5,28,hour = int(folderfiles[i+j+1][12:14]), 
                                        minute = int(folderfiles[i+j+1][14:16]), 
                                        second = int(folderfiles[i+j+1][16:18]) )
                        
                        time_diff = dt_f - dt_i
                        
                        if  dt_min < time_diff < dt_max: #las imagenes estan bien espaciadas en el tiempo
                            if j == 0:
                                image_sequence.append(folderfiles[i+j])
                                image_sequence.append(folderfiles[i+j+1])
                            if j>0:
                                image_sequence.append(folderfiles[i+j+1])          
                        else:
                            complete_seq = False
                        
                if complete_seq: 
                    writer.writerow(image_sequence)
    
def image_sequence_generator_folders_cosangs(path, in_channel, out_channel, min_time_diff, max_time_diff, csv_path, 
                                             folders=None,
                                             meta_path='/clusteruy/home03/DeepCloud/deepCloud/data/raw/meta',
                                             region=None):
    """Recieves a folder with images named as ART_2020XXX_hhmmss.npy and it generates a csv file with the 
    available sequences of a specified length. Images from Dawn/Dusk are not included.

    Args:
        path (str): path to folder containing images '/train'
        in_channel (int): Quantity of input images in sequence
        out_channel (int): Quantity of output images in sequence
        min_time_diff (int): Images separated by less than this time cannot be a sequence
        max_time_diff (int): Images separated by more than this time cannot be a sequence
        csv_path (int): Path and name og generated csv. 
        folders (list): Containing the folders for generating the sequences. If None, whole train is used
    """    
    #folder names
    if folders is None:
        folders = os.listdir(path)
    
    dt_min =timedelta(minutes = min_time_diff)
    dt_max =timedelta(minutes = max_time_diff)
    
    with open(csv_path, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        for folder in folders:
            print(folder)
            folderfiles = sorted([f for f in listdir(join(path,folder)) if isfile(join(path,folder, f))])
            len_day = len(folderfiles)
            for i in range(len_day - (in_channel+out_channel)): #me fijo si puedo completar un conjunto de datos
                complete_seq = True
                image_sequence = []
                for j in range(in_channel+out_channel-1): #veo si puede rellenar un dato
                    if complete_seq:
                        dt_i = datetime(1997,5,28,hour = int(folderfiles[i+j][12:14]), 
                                        minute = int(folderfiles[i+j][14:16]), 
                                        second = int(folderfiles[i+j][16:18]) )
                        dt_f = datetime(1997,5,28,hour = int(folderfiles[i+j+1][12:14]), 
                                        minute = int(folderfiles[i+j+1][14:16]), 
                                        second = int(folderfiles[i+j+1][16:18]) )
                        
                        time_diff = dt_f - dt_i

                        #si la primera o ultima imagen de la secuencia no cumple con los cosangs se descarta la sec
                        if j == 0 or j == in_channel+out_channel-2:
                            aux = 0
                            if j == in_channel+out_channel-2:
                                aux = 1
                            _, cosangs_thresh = get_cosangs_mask(meta_path=meta_path,
                                            img_name=folderfiles[i+j+aux])
                            if region is None:
                                img = cosangs_thresh[1550:1550+256, 1600:1600+256] # cut montevideo
                            elif region == 'uru':
                                img = cosangs_thresh[1205:1205+512, 1450:1450+512]
                            elif region == 'region3':
                                img = cosangs_thresh[800:800+1024, 1250:1250+1024]
                            if (np.mean(img) != 1.0):
                                complete_seq = False
                        
                        if  dt_min < time_diff < dt_max: #las imagenes estan bien espaciadas en el tiempo
                            if j == 0:
                                image_sequence.append(folderfiles[i+j])
                                image_sequence.append(folderfiles[i+j+1])
                            if j>0:
                                image_sequence.append(folderfiles[i+j+1])          
                        else:
                            complete_seq = False
                
                if complete_seq: 
                    writer.writerow(image_sequence)

def sequence_df_generator_folders(path, in_channel,out_channel, min_time_diff, max_time_diff):
    """Generates DataFrame that contains all possible sequences for images separated by day

    Args:
        path (str): path to folder containing images '/train'
        in_channel (int): Quantity of input images in sequence
        out_channel (int): Quantity of output images in sequence
        min_time_diff (int): Images separated by less than this time cannot be a sequence
        max_time_diff (int): Images separated by more than this time cannot be a sequence

    Returns:
        [pd.DataFrame]: Rows contain all sequences
    """    
    #folder names
    folders = os.listdir(path)
    
    dt_min =timedelta(minutes = min_time_diff)
    dt_max =timedelta(minutes = max_time_diff)
    
    sequences_df = []

    for folder in folders:
        folderfiles = sorted([f for f in listdir(join(path,folder)) if isfile(join(path,folder, f))])
        len_day = len(folderfiles)
        for i in range(len_day - (in_channel+out_channel-1)): #me fijo si puedo completar un conjunto de datos
            complete_seq = True
            image_sequence = []
            for j in range(in_channel+out_channel-1): #veo si puede rellenar un dato
                if complete_seq:
                    dt_i = datetime(1997,5,28,hour = int(folderfiles[i+j][12:14]), 
                                    minute = int(folderfiles[i+j][14:16]), 
                                    second = int(folderfiles[i+j][16:18]) )
                    dt_f = datetime(1997,5,28,hour = int(folderfiles[i+j+1][12:14]), 
                                    minute = int(folderfiles[i+j+1][14:16]), 
                                    second = int(folderfiles[i+j+1][16:18]) )
                    
                    time_diff = dt_f - dt_i
                    
                    if  dt_min < time_diff < dt_max: #las imagenes estan bien espaciadas en el tiempo
                        if j == 0:
                            image_sequence.append(folderfiles[i+j])
                            image_sequence.append(folderfiles[i+j+1])
                        if j>0:
                            image_sequence.append(folderfiles[i+j+1])          
                    else:
                        complete_seq = False
                    
            if complete_seq: 
                sequences_df.append(image_sequence)
                
    sequences_df = pd.DataFrame(sequences_df)
    return sequences_df

def sequence_df_generator(path, in_channel,out_channel, min_time_diff, max_time_diff, csv_path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    days_list = []
    dict_day = {}
    for i in range(len(onlyfiles)):
        day = int(onlyfiles[i][8:11]) #dia el anio
        if day not in(days_list):
            days_list.append(day)
        if day in dict_day.keys():
            dict_day[day].append(onlyfiles[i])
        else:
            dict_day[day] = []
            dict_day[day].append(onlyfiles[i])
    dt_min =timedelta(minutes = min_time_diff)
    dt_max =timedelta(minutes = max_time_diff)
    sequences_df = []    
    days_in_folder = sorted(dict_day.keys())
    for day in days_in_folder: #recorro cada dia por separado
        len_day = len(dict_day[day]) 
        day_images_sorted = sorted(dict_day[day])
        for i in range(len_day - (in_channel+out_channel)): #me fijo si puedo completar un conjunto de datos
            complete_seq = True
            image_sequence = []
            for j in range(in_channel+out_channel-1): #veo si puede rellenar un dato
                if complete_seq:
                    dt_i = datetime(1997,5,28,hour = int(day_images_sorted[i+j][12:14]), 
                                    minute = int(day_images_sorted[i+j][14:16]), 
                                    second = int(day_images_sorted[i+j][16:18]) )
                    dt_f = datetime(1997,5,28,hour = int(day_images_sorted[i+j+1][12:14]), 
                                    minute = int(day_images_sorted[i+j+1][14:16]), 
                                    second = int(day_images_sorted[i+j+1][16:18]) )
                    time_diff = dt_f - dt_i
                    if  dt_min < time_diff < dt_max: #las imagenes estan bien espaciadas en el tiempo
                        if j == 0:
                            image_sequence.append(day_images_sorted[i+j])
                            image_sequence.append(day_images_sorted[i+j+1])
                        if j>0:
                            image_sequence.append(day_images_sorted[i+j+1])          
                    else:
                        complete_seq = False
            if complete_seq: 
                sequences_df.append(image_sequence)       
    sequences_df = pd.DataFrame(sequences_df)
    return sequences_df

def data_separator_by_folder(data_path):
    """Takes a folder with images ART_2020XXX_hhmmss.npy and it separates it in folders named 2020XXX

    Args:
        data_path (string): path to folder that contains images /../
    """    

    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    dict_day = {}

    for i in range(len(onlyfiles)):
        day = int(onlyfiles[i][8:11]) #dia el anio

        if day in dict_day.keys():
            dict_day[day].append(onlyfiles[i])
        else:
            dict_day[day] = []
            dict_day[day].append(onlyfiles[i])
            
    print('El dataset de mvd tiene',len(dict_day.keys()), 'dias')

    for day in dict_day.keys():
        first = True
        for filename in dict_day[day]:
            if first:
                try:
                    os.mkdir(data_path+ filename[4:11])
                except OSError:
                    print ("Creation of the directory failed" )
                else:
                    print ("Successfully created the directory ")
                first = False
            shutil.move(data_path + filename, data_path+ filename[4:11]+ '/')

def clear_lines(num_lines):
    for _ in range(num_lines):
        sys.stdout.write("\033[F") #back to previous line 
        sys.stdout.write("\033[K") #clear line 

def save_pickle_dict(path='reports/model_training', name='', dict_=None):
    with open(os.path.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(dict_, f, pickle.HIGHEST_PROTOCOL) 

def load_pickle_dict(path='reports/model_training', name=''):
    with open(os.path.join(path, name + '.pkl'), 'rb') as f:
        return pickle.load(f)
