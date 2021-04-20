# USAGE:
#   Utility functions and functions that perform part of the computation 
#   for functions in other modules. Should be used as little as possible.
#

import datetime
import numpy as np
import csv

import src.lib.preprocessing_functions as pf


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
            if (found_xmin == False and found_xmin_aux == True and np.isnan(image[xmin_aux][ymin_aux+i]) ):
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
    
    M,N = error_array.shape
    fieldnames = []
    fieldnames.append('timestamp')
    for i in range(N):
        #fieldnames.append(str(10*(i+1)) + 'min')
        fieldnames.append(str(10*(i)) + 'min')
    
    with open( filename + '.csv', 'w', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(M):
            row_dict = {}
            row_dict['timestamp'] = time_stamp[i]
            for j in range (N):
                #row_dict[str(10*(j+1)) + 'min']  = error_array[i,j]
                row_dict[str(10*(j)) + 'min']  = error_array[i,j]
            
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
        dtime = datetime.datetime(year =int(img_name[6:10]) ,month = int(img_name[3:5]), day = int(img_name[0:2]), 
                          hour=int(img_name[11:13]), minute=int(img_name[14:16]) ,second=int(img_name[17:]))
        
    cosangs, _ = pf.get_cosangs(dtime, lats, lons)
    
    
    cosangs_thresh = cosangs.copy()
    
    cosangs_thresh[(0 < cosangs) & (cosangs <= 0.15)] = 0.5
    cosangs_thresh[0.15 < cosangs] = 1    
    
    return cosangs, cosangs_thresh  
