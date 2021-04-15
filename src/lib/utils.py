# USAGE:
#   Utility functions and functions that perform part of the computation 
#   for functions in other modules. Should be used as little as possible.
#

import datetime
import numpy as np

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
    step = 5
    found_xmin = found_ymin = found_xmax = found_ymax = False
    xmin = ymin = xmax = ymax = 0
    while(not (found_xmin and found_ymin and found_xmax and found_ymax)):
        range_x = len(image[0]) - (xmin+xmax)
        range_y = len(image[0]) - (ymin+ymax)
        found_xmin_aux = found_ymin_aux = found_xmax_aux = found_ymax_aux = True
        for i in range(range_x):
            if (found_ymin == False and found_ymin_aux == True and np.isnan(image[xmin+i][ymin])):
                ymin += step
                found_ymin_aux = False
            if (found_ymax == False and found_ymax_aux == True and np.isnan(image[-(xmax+i+1)][-(ymax+1)])):
                ymax += step
                found_ymax_aux = False
            if (i == range_x-1):
                found_ymin = found_ymin_aux
                found_ymax = found_ymax_aux

        for i in range(range_y):
            if (found_xmin == False and found_xmin_aux == True and np.isnan(image[xmin][ymin+i]) ):
                xmin += step
                found_xmin_aux = False
            if (found_xmax == False and found_xmax_aux == True and np.isnan(image[-(xmax+1)][-(ymax+i+1)])):
                xmax += step
                found_xmax_aux = False
            if (i == range_y-1):
                found_xmin = found_xmin_aux
                found_xmax = found_xmax_aux
                
    return xmin, xmax, ymin, ymax
