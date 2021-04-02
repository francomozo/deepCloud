import pandas as pd
import datetime as datetime
import numpy as np
import cv2 as cv


def persitence (image, img_timestamp, predict_horizon):
    """Takes an image and uses it as the prediction for the next time stamps

    Args:
        image (array): Image used as prediction
        predict_horizon (int): Length of the prediction horizon. 

    Returns:
        [list]: list containing precitions
    """    
    
    predictions = [np.array(image) for i in range(predict_horizon)]
    predict_timestamp = pd.date_range(start = img_timestamp+datetime.timedelta(minutes = 10),
                                      periods= predict_horizon, freq = '10min')

    return predictions , predict_timestamp

def noisy_persitence (image, img_timestamp, predict_horizon, sigma):
    """Takes an image adds gaussanian noise and uses it as the prediction for the next time stamps. 
    Used only to have another model for the bar chart in visualization. 

    Args:
        image (array): Image used as prediction
        predict_horizon (int): Length of the prediction horizon. 
        sigma (int): standard deviation of the gauss noise

    Returns:
        [list]: list containing predictions
    """    
    M,N = image.shape
    
    predictions = []
    
    for _ in range(predict_horizon): 
        noisy_pred = np.clip(image + np.random.normal(0,sigma,(M,N)), 0,255)
        predictions.append(noisy_pred)
        
    predict_timestamp = pd.date_range(start = img_timestamp+datetime.timedelta(minutes = 10),
                                      periods= predict_horizon, freq = '10min')

    return predictions , predict_timestamp

def blurred_persitence (image, img_timestamp, predict_horizon, kernel_size = (5,5)):
    """Takes an image and blurs it with a gaussanian window and uses it as the prediction for the next time stamps. 

    Args:
        image (array): Image used as prediction
        predict_horizon (int): Length of the prediction horizon. 
        kernel_size (tuple): size of kernel

    Returns:
        [list]: list containing predictions
    """    

    predictions = []
    
    for _ in range(predict_horizon): 
        blurred_pred = cv.GaussianBlur(image,kernel_size, 0 )
        predictions.append(blurred_pred)
        image = blurred_pred
        
    predict_timestamp = pd.date_range(start = img_timestamp+datetime.timedelta(minutes = 10),
                                      periods= predict_horizon, freq = '10min')

    return predictions , predict_timestamp