import pandas as pd
import datetime as datetime
import numpy as np


#Podria estar bueno que le entre tambien el time stamp de la imagen asi se lo agrega a las predicciones

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

def gauss_persitence (image, img_timestamp, predict_horizon, sigma):
    """Takes an image adds gaussanian noise and uses it as the prediction for the next time stamps

    Args:
        image (array): Image used as prediction
        predict_horizon (int): Length of the prediction horizon. 
        sigma (int): standard deviation of the gauss noise

    Returns:
        [list]: list containing predictions
    """    
    M,N = image.shape
    
    predictions = []
    
    for i in range(predict_horizon): #la i no se usa, ver mejor opcion
        noisy_pred = np.clip(image + np.random.normal(0,sigma,(M,N)), 0,255)
        predictions.append(noisy_pred)
        
    predict_timestamp = pd.date_range(start = img_timestamp+datetime.timedelta(minutes = 10),
                                      periods= predict_horizon, freq = '10min')

    return predictions , predict_timestamp