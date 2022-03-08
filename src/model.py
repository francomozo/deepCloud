# USAGE:
#   ML Models, Persistence, State of the Art models implementations.
#

import pandas as pd
import datetime as datetime
import numpy as np
import cv2 as cv
import yaml
import torch
from src import evaluate

class Persistence:
    """ Class that predicts the next images using naive prediction.
    """    
    def predict(self, image, predict_horizon, img_timestamp=None, predict_direct=False):
        """Takes an image and predicts the next images on the predict_horizon depending on class instance
            normal: identical image
            noisy: adds gaussanian noise 
            blurred: blurs it with a gaussanian window

        Args:
            image (array): Image used as prediction
            img_timestamp (datetime): time stamp of image
            predict_horizon (int): Length of the prediction horizon.
            predict_direct (bool): True to return only the last prediction

        Returns:
            [Numpy array], [list]: Array containing preditions and list containing timestamps
        """   
        if torch.is_tensor(image): 
            image = image.numpy()
                
        predictions = []
        M,N = image.shape

        for i in range(predict_horizon): 
            if (isinstance(self, NoisyPersistence)):
                predictions.append(np.clip(image + np.random.normal(0,self.sigma,(M,N)), 0,255))
            elif (isinstance(self, BlurredPersistence)):
                if self.kernel_size_list:
                    kernel_size = self.kernel_size_list[i]
                else:
                    kernel_size = self.kernel_size
                blurred_pred = cv.GaussianBlur(image, kernel_size, 0)
                predictions.append(blurred_pred)
                if not self.kernel_size_list:
                    image = blurred_pred
            else:
                predictions.append(np.array(image))

        if predict_direct:
            return np.array(predictions[-1])[np.newaxis]
        else:
            if img_timestamp is not None:
                predict_timestamp = pd.date_range(start = img_timestamp,
                                            periods= predict_horizon+1, freq = '10min')
                return np.array(predictions), predict_timestamp
            else:
                return np.array(predictions)

class NoisyPersistence(Persistence):
    """Sub class of Persistence, adds white noise to predictions.

    Args:
        Persistence ([type]): [description]
    """    
    def __init__(self, sigma):
        #sigma (int): standard deviation of the gauss noise
        self.sigma = sigma
        
class BlurredPersistence(Persistence):
    """Sub class of Persistence, returns predictions after passign through a gauss filter.

    Args:
        Persistence ([type]): [description]
    """    
    def __init__(self, kernel_size = (0,0), kernel_size_list = None):
        #kernel_size (tuple): size of kernel
        self.kernel_size = kernel_size
        self.kernel_size_list = kernel_size_list

class Cmv:
    def __init__(self, kernel_size = (0,0), kernel_size_list = None, dcfg=None):
        # Load configuration
        if dcfg is None:
            stream = open("les-prono/admin_scripts/config.yaml", 'r')
            self.dcfg = yaml.load(stream, yaml.FullLoader)  # dict
        else:
            self.dcfg = dcfg
        self.kernel_size = kernel_size
        self.kernel_size_list = kernel_size_list

    def predict(self, imgi, imgf,period, delta_t, predict_horizon, imgf_ts=None, predict_direct=False):
        """Predicts next image using openCV optical Flow

        Args:
            imgi (numpy.ndarray): first image used for prediction
            imgf (numpy.ndarray): last image used for prediction
            period (int): time difference between imgi and imgf in seconds
            delta_t (int): time passed between imgf and predicted image in seconds
            predict_horizon (int): Length of the prediction horizon (Cuantity of images returned)
            predict_direct (bool): True to return only the last prediction

        Returns:
            [Numpy array]: Numpy array with predicted images
        """    
        
        if torch.is_tensor(imgi): 
            imgi = imgi.numpy()
            imgf = imgf.numpy()
            
        #get_cmv (dcfg, imgi,imgf, period)
        cmvcfg = self.dcfg["algorithm"]["cmv"]
        pyr_scale=cmvcfg["pyr_scale"]
        levels=cmvcfg["levels"]
        winsize=cmvcfg["winsize"]
        iterations=cmvcfg["iterations"]
        poly_n=cmvcfg["poly_n"]
        poly_sigma=cmvcfg["poly_sigma"]
 
        flow = cv.calcOpticalFlowFarneback(
            imgi,
            imgf,
            None,
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )
        cmv = - flow / period

        #get_mapping(cmv, delta_t)
        base_img = imgf  #base_img imagen a la que le voy a aplicar el campo
        predictions = []

        for i in range(predict_horizon):
            i_idx, j_idx = np.meshgrid(
                np.arange(cmv.shape[1]), np.arange(cmv.shape[0])
            )
            if (isinstance(self, Cmv1)):
                # img(t) + k*cmv estático -> img(t+k)
                map_i = i_idx + cmv[:, :, 0] * (delta_t * (i+1)) #cmv[...] * x , donde x es la cantidad de segundos hacia adelante
                map_j = j_idx + cmv[:, :, 1] * (delta_t * (i+1))
            elif (isinstance(self, Cmv2)):
                # img(t+k) + cmv estático -> img(t+k+1)
                map_i = i_idx + cmv[:, :, 0] * delta_t 
                map_j = j_idx + cmv[:, :, 1] * delta_t 
            map_x, map_y = map_i.astype(np.float32), map_j.astype(np.float32)

            #project_cmv(cmv, base_img, delta_t, show_for_debugging=False)
            #map_x, map_y = get_mapping(cmv, delta_t)
            next_img = cv.remap(
                base_img,
                map_x,
                map_y,
                cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=np.nan,  # valor que se agrega al mover los bordes
                #borderValue=0,
            )
            if self.kernel_size[0] == 0 and self.kernel_size[1] == 0 and not self.kernel_size_list:
                predictions.append(next_img)
            else: #add blur to prediction
                next_img_aux = np.copy(next_img)
                aux = np.ones_like(next_img_aux)
                aux[np.isnan(next_img_aux)]=np.nan
                next_img_aux[np.isnan(next_img_aux)]=0
                if self.kernel_size_list:
                    kernel_size = self.kernel_size_list[i]
                else:
                    kernel_size = self.kernel_size
                blurred_pred = cv.GaussianBlur(next_img_aux, kernel_size, 0)
                blurred_pred = blurred_pred * aux
                predictions.append(blurred_pred)
                
            
            # if show_for_debugging:
            #     show_for_debugging2(base_img, next_img, cmv, delta_t)
        
            if (isinstance(self, Cmv2)):
                base_img = next_img

        if predict_direct:
            return np.array(predictions[-1])[np.newaxis]
        else:
            if imgf_ts is not None:
                predict_timestamp = pd.date_range(start = imgf_ts,
                                            periods= predict_horizon+1, freq = str(delta_t//60) +'min') 
                return np.array(predictions), predict_timestamp
            else:
                return np.array(predictions)
        
    def predict_optimize_blur(self, imgi, imgf,period, delta_t, predict_horizon, imgf_ts=None,
                             start_blur=7, range_blur=75, gt=None, errors_blurred_cmv_dict=None):
        """Optimization of Kernel size for blurring

        Args:
            imgi (numpy.ndarray): first image used for prediction
            imgf (numpy.ndarray): last image used for prediction
            period (int): time difference between imgi and imgf in seconds
            delta_t (int): time passed between imgf and predicted image in seconds
            predict_horizon (int): Length of the prediction horizon (Cuantity of images returned)
            start_blur (int): Kernel size to start optimization
            range_blur (int): Amount of kernel sizes to test 
            gt: Grount trtth to evaluate predictions 
            errors_blurred_cmv_dict (dictionary): Dictionary to save errors for each kernel size

        Returns:
            [dictionary]: Dictionary with error for each kernel size
        """    

        if torch.is_tensor(imgi): 
            imgi = imgi.numpy()
            imgf = imgf.numpy()

        #get_cmv (dcfg, imgi,imgf, period)
        cmvcfg = self.dcfg["algorithm"]["cmv"]
        pyr_scale=cmvcfg["pyr_scale"]
        levels=cmvcfg["levels"]
        winsize=cmvcfg["winsize"]
        iterations=cmvcfg["iterations"]
        poly_n=cmvcfg["poly_n"]
        poly_sigma=cmvcfg["poly_sigma"]

        flow = cv.calcOpticalFlowFarneback(
            imgi,
            imgf,
            None,
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )
        cmv = - flow / period

        #get_mapping(cmv, delta_t)
        base_img = imgf  #base_img imagen a la que le voy a aplicar el campo

        for i in range(predict_horizon):
            i_idx, j_idx = np.meshgrid(
                np.arange(cmv.shape[1]), np.arange(cmv.shape[0])
            )
            if (isinstance(self, Cmv1)):
                # img(t) + k*cmv estático -> img(t+k)
                map_i = i_idx + cmv[:, :, 0] * (delta_t * (i+1)) #cmv[...] * x , donde x es la cantidad de segundos hacia adelante
                map_j = j_idx + cmv[:, :, 1] * (delta_t * (i+1))
            elif (isinstance(self, Cmv2)):
                # img(t+k) + cmv estático -> img(t+k+1)
                map_i = i_idx + cmv[:, :, 0] * delta_t 
                map_j = j_idx + cmv[:, :, 1] * delta_t 
            map_x, map_y = map_i.astype(np.float32), map_j.astype(np.float32)

            #project_cmv(cmv, base_img, delta_t, show_for_debugging=False)
            #map_x, map_y = get_mapping(cmv, delta_t)
            next_img = cv.remap(
                base_img,
                map_x,
                map_y,
                cv.INTER_LINEAR,
                borderMode=cv.BORDER_CONSTANT,
                borderValue=np.nan,  # valor que se agrega al mover los bordes
                #borderValue=0,
            )
            if (isinstance(self, Cmv2)):
                base_img = next_img

        aux_nans = np.ones_like(next_img)
        aux_nans[np.isnan(next_img)]=np.nan
        for i in range(range_blur):
            j = start_blur+i*8
            #key = f'{j}x{j}'
            blurred_pred = np.copy(next_img)
            blurred_pred[np.isnan(blurred_pred)]=0
            blurred_pred = cv.GaussianBlur(blurred_pred, (j,j), 0)
            blurred_pred = blurred_pred * aux_nans
            predict_errors = evaluate.evaluate_image(
                                predictions = np.array([blurred_pred]), 
                                gt = gt, 
                                gt_ts = None,
                                metric='RMSE', dynamic_window=False,
                                evaluate_day_pixels = False, 
                                error_percentage = True)
            predict_errors = np.array(predict_errors)
            if errors_blurred_cmv_dict.get(j) is not None:
                errors_blurred_cmv_dict[j] += predict_errors[-1]
            else:
                errors_blurred_cmv_dict[j] = 0 
        return errors_blurred_cmv_dict


class Cmv1(Cmv):
    pass

class Cmv2(Cmv):
    pass


def persistence(image, img_timestamp, predict_horizon):
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

def noisy_persistence(image, img_timestamp, predict_horizon, sigma):
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

def blurred_persistence(image, img_timestamp, predict_horizon, kernel_size = (5,5)):
    """Takes an image and blurs it with a gaussanian window and uses it as 
    the prediction for the next time stamps. 

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


# img(t) + k*cmv estático -> img(t+k)

def cmv1(dcfg, imgi,imgf, period,delta_t, predict_horizon):
    """Predicts next image using openCV optical Flow

    Args:
        dcfg (dict): dictionary containing configuration 
        imgi (numpy.ndarray): first image used for prediction
        imgf (numpy.ndarray): last image used for prediction
        period (int): time difference between imgi and imgf in seconds
        delta_t (int): time passed between imgf and predicted image in seconds
        predict_horizon (int): Length of the prediction horizon (Cuantity of images returned)

    Returns:
        [list]: List with predicted images
    """    
    
    #get_cmv (dcfg, imgi,imgf, period)
    
    cmvcfg = dcfg["algorithm"]["cmv"]
    flow = cv.calcOpticalFlowFarneback(
        imgi,
        imgf,
        None,
        pyr_scale=cmvcfg["pyr_scale"],
        levels=cmvcfg["levels"],
        winsize=cmvcfg["winsize"],
        iterations=cmvcfg["iterations"],
        poly_n=cmvcfg["poly_n"],
        poly_sigma=cmvcfg["poly_sigma"],
        flags=0,
    )
    cmv = - flow / period
    
    #get_mapping(cmv, delta_t)
    
    base_img = imgf  #base_img imagen a la que le voy a aplicar el campo
    
    predictions = []
    for i in range(predict_horizon):
    
    
        i_idx, j_idx = np.meshgrid(
            np.arange(cmv.shape[1]), np.arange(cmv.shape[0])
        )
        map_i = i_idx + cmv[:, :, 0] * (delta_t * (i+1)) #cmv[...] * x , donde x es la cantidad de segundos hacia adelante
        map_j = j_idx + cmv[:, :, 1] * (delta_t * (i+1))
        map_x, map_y = map_i.astype(np.float32), map_j.astype(np.float32)

        #project_cmv(cmv, base_img, delta_t, show_for_debugging=False)
        

        #map_x, map_y = get_mapping(cmv, delta_t)
        next_img = cv.remap(
            base_img,
            map_x,
            map_y,
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            #borderValue=np.nan,  # valor que se agrega al mover los bordes
            borderValue=0,
        )
        # if show_for_debugging:
        #     show_for_debugging2(base_img, next_img, cmv, delta_t)
    
        predictions.append(next_img)
        
    return predictions




# img(t+k) + cmv estático -> img(t+k+1)

def cmv2(dcfg, imgi,imgf, period,delta_t, predict_horizon):
    """Predicts next image using openCV optical Flow

    Args:
        dcfg (dict): dictionary containing configuration 
        imgi (numpy.ndarray): first image used for prediction
        imgf (numpy.ndarray): last image used for prediction
        period (int): time difference between imgi and imgf in seconds
        delta_t (int): time passed between imgf and predicted image in seconds
        predict_horizon (int): Length of the prediction horizon (Cuantity of images returned)

    Returns:
        [list]: List with predicted images
    """    
    
    #get_cmv (dcfg, imgi,imgf, period)
    
    cmvcfg = dcfg["algorithm"]["cmv"]
    flow = cv.calcOpticalFlowFarneback(
        imgi,
        imgf,
        None,
        pyr_scale=cmvcfg["pyr_scale"],
        levels=cmvcfg["levels"],
        winsize=cmvcfg["winsize"],
        iterations=cmvcfg["iterations"],
        poly_n=cmvcfg["poly_n"],
        poly_sigma=cmvcfg["poly_sigma"],
        flags=0,
    )
    cmv = - flow / period
    
    #get_mapping(cmv, delta_t)
    
    base_img = imgf  #base_img imagen a la que le voy a aplicar el campo
    
    predictions = []
    
    for i in range(predict_horizon):
    
    
        i_idx, j_idx = np.meshgrid(
            np.arange(cmv.shape[1]), np.arange(cmv.shape[0])
        )
        map_i = i_idx + cmv[:, :, 0] * delta_t  #cmv[...] * x , donde x es la cantidad de segundos hacia adelante
        map_j = j_idx + cmv[:, :, 1] * delta_t 
        map_x, map_y = map_i.astype(np.float32), map_j.astype(np.float32)

        #project_cmv(cmv, base_img, delta_t, show_for_debugging=False)
        

        #map_x, map_y = get_mapping(cmv, delta_t)
        next_img = cv.remap(
            base_img,
            map_x,
            map_y,
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            #borderValue=np.nan,  # valor que se agrega al mover los bordes
            borderValue=0,
        )
        # if show_for_debugging:
        #     show_for_debugging2(base_img, next_img, cmv, delta_t)
    
        predictions.append(next_img)
        base_img = next_img
        
    return predictions