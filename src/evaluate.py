# USAGE:
#   Model predictions evaluation functions
#

import numpy as np
from skimage.metrics import structural_similarity as ssim
import skimage.metrics
import math
import torch

import src.lib.utils as utils

def evaluate_image(predictions, gt,gt_ts, metric, pixel_max_value =100, 
                   small_eval_window = False,window_pad_height=0,window_pad_width =0 ,
                   dynamic_window = False, evaluate_day_pixels = True):
    """
    Evaluates the precision of the prediction compared to the gorund truth using different metrics

    Args:
        - predictions (Numpy Array): Numpy array containing the predicted images
        - gt (pyTorch Tensor): tensor containing the ground truth images corresponding to the predicted ones
        - gt_ts (list): Contains strings with date and time of the gt images, for cosz 
        - metric (string): 
        RMSE, 
        MSE, 
        PSNR, 
        SSIM : Compute the mean structural similarity index between two images.
        NRMSE = rmse( im_true - im_pred ) / || im_true ||
        'ReRMSE' : Relative RMSE rmse(pred-gt) / rmse( gt - mean(gt))
        'FS': Forecast skill ,realtive comparison with persistence
        - pixel_max_value (int): Maximum value a pixel can take (used for PSNR)
        - small_eval_window(bool) : If true the evaluation window shrinks by window_pad_height rows and
                                    window_pad_width columns
        - window_pad_height(int) : If M,N size of image -> eval window is [w_p_h//2 : M - w_p_h//2]
        - window_pad_width(int) : If M,N size of image -> eval window is [w_p_w//2 : N - w_p_w//2]
        - dynamic_window(bool) : generate biggest window without nans in prediction and evaluate only in 
                                those pixels
        -evaluate_day_pixels: generate cosz map and evaluate only in pixels with cosz over 

    Returns:
        [list]: list containing the erorrs of each predicted image 
        compared to the according ground truth image
    """   

    if torch.is_tensor(gt): 
        gt = gt.numpy()

    #length must be the same
    C, M, N = predictions.shape
    len_gt = gt.shape[0]
     
    if (C != len_gt):
        raise ValueError('Predictions and Ground truth must have the same length. len(predictions)=',C, ', len(gt)=',len_gt)
 
    error= [] 
    
    if (dynamic_window):
        xmin, xmax, ymin, ymax = utils.find_inner_image(predictions[-1])
        pi = xmin
        pf = M - xmax
        qi = ymin
        qf = N - ymax
    
    elif (small_eval_window):
        pi = window_pad_height//2
        pf = M -window_pad_height//2
        qi = window_pad_width//2
        qf = N - window_pad_width//2
    else:
        pi,pf,qi,qf = 0,M,0,N
        
    #Check for NANs in last image    
    if (math.isnan(np.sum(predictions[-1][pi:pf,qi:qf]  ))) :
        raise ValueError('Last prediction has np.nan values')
        
    cosangs_map = np.ones((pf-pi, qf-qi))
    
    for i in range (C):
        
        if (predictions[i,pi:pf,qi:qf].shape != gt[i,pi:pf,qi:qf].shape):
            raise ValueError('Input images must have the same dimensions.')
        
        if evaluate_day_pixels:
            _ , cosangs_thresh = utils.get_cosangs_mask(meta_path='data/meta',
                                                            img_name=gt_ts[i])

            cosangs_map = cosangs_thresh[pi:pf,qi:qf] 
        
            
        if(metric == 'RMSE'):
            rmse = np.sqrt(np.mean((predictions[i,pi:pf,qi:qf]*(cosangs_map==1) - gt[i,pi:pf,qi:qf]*(cosangs_map==1))**2))
            error.append(rmse )   
        elif (metric == 'MSE' ):
            error.append(np.mean((predictions[i,pi:pf,qi:qf]*(cosangs_map==1)-gt[i,pi:pf,qi:qf]*(cosangs_map==1))**2) ) 
        elif (metric == 'PSNR' ):            
            mse = np.mean((predictions[i,pi:pf,qi:qf]*(cosangs_map==1)-gt[i,pi:pf,qi:qf]**(cosangs_map==1))**2)
            if (mse != 0 ):
                error.append(10* np.log10(pixel_max_value**2/mse)) 
            else:
                error.append(20*np.log10(pixel_max_value))
     
        elif (metric == 'SSIM'):
            error.append(ssim(predictions[i,pi:pf,qi:qf]*(cosangs_map==1),gt[i,pi:pf,qi:qf]*(cosangs_map==1)))
        elif (metric == 'NRMSE'):
            nrmse = skimage.metrics.normalized_root_mse(gt[i,pi:pf,qi:qf]*(cosangs_map==1),predictions[i,pi:pf,qi:qf]*(cosangs_map==1))
            error.append(nrmse)
        elif (metric == 'ReRMSE'):
            eps = 0.0001
            re_rmse = np.sqrt(np.mean((predictions[i,pi:pf,qi:qf]*(cosangs_map==1)-gt[i,pi:pf,qi:qf]*(cosangs_map==1))**2))/(np.sqrt(np.mean((np.mean(gt[i,pi:pf,qi:qf]*(cosangs_map==1))-gt[i,pi:pf,qi:qf]*(cosangs_map==1))**2))+eps)
            error.append(re_rmse)
        elif (metric == 'FS'):
            rmse = np.sqrt(np.mean((predictions[i,pi:pf,qi:qf]*(cosangs_map==1)-gt[i,pi:pf,qi:qf]*(cosangs_map==1))**2))
            rmse_persistence = np.sqrt(np.mean((predictions[0,pi:pf,qi:qf]*(cosangs_map==1) -gt[i,pi:pf,qi:qf]*(cosangs_map==1) )**2))
            if rmse_persistence == 0 :
                fs = 1
                error.append(fs)
            else: 
                fs = 1 - rmse/rmse_persistence
                error.append(fs)

    return error

def evaluate_pixel(predictions,gt,metric,pixel_max_value =255,pixel= (0,0)):
    """
     Measures the error of the prediction compared to the gorund truth for 
     a specific pixel

    Args:
        predictions (list): list containing the predicted images
        gt (list): list containing the ground truth images corresponding to the predicted ones
        metric (string): RMSE, MSE, PSNR
        pixel_max_value (int): Maximum value a pixel can take (used for PSNR). Defaults to 255.
        pixel (tuple): Coordinates of the pixel to compare. Defaults to (0,0).

    Raises:
        ValueError: Predictions and Ground truth must have the same length.

    Returns:
        [list]: list containing the erorrs of each predicted pixel
        compared to the according ground truth pixel
    """    
    
    #length must be the same
    len_pred = len(predictions)
    len_gt = len(gt)
    if (len_pred != len_gt):
        raise ValueError('Predictions and Ground truth must have the same length.')
    
    n,m = pixel
    
    if type(predictions) == np.ndarray : #case the prediction input is an array instead of a list
        len_pred=1
    
    error= []  
    for i in range (len_pred):

        if(metric == 'RMSE'):
            error.append(abs(predictions[i][n,m]-gt[i][n,m]))     
        elif (metric == 'MSE' ):
            error.append((predictions[i][n,m]-gt[i][n,m])**2) 
        elif (metric == 'PSNR'):
            mse = np.mean((predictions[i][n,m]-gt[i][n,m])**2)
            if (mse != 0 ):
                error.append(10* np.log10(pixel_max_value**2/mse)) 
            else:
                error.append(20*np.log10(pixel_max_value))
            
    return error

