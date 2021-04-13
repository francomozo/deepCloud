# USAGE:
#   Model predictions evaluation functions
#

import numpy as np
from skimage.metrics import structural_similarity as ssim
import skimage.metrics

def evaluate_image(predictions, gt, metric, pixel_max_value =255):
    """
    Evaluates the precision of the prediction compared to the gorund truth using different metrics

    Args:
        - predictions (list): list containing the predicted images
        - gt (list): list containing the ground truth images corresponding to the predicted ones
        - metric (string): 
        RMSE, 
        MSE, 
        PSNR, 
        SSIM : Compute the mean structural similarity index between two images.
        NRMSE = || im_true - im_pred || / || im_true ||
        'ReRMSE' : Relative RMSE rmse(pred-gt) / rmse( gt - mean(gt))
        'FS': Forecast skill ,realtive comparison with persistence
        - pixel_max_value (int): Maximum value a pixel can take (used for PSNR)

    Returns:
        [list]: list containing the erorrs of each predicted image 
        compared to the according ground truth image
    """   

    #length must be the same
    len_pred = len(predictions)
    len_gt = len(gt)
    
    if (len_pred != len_gt):
        raise ValueError('Predictions and Ground truth must have the same length. len(predictions)=',len_pred, ', len(gt)=',len_gt)
      
        
    if type(predictions) == np.ndarray : #case the prediction input is an array instead of a list
        len_pred=1
    
    error= [] 
    
    for i in range (len_pred):
        
        if (predictions[i].shape != gt[i].shape):
            raise ValueError('Input images must have the same dimensions.')
        
        if(metric == 'RMSE'):
            error.append(np.sqrt(np.mean((predictions[i]-gt[i])**2)) )   
        elif (metric == 'MSE' ):
            error.append(np.mean((predictions[i]-gt[i])**2) ) 
        elif (metric == 'PSNR' ):            
            mse = np.mean((predictions[i]-gt[i])**2)
            if (mse != 0 ):
                error.append(10* np.log10(pixel_max_value**2/mse)) 
            else:
                error.append(20*np.log10(pixel_max_value))
     
        elif (metric == 'SSIM'):
            error.append(ssim(predictions[i] , gt[i]))
        elif (metric == 'NRMSE'):
            nrmse = skimage.metrics.normalized_root_mse(gt[i],predictions[i])
            error.append(nrmse)
        elif (metric == 'ReRMSE'):
            eps = 0.0001
            re_rmse = np.sqrt(np.mean((predictions[i]-gt[i])**2))/(np.sqrt(np.mean((np.mean(gt[i])-gt[i])**2))+eps)
            error.append(re_rmse)
        elif (metric == 'FS'):
            rmse = np.sqrt(np.mean((predictions[i]-gt[i])**2))
            rmse_persistence = np.sqrt(np.mean((predictions[0]-gt[i])**2))
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

