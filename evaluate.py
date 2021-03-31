import numpy as np
from skimage.measure import compare_ssim as ssim

def evaluate_image(predictions, gt, metric, pixel_max_value =255):
    """
    Evaluates the precision of the prediction compared to the gorund truth using different metrics

    Args:
        predictions (list): list containing the predicted images
        gt (list): list containing the ground truth images corresponding to the predicted ones
        metric (string): RMSE, MSE, PSNR, SSIM 
        pixel_max_value (int): Maximum value a pixel can take (used for PSNR)

    Returns:
        [list]: list containing the erorrs of each predicted image 
        compared to the according ground truth image
    """   

    #length must be the same
    len_pred = len(predictions)
    len_gt = len(gt)
    
    if (len_pred != len_gt):
            raise ValueError('Predictions and Ground truth must have the same length.')
    
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
            error.append(10* np.log10(pixel_max_value**2/mse))       
        elif (metric == 'SSIM'):
            error.append(ssim(predictions[i] , gt[i]))
            

    return error

def evaluate_pixel(predictions,gt,metric,pixel= (0,0)):
    
    #length must be the same
    len_pred = len(predictions)
    len_gt = len(gt)
    if (len_pred != len_gt):
        raise ValueError('Predictions and Ground truth must have the same length.')
    
    n,m = pixel
    
    error= []  
    for i in range (len_pred):

        if(metric == 'RMSE'):
            error.append(abs(predictions[i][n,m]-gt[i][n,m]))     
        elif (metric == 'MSE' ):
            error.append((predictions[i][n,m]-gt[i][n,m])**2) 
    return error

