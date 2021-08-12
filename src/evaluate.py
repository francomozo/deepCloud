# USAGE:
#   Model predictions evaluation functions
#

import numpy as np
from skimage.metrics import structural_similarity as ssim
import skimage.metrics
import math
import torch
import time
from tqdm import tqdm
import cv2 as cv

import src.lib.utils as utils
from src import model

def evaluate_image(predictions, gt, gt_ts, metric, pixel_max_value=100, 
                   window_pad=0, window_pad_height=0, window_pad_width=0,
                   dynamic_window=False, evaluate_day_pixels=True, error_percentage=False):
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
        - window_pad(int) : evaluation window shrinks by window_pad rows and columns. 
                            Eval window is [w_p//2 : M-w_p//2, w_p//2 : N-w_p//2]
        - window_pad_height(int) : If M,N size of image -> eval window is [w_p_h//2 : M - w_p_h//2]
        - window_pad_width(int) : If M,N size of image -> eval window is [w_p_w//2 : N - w_p_w//2]
        - dynamic_window(bool) : generate biggest window without nans in prediction and evaluate only in 
                                those pixels
        -evaluate_day_pixels(bool): generate cosz map and evaluate only in pixels with cosz over 
        -error_percentage(bool): if true, return error in percentage value

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
    
    elif (window_pad_height or window_pad_width):
        pi = window_pad_height//2
        pf = M - window_pad_height//2
        qi = window_pad_width//2
        qf = N - window_pad_width//2
        
    elif (window_pad):
        pi = window_pad//2
        pf = M - window_pad//2
        qi = window_pad//2
        qf = N - window_pad//2
    else:
        pi,pf,qi,qf = 0,M,0,N
        
    nan_in_pred = False
    #Check for NANs in last image    
    if (math.isnan(np.sum(predictions[-1][pi:pf,qi:qf]  ))) :
        nan_in_pred = True
        # raise ValueError('Last prediction has np.nan values')
        
    for i in range (C):
        
        cosangs_map = np.ones((pf-pi, qf-qi))
        pred = predictions[i,pi:pf,qi:qf]
        gt_aux = gt[i,pi:pf,qi:qf]
        
        if (pred.shape != gt_aux.shape):
            raise ValueError('Input images must have the same dimensions.')
        
        if evaluate_day_pixels:
            _ , cosangs_thresh = utils.get_cosangs_mask(meta_path='data/meta',
                                                            img_name=gt_ts[i])
            cosangs_map = cosangs_thresh[pi:pf,qi:qf] 
        
        if nan_in_pred:
            gt_aux = gt_aux[np.logical_not(np.isnan(pred))]
            cosangs_map = cosangs_map[np.logical_not(np.isnan(pred))]
            pred = pred[np.logical_not(np.isnan(pred))]
            
        if(metric == 'RMSE'):
            rmse = np.sqrt(np.mean(((pred - gt_aux)*(cosangs_map==1))**2))
            error.append(rmse )   
        elif (metric == 'MSE' ):
            error.append(np.mean(((pred-gt_aux)*(cosangs_map==1))**2) ) 
        elif (metric == 'MAE'):
            error.append(np.mean( np.absolute((pred-gt_aux)*(cosangs_map==1)) ))
        elif (metric == 'MBD'):
            error.append(np.mean( (pred-gt_aux)*(cosangs_map==1) ))
        elif (metric == 'PSNR' ):            
            mse = np.mean(((pred-gt_aux)*(cosangs_map==1))**2)
            if (mse != 0 ):
                error.append(10* np.log10(pixel_max_value**2/mse)) 
            else:
                error.append(20*np.log10(pixel_max_value))
     
        elif (metric == 'SSIM'):
            error.append(ssim(pred*(cosangs_map==1),gt_aux*(cosangs_map==1) , win_size=101  ))
        elif (metric == 'NRMSE'):
            nrmse = skimage.metrics.normalized_root_mse(gt_aux*(cosangs_map==1),pred*(cosangs_map==1))
            error.append(nrmse)
        elif (metric == 'ReRMSE'):
            eps = 0.0001
            re_rmse = np.sqrt(np.mean(((pred-gt_aux)*(cosangs_map==1))**2))/(np.sqrt(np.mean((np.mean(gt_aux)-gt_aux)**2))+eps)
            error.append(re_rmse)
        elif (metric == 'FS'):
            rmse = np.sqrt(np.mean(((pred-gt_aux)*(cosangs_map==1))**2))
            rmse_persistence = np.sqrt(np.mean(((predictions[0,pi:pf,qi:qf] -gt_aux )*(cosangs_map==1))**2))
            if rmse_persistence == 0 :
                fs = 1
                error.append(fs)
            else: 
                fs = 1 - rmse/rmse_persistence
                error.append(fs)

        if (error_percentage):
            error[i] = error[i]/np.mean(gt_aux)

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


def evaluate_model(model_instance, loader, predict_horizon, 
                   device=None, metric='RMSE', error_percentage=False,
                   window_pad=0, window_pad_height=0, window_pad_width=0):
    """
    Evaluates performance of model_instance on loader data. 

    Args:
        model_instance (object): Model to evaluate. Persistence, CMV or nn.Module
        loader (Dataloader): Dataloader with data for evaluation
        predict_horizon (int): Number of images to predict 
        device (string, optional): Device for pytorch. "cpu" or "cuda". Defaults to None.
        metric (str, optional): Metric for evaluation. Defaults to 'RMSE'.
        error_percentage(bool): if true, return error in percentage value
        window_pad(int) : evaluation window shrinks by window_pad rows and columns. 
                          Eval window is [w_p//2 : M-w_p//2, w_p//2 : N-w_p//2]
        window_pad_height(int) : If M,N size of image -> eval window is [w_p_h//2 : M - w_p_h//2]
        window_pad_width(int) : If M,N size of image -> eval window is [w_p_w//2 : N - w_p_w//2]

    Returns:
        [np.array]: Array of errors in evaluation with shape (len(loader), predict_horizon)
    """
    error_list =[]

    per_predict_time= []
    eval_time = []
    cmv_predict_time = []
    
    with tqdm(loader, desc=f'Status', unit='sequences') as loader_pbar:
        for idx, (inputs, targets) in enumerate(loader_pbar):
            inputs = inputs.squeeze()
            targets = targets.squeeze()
            
            # predict depending on model
            if (isinstance(model_instance, list)):
                # direct NNs models
                predictions = []
                with torch.no_grad():
                    inputs = inputs.to(device=device)
                    for i in range(predict_horizon):
                        prediction = model_instance[i](inputs.unsqueeze(0))
                        predictions.append(prediction.cpu().detach().numpy().squeeze())
                predictions = np.array(predictions) 
                dynamic_window = False
                
            elif (isinstance(model_instance, model.Persistence)):
                start = time.time()
                predictions = model_instance.predict(
                                        image=inputs[1], 
                                        predict_horizon=predict_horizon)
                end = time.time()
                per_predict_time.append(end-start)
                dynamic_window = False

            elif (isinstance(model_instance, model.Cmv)):
                start = time.time()
                predictions = model_instance.predict(
                                        imgi=inputs[0], 
                                        imgf=inputs[1],
                                        period=10*60, delta_t=10*60, 
                                        predict_horizon=predict_horizon) 
                end = time.time()
                cmv_predict_time.append(end-start)
                dynamic_window = False # true for dynamic_window

            elif (isinstance(model_instance, torch.nn.Module) and model_instance.n_classes == 1):
                # recursive NN model
                predictions = []
                with torch.no_grad():
                    for i in range(predict_horizon):
                        inputs = inputs.to(device=device)
                        prediction = model_instance(inputs.unsqueeze(0))
                        predictions.append(prediction.cpu().detach().numpy().squeeze())
                        inputs = torch.cat((inputs[1:], prediction.squeeze(0)))
                predictions = np.array(predictions) 
                dynamic_window = False

            elif (isinstance(model_instance, str) and model_instance == "gt_blur"):
                predictions = []
                kernel_size_list = [(35,35),(73,73),(105,105),(137,137),(169,169),(201,201)]
                for i in range(predict_horizon):
                    predictions.append(cv.GaussianBlur(targets.cpu().detach().numpy()[i], kernel_size_list[i], 0))
                predictions = np.array(predictions)
                dynamic_window = False

            # evaluate
            if not (isinstance(model_instance, torch.nn.Module) or isinstance(model_instance, list) or isinstance(model_instance, str)):
                predictions = predictions[1:]
            start = time.time()
            predict_errors = evaluate_image(
                                        predictions = predictions, 
                                        gt = targets.cpu().detach().numpy(), 
                                        gt_ts = None,
                                        metric=metric, dynamic_window=dynamic_window,
                                        evaluate_day_pixels = False, 
                                        error_percentage = error_percentage,
                                        window_pad=window_pad, 
                                        window_pad_height=window_pad_height, 
                                        window_pad_width=window_pad_width)
            error_list.append(predict_errors)
            end = time.time()
            eval_time.append(end-start)

    if (isinstance(model_instance, model.Persistence)):
        print(f'Persistence predict time: {np.sum(per_predict_time):.2f} seconds.')
    elif (isinstance(model_instance, model.Cmv)):
        print(f'Cmv predict time: {np.sum(cmv_predict_time):.2f} seconds.')
    print(f'Evaluation time: {np.sum(eval_time):.2f} seconds.')
    return np.array(error_list)