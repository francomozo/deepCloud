import numpy

def evaluate_image(predictions, gt, metric):
    #length must be the same
    len_pred = len(predictions)
    len_gt = len(gt)
    
    error= [] 
    for i in range (len_pred):
        if(metric == 'RMSE'):
            error.append(np.sqrt(np.mean((predictions[i]-gt[i])**2)) )        

    return error

def evaluate_pixel(predictions,gt,metric,pixel= (0,0)):
    
    #length must be the same
    len_pred = len(predictions)
    len_gt = len(gt)
    
    n,m = pixel
    error= []  
    for i in range (len_pred):
        if(metric == 'RMSE'):
            error.append(np.sqrt(np.mean((predictions[i][n,m]-gt[i][n,m])**2)) )        

    return error

