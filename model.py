
#Podria estar bueno que le entre tambien el time stamp de la imagen asi se lo agrega a las predicciones

def persitence (image, predict_horizon):
    """Takes an image and uses it as the prediction for the next time stamps

    Args:
        image (array): Image used as prediction
        predict_horizon (int): Length of the prediction horizon. 

    Returns:
        [list]: list containing precitions
    """    
    
    predictions = [image for i in range(predict_horizon)]
    return predictions 