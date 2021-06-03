# USAGE:
#   Barcharts, plots, graphs, histograms, functions to view dataset, etc
#

import csv
import time

import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#rolo graph
def matrix_graph (error_array):
    """
    Takes an array where the rows advance with time of the day and each column 
    is the error of the prediction.

    Args:
        error_array (array): Array containing the values of the error of a prediction
    """    
    fig, (ax1) = plt.subplots(figsize=(13, 3), ncols=1)
    grafica = ax1.imshow(error_array, interpolation='none')
    #grafica = ax1.imshow(error_array[70:100], interpolation='none')
    fig.colorbar(grafica, ax=ax1)
    ax1.title.set_text('One day error')
    ax1.set_xlabel('Prediction horizon ') 
    ax1.set_ylabel('Day time')

    plt.show()

    
def barchart_compare2(model1_values,
                      model1_name,
                      model2_values,
                      model2_name,
                      error_metric='RMSE'):
    """Takes the errors list of different models and plots them in a bar chart

    Args:
        model1_values (list): [description]
        model2_values (list): [description]

    Raises:
        ValueError: If lists don't have the same length
    """   

    labels = []
    if (len(model1_values) != len(model2_values)):
        raise ValueError('Lists must have the same length.')
    
    for i in range(len(model1_values)):
        labels.append(str(10* (i+1)))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(x - width/2, model1_values, width, label=model1_name)
    ax.bar(x + width/2, model2_values, width, label=model2_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error Metric: ' + error_metric)
    ax.set_xlabel('Predict horizon (min)')
    ax.set_title('Error metric comparisson')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()        
    
def barchart_compare3(model1_values,model1_name,model2_values,model2_name,model3_values,model3_name ):
    """Takes the errors list of different models and plots them in a bar chart

    Args:
        model1_values (list): [description]
        model2_values (list): [description]
        model3_values (list): [description]

    Raises:
        ValueError: If lists don't have the same length
    """   

    labels = []
    if (len(model1_values) != len(model2_values) or len(model1_values) != len(model3_values) ) or len(model3_values) != len(model2_values):
        raise ValueError('Lists must have the same length.')
    
    for i in range(len(model1_values)):
        labels.append(str(10* (i+1)))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.bar(x - width/2, model1_values, width/2, label=model1_name)
    ax.bar(x, model2_values, width/2, label=model2_name)
    ax.bar(x + width/2, model3_values, width/2, label=model3_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error metric')
    ax.set_xlabel('Predict horizon (min)')
    ax.set_title('Error metric comparisson')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show() 
    
def plot_graph(model_values, model, error_metric='RMSE'):
    """Plots the errors of the predictions for a generated sequence

    Args:
        model_values (list): List containing the values of the errors 
    """    
    labels = []
    for i in range(len(model_values)):
        labels.append(str(10* (i+1)))
    
    plt.plot(labels, model_values, "r.")
    plt.title(model + 'Model Error')
    plt.xlabel('Predict horizon (min)') 
    plt.ylabel('Error Metric: ' + error_metric) 
    plt.show()
    
def show_image_list(images_list, rows):
    """ Shows the images passed in a grid

    Args:
        images_list (list): Each element is a numpy array 
        rows (int): Number of rows in the grid
    """
    num = 0
    len_list = len(images_list)
    
    if len_list % rows == 0: cols = len_list//rows
    else: cols = len_list//rows + 1
    
     
    plt.figure(figsize=(10, 5))
    for img in images_list:

        plt.subplot(rows,cols ,num+1)
        plt.title('img' + str(num))
        plt.axis('off')
        plt.imshow(img)
        num += 1
        
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.tight_layout()
    plt.show()
    
def show_sample_dict(sample_dict , rows):
    images_array = sample_dict['images']
    ts_list = sample_dict['time_stamps']
    
    C = images_array.shape[0]
    
    if C % rows == 0: cols = C//rows
    else: cols = C//rows + 1
    
    plt.figure(figsize=(10, 5))
    for i in range(C):

        plt.subplot(rows,cols ,i+1)
        plt.title(ts_list[i])
        plt.axis('off')
        plt.imshow(images_array[i,:,:])
        
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.tight_layout()
    plt.show()
    
def plot_day_images(dataset, sleep_secs=0, start=0):
    """Shows images from dataset on a Jupyter Notebook

    Args:
        dataset (class): Dataset of one day of images
        sleep_secs (int): Time between each image in seconds
        start (int): Where to start in the images sequence.
    """
    
    for n in range(start, len(dataset)):
        sample = dataset[n]
        image, time_stamp = sample['image'], sample['time_stamp']
        
        fig, ax = plt.subplots()
        
        ax.title.set_text(f'Image: {n+1} {time_stamp}')
        fig.tight_layout()
        
        plt.imshow(image, cmap='gray')
        plt.show()
        
        time.sleep(sleep_secs)
        IPython.display.clear_output(wait=True)
        
def plot_histogram(values,bins, normalize = True):     
    """Takes a list of values or an image and it plots the histogram, showing the mean and standard 
    deviation.

    Args:
        values (list / np.array): Values to plot in histogram
        bins (int): Quanitity of bins to separate the values
        normalize (bool): if True the histogram is normalize
    """    
    plt.figure(figsize=(10, 5))
    
    if (type(values) == np.ndarray):
        M,N = values.shape
        values = np.reshape(values,(M*N))
             
    plt.hist(values, bins=bins, density=normalize)
    l1 = plt.axvline(np.mean(values), c='r')
    l2 = plt.axvline(np.mean(values)+np.std(values), c='g')
    l3 = plt.axvline(np.mean(values)-np.std(values), c='g')
    plt.legend((l1, l2, l3), ['mean of values', 'std of values'])
    if (normalize): plt.ylabel('p(x)')
    else: plt.ylabel('Quantity')
    
    plt.xlabel('value')
    plt.show()
    
def show_image_w_colorbar (image):
    """
    Shows the image with a colorbar 

    Args:
        image (array): Array containing the values of the image
    """    
    fig, (ax1) = plt.subplots(figsize=(13, 3), ncols=1)
    image_ = ax1.imshow(image, interpolation='none')
    #grafica = ax1.imshow(error_array[70:100], interpolation='none')
    fig.colorbar(image_, ax=ax1)
    ax1.title.set_text('Image')
    plt.show()
    
    
def show_images_diff(img1,img2):
    """
    Shows the difference between two images with a colorbar 

    Args:
        img1(array): Array containing the values of the image 1
        img2(array): Array containing the values of the image 2
    """    
    if (img1.shape != img2.shape):
        raise ValueError('Images must have the same shape, img1:',img1.shape,'img2:',img2.shape)
    diff = abs(img1-img2)
    fig, (ax1) = plt.subplots(figsize=(14, 4), ncols=1)
    image_ = ax1.imshow(diff, interpolation='none')
    #grafica = ax1.imshow(error_array[70:100], interpolation='none')
    fig.colorbar(image_, ax=ax1)
    ax1.title.set_text('Image')
    plt.show()
