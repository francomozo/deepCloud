import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time
import IPython

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
    
def save_errorarray_as_csv(error_array, time_stamp, filename):
    """ Generates a CSV file with the error of the predictions at the different times of the day

    Args:
        error_array (array): Array containing the values of the error of a prediction
        time_stamp (list): Contains the diferent timestamps of the day
        filename (string): path and name of the generated file
    """    
    
    M,N = error_array.shape
    fieldnames = []
    fieldnames.append('timestamp')
    for i in range(N):
        fieldnames.append(str(10*(i+1)) + 'min')
    
    with open( filename + '.csv', 'w', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(M):
            row_dict = {}
            row_dict['timestamp'] = time_stamp[i]
            for j in range (N):
                row_dict[str(10*(j+1)) + 'min']  = error_array[i,j]
            
            writer.writerow(row_dict)
        

    
def barchart_compare2(model1_values,model1_name,model2_values,model2_name ):
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
    ax.set_ylabel('Error metric')
    ax.set_xlabel('Time (min)')
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
    ax.set_xlabel('Time (min)')
    ax.set_title('Error metric comparisson')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show() 
    
def plot_graph (model_values):
    """Plots the errors of the predictions for a generated sequence

    Args:
        model_values (list): List containing the values of the errors 
    """    
    labels = []
    for i in range(len(model_values)):
        labels.append(str(10* (i+1)))
    
    plt.plot(labels, model_values, "r.")
    plt.title('Model Error')
    plt.xlabel('Time (min)') 
    plt.ylabel('Error Metric') 
    plt.show()
    
def show_image_list (images_list,rows):
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
        
def plot_histogram(values,bins):     
    
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=bins, density=True)
    l1 = plt.axvline(np.mean(values), c='r')
    l2 = plt.axvline(np.mean(values)+np.std(values), c='g')
    l3 = plt.axvline(np.mean(values)-np.std(values), c='g')
    plt.legend((l1, l2, l3), ['mean of values', 'std of values'])
    plt.ylabel('p(x)')
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