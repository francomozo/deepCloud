import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
    
    
def barchart_compare(model1_values,model2_values):
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

    #labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    
    #men_means = [20, 34, 30, 35, 27]
    #women_means = [25, 32, 34, 20, 25]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    #rects1 = ax.bar(x - width/2, model1_values, width, label='Model 1')
    #rects2 = ax.bar(x + width/2, model2_values, width, label='Model 2')

    ax.bar(x - width/2, model1_values, width, label='Model 1')
    ax.bar(x + width/2, model2_values, width, label='Model 2')


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Error metric')
    ax.set_xlabel('Time (min)')
    ax.set_title('Error metric comparisson')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()
    
    
def plot_graph (model_values):
    labels = []
    for i in range(len(model_values)):
        labels.append(str(10* (i+1)))
    
    plt.plot(labels, model_values, "r.")
    plt.title('Model Error')
    plt.xlabel('Time (min)') 
    plt.ylabel('Error Metric') 
    plt.show()