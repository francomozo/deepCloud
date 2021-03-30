import matplotlib.pyplot as plt

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