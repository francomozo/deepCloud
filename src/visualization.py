# USAGE:
#   Barcharts, plots, graphs, histograms, functions to view dataset, etc
#

import os
import pickle
import time

import IPython
import matplotlib.pyplot as plt
import numpy as np

from src.lib.latex_options import Colors, Linestyles


def show_seq_and_pred(sequence_array, time_list, prediction_t, fig_name=None, save_fig=False, grid=True):
    """ Shows the images passed in a grid
    Args:
        sequence_array (array)
    """
    nbof_frames = sequence_array.shape[0]
    if np.max(sequence_array[0]) > 1.1:
        vmax = 100
    else:
        vmax = 1
    
    fontsize = 22 # 22 generates the font more like the latex text
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # plt.figure(figsize=(25, 5))
    fig, ax = plt.subplots(1, nbof_frames, figsize=(6 * nbof_frames, 5))
    plt.subplots_adjust(wspace=0.01)
    for i in range(nbof_frames):
        if i < nbof_frames - 2:
            ax[i].imshow(sequence_array[i], vmin=0, vmax=vmax)
            input_nbr = i - (nbof_frames - 3)
            ax[i].set_title(rf'{time_list[i]} ($t_{{{input_nbr}}}$)',
                            fontsize=fontsize)
            if i == 1:
                ax[i].set_xlabel('Inputs', fontsize=fontsize)
            
            ax[i].grid(grid)
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])

            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
        if i == nbof_frames - 2:
            im = ax[i].imshow(sequence_array[i], vmin=0, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=fontsize)
            ax[i].set_title(rf'{time_list[i]} ($t_{{{prediction_t}}}$)', fontsize=fontsize)
            ax[i].set_xlabel('Ground Truth', fontsize=fontsize)
            ax[i].grid(grid)
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
        if i == nbof_frames - 1:
            im = ax[i].imshow(sequence_array[i], vmin=0, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=fontsize)
            
            ax[i].set_title(rf'{time_list[-1]} ($\hat{{t_{{{prediction_t}}}}}$)', fontsize=fontsize)
            ax[i].set_xlabel('Prediction', fontsize=fontsize)

            ax[i].grid(grid)
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
        if nbof_frames == 5:
            line = plt.Line2D((.59, .59),(.1, 1), color=Colors.concrete, linestyle=Linestyles.dashed, linewidth=3)
        else:
            line = plt.Line2D((.515, .515),(.1, 1), color=Colors.concrete, linestyle=Linestyles.dashed, linewidth=3)
        fig.add_artist(line)
    if save_fig:
        fig.tight_layout() 
        fig.savefig(fig_name)
    plt.show()


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


def show_image_w_colorbar(image, title=None, fig_name=None, save_fig=False, bar_max=None, colormap='viridis'):
    """
    Shows the image with a colorbar 

    Args:
        image (array): Array containing the values of the image
    """
    fontsize = 16
    fig = plt.figure()
    fig.set_size_inches(6, 6)
    ax = fig.add_subplot(1, 1, 1)

    if bar_max:
        image_ = ax.imshow(image, cmap=colormap, interpolation='none', vmin=0, vmax=bar_max)
    else:
        image_ = ax.imshow(image, cmap=colormap, interpolation='none')

    cbar = plt.colorbar(image_, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize)
    
    if title:
        ax.title.set_text('Image')
     # Thicknes and axis colors
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.0)
    ax.tick_params(direction='out', length=6, width=1, colors='k')
    
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    if save_fig:
        fig.tight_layout() 
        fig.savefig(fig_name)
    plt.show()
    

def error_maps_for_5_horizons(error_maps_list, vmax=None, colormap='coolwarm', fig_name=None, save_fig=False):
    """ Shows the images passed in a grid
    Args:
        sequence_array (array)
    """
    if len(error_maps_list) != 5:
        raise ValueError('Must input 5 Maps')
    
    fontsize = 22 # 22 generates the font more like the latex text
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax = plt.subplots(1, 5, figsize=(30, 5))
    plt.subplots_adjust(wspace=0.01)
    for i in range(len(error_maps_list)):
        if i < 4:
            if vmax:
                ax[i].imshow(error_maps_list[i], vmin=0, vmax=vmax, cmap=colormap)
            else:
                ax[i].imshow(error_maps_list[i], cmap=colormap)
            if i == 0:
                ax[i].set_title(f'1 Hour', fontsize=fontsize)
            else:
                ax[i].set_title(f'{i + 1} Hours', fontsize=fontsize)

            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])

            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)

        elif i == 4:
            if vmax:
                im = ax[i].imshow(error_maps_list[i], vmin=0, vmax=vmax, cmap=colormap)
            else:
                im = ax[i].imshow(error_maps_list[i], cmap=colormap)
            cbar = plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=fontsize)

            ax[i].set_title(f'5 Hours', fontsize=fontsize)

            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)

    if save_fig:
        fig.tight_layout() 
        fig.savefig(fig_name)
    plt.show()


def make_plots_from_dict(load_paths, save_folder=None):
    """
    Load results from dictionary in a pickle file (or multiple dictionaries from multiple files) and display graphs

    Args:
        load_paths(str or list of strings): path of files containing dictionaries
        save_folder(str): folder to save graphs
    """   
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
    if(not isinstance(load_paths, list)):
        load_paths = [load_paths]
    
    errors_metrics_list = []
    for load_path in load_paths:
        errors_file = open(load_path, "rb") 
        errors_metrics_list.append(pickle.load(errors_file))
        errors_file.close()

    for metric, errors_metric in errors_metrics_list[0].items():
        error_list = []
        models = []
        for i in range(len(errors_metrics_list)-1):
            errors_metric.update(errors_metrics_list[i+1][metric])

        for model_name, error in errors_metric.items():
            error_list.append(errors_metric[model_name])
            models.append(model_name)
        
        if save_folder:
            save_file = os.path.join(save_folder, metric)
        else:
            save_file = None
        plot_graph_multiple(error_list, models, error_metric=metric, save_file=save_file)
        
        
def error_maps_for_3_horizons(error_maps_list, vmax=None, colormap='coolwarm', fig_name=None, save_fig=False):
    """ Shows the images passed in a grid
    Args:
        sequence_array (array)
    """
    if len(error_maps_list) != 3:
        raise ValueError('Must input 3 Maps')
    
    fontsize = 22 # 22 generates the font more like the latex text
    
    text = {0:'30min', 1:'60min', 2:'90min'}
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.01)
    for i in range(len(error_maps_list)):
        if i < 2:
            if vmax:
                ax[i].imshow(error_maps_list[i], vmin=0, vmax=vmax, cmap=colormap)
            else:
                ax[i].imshow(error_maps_list[i], cmap=colormap)
            if i == 0:
                ax[i].set_title(text[i], fontsize=fontsize)
            else:
                ax[i].set_title(text[i], fontsize=fontsize)

            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])

            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)

        elif i == 2:
            if vmax:
                im = ax[i].imshow(error_maps_list[i], vmin=0, vmax=vmax, cmap=colormap)
            else:
                im = ax[i].imshow(error_maps_list[i], cmap=colormap)
            cbar = plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=fontsize)

            ax[i].set_title(text[i], fontsize=fontsize)

            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            for tic in ax[i].xaxis.get_major_ticks():
                tic.tick1line.set_visible(False)
            for tic in ax[i].yaxis.get_major_ticks():
                tic.tick1line.set_visible(False)

    if save_fig:
        fig.tight_layout() 
        fig.savefig(fig_name)
    plt.show()
