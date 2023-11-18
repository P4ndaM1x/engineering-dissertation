import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utils import make_dataframe, moving_average

def plot_metric_history(fit_history, metric = 'loss', moving_average_window = (10,), title = None):
    
    if isinstance(fit_history, pd.core.frame.DataFrame):
        df = fit_history
    else:
        df = make_dataframe(fit_history)
    
    plt.figure()
    
    plt.plot(df['epoch'], df[metric], label=f'training set {metric}')
    plt.plot(df['epoch'], df[f'val_{metric}'], linestyle='dotted', linewidth=0.5, label=f'validation set {metric}')
    if moving_average_window is not None:
        for w in moving_average_window:
            plt.plot(df['epoch'][w-1:], moving_average(df[f'val_{metric}'], w), linestyle='dashed', label=f'validation set {metric} moving average, w={w}')
    
    if title is None:
        plt.title(f'{metric} history')
    else:
        plt.title(title)
    plt.ylabel(f'{metric} value')
    plt.xlabel('epoch no.')
    plt.legend()
    plt.show()


def plot_metric_history_zoomed(fit_history, metric = 'loss', mode = 'min', max = 1, min = 0, moving_average_window = (10,), threshold_multiplier = 10, legend_loc = 'lower left', title = None):

    if isinstance(fit_history, pd.core.frame.DataFrame):
        df = fit_history
    else:
        df = make_dataframe(fit_history)

    if mode == 'min':
        train_1st_percentile = np.percentile(df[metric].values, 1)
        val_1st_percentile = np.percentile(df[f'val_{metric}'].values, 1)
        print(f'1st percentile of train {metric}:       {train_1st_percentile:.4e}')
        print(f'1st percentile of validation {metric}:  {val_1st_percentile:.4e}')
        threshold = np.maximum(train_1st_percentile, val_1st_percentile)
    else:
        train_99th_percentile = np.percentile(df[metric].values, 99)
        val_99th_percentile = np.percentile(df[f'val_{metric}'].values, 99)
        print(f'99th percentile of train {metric}:       {train_99th_percentile:.4e}')
        print(f'99th percentile of validation {metric}:  {val_99th_percentile:.4e}')
        threshold = np.minimum(train_99th_percentile, val_99th_percentile)
        
    fig, axes = plt.subplots(3, sharex=True, figsize=(8,6))
    if title is None:
        axes[0].set_title(f'{metric} history')
    else:
        axes[0].set_title(title)
        
    for axis in axes:
        axis.plot(df['epoch'], df[metric], label='training set')
        axis.plot(df['epoch'], df[f'val_{metric}'], linestyle='dotted', linewidth=0.5, label='validation set')
        if moving_average_window is not None:
            for w in moving_average_window:
                axis.plot(df['epoch'][w-1:], moving_average(df[f'val_{metric}'], w), linestyle='dashed', label=f'validation set {metric} moving average, w={w}')
        if mode == 'min':
            axis.set_ylim((min, threshold_multiplier*threshold))
            threshold_multiplier /= 2
        else:
            axis.set_ylim((threshold_multiplier*threshold, max))
            threshold_multiplier /= 1.1
    axes[-1].legend(loc=legend_loc)
    axes[-1].set_xlabel("epoch no.")
    fig.supylabel(f"{metric} value")


def plot_loss_by_parameter(loss, parameter, loss_label='', parameter_label='', statistic='mean', bins=25, ylim=None):
    from scipy.stats import binned_statistic
    
    bin_means, bin_edges, bin_number = binned_statistic(parameter, loss, statistic=statistic, bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    plt.figure()
    plt.plot(parameter, loss, '.', label='loss value', alpha=0.5)
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='red', lw=4, label=f'binned {statistic}')
    plt.xlabel(parameter_label)
    plt.ylabel(loss_label)
    plt.grid(linestyle='--', linewidth=0.5) # axis='y', 
    if ylim:
        plt.ylim(ylim)
    plt.legend()


def plot_steps_by_parameter(x_data, y_data, parameter, metric_func, metric_label='', parameter_label='', bins=25, ylim=None, filter=lambda data: data):
    
    import tensorflow.keras.utils as tf_utils
    tf_utils.disable_interactive_logging()
    
    x_data=filter(x_data)
    y_data=filter(y_data)
    parameter=filter(parameter)

    _, bin_edges = np.histogram(parameter, bins=bins)
    bars = []
    for i in range(bins):
        indexes = np.where(np.logical_and(parameter>=bin_edges[i], parameter<=bin_edges[i+1]))
        bars.append( float(metric_func(np.asarray(y_data)[indexes], model.predict(x_data[indexes]))) )

    plt.figure()
    # plt.hlines(bars, bin_edges[:-1], bin_edges[1:], colors='red', lw=4, label=f'binned')
    # plt.bar(bin_edges[:-1], bars, width=bin_edges[1]-bin_edges[0], align='edge')
    bars.insert(0, bars[0])
    plt.step(bin_edges, bars, where='pre', color='red')
    plt.fill_between(bin_edges, bars, step="pre", alpha=0.5)
    offset = 0.001
    plt.xlabel(parameter_label)
    plt.ylabel(metric_label)
    plt.grid(linestyle='--', linewidth=0.5)
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim(((1-offset)*np.min(bars), (1+offset)*np.max(bars)))
    plt.show()
    
    tf_utils.enable_interactive_logging()
