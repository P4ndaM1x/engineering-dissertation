import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utils import make_dataframe, moving_average, validation_set, train_set

def plot_metric_history(fit_history, metric = 'loss', moving_average_window = (20,), ylim=None, hline=None, legend_loc = 'upper right', title = None, figsize=[8,4], dpi=100):
    
    if isinstance(fit_history, pd.core.frame.DataFrame):
        df = fit_history
    else:
        df = make_dataframe(fit_history)
    
    plt.figure(dpi=dpi, figsize=figsize)
    
    plt.plot(df['epoch'], df[metric], label=f'training set')
    plt.plot(df['epoch'], df[f'val_{metric}'], linestyle='dotted', linewidth=0.5, label=f'validation set')
    if moving_average_window is not None:
        for w in moving_average_window:
            plt.plot(df['epoch'][w-1:], moving_average(df[f'val_{metric}'], w), linestyle=(0, (1, 1)), label=f'validation set - moving average, $w={w}$')
    if hline:
        plt.axhline(y=hline, color='grey', alpha=0.5, linestyle=(0, (1, 1)), label=f'{metric}$ = 0.005$')
    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)
    plt.ylabel(f'{metric} value')
    plt.xlabel('epoch no.')
    plt.legend(loc=legend_loc)
    plt.show()


def plot_metric_history_zoomed(fit_history, metric = 'loss', mode = 'min', max = 1, min = 0, moving_average_window = (10,), threshold_multiplier = 10, subplots=3, legend_loc = 'lower left', title = None, dpi=100):

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
    
    if subplots < 3:
        height_ratios = [2, 1] if  mode == 'min' else [1, 2]
        fig, axes = plt.subplots(2, sharex=True, figsize=(8,6),  gridspec_kw={'height_ratios': height_ratios}, constrained_layout=True)
    else:
        fig, axes = plt.subplots(subplots, sharex=True, figsize=(8,6), constrained_layout=True)
    fig.set_dpi(dpi)
    if title is not None:
        # axes[0].set_title(f'{metric} history')
    # else:
        axes[0].set_title(title)
        
    for axis in axes:
        # axis.grid(axis='x', linestyle='--', linewidth=0.5)
        axis.plot(df['epoch'], df[metric], label='training set')
        axis.plot(df['epoch'], df[f'val_{metric}'], linestyle='dotted', linewidth=0.5, label='validation set')
        if moving_average_window is not None:
            for w in moving_average_window:
                axis.plot(df['epoch'][w-1:], moving_average(df[f'val_{metric}'], w), linestyle=(0, (1, 1)), label=f'validation set - moving average, $w={w}$')
        if mode == 'min':
            axis.set_ylim((min, threshold_multiplier*threshold))
            threshold_multiplier /= 2
        else:
            axis.set_ylim((threshold_multiplier*threshold, max))
            threshold_multiplier /= 1.2
    if 'lower' in legend_loc:
        axes[-1].legend(loc=legend_loc)
    if 'upper' in legend_loc:
        axes[0].legend(loc=legend_loc)
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


def plot_steps_by_parameter(model, x_data, y_data, parameter, metric_func, metric_label='', parameter_label='', bins=25, ylim=None, show_train=True, show_val=True, dpi=100):
    
    import tensorflow.keras.utils as tf_utils
    tf_utils.disable_interactive_logging()
    
    x_data_train=train_set(x_data)
    y_data_train=train_set(y_data)
    parameter_train=train_set(parameter)
    
    x_data_val=validation_set(x_data)
    y_data_val=validation_set(y_data)
    parameter_val=validation_set(parameter)


    _, bin_edges_train = np.histogram(parameter_train, bins=bins)
    _, bin_edges_val = np.histogram(parameter_val, bins=bins)
    bars_train = []
    bars_val = []
    for i in range(bins):
        indexes = np.where(np.logical_and(parameter_train>=bin_edges_train[i], parameter_train<=bin_edges_train[i+1]))
        bars_train.append( float(metric_func(np.asarray(y_data_train)[indexes], model.predict(x_data_train[indexes]))) )
        indexes = np.where(np.logical_and(parameter_val>=bin_edges_val[i], parameter_val<=bin_edges_val[i+1]))
        bars_val.append( float(metric_func(np.asarray(y_data_val)[indexes], model.predict(x_data_val[indexes]))) )

    plt.figure(dpi=dpi)
    # plt.hlines(bars, bin_edges[:-1], bin_edges[1:], colors='red', lw=4, label=f'binned')
    # plt.bar(bin_edges[:-1], bars, width=bin_edges[1]-bin_edges[0], align='edge')
    
    if show_train:
        bars_train.insert(0, bars_train[0])
        plt.step(bin_edges_train, bars_train, where='pre', label='training set')
        plt.fill_between(bin_edges_train, bars_train, step="pre", alpha=0.4)
    
    if show_val:
        bars_val.insert(0, bars_val[0])
        plt.step(bin_edges_val, bars_val, where='pre', ls=(0, (5, 3)), label='validation set')
        plt.fill_between(bin_edges_val, bars_val, step="pre", alpha=0.4)
    
    plt.xlabel(parameter_label)
    plt.ylabel(metric_label)
    plt.grid(linestyle='--', linewidth=0.5)
    offset = 0.001
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim(((1-offset)*np.min(bars_train), (1+offset)*np.max(bars_train)))
    if show_train and show_val:
        plt.legend()
    plt.show()
    
    tf_utils.enable_interactive_logging()
