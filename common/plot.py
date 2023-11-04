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


def plot_metric_history_zoomed(fit_history, metric = 'loss', moving_average_window = (10,), threshold_multiplier = 10, title = None):

    if isinstance(fit_history, pd.core.frame.DataFrame):
        df = fit_history
    else:
        df = make_dataframe(fit_history)

    train_1st_percentile = np.percentile(df[metric].values, 1)
    val_1st_percentile = np.percentile(df[f'val_{metric}'].values, 1)
    print(f'1st percentile of train {metric}:       {train_1st_percentile:.4e}')
    print(f'1st percentile of validation {metric}:  {val_1st_percentile:.4e}')
    
    threshold = np.maximum(train_1st_percentile, val_1st_percentile)
    fig, axes = plt.subplots(3, sharex=True, figsize=(8,6))
    
    if title is None:
        axes[0].set_title(f'{metric} history')
    else:
        axes[0].set_title(title)
        
    for axis in axes:
        axis.set_ylim((0, threshold_multiplier*threshold))
        axis.plot(df['epoch'], df[metric], label='training set')
        axis.plot(df['epoch'], df[f'val_{metric}'], linestyle='dotted', linewidth=0.5, label='validation set')
        if moving_average_window is not None:
            for w in moving_average_window:
                axis.plot(df['epoch'][w-1:], moving_average(df[f'val_{metric}'], w), linestyle='dashed', label=f'validation set {metric} moving average, w={w}')
        threshold_multiplier /= 2
    axes[-1].legend(loc="lower left")
    axes[-1].set_xlabel("epoch no.")
    fig.supylabel(f"{metric} value")
