import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

def print_advanced_stats(dataframe, label, use_scientific_notation = False):
    values = dataframe[label].to_numpy()
    print(f'--- {label} stats ---')
    if use_scientific_notation:
        print(f'min({label}):      {np.min(values):.5e}')
        print(f'max({label}):      {np.max(values):.5e}')
        print(f'med({label}):      {np.median(values):.5e}')
        print(f'std({label}):      {np.std(values):.5e}')
        print(f'kurtosis({label}): {sc.kurtosis(values):.5e}')
        print(f'skew({label}):     {sc.skew(values):.5e}')
    else:
        print(f'min({label}):      {np.min(values)}')
        print(f'max({label}):      {np.max(values)}')
        print(f'med({label}):      {np.median(values)}')
        print(f'std({label}):      {np.std(values)}')
        print(f'kurtosis({label}): {sc.kurtosis(values)}')
        print(f'skew({label}):     {sc.skew(values)}')
        
def print_multiple_advanced_stats(dataframe, labels):
    # print('label & minimum & maksimum & mediana & odchylenie standardowe & kurtoza & współczynnik skośności')
    # for label in labels:
    #     values = dataframe[label].to_numpy()
    #     print(f'{label} & {np.min(values):.2f} & {np.max(values):.2f} & {np.median(values):.5f} & {np.std(values):.2f} & {sc.kurtosis(values):.2f} & {sc.skew(values):.2e} \\\ ')
    print('współrzędna & mediana & odchylenie standardowe & kurtoza & współczynnik skośności \\\ ')
    for label in labels:
        values = dataframe[label].to_numpy()
        print(f'{label} & {np.median(values):.2f} & {np.std(values):.1f} & {sc.kurtosis(values):.2f} & {sc.skew(values):.2e} \\\ ')
    

def plot_histogram(dataframe, label, bins=20, dpi=100):
    values = dataframe[label]
    fig, ax1 = plt.subplots()
    fig.set_dpi(dpi)
     
    ax1.hist(values, edgecolor = 'black', weights=np.ones(len(values)) / len(values), alpha=1, lw=0.5, bins=bins)
    ax1.yaxis.set_major_formatter(tick.PercentFormatter(1))
    ax1.set_xlabel(f'{label} value')
    ax1.set_ylabel('occurrences percentage')
    ax1.grid(axis='y', linestyle='--', linewidth=0.5)
    
    ax2 = ax1.twinx() 
    ax2.hist(values, bins=bins, alpha=0)
    ax2.set_ylabel('occurrences count')
    plt.show()
    
def plot_multiple_histograms(dataframe, labels, xlabel=None, range=None, dpi=100):
    values = dataframe[labels]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.set_dpi(dpi)

    if range is None:
        ax1.hist(values, edgecolor = 'black', lw=0.5, histtype='bar', label=labels, weights=np.ones((len(values), len(labels))) / len(values))
        ax2.hist(values, edgecolor = 'black', lw=0.5, histtype='bar', label=labels, alpha=1)
    else:
        ax1.hist(values, edgecolor = 'black', lw=0.5, histtype='bar', label=labels, weights=np.ones((len(values), len(labels))) / len(values), range=range)
        ax2.hist(values, edgecolor = 'black', lw=0.5, histtype='bar', label=labels, alpha=1, range=range)
    ax1.yaxis.set_major_formatter(tick.PercentFormatter(1))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('occurrences percentage')
    ax2.set_ylabel('occurrences count')
    ax1.grid(axis='y', linestyle='--', linewidth=0.5)
    ax2.grid(axis='y', linestyle='--', linewidth=0.5)
    ax1.legend()
    ax2.legend()
    plt.show()

def get_count(data):
    unique, counts = np.unique(data, return_counts=True)
    return dict(zip(unique, counts))
    
def get_percentage(data):
    data_size = len(data)
    unique, counts = np.unique(data, return_counts=True)
    return dict(zip(unique, np.round(counts*100/data_size, 2)))

def print_count_and_percentage_stats(property, property_label, tail_filter):
    print(f'{property_label} - training set stats')
    print('counts:    ', get_count(property))
    print('percentage:', get_percentage(property))
    print()
    print(f'{property_label} - validation set stats')
    print('counts:    ', get_count(property[-tail_filter:]))
    print('percentage:', get_percentage(property[-tail_filter:]))
    