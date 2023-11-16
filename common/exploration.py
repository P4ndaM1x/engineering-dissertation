import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

def print_basic_stats(dataframe, label, use_scientific_notation = False):
    values = np.asarray(dataframe[label])
    print(f'--- {label} stats ---')
    if use_scientific_notation:
        print(f'min({label}): {np.min(values):.5e}')
        print(f'max({label}): {np.max(values):.5e}')
        print(f'avg({label}): {np.average(values):.5e}')
        print(f'med({label}): {np.median(values):.5e}')
    else:
        print(f'min({label}): {np.min(values)}')
        print(f'max({label}): {np.max(values)}')
        print(f'avg({label}): {np.average(values)}')
        print(f'med({label}): {np.median(values)}')

def plot_histogram(dataframe, label):
    plt.figure()
    values = dataframe[label]
    plt.hist(values, edgecolor = 'black', weights=np.ones(len(values)) / len(values))
    plt.gca().yaxis.set_major_formatter(tick.PercentFormatter(1))
    plt.xlabel(f'{label} values')
    plt.ylabel('occurrences percentage')
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.show()
    
def plot_multiple_histograms(dataframe, labels, range=None):
    plt.figure()
    values = dataframe[labels]
    if range is None:
        plt.hist(values, edgecolor = 'black', histtype='bar', weights=np.ones((len(values), len(labels))) / len(values), label=labels)
    else:
        plt.hist(values, edgecolor = 'black', histtype='bar', weights=np.ones((len(values), len(labels))) / len(values), label=labels, range=range)
    plt.gca().yaxis.set_major_formatter(tick.PercentFormatter(1))
    plt.legend()
    plt.ylabel('occurrences percentage')
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
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
    