import numpy as np
import pandas as pd

def make_dataframe(fit_history):
    df = pd.DataFrame(fit_history.history)
    df.insert(0, 'epoch', fit_history.epoch)
    return df

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def validation_set(data, validation_split = 0.2):
    validation_len = int(validation_split*len(data))
    return data[-validation_len-1:]

def train_set(data, validation_split = 0.2):
    train_len = int((1-validation_split)*len(data))
    return data[:train_len]
