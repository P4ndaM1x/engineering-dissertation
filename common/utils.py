import numpy as np
import pandas as pd

def make_dataframe(fit_history):
    df = pd.DataFrame(fit_history.history)
    df.insert(0, 'epoch', fit_history.epoch)
    return df

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w